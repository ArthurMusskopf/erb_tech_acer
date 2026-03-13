"""
ERB Tech - Tela 2: Boletos / Fila operacional de workflow

Objetivo desta versão:
- Usar faturas_workflow como fonte da verdade da fila operacional
- Exibir KPIs e EDA básica das faturas parseadas
- Mostrar tabela principal com status de validação/cálculo/emissão
- Permitir validar a NF, ajustar info_clientes e calcular a NF selecionada
- Permitir validação em massa da fila filtrada ou selecionada
- Permitir cálculo em massa e individual, salvando em boletos_calculados
- Permitir emissão placeholder em massa e individual
- Exibir memorial e exportar Excel

Observações:
- O cálculo só é habilitado quando:
  1) o cadastro mínimo em info_clientes está OK
  2) a NF está com status_validacao = "validada"
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))

from utils.bigquery_client import (
    execute_query,
    upsert_dataframe,
    TABLE_FATURA_ITENS,
    TABLE_MEDIDORES,
    TABLE_CLIENTES,
    TABLE_FATURAS_WORKFLOW,
    TABLE_BOLETOS,
)
from utils.calc_engine import calculate_boletos, infer_n_fases, compute_custo_disp
from utils.boletos_adapter import calc_to_boletos_schema

st.set_page_config(page_title="Boletos - ERB Tech", page_icon="💰", layout="wide")
st.title("💰 Boletos / Fila operacional")


# -----------------------------------------------------------------------------
# Helpers robustos
# -----------------------------------------------------------------------------
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _num_or_default(x, default: float) -> float:
    v = pd.to_numeric(x, errors="coerce")
    if pd.isna(v):
        return float(default)
    return float(v)


def _int_or_default(x, default: int) -> int:
    v = pd.to_numeric(x, errors="coerce")
    if pd.isna(v):
        return int(default)
    return int(round(float(v), 0))


def _to_iso_date(v: str) -> str:
    v = str(v or "").strip()
    try:
        return datetime.strptime(v, "%d/%m/%Y").strftime("%Y-%m-%d")
    except Exception:
        try:
            return datetime.strptime(v, "%Y-%m-%d").strftime("%Y-%m-%d")
        except Exception:
            return (_now_utc() + timedelta(days=5)).strftime("%Y-%m-%d")


def _status_norm(x: Any) -> str:
    return str(x or "").strip().lower()


def _cadastro_motivo(cli_row: Optional[pd.Series]) -> Optional[str]:
    """
    Regras mínimas para o calc_engine não filtrar quando only_registered_clients=True:
    - desconto_contratado != NULL
    - custo_disp != NULL
    - status == 'Ativo'
    """
    if cli_row is None:
        return "UC não cadastrada em info_clientes"
    if pd.isna(cli_row.get("desconto_contratado")):
        return "desconto_contratado vazio"
    if pd.isna(cli_row.get("custo_disp")):
        return "custo_disp vazio"
    stt = _status_norm(cli_row.get("status"))
    if not stt:
        return "status vazio"
    if stt != "ativo":
        return f"status '{cli_row.get('status')}'"
    return None


def _upsert_workflow_status(
    nf: str,
    *,
    status_parse: Optional[str] = None,
    status_validacao: Optional[str] = None,
    validado_por: Optional[str] = None,
    validado_em: Optional[datetime] = None,
    status_calculo: Optional[str] = None,
    calculado_em: Optional[datetime] = None,
    status_emissao: Optional[str] = None,
    emitido_em: Optional[datetime] = None,
    observacoes_append: Optional[str] = None,
) -> None:
    """
    Atualiza a linha da NF em faturas_workflow sem perder os demais campos.
    """
    if not nf:
        return

    q = f"""
    SELECT *
    FROM `{TABLE_FATURAS_WORKFLOW}`
    WHERE id = @nf
    LIMIT 1
    """
    df_current = execute_query(q, {"nf": str(nf)})
    if df_current is None or df_current.empty:
        return

    row = df_current.iloc[0].to_dict()

    if status_parse is not None:
        row["status_parse"] = status_parse

    if status_validacao is not None:
        row["status_validacao"] = status_validacao
        row["validado_por"] = validado_por
        row["validado_em"] = validado_em

    if status_calculo is not None:
        row["status_calculo"] = status_calculo
        row["calculado_em"] = calculado_em

    if status_emissao is not None:
        row["status_emissao"] = status_emissao
        row["emitido_em"] = emitido_em

    if observacoes_append:
        obs_atual = str(row.get("observacoes") or "").strip()
        if obs_atual:
            row["observacoes"] = f"{obs_atual} || {observacoes_append}"
        else:
            row["observacoes"] = observacoes_append

    row["updated_at"] = _now_utc()
    upsert_dataframe(pd.DataFrame([row]), TABLE_FATURAS_WORKFLOW, key_column="id")


def _upsert_boleto_status(nf: str, status: str) -> None:
    """
    Atualiza status em boletos_calculados sem perder os demais campos.
    """
    if not nf:
        return

    q = f"""
    SELECT *
    FROM `{TABLE_BOLETOS}`
    WHERE id = @nf
    LIMIT 1
    """
    df_current = execute_query(q, {"nf": str(nf)})
    if df_current is None or df_current.empty:
        return

    row = df_current.iloc[0].to_dict()
    row["status"] = status
    row["updated_at"] = _now_utc()
    upsert_dataframe(pd.DataFrame([row]), TABLE_BOLETOS, key_column="id")


def _save_calc_to_bigquery(nf: str, df_calc: pd.DataFrame, df_it: pd.DataFrame) -> None:
    """
    Salva o cálculo normalizado em boletos_calculados e sincroniza o workflow.
    """
    if df_calc is None or df_calc.empty:
        return

    df_bq = calc_to_boletos_schema(df_calc, df_it, status="calculado")
    upsert_dataframe(df_bq, TABLE_BOLETOS, key_column="id")
    _upsert_workflow_status(
        str(nf),
        status_calculo="calculada",
        calculado_em=_now_utc(),
    )


# -----------------------------------------------------------------------------
# BigQuery loads
# -----------------------------------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def load_workflow_queue(limit: int = 1000) -> pd.DataFrame:
    q = f"""
    SELECT
        id,
        nota_fiscal,
        unidade_consumidora,
        cliente_numero,
        nome,
        cnpj_cpf,
        referencia,
        vencimento,
        classe_modalidade,
        grupo_subgrupo_tensao,
        total_pagar,
        arquivo_nome_original,
        arquivo_hash,
        pdf_uri,
        is_inedita,
        duplicada_de,
        status_parse,
        status_validacao,
        validado_por,
        validado_em,
        status_calculo,
        calculado_em,
        status_emissao,
        emitido_em,
        observacoes,
        created_at,
        updated_at
    FROM `{TABLE_FATURAS_WORKFLOW}`
    ORDER BY COALESCE(updated_at, created_at) DESC
    LIMIT {int(limit)}
    """
    return execute_query(q)


@st.cache_data(ttl=60, show_spinner=False)
def load_invoice_data(nf: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    q1 = f"SELECT * FROM `{TABLE_FATURA_ITENS}` WHERE numero = @nf"
    q2 = f"SELECT * FROM `{TABLE_MEDIDORES}` WHERE nota_fiscal_numero = @nf"
    return execute_query(q1, {"nf": str(nf)}), execute_query(q2, {"nf": str(nf)})


@st.cache_data(ttl=60, show_spinner=False)
def load_clientes_for_ucs(ucs: List[str]) -> pd.DataFrame:
    if not ucs:
        return pd.DataFrame()
    q = f"""
    SELECT unidade_consumidora, desconto_contratado, subvencao, status, n_fases, custo_disp
    FROM `{TABLE_CLIENTES}`
    WHERE unidade_consumidora IN UNNEST(@ucs)
    """
    return execute_query(q, {"ucs": ucs})


@st.cache_data(ttl=60, show_spinner=False)
def load_cliente(uc: str) -> pd.DataFrame:
    q = f"SELECT * FROM `{TABLE_CLIENTES}` WHERE unidade_consumidora = @uc LIMIT 1"
    return execute_query(q, {"uc": str(uc)})


@st.cache_data(ttl=60, show_spinner=False)
def load_calculated_queue(limit: int = 1000) -> pd.DataFrame:
    q = f"""
    SELECT
        id,
        nota_fiscal,
        unidade_consumidora,
        cliente_numero,
        nome,
        periodo,
        consumo_kwh,
        injetada_kwh,
        desconto_contratado,
        valor_final,
        status,
        validado_por,
        validado_em,
        created_at,
        updated_at
    FROM `{TABLE_BOLETOS}`
    ORDER BY COALESCE(updated_at, created_at) DESC
    LIMIT {int(limit)}
    """
    return execute_query(q)


# -----------------------------------------------------------------------------
# Estado local de memorial
# -----------------------------------------------------------------------------
if "calc_nf" not in st.session_state:
    st.session_state.calc_nf = {}


# -----------------------------------------------------------------------------
# Fila do workflow
# -----------------------------------------------------------------------------
df_wf = load_workflow_queue()
if df_wf is None or df_wf.empty:
    st.warning("Nenhuma fatura encontrada em faturas_workflow. Faça upload na Tela 1.")
    st.stop()

df_wf = df_wf.copy()
df_wf["id"] = df_wf["id"].astype(str)
df_wf["nota_fiscal"] = df_wf["nota_fiscal"].fillna(df_wf["id"]).astype(str)
df_wf["unidade_consumidora"] = df_wf["unidade_consumidora"].fillna("").astype(str)

ucs = sorted([x for x in df_wf["unidade_consumidora"].dropna().unique().tolist() if str(x).strip()])
df_cli_all = load_clientes_for_ucs(ucs)

df_diag = df_wf.merge(df_cli_all, on="unidade_consumidora", how="left", suffixes=("", "_cli"))


def _motivo_from_row(r: pd.Series) -> str:
    if pd.isna(r.get("desconto_contratado")):
        return "desconto_contratado vazio"
    if pd.isna(r.get("custo_disp")):
        return "custo_disp vazio"
    stt = _status_norm(r.get("status"))
    if not stt:
        return "status vazio"
    if stt != "ativo":
        return f"status '{r.get('status')}'"
    return ""


df_diag["motivo_bloqueio"] = df_diag.apply(_motivo_from_row, axis=1)
df_diag["calculavel"] = df_diag["motivo_bloqueio"].eq("")


# -----------------------------------------------------------------------------
# KPIs e EDA básica do parseado
# -----------------------------------------------------------------------------
total_faturas = len(df_diag)
total_clientes = int(df_diag["cliente_numero"].nunique()) if "cliente_numero" in df_diag.columns else 0
total_ucs = int(df_diag["unidade_consumidora"].nunique()) if "unidade_consumidora" in df_diag.columns else 0
total_validadas = int((df_diag["status_validacao"].fillna("") == "validada").sum())
total_calculadas = int((df_diag["status_calculo"].fillna("") == "calculada").sum())
total_emitidas = int((df_diag["status_emissao"].fillna("") == "emitido").sum())

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Faturas", total_faturas)
m2.metric("Clientes", total_clientes)
m3.metric("UCs", total_ucs)
m4.metric("Validadas", total_validadas)
m5.metric("Calculadas", total_calculadas)
m6.metric("Emitidas", total_emitidas)

with st.expander("📊 EDA básica dos dados parseados"):
    total_pagar_num = pd.to_numeric(df_diag.get("total_pagar"), errors="coerce")
    resumo = {
        "Total de faturas": total_faturas,
        "Total de associados": total_clientes,
        "Total de UCs": total_ucs,
        "Associados com múltiplas UCs": int((df_diag.groupby("cliente_numero")["unidade_consumidora"].nunique() > 1).sum()) if "cliente_numero" in df_diag.columns and "unidade_consumidora" in df_diag.columns else 0,
        "Valor médio da fatura": None if total_pagar_num.dropna().empty else float(total_pagar_num.mean()),
        "Maior valor de fatura": None if total_pagar_num.dropna().empty else float(total_pagar_num.max()),
        "Menor valor de fatura": None if total_pagar_num.dropna().empty else float(total_pagar_num.min()),
        "Faturas calculáveis (cadastro OK)": int(df_diag["calculavel"].sum()),
        "Faturas bloqueadas por cadastro": int((~df_diag["calculavel"]).sum()),
    }
    resumo_df = pd.DataFrame({"indicador": list(resumo.keys()), "valor": list(resumo.values())})
    st.dataframe(resumo_df, hide_index=True, width="stretch")

    if "cliente_numero" in df_diag.columns and "unidade_consumidora" in df_diag.columns:
        top_multiplas = (
            df_diag.groupby(["cliente_numero", "nome"], dropna=False)["unidade_consumidora"]
            .nunique()
            .reset_index(name="qtd_ucs")
            .sort_values("qtd_ucs", ascending=False)
            .head(10)
        )
        st.markdown("**Clientes com mais UCs**")
        st.dataframe(top_multiplas, hide_index=True, width="stretch")


# -----------------------------------------------------------------------------
# Filtros da fila
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("## 📋 Fila operacional")

f1, f2, f3 = st.columns(3)

status_validacao_opts = sorted([x for x in df_diag["status_validacao"].dropna().astype(str).unique().tolist() if x])
status_calculo_opts = sorted([x for x in df_diag["status_calculo"].dropna().astype(str).unique().tolist() if x])
status_emissao_opts = sorted([x for x in df_diag["status_emissao"].dropna().astype(str).unique().tolist() if x])

with f1:
    filtro_validacao = st.multiselect(
        "Filtrar status_validacao",
        options=status_validacao_opts,
        default=[],
    )
with f2:
    filtro_calculo = st.multiselect(
        "Filtrar status_calculo",
        options=status_calculo_opts,
        default=[],
    )
with f3:
    filtro_emissao = st.multiselect(
        "Filtrar status_emissao",
        options=status_emissao_opts,
        default=[],
    )

df_fila = df_diag.copy()
if filtro_validacao:
    df_fila = df_fila[df_fila["status_validacao"].astype(str).isin(filtro_validacao)]
if filtro_calculo:
    df_fila = df_fila[df_fila["status_calculo"].astype(str).isin(filtro_calculo)]
if filtro_emissao:
    df_fila = df_fila[df_fila["status_emissao"].astype(str).isin(filtro_emissao)]

cols_fila = [
    "id",
    "referencia",
    "unidade_consumidora",
    "cliente_numero",
    "nome",
    "total_pagar",
    "status_validacao",
    "status_calculo",
    "status_emissao",
    "calculavel",
    "motivo_bloqueio",
    "arquivo_nome_original",
    "updated_at",
]
cols_fila = [c for c in cols_fila if c in df_fila.columns]

st.dataframe(
    df_fila[cols_fila].sort_values(["referencia", "id"], ascending=[False, False]),
    hide_index=True,
    width="stretch",
)

if df_fila.empty:
    st.info("Nenhuma fatura atende aos filtros selecionados.")
    st.stop()

st.markdown("---")


# -----------------------------------------------------------------------------
# Validação em massa
# -----------------------------------------------------------------------------
st.markdown("## ✅ Validação em massa")

nfs_filtradas = df_fila["id"].dropna().astype(str).tolist()
nfs_calculaveis_filtradas = df_fila[df_fila["calculavel"] == True]["id"].dropna().astype(str).tolist()

nfs_selecionadas = st.multiselect(
    "Selecione NFs para ação em massa",
    options=nfs_filtradas,
    default=[],
)

b1, b2, b3 = st.columns(3)
validar_todas_btn = b1.button(
    f"✅ Validar todas as calculáveis filtradas ({len(nfs_calculaveis_filtradas)})",
    width="stretch",
)
validar_sel_btn = b2.button(
    f"✅ Validar selecionadas ({len(nfs_selecionadas)})",
    width="stretch",
    disabled=(len(nfs_selecionadas) == 0),
)
pendente_sel_btn = b3.button(
    f"⏳ Marcar selecionadas como pendente ({len(nfs_selecionadas)})",
    width="stretch",
    disabled=(len(nfs_selecionadas) == 0),
)

if validar_todas_btn:
    try:
        for nf in nfs_calculaveis_filtradas:
            _upsert_workflow_status(
                str(nf),
                status_validacao="validada",
                validado_por="app_boletos_massa",
                validado_em=_now_utc(),
                observacoes_append="validacao_em_massa_tela_2",
            )
        st.success(f"✅ {len(nfs_calculaveis_filtradas)} NF(s) calculáveis foram validadas.")
        st.cache_data.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Falha na validação em massa: {e}")

if validar_sel_btn:
    try:
        validadas = 0
        puladas = []
        for nf in nfs_selecionadas:
            row_nf = df_fila[df_fila["id"].astype(str) == str(nf)]
            if row_nf.empty:
                continue
            row_nf = row_nf.iloc[0]
            if bool(row_nf.get("calculavel")):
                _upsert_workflow_status(
                    str(nf),
                    status_validacao="validada",
                    validado_por="app_boletos_massa",
                    validado_em=_now_utc(),
                    observacoes_append="validacao_selecionadas_tela_2",
                )
                validadas += 1
            else:
                puladas.append(str(nf))

        if puladas:
            st.warning(f"⚠️ {len(puladas)} NF(s) foram puladas por cadastro incompleto: {', '.join(puladas[:10])}")
        st.success(f"✅ {validadas} NF(s) selecionadas foram validadas.")
        st.cache_data.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Falha ao validar selecionadas: {e}")

if pendente_sel_btn:
    try:
        for nf in nfs_selecionadas:
            _upsert_workflow_status(
                str(nf),
                status_validacao="pendente",
                observacoes_append="retorno_para_pendente_em_massa_tela_2",
            )
        st.info(f"ℹ️ {len(nfs_selecionadas)} NF(s) voltaram para PENDENTE.")
        st.cache_data.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Falha ao atualizar selecionadas: {e}")

st.markdown("---")


# -----------------------------------------------------------------------------
# Cálculo e emissão em massa
# -----------------------------------------------------------------------------
st.markdown("## 🧮 Cálculo e emissão em massa")

nfs_validadas_filtradas = (
    df_fila[
        (df_fila["status_validacao"].astype(str) == "validada")
        & (df_fila["calculavel"] == True)
    ]["id"]
    .dropna()
    .astype(str)
    .tolist()
)

nfs_calculadas_filtradas = (
    df_fila[df_fila["status_calculo"].astype(str) == "calculada"]["id"]
    .dropna()
    .astype(str)
    .tolist()
)

c1, c2, c3 = st.columns(3)
calc_todas_btn = c1.button(
    f"🧮 Calcular todas as validadas filtradas ({len(nfs_validadas_filtradas)})",
    type="primary",
    width="stretch",
)
calc_sel_btn = c2.button(
    f"🧮 Calcular selecionadas ({len(nfs_selecionadas)})",
    width="stretch",
    disabled=(len(nfs_selecionadas) == 0),
)
emitir_sel_btn = c3.button(
    f"🧾 Marcar selecionadas como emitidas ({len(nfs_selecionadas)})",
    width="stretch",
    disabled=(len(nfs_selecionadas) == 0),
)

if calc_todas_btn:
    try:
        ok_count = 0
        erros = []
        for nf in nfs_validadas_filtradas:
            row_nf = df_fila[df_fila["id"].astype(str) == str(nf)].iloc[0]
            uc_nf = str(row_nf["unidade_consumidora"])

            df_it, df_med = load_invoice_data(str(nf))
            df_cli = load_cliente(uc_nf)

            if df_it is None or df_it.empty:
                erro = "NF sem itens em fatura_itens."
                erros.append({"nf": nf, "erro": erro})
                _upsert_workflow_status(str(nf), status_calculo="erro_calculo", calculado_em=_now_utc(), observacoes_append=erro)
                continue

            res = calculate_boletos(df_itens=df_it, df_medidores=df_med, df_clientes=df_cli)
            if res.df_boletos is None or res.df_boletos.empty:
                erro = getattr(res, "missing_reason", "Cálculo vazio")
                erros.append({"nf": nf, "erro": erro})
                _upsert_workflow_status(str(nf), status_calculo="erro_calculo", calculado_em=_now_utc(), observacoes_append=f"erro_calculo_massa: {erro}")
                continue

            _save_calc_to_bigquery(str(nf), res.df_boletos, df_it)
            ok_count += 1

        if erros:
            st.warning(f"⚠️ {len(erros)} NF(s) falharam no cálculo em massa.")
            st.dataframe(pd.DataFrame(erros), hide_index=True, width="stretch")

        st.success(f"✅ {ok_count} NF(s) calculadas e salvas em boletos_calculados.")
        st.cache_data.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Falha no cálculo em massa: {e}")

if calc_sel_btn:
    try:
        ok_count = 0
        erros = []
        for nf in nfs_selecionadas:
            row_nf_df = df_fila[df_fila["id"].astype(str) == str(nf)]
            if row_nf_df.empty:
                continue

            row_nf = row_nf_df.iloc[0]
            if not (str(row_nf.get("status_validacao") or "") == "validada" and bool(row_nf.get("calculavel"))):
                erros.append({"nf": nf, "erro": "NF não validada ou com cadastro incompleto"})
                continue

            uc_nf = str(row_nf["unidade_consumidora"])
            df_it, df_med = load_invoice_data(str(nf))
            df_cli = load_cliente(uc_nf)

            if df_it is None or df_it.empty:
                erro = "NF sem itens em fatura_itens."
                erros.append({"nf": nf, "erro": erro})
                _upsert_workflow_status(str(nf), status_calculo="erro_calculo", calculado_em=_now_utc(), observacoes_append=erro)
                continue

            res = calculate_boletos(df_itens=df_it, df_medidores=df_med, df_clientes=df_cli)
            if res.df_boletos is None or res.df_boletos.empty:
                erro = getattr(res, "missing_reason", "Cálculo vazio")
                erros.append({"nf": nf, "erro": erro})
                _upsert_workflow_status(str(nf), status_calculo="erro_calculo", calculado_em=_now_utc(), observacoes_append=f"erro_calculo_selecionadas: {erro}")
                continue

            _save_calc_to_bigquery(str(nf), res.df_boletos, df_it)
            ok_count += 1

        if erros:
            st.warning(f"⚠️ {len(erros)} NF(s) não foram calculadas.")
            st.dataframe(pd.DataFrame(erros), hide_index=True, width="stretch")

        st.success(f"✅ {ok_count} NF(s) selecionadas foram calculadas e salvas.")
        st.cache_data.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Falha ao calcular selecionadas: {e}")

if emitir_sel_btn:
    try:
        emitidas = 0
        puladas = []
        for nf in nfs_selecionadas:
            row_nf_df = df_fila[df_fila["id"].astype(str) == str(nf)]
            if row_nf_df.empty:
                continue
            row_nf = row_nf_df.iloc[0]

            if str(row_nf.get("status_calculo") or "") != "calculada":
                puladas.append(str(nf))
                continue

            _upsert_workflow_status(
                str(nf),
                status_emissao="emitido",
                emitido_em=_now_utc(),
                observacoes_append="emissao_placeholder_tela_2",
            )
            _upsert_boleto_status(str(nf), "emitido")
            emitidas += 1

        if puladas:
            st.warning(f"⚠️ {len(puladas)} NF(s) foram puladas por ainda não estarem calculadas: {', '.join(puladas[:10])}")

        st.success(f"✅ {emitidas} NF(s) marcadas como emitidas.")
        st.cache_data.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Falha ao marcar emissão: {e}")

st.markdown("---")


# -----------------------------------------------------------------------------
# Base de cálculos salvos
# -----------------------------------------------------------------------------
st.markdown("## 📦 Base de cálculos salvos")

df_calc_base = load_calculated_queue()

if df_calc_base is not None and not df_calc_base.empty:
    valor_final_num = pd.to_numeric(df_calc_base.get("valor_final"), errors="coerce")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Registros calculados", len(df_calc_base))
    d2.metric("Clientes calculados", int(df_calc_base["cliente_numero"].nunique()) if "cliente_numero" in df_calc_base.columns else 0)
    d3.metric("UCs calculadas", int(df_calc_base["unidade_consumidora"].nunique()) if "unidade_consumidora" in df_calc_base.columns else 0)
    d4.metric("Valor total calculado", "R$ " + f"{float(valor_final_num.fillna(0).sum()):,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

    with st.expander("Ver tabela de calculados"):
        st.dataframe(df_calc_base, hide_index=True, width="stretch")
else:
    st.info("Nenhum cálculo salvo em boletos_calculados até o momento.")

st.markdown("---")


# -----------------------------------------------------------------------------
# Seleção NF
# -----------------------------------------------------------------------------
df_fila["label"] = df_fila.apply(
    lambda r: (
        f"{r.get('referencia', '')} | NF {r.get('id', '')} | "
        f"UC {r.get('unidade_consumidora', '')} | "
        f"{str(r.get('nome') or '')[:40]} | "
        f"val={r.get('status_validacao', '')} | calc={r.get('status_calculo', '')}"
    ),
    axis=1,
)

nf_label = st.selectbox("Selecione a NF", options=df_fila["label"].tolist())
row_sel = df_fila[df_fila["label"] == nf_label].iloc[0].to_dict()

nf_sel = str(row_sel["id"])
uc_sel = str(row_sel["unidade_consumidora"])
classe_mod = str(row_sel.get("classe_modalidade") or "")
status_validacao_nf = str(row_sel.get("status_validacao") or "")
status_calculo_nf = str(row_sel.get("status_calculo") or "")
status_emissao_nf = str(row_sel.get("status_emissao") or "")

st.caption(
    f"NF: {nf_sel} | UC: {uc_sel} | Classe/Modalidade: {classe_mod or '-'} | "
    f"Validação: {status_validacao_nf or '-'} | Cálculo: {status_calculo_nf or '-'} | Emissão: {status_emissao_nf or '-'}"
)


# -----------------------------------------------------------------------------
# Cadastro mínimo (info_clientes)
# -----------------------------------------------------------------------------
st.markdown("## 👤 Cadastro mínimo (info_clientes)")

df_cli = load_cliente(uc_sel)
cli_row = df_cli.iloc[0] if df_cli is not None and not df_cli.empty else None
motivo = _cadastro_motivo(cli_row)

if motivo is None:
    st.success("✅ Cadastro mínimo OK (desconto_contratado + custo_disp + status=Ativo).")
else:
    st.error(f"Cadastro insuficiente para cálculo: **{motivo}**")

n_sug = infer_n_fases(classe_mod) or 3
custo_sug = int(compute_custo_disp(n_sug) or 100)

desconto_cur = _num_or_default(cli_row.get("desconto_contratado") if cli_row is not None else None, 0.15)
subv_cur = _num_or_default(cli_row.get("subvencao") if cli_row is not None else None, 0.0)
status_cur = str((cli_row.get("status") if cli_row is not None else None) or "Ativo")
n_cur = _int_or_default(cli_row.get("n_fases") if cli_row is not None else None, int(n_sug))
custo_cur = _int_or_default(cli_row.get("custo_disp") if cli_row is not None else None, int(custo_sug))

with st.form("form_cadastro_minimo"):
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    with c1:
        desconto = st.number_input(
            "desconto_contratado",
            min_value=0.0,
            max_value=1.0,
            value=float(desconto_cur),
            step=0.01,
        )
    with c2:
        subvencao = st.number_input("subvencao", min_value=0.0, value=float(subv_cur), step=50.0)
    with c3:
        n_fases = st.selectbox("n_fases (opcional)", options=[1, 2, 3], index=[1, 2, 3].index(int(n_cur)))
    with c4:
        custo_disp = st.number_input("custo_disp (kWh)", min_value=0, value=int(custo_cur), step=10)
    with c5:
        status = st.selectbox(
            "status",
            options=["Ativo", "Inativo"],
            index=0 if _status_norm(status_cur) == "ativo" else 1,
        )

    salvar = st.form_submit_button("💾 Salvar/Atualizar info_clientes")

if salvar:
    payload = {
        "unidade_consumidora": uc_sel,
        "desconto_contratado": float(desconto),
        "subvencao": float(subvencao),
        "status": str(status),
        "n_fases": int(n_fases),
        "custo_disp": int(custo_disp),
        "updated_at": _now_utc(),
    }
    try:
        upsert_dataframe(pd.DataFrame([payload]), TABLE_CLIENTES, key_column="unidade_consumidora")
        _upsert_workflow_status(
            str(nf_sel),
            status_validacao="pendente",
            observacoes_append="cadastro_info_clientes_atualizado_na_tela_2",
        )
        st.success("✅ Cadastro salvo. Recarregando…")
        st.cache_data.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Falha ao salvar info_clientes: {e}")

st.markdown("---")


# -----------------------------------------------------------------------------
# Ações da fatura
# -----------------------------------------------------------------------------
st.markdown("## ✅ Ações da fatura")

a1, a2, a3 = st.columns([1, 1, 1])
validar_btn = a1.button("✅ Validar dados desta NF", width="stretch")
manter_pendente_btn = a2.button("⏳ Manter como pendente", width="stretch")
emitir_btn = a3.button("🧾 Marcar esta NF como emitida", width="stretch", disabled=(status_calculo_nf != "calculada"))

if validar_btn:
    try:
        _upsert_workflow_status(
            str(nf_sel),
            status_validacao="validada",
            validado_por="app_boletos",
            validado_em=_now_utc(),
            observacoes_append="dados_validados_na_tela_2",
        )
        st.success(f"✅ NF {nf_sel} marcada como VALIDADA no workflow.")
        st.cache_data.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Falha ao validar NF no workflow: {e}")

if manter_pendente_btn:
    try:
        _upsert_workflow_status(
            str(nf_sel),
            status_validacao="pendente",
            observacoes_append="mantida_pendente_na_tela_2",
        )
        st.info(f"ℹ️ NF {nf_sel} mantida como PENDENTE.")
        st.cache_data.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Falha ao atualizar workflow: {e}")

if emitir_btn:
    try:
        _upsert_workflow_status(
            str(nf_sel),
            status_emissao="emitido",
            emitido_em=_now_utc(),
            observacoes_append="emissao_placeholder_individual_tela_2",
        )
        _upsert_boleto_status(str(nf_sel), "emitido")
        st.success(f"✅ NF {nf_sel} marcada como EMITIDA.")
        st.cache_data.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Falha ao marcar emissão: {e}")

st.markdown("---")


# -----------------------------------------------------------------------------
# Cálculo NF selecionada
# -----------------------------------------------------------------------------
st.markdown("## 🧮 Cálculo da NF selecionada")

calc_disabled = (_cadastro_motivo(cli_row) is not None) or (status_validacao_nf != "validada")
calc_btn = st.button("Calcular esta NF", type="primary", width="stretch", disabled=calc_disabled)

if _cadastro_motivo(cli_row) is not None:
    st.info("Preencha e salve o cadastro mínimo acima para habilitar o cálculo.")
elif status_validacao_nf != "validada":
    st.info("Valide a NF antes de calcular. O cálculo nesta etapa é liberado apenas para faturas validadas.")

if calc_btn:
    df_it, df_med = load_invoice_data(nf_sel)
    if df_it is None or df_it.empty:
        st.error("NF sem itens em fatura_itens.")
    else:
        df_cli2 = load_cliente(uc_sel)
        if df_cli2 is None or df_cli2.empty:
            st.error("UC não encontrada em info_clientes após salvar.")
        else:
            with st.spinner("Calculando (fiel ao Excel)..."):
                res = calculate_boletos(df_itens=df_it, df_medidores=df_med, df_clientes=df_cli2)

            if res.df_boletos is None or res.df_boletos.empty:
                _upsert_workflow_status(
                    str(nf_sel),
                    status_calculo="erro_calculo",
                    calculado_em=_now_utc(),
                    observacoes_append=f"erro_calculo_tela_2: {getattr(res, 'missing_reason', 'vazio')}",
                )
                if getattr(res, "missing_clientes", None):
                    st.error(f"Cálculo retornou vazio. Filtrados: {res.missing_clientes} | {res.missing_reason}")
                else:
                    st.error("Cálculo retornou vazio. Investigar parse/medidores/itens.")
            else:
                st.session_state.calc_nf[nf_sel] = res.df_boletos.copy()
                _save_calc_to_bigquery(str(nf_sel), res.df_boletos, df_it)
                st.success("✅ Cálculo concluído e salvo em boletos_calculados.")
                st.cache_data.clear()

df_calc = st.session_state.calc_nf.get(nf_sel)
if df_calc is None or df_calc.empty:
    st.stop()

st.markdown("### 🧾 Memorial (tabela)")
st.dataframe(df_calc, hide_index=True, width="stretch")

buf = io.BytesIO()
df_calc.to_excel(buf, index=False)
buf.seek(0)
st.download_button(
    "⬇️ Exportar memorial (Excel)",
    data=buf,
    file_name=f"memorial_nf_{nf_sel}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    width="stretch",
)
