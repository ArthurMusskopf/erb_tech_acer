"""
ERB Tech - Tela 2: Boletos (por Fatura / NF)

Objetivo:
- Listar NFs disponíveis no BigQuery
- Destacar NFs que não podem ser calculadas por falta de cadastro mínimo (info_clientes)
- Permitir preencher e salvar manualmente os campos necessários
- Calcular a NF selecionada e exibir memorial/exportar

Correções:
- Evita ValueError: cannot convert float NaN to integer (n_fases / custo_disp)
- Não depende de n_fases para calcular (apenas ajuda sugestão de custo_disp)
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta
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
)

from utils.calc_engine import calculate_boletos, infer_n_fases, compute_custo_disp

st.set_page_config(page_title="Boletos - ERB Tech", page_icon="💰", layout="wide")
st.title("💰 Boletos (por Fatura / NF)")


# -----------------------------------------------------------------------------
# Helpers robustos (NAN-safe)
# -----------------------------------------------------------------------------
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
            return (datetime.utcnow() + timedelta(days=5)).strftime("%Y-%m-%d")


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


# -----------------------------------------------------------------------------
# BigQuery loads
# -----------------------------------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def load_nf_list(limit: int = 400) -> pd.DataFrame:
    q = f"""
    SELECT
      numero,
      referencia,
      unidade_consumidora,
      ANY_VALUE(nome) AS nome,
      ANY_VALUE(vencimento) AS vencimento,
      ANY_VALUE(classe_modalidade) AS classe_modalidade
    FROM `{TABLE_FATURA_ITENS}`
    WHERE numero IS NOT NULL
    GROUP BY numero, referencia, unidade_consumidora
    ORDER BY referencia DESC, numero DESC
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


# -----------------------------------------------------------------------------
# UI topo: lista de NFs + bloqueios por cadastro
# -----------------------------------------------------------------------------
df_nfs = load_nf_list()
if df_nfs is None or df_nfs.empty:
    st.warning("Nenhuma NF encontrada em BigQuery. Faça upload na aba 'Upload Faturas'.")
    st.stop()

df_nfs = df_nfs.copy()
df_nfs["numero"] = df_nfs["numero"].astype(str)
df_nfs["unidade_consumidora"] = df_nfs["unidade_consumidora"].astype(str)

ucs = sorted(df_nfs["unidade_consumidora"].dropna().unique().tolist())
df_cli_all = load_clientes_for_ucs(ucs)

df_diag = df_nfs.merge(df_cli_all, on="unidade_consumidora", how="left", suffixes=("", "_cli"))

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

bloq = df_diag[~df_diag["calculavel"]].copy()
ok = df_diag[df_diag["calculavel"]].copy()

colA, colB, colC = st.columns([1, 1, 2])
colA.metric("Total NFs", len(df_diag))
colB.metric("Calculáveis", len(ok))
colC.metric("Bloqueadas (cadastro)", len(bloq))

if len(bloq) > 0:
    st.warning("⚠️ Existem faturas bloqueadas por falta de cadastro mínimo em info_clientes.")
    with st.expander("Ver NFs bloqueadas e motivo"):
        st.dataframe(
            bloq[["referencia", "numero", "unidade_consumidora", "nome", "motivo_bloqueio"]]
            .sort_values(["referencia", "numero"], ascending=[False, False]),
            hide_index=True,
            width="stretch",
        )
else:
    st.success("✅ Todas as faturas listadas têm cadastro mínimo para cálculo.")

st.markdown("---")


# -----------------------------------------------------------------------------
# Seleção NF
# -----------------------------------------------------------------------------
df_diag["label"] = df_diag.apply(
    lambda r: f"{r['referencia']} | NF {r['numero']} | UC {r['unidade_consumidora']} | {str(r.get('nome') or '')[:40]}",
    axis=1,
)

nf_label = st.selectbox("Selecione a NF", options=df_diag["label"].tolist())
row_sel = df_diag[df_diag["label"] == nf_label].iloc[0].to_dict()

nf_sel = str(row_sel["numero"])
uc_sel = str(row_sel["unidade_consumidora"])
classe_mod = str(row_sel.get("classe_modalidade") or "")

st.caption(f"NF: {nf_sel} | UC: {uc_sel} | Classe/Modalidade: {classe_mod or '-'}")


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

# sugestões para ajudar preenchimento (não obrigatórias)
n_sug = infer_n_fases(classe_mod) or 3
custo_sug = int(compute_custo_disp(n_sug) or 100)

# valores atuais (NAN-safe)
desconto_cur = _num_or_default(cli_row.get("desconto_contratado") if cli_row is not None else None, 0.15)
subv_cur = _num_or_default(cli_row.get("subvencao") if cli_row is not None else None, 0.0)
status_cur = str((cli_row.get("status") if cli_row is not None else None) or "Ativo")

n_cur = _int_or_default(cli_row.get("n_fases") if cli_row is not None else None, int(n_sug))
custo_cur = _int_or_default(cli_row.get("custo_disp") if cli_row is not None else None, int(custo_sug))

with st.form("form_cadastro_minimo"):
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    with c1:
        desconto = st.number_input("desconto_contratado", min_value=0.0, max_value=1.0, value=float(desconto_cur), step=0.01)
    with c2:
        subvencao = st.number_input("subvencao", min_value=0.0, value=float(subv_cur), step=50.0)
    with c3:
        n_fases = st.selectbox("n_fases (opcional)", options=[1, 2, 3], index=[1, 2, 3].index(int(n_cur)))
    with c4:
        custo_disp = st.number_input("custo_disp (kWh)", min_value=0, value=int(custo_cur), step=10)
    with c5:
        status = st.selectbox("status", options=["Ativo", "Inativo"], index=0 if _status_norm(status_cur) == "ativo" else 1)

    salvar = st.form_submit_button("💾 Salvar/Atualizar info_clientes")

if salvar:
    payload = {
        "unidade_consumidora": uc_sel,
        "desconto_contratado": float(desconto),
        "subvencao": float(subvencao),
        "status": str(status),
        "n_fases": int(n_fases),
        "custo_disp": int(custo_disp),
        "updated_at": datetime.utcnow(),
    }
    try:
        upsert_dataframe(pd.DataFrame([payload]), TABLE_CLIENTES, key_column="unidade_consumidora")
        st.success("✅ Cadastro salvo. Recarregando…")
        st.cache_data.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Falha ao salvar info_clientes: {e}")

st.markdown("---")


# -----------------------------------------------------------------------------
# Cálculo NF selecionada
# -----------------------------------------------------------------------------
st.markdown("## 🧮 Cálculo da NF selecionada")

calc_disabled = (_cadastro_motivo(cli_row) is not None)
calc_btn = st.button("Calcular esta NF", type="primary", width="stretch", disabled=calc_disabled)

if calc_disabled:
    st.info("Preencha e salve o cadastro mínimo acima para habilitar o cálculo.")

if "calc_nf" not in st.session_state:
    st.session_state.calc_nf = {}

if calc_btn:
    df_it, df_med = load_invoice_data(nf_sel)
    if df_it is None or df_it.empty:
        st.error("NF sem itens em fatura_itens.")
    else:
        # recarrega cadastro atualizado
        df_cli2 = load_cliente(uc_sel)
        if df_cli2 is None or df_cli2.empty:
            st.error("UC não encontrada em info_clientes após salvar.")
        else:
            with st.spinner("Calculando (fiel ao Excel)..."):
                res = calculate_boletos(df_itens=df_it, df_medidores=df_med, df_clientes=df_cli2)
            if res.df_boletos is None or res.df_boletos.empty:
                if getattr(res, "missing_clientes", None):
                    st.error(f"Cálculo retornou vazio. Filtrados: {res.missing_clientes} | {res.missing_reason}")
                else:
                    st.error("Cálculo retornou vazio. Investigar parse/medidores/itens.")
            else:
                st.session_state.calc_nf[nf_sel] = res.df_boletos.copy()
                st.success("✅ Cálculo concluído.")

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
