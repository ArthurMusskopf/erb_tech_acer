"""
ERB Tech - Tela 2: Boletos (lote + correção de cadastro info_clientes)

Objetivo:
- Após upload, calcular todas as NFs possíveis
- Destacar NFs que não puderem ser calculadas (cadastro incompleto ou outros)
- Permitir preencher e salvar (BigQuery) os campos necessários em info_clientes
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))

import io
from utils.bigquery_client import (
    execute_query,
    upsert_dataframe,
    TABLE_FATURA_ITENS,
    TABLE_MEDIDORES,
    TABLE_CLIENTES,
    TABLE_BOLETOS,
)
from utils.boletos_adapter import calc_to_boletos_schema

from utils.calc_engine import calculate_boletos, infer_n_fases, compute_custo_disp


st.set_page_config(page_title="Boletos - ERB Tech", page_icon="💰", layout="wide")
st.title("💰 Boletos — Cálculo em Lote + Cadastro (info_clientes)")


# --------------------------------------------------------------------------------------
# Queries
# --------------------------------------------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def load_nf_list(limit: int = 300) -> pd.DataFrame:
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
    SELECT
      unidade_consumidora,
      desconto_contratado,
      subvencao,
      status,
      n_fases,
      custo_disp
    FROM `{TABLE_CLIENTES}`
    WHERE unidade_consumidora IN UNNEST(@ucs)
    """
    return execute_query(q, {"ucs": ucs})


@st.cache_data(ttl=60, show_spinner=False)
def load_cliente(uc: str) -> pd.DataFrame:
    q = f"SELECT * FROM `{TABLE_CLIENTES}` WHERE unidade_consumidora = @uc LIMIT 1"
    return execute_query(q, {"uc": str(uc)})


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _status_norm(x: Any) -> str:
    return str(x or "").strip().lower()


def _cadastro_motivo(cli_row: Optional[pd.Series]) -> Optional[str]:
    """
    Regras mínimas para o calc_engine não filtrar (only_registered_clients=True):
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


def _render_calc_list(row: Dict[str, Any]) -> None:
    v = float(pd.to_numeric(row.get("valor_total_boleto"), errors="coerce") or 0.0)
    kwh = float(pd.to_numeric(row.get("med_inj_tusd"), errors="coerce") or 0.0)
    desc = float(pd.to_numeric(row.get("desconto_contratado"), errors="coerce") or 0.0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Valor (R$)", f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    col2.metric("Base (kWh)", f"{kwh:,.0f}".replace(",", "X").replace(".", ",").replace("X", "."))
    col3.metric("Desconto", f"{desc*100:.2f}%".replace(".", ","))
    col4.metric("Check", str(row.get("check") or "-"))

    st.markdown("**Energia**")
    st.markdown(
        "\n".join([
            f"- custo_disp: {row.get('custo_disp')}",
            f"- medidores_apurado: {row.get('medidores_apurado')}",
            f"- injetada: {row.get('injetada')}",
            f"- boleto: {row.get('boleto')} | gerador: {row.get('gerador')}",
        ])
    )

    st.markdown("**Tarifas (por kWh)**")
    st.markdown(
        "\n".join([
            f"- tarifa_cheia_trib2 (TE+TUSD ajustada): {row.get('tarifa_cheia_trib2')}",
            f"- tarifa_cheia (inj TE+TUSD): {row.get('tarifa_cheia')}",
            f"- tarifa_total_boleto: {row.get('tarifa_total_boleto')}",
        ])
    )

    with st.expander("JSON completo (debug)"):
        st.json(row)


# --------------------------------------------------------------------------------------
# Topo: lista + diagnóstico
# --------------------------------------------------------------------------------------
df_nfs = load_nf_list()
if df_nfs is None or df_nfs.empty:
    st.warning("Nenhuma NF encontrada. Faça upload na Tela 1.")
    st.stop()

df_nfs = df_nfs.copy()
df_nfs["numero"] = df_nfs["numero"].astype(str)
df_nfs["unidade_consumidora"] = df_nfs["unidade_consumidora"].astype(str)

ucs = sorted(df_nfs["unidade_consumidora"].dropna().unique().tolist())
df_cli_all = load_clientes_for_ucs(ucs)

# merge para montar status de cálculo por NF
df_diag = df_nfs.merge(df_cli_all, on="unidade_consumidora", how="left", suffixes=("", "_cli"))

def _motivo_row(r: pd.Series) -> str:
    if pd.isna(r.get("desconto_contratado")) and pd.isna(r.get("custo_disp")) and (pd.isna(r.get("status"))):
        return "UC não cadastrada (ou sem campos mínimos)"
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

df_diag["motivo_bloqueio"] = df_diag.apply(_motivo_row, axis=1)
df_diag["calculavel"] = df_diag["motivo_bloqueio"].eq("")

bloq = df_diag[~df_diag["calculavel"]].copy()
ok = df_diag[df_diag["calculavel"]].copy()

colA, colB, colC = st.columns([1, 1, 2])
colA.metric("Total de faturas (NFs)", len(df_diag))
colB.metric("Calculáveis", len(ok))
colC.metric("Bloqueadas (cadastro)", len(bloq))

if len(bloq) > 0:
    st.warning("⚠️ Existem faturas que não podem ser calculadas por cadastro incompleto em info_clientes.")
    with st.expander("Ver faturas bloqueadas e motivo"):
        st.dataframe(
            bloq[["referencia", "numero", "unidade_consumidora", "nome", "motivo_bloqueio"]]
            .sort_values(["referencia", "numero"], ascending=[False, False]),
            hide_index=True,
            width="stretch",
        )
else:
    st.success("✅ Todas as faturas listadas possuem cadastro mínimo para cálculo (info_clientes).")

st.markdown("---")


# --------------------------------------------------------------------------------------
# Seleção NF
# --------------------------------------------------------------------------------------
df_diag["label"] = df_diag.apply(
    lambda r: f"{r['referencia']} | NF {r['numero']} | UC {r['unidade_consumidora']} | {str(r.get('nome') or '')[:40]}",
    axis=1,
)

nf_label = st.selectbox("Selecione a NF para conferência/cálculo", options=df_diag["label"].tolist())
row_sel = df_diag[df_diag["label"] == nf_label].iloc[0].to_dict()

nf_sel = str(row_sel["numero"])
uc_sel = str(row_sel["unidade_consumidora"])
classe_mod = str(row_sel.get("classe_modalidade") or "")

st.caption(f"NF: {nf_sel} | UC: {uc_sel} | Classe/Modalidade: {classe_mod or '-'}")


# --------------------------------------------------------------------------------------
# Cadastro: preencher e salvar no BigQuery
# --------------------------------------------------------------------------------------
st.markdown("## 👤 Cadastro mínimo (info_clientes)")

df_cli_one = load_cliente(uc_sel)
cli_row = df_cli_one.iloc[0] if df_cli_one is not None and not df_cli_one.empty else None
motivo = _cadastro_motivo(cli_row)

if motivo is None:
    st.success("✅ Cadastro mínimo OK (desconto_contratado + custo_disp + status=Ativo).")
else:
    st.error(f"Cadastro insuficiente para cálculo: **{motivo}**")

# Sugestões automáticas (p/ ajudar)
n_sug = infer_n_fases(classe_mod) or 3
custo_sug = int(compute_custo_disp(n_sug) or 100)

desconto_cur = float(pd.to_numeric(cli_row.get("desconto_contratado"), errors="coerce") or 0.15) if cli_row is not None else 0.15
subv_cur = float(pd.to_numeric(cli_row.get("subvencao"), errors="coerce") or 0.0) if cli_row is not None else 0.0
status_cur = str(cli_row.get("status") or "Ativo") if cli_row is not None else "Ativo"
n_cur = int(pd.to_numeric(cli_row.get("n_fases"), errors="coerce") or n_sug) if cli_row is not None else n_sug
custo_cur = int(pd.to_numeric(cli_row.get("custo_disp"), errors="coerce") or custo_sug) if cli_row is not None else custo_sug

with st.form("form_cadastro_minimo"):
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    with c1:
        desconto = st.number_input("desconto_contratado", min_value=0.0, max_value=1.0, value=float(desconto_cur), step=0.01)
    with c2:
        subvencao = st.number_input("subvencao", min_value=0.0, value=float(subv_cur), step=50.0)
    with c3:
        n_fases = st.selectbox("n_fases", options=[1, 2, 3], index=[1, 2, 3].index(int(n_cur)))
    with c4:
        custo_disp = st.number_input("custo_disp (kWh)", min_value=0, value=int(custo_cur), step=10)
    with c5:
        status = st.selectbox("status", options=["Ativo", "Inativo"], index=0 if _status_norm(status_cur) == "ativo" else 1)

    salvar = st.form_submit_button("💾 Salvar no BigQuery (info_clientes)")

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
        st.success("✅ Cadastro salvo. Recarregando para habilitar cálculo…")
        st.cache_data.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Falha ao salvar info_clientes: {e}")


st.markdown("---")


# --------------------------------------------------------------------------------------
# Cálculo individual
# --------------------------------------------------------------------------------------
st.markdown("## 🧮 Cálculo da NF selecionada")

calc_disabled = (_cadastro_motivo(cli_row) is not None)
calc_btn = st.button("Calcular esta NF", type="primary", width="stretch", disabled=calc_disabled)

if calc_disabled:
    st.info("Preencha e salve o cadastro mínimo acima para habilitar o cálculo.")

if "calc_result_nf" not in st.session_state:
    st.session_state.calc_result_nf = {}

if calc_btn:
    df_it, df_med = load_invoice_data(nf_sel)

    # recarrega cadastro após salvar
    df_cli_one = load_cliente(uc_sel)
    if df_cli_one is None or df_cli_one.empty:
        st.error("UC não encontrada em info_clientes mesmo após salvar.")
    else:
        try:
            res = calculate_boletos(df_itens=df_it, df_medidores=df_med, df_clientes=df_cli_one)
            if res.df_boletos is None or res.df_boletos.empty:
                # motivo explícito (do próprio motor)
                if res.missing_clientes:
                    st.error(f"Cálculo retornou vazio. UCs filtradas: {res.missing_clientes} | motivo: {res.missing_reason}")
                else:
                    st.error("Cálculo retornou vazio (sem missing_clientes). Investigar parse/medidores/itens.")
            else:
                st.session_state.calc_result_nf[nf_sel] = res.df_boletos.copy()
                st.success("✅ Cálculo concluído.")
        except Exception as e:
            st.error(f"Erro ao calcular: {e}")

df_calc_nf = st.session_state.calc_result_nf.get(nf_sel)
if df_calc_nf is not None and not df_calc_nf.empty:
    row = df_calc_nf.iloc[0].to_dict()
    _render_calc_list(row)

save_one = st.button("💾 Salvar cálculo desta NF em boletos_calculados", width="stretch")
if save_one:
    try:
        df_it, _ = load_invoice_data(nf_sel)
        df_bq = calc_to_boletos_schema(df_calc_nf, df_it, status="calculado")
        upsert_dataframe(df_bq, TABLE_BOLETOS, key_column="id")
        st.success("✅ Salvo em boletos_calculados.")
    except Exception as e:
        st.error(f"Erro ao salvar cálculo: {e}")
        
    buf = io.BytesIO()
    df_calc_nf.to_excel(buf, index=False)
    buf.seek(0)
    st.download_button(
        "⬇️ Exportar memorial (Excel)",
        data=buf,
        file_name=f"memorial_nf_{nf_sel}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )

st.markdown("---")


# --------------------------------------------------------------------------------------
# Cálculo em lote (todas as NFs calculáveis)
# --------------------------------------------------------------------------------------
st.markdown("## 🚀 Cálculo em lote (todas as NFs calculáveis)")

max_calc = st.number_input("Máximo de NFs para calcular no lote", min_value=1, max_value=100, value=20, step=1)
run_batch = st.button("Calcular lote (somente as calculáveis)", width="stretch", disabled=(len(ok) == 0))

if "batch_results" not in st.session_state:
    st.session_state.batch_results = pd.DataFrame()
if "batch_failures" not in st.session_state:
    st.session_state.batch_failures = pd.DataFrame()

if run_batch:
    rows_ok = ok.head(int(max_calc)).to_dict("records")
    out_rows = []
    fail_rows = []

    with st.spinner(f"Calculando {len(rows_ok)} NF(s)…"):
        for rr in rows_ok:
            nf = str(rr["numero"])
            uc = str(rr["unidade_consumidora"])

            try:
                df_it, df_med = load_invoice_data(nf)
                df_cli = load_cliente(uc)
                res = calculate_boletos(df_itens=df_it, df_medidores=df_med, df_clientes=df_cli)

                if res.df_boletos is None or res.df_boletos.empty:
                    fail_rows.append({
                        "numero": nf,
                        "unidade_consumidora": uc,
                        "motivo": f"vazio | missing={res.missing_clientes} | reason={res.missing_reason}",
                    })
                else:
                    r0 = res.df_boletos.iloc[0].to_dict()
                    out_rows.append({
                        "referencia": rr.get("referencia"),
                        "numero": nf,
                        "unidade_consumidora": uc,
                        "nome": rr.get("nome"),
                        "valor_total_boleto": float(pd.to_numeric(r0.get("valor_total_boleto"), errors="coerce") or 0.0),
                        "med_inj_tusd": float(pd.to_numeric(r0.get("med_inj_tusd"), errors="coerce") or 0.0),
                        "check": r0.get("check"),
                    })

            except Exception as e:
                fail_rows.append({
                    "numero": nf,
                    "unidade_consumidora": uc,
                    "motivo": f"erro: {e}",
                })

    st.session_state.batch_results = pd.DataFrame(out_rows)
    st.session_state.batch_failures = pd.DataFrame(fail_rows)

df_batch = st.session_state.batch_results
df_fail = st.session_state.batch_failures

if df_batch is not None and not df_batch.empty:
    st.success(f"✅ Lote calculado: {len(df_batch)} NF(s)")
    st.dataframe(df_batch.sort_values(["referencia", "numero"], ascending=[False, False]), hide_index=True, width="stretch")

if df_fail is not None and not df_fail.empty:
    st.warning(f"⚠️ Falhas no lote: {len(df_fail)} NF(s)")
    st.dataframe(df_fail, hide_index=True, width="stretch")
