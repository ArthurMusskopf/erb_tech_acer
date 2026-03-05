"""
ERB Tech - Tela 2: Boletos (por Fatura / NF)

Correção v1 (cadastro incompleto):
- Mostra aviso no topo com NFs que não podem ser calculadas (e motivo)
- Permite preencher os dados faltantes do associado (info_clientes) diretamente na tela
- Após salvar, recalcula normalmente usando o calc_engine

Motivo da correção:
- calc_engine filtra fora UCs com custo_disp/desconto_contratado nulos quando only_registered_clients=True
"""

from __future__ import annotations

import io
import re
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))

from utils.bigquery_client import (
    execute_query,
    upsert_dataframe,
    TABLE_FATURA_ITENS,
    TABLE_MEDIDORES,
    TABLE_CLIENTES,
    TABLE_BOLETOS,
)

from utils.calc_engine import calculate_boletos, infer_n_fases, compute_custo_disp
from utils.sicoob_client import (
    get_sicoob_config_from_secrets,
    SicoobCobrancaV3Client,
    build_boleto_payload_from_row,
)

st.set_page_config(page_title="Boletos - ERB Tech", page_icon="💰", layout="wide")
st.title("💰 Boletos (por Fatura / NF)")


# -----------------------------------------------------------------------------
# Queries / loads
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
      ANY_VALUE(total_pagar) AS total_pagar
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
    SELECT *
    FROM `{TABLE_CLIENTES}`
    WHERE unidade_consumidora IN UNNEST(@ucs)
    """
    return execute_query(q, {"ucs": ucs})


@st.cache_data(ttl=60, show_spinner=False)
def load_cliente(uc: str) -> pd.DataFrame:
    q = f"SELECT * FROM `{TABLE_CLIENTES}` WHERE unidade_consumidora = @uc LIMIT 1"
    return execute_query(q, {"uc": str(uc)})


def _to_iso_date(v: str) -> str:
    v = str(v or "").strip()
    try:
        return datetime.strptime(v, "%d/%m/%Y").strftime("%Y-%m-%d")
    except Exception:
        try:
            return datetime.strptime(v, "%Y-%m-%d").strftime("%Y-%m-%d")
        except Exception:
            return (datetime.utcnow() + timedelta(days=5)).strftime("%Y-%m-%d")


def _norm_status(x: Any) -> str:
    return str(x or "").strip().lower()


def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df2 = df.copy() if df is not None else pd.DataFrame()
    for c in cols:
        if c not in df2.columns:
            df2[c] = None
    return df2


def _calc_block_reason(cli_row: Optional[pd.Series]) -> Optional[str]:
    """
    Regras mínimas para o calc_engine não filtrar:
    - precisa existir registro
    - desconto_contratado não pode ser NULL
    - custo_disp não pode ser NULL
    - status precisa ser 'Ativo' (case-insensitive)
    """
    if cli_row is None:
        return "UC não cadastrada em info_clientes"

    # campos críticos (o calc_engine filtra se custo_disp/desconto estiverem nulos)
    if pd.isna(cli_row.get("desconto_contratado")):
        return "desconto_contratado vazio"
    if pd.isna(cli_row.get("custo_disp")):
        return "custo_disp vazio"

    stt = _norm_status(cli_row.get("status"))
    if stt and stt != "ativo":
        return f"status '{cli_row.get('status')}'"

    # se status estiver NULL, a gente também bloqueia para evitar resultado estranho
    if not stt:
        return "status vazio"

    return None


def _build_warning_table(df_nfs: pd.DataFrame, df_cli_all: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna df com colunas: numero, referencia, uc, nome, can_calc, motivo
    """
    if df_nfs is None or df_nfs.empty:
        return pd.DataFrame()

    df_cli_all = df_cli_all.copy() if df_cli_all is not None else pd.DataFrame()
    if not df_cli_all.empty and "unidade_consumidora" in df_cli_all.columns:
        df_cli_all["unidade_consumidora"] = df_cli_all["unidade_consumidora"].astype(str)

    rows = []
    for _, r in df_nfs.iterrows():
        nf = str(r.get("numero"))
        uc = str(r.get("unidade_consumidora"))
        ref = str(r.get("referencia"))
        nome = str(r.get("nome") or "")

        cli_row = None
        if not df_cli_all.empty and "unidade_consumidora" in df_cli_all.columns:
            hit = df_cli_all[df_cli_all["unidade_consumidora"] == uc]
            if not hit.empty:
                cli_row = hit.iloc[0]

        motivo = _calc_block_reason(cli_row)
        rows.append({
            "numero": nf,
            "referencia": ref,
            "unidade_consumidora": uc,
            "nome": nome,
            "can_calc": motivo is None,
            "motivo": "" if motivo is None else motivo,
        })

    return pd.DataFrame(rows)


def _render_calc_summary(row: Dict[str, Any]) -> None:
    """
    Exibe um resumo em lista (sem explodir colunas em tabela).
    """
    st.markdown("### ✅ Resultado do cálculo (Resumo)")
    col1, col2, col3, col4 = st.columns(4)
    v = float(pd.to_numeric(row.get("valor_total_boleto"), errors="coerce") or 0.0)
    kwh = float(pd.to_numeric(row.get("med_inj_tusd"), errors="coerce") or 0.0)
    desc = float(pd.to_numeric(row.get("desconto_contratado"), errors="coerce") or 0.0)

    col1.metric("Valor do boleto (R$)", f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    col2.metric("Base (kWh)", f"{kwh:,.0f}".replace(",", "X").replace(".", ",").replace("X", "."))
    col3.metric("Desconto", f"{desc*100:.2f}%".replace(".", ","))
    col4.metric("Check", str(row.get("check") or "-"))

    st.markdown("#### ⚡ Energia")
    st.markdown(
        "\n".join([
            f"- **custo_disp (kWh):** {row.get('custo_disp')}",
            f"- **medidores_apurado (kWh):** {row.get('medidores_apurado')}",
            f"- **injetada (kWh):** {row.get('injetada')}",
            f"- **boleto:** {row.get('boleto')} | **gerador:** {row.get('gerador')}",
        ])
    )

    st.markdown("#### 💳 Tarifas (por kWh)")
    st.markdown(
        "\n".join([
            f"- **tarifa_cheia_trib2 (TE+TUSD ajustada):** {row.get('tarifa_cheia_trib2')}",
            f"- **tarifa_cheia (injetada TE+TUSD):** {row.get('tarifa_cheia')}",
            f"- **tarifa_total_boleto:** {row.get('tarifa_total_boleto')}",
        ])
    )

    with st.expander("Ver JSON completo (debug)"):
        st.json(row)


# -----------------------------------------------------------------------------
# Carrega NFs e monta aviso topo
# -----------------------------------------------------------------------------
df_nfs = load_nf_list()
if df_nfs is None or df_nfs.empty:
    st.warning("Nenhuma NF encontrada em BigQuery. Faça upload na Tela 1.")
    st.stop()

df_nfs = df_nfs.copy()
df_nfs["numero"] = df_nfs["numero"].astype(str)
df_nfs["unidade_consumidora"] = df_nfs["unidade_consumidora"].astype(str)

ucs = sorted(df_nfs["unidade_consumidora"].dropna().unique().tolist())
df_cli_all = load_clientes_for_ucs(ucs)
df_cli_all = _ensure_cols(df_cli_all, ["unidade_consumidora", "desconto_contratado", "custo_disp", "status", "subvencao", "n_fases"])

df_warn = _build_warning_table(df_nfs, df_cli_all)
df_block = df_warn[df_warn["can_calc"] == False].copy()

if not df_block.empty:
    st.warning(f"⚠️ {len(df_block)} fatura(s) NÃO podem ser calculadas por falta de cadastro (info_clientes).")
    with st.expander("Ver lista e motivo"):
        st.dataframe(
            df_block[["referencia", "numero", "unidade_consumidora", "nome", "motivo"]].sort_values(["referencia", "numero"], ascending=[False, False]),
            hide_index=True,
            width="stretch",
        )
else:
    st.success("✅ Todas as faturas listadas têm cadastro mínimo para cálculo (info_clientes).")

st.markdown("---")

# -----------------------------------------------------------------------------
# Seleção NF
# -----------------------------------------------------------------------------
df_nfs["label"] = df_nfs.apply(
    lambda r: f"{r['referencia']} | NF {r['numero']} | UC {r['unidade_consumidora']} | {str(r.get('nome') or '')[:40]}",
    axis=1,
)

nf_label = st.selectbox("Selecione a NF", options=df_nfs["label"].tolist())
nf_sel = str(df_nfs.loc[df_nfs["label"] == nf_label, "numero"].iloc[0])

df_it, df_med = load_invoice_data(nf_sel)
if df_it is None or df_it.empty:
    st.error("NF sem itens em fatura_itens.")
    st.stop()

uc = str(df_it["unidade_consumidora"].dropna().iloc[0]) if "unidade_consumidora" in df_it.columns and df_it["unidade_consumidora"].notna().any() else ""
if not uc:
    st.error("UC não encontrada nos itens dessa NF.")
    st.stop()

# carrega cliente (pode vir sem colunas se schema incompleto)
df_cli = load_cliente(uc)
df_cli = _ensure_cols(df_cli, ["unidade_consumidora", "desconto_contratado", "custo_disp", "status", "subvencao", "n_fases"])

cli_row = df_cli.iloc[0] if df_cli is not None and not df_cli.empty else None
motivo_nf = _calc_block_reason(cli_row)

# -----------------------------------------------------------------------------
# Form de preenchimento (quando faltar dados)
# -----------------------------------------------------------------------------
st.markdown(f"### 👤 Cadastro do associado (UC {uc})")

if motivo_nf is None:
    st.success("✅ Cadastro mínimo OK para calcular.")
else:
    st.error(f"Cadastro insuficiente para cálculo: **{motivo_nf}**")

# Sugestões padrão a partir da fatura
classe_mod = None
if "classe_modalidade" in df_it.columns and df_it["classe_modalidade"].notna().any():
    classe_mod = str(df_it["classe_modalidade"].dropna().iloc[0])

n_fases_sug = infer_n_fases(classe_mod) or 3
custo_sug = compute_custo_disp(n_fases_sug) or 100.0

# valores atuais (se existirem)
desconto_cur = float(pd.to_numeric(cli_row.get("desconto_contratado"), errors="coerce") or 0.15) if cli_row is not None else 0.15
subv_cur = float(pd.to_numeric(cli_row.get("subvencao"), errors="coerce") or 0.0) if cli_row is not None else 0.0
n_fases_cur = int(pd.to_numeric(cli_row.get("n_fases"), errors="coerce") or n_fases_sug) if cli_row is not None else n_fases_sug
custo_cur = float(pd.to_numeric(cli_row.get("custo_disp"), errors="coerce") or (compute_custo_disp(n_fases_cur) or custo_sug)) if cli_row is not None else custo_sug
status_cur = str(cli_row.get("status") or "Ativo") if cli_row is not None else "Ativo"

with st.form("form_cliente_minimo"):
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    with c1:
        desconto = st.number_input("desconto_contratado", min_value=0.0, max_value=1.0, value=float(desconto_cur), step=0.01)
    with c2:
        subvencao = st.number_input("subvencao", min_value=0.0, value=float(subv_cur), step=50.0)
    with c3:
        n_fases = st.selectbox("n_fases", options=[1, 2, 3], index=[1, 2, 3].index(int(n_fases_cur)))
    with c4:
        custo_disp = st.number_input("custo_disp (kWh)", min_value=0.0, value=float(custo_cur), step=10.0)
    with c5:
        status = st.selectbox("status", options=["Ativo", "Inativo"], index=0 if _norm_status(status_cur) == "ativo" else 1)

    salvar = st.form_submit_button("💾 Salvar/Atualizar info_clientes")

if salvar:
    payload = {
        "unidade_consumidora": uc,
        "desconto_contratado": float(desconto),
        "subvencao": float(subvencao),
        "n_fases": int(n_fases),
        "custo_disp": float(custo_disp),
        "status": str(status),
        "updated_at": datetime.utcnow(),
    }
    try:
        upsert_dataframe(pd.DataFrame([payload]), TABLE_CLIENTES, key_column="unidade_consumidora")
        st.success("✅ Cadastro salvo em info_clientes. Agora você pode calcular.")
        st.cache_data.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Falha ao salvar info_clientes: {e}")

st.markdown("---")

# -----------------------------------------------------------------------------
# Botões principais
# -----------------------------------------------------------------------------
c1, c2, c3 = st.columns([1, 1, 1])

calc_disabled = motivo_nf is not None
calc_btn = c1.button("🧮 Calcular", type="primary", width="stretch", disabled=calc_disabled)
save_btn = c2.button("💾 Salvar cálculo", width="stretch", disabled=True)  # mantido desabilitado neste patch simples
emit_btn = c3.button("🏦 Emitir + PDF", width="stretch", disabled=True)   # mantido desabilitado neste patch simples

if calc_disabled:
    st.info("Preencha e salve o cadastro mínimo acima para habilitar o cálculo (desconto_contratado, custo_disp e status=Ativo).")

# -----------------------------------------------------------------------------
# Cálculo
# -----------------------------------------------------------------------------
if "calc_nf" not in st.session_state:
    st.session_state.calc_nf = {}

if calc_btn:
    # recarrega cliente após salvar
    df_cli = load_cliente(uc)
    df_cli = _ensure_cols(df_cli, ["unidade_consumidora", "desconto_contratado", "custo_disp", "status", "subvencao", "n_fases"])

    with st.spinner("Calculando (fiel ao Excel)..."):
        res = calculate_boletos(df_itens=df_it, df_medidores=df_med, df_clientes=df_cli)
    st.session_state.calc_nf[nf_sel] = res.df_boletos.copy()
    st.success("✅ Cálculo concluído.")

df_calc = st.session_state.calc_nf.get(nf_sel)
if df_calc is None or df_calc.empty:
    st.stop()

row = df_calc.iloc[0].to_dict()
_render_calc_summary(row)

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
