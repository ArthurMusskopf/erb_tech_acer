"""ERB Tech - Sistema de Gestão de Faturas ACER"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

from typing import Optional, Dict, Any, Tuple, List

# =============================================================================
# Tentamos importar o cliente BigQuery do teu projeto (ajuste se necessário)
# =============================================================================

BQ_OK = False
BQ_IMPORT_ERROR = None

execute_query = None
upsert_dataframe = None

TABLE_FATURA_ITENS = None
TABLE_MEDIDORES = None
TABLE_CLIENTES = None
TABLE_BOLETOS = None

IMPORTS_CANDIDATOS = [
    "cliente_bigquery",      # ex.: cliente_bigquery.py
    "bigquery_client",       # ex.: bigquery_client.py
    "bq_client",             # ex.: bq_client.py
]

for mod in IMPORTS_CANDIDATOS:
    try:
        m = __import__(mod, fromlist=[
            "execute_query",
            "upsert_dataframe",
            "TABLE_FATURA_ITENS",
            "TABLE_MEDIDORES",
            "TABLE_CLIENTES",
            "TABLE_BOLETOS",
        ])
        execute_query = getattr(m, "execute_query")
        upsert_dataframe = getattr(m, "upsert_dataframe")

        TABLE_FATURA_ITENS = getattr(m, "TABLE_FATURA_ITENS")
        TABLE_MEDIDORES = getattr(m, "TABLE_MEDIDORES")
        TABLE_CLIENTES = getattr(m, "TABLE_CLIENTES")
        TABLE_BOLETOS = getattr(m, "TABLE_BOLETOS")

        BQ_OK = True
        break
    except Exception as e:
        BQ_IMPORT_ERROR = e

# =============================================================================
# Calc engine
# =============================================================================
try:
    from calc_engine import calculate_boletos
    CALC_OK = True
    CALC_IMPORT_ERROR = None
except Exception as e:
    CALC_OK = False
    CALC_IMPORT_ERROR = e


# =============================================================================
# Streamlit config
# =============================================================================

st.set_page_config(page_title="ERB Tech - ACER", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1E88E5; }
    .metric-card { background: #f8f9fa; border-radius: 10px; padding: 1rem; border-left: 4px solid #1E88E5; }
    .small-note { color: #6c757d; font-size: 0.9rem; }
    .warn { background: #fff3cd; border: 1px solid #ffeeba; padding: 0.75rem; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Helpers BigQuery
# =============================================================================

@st.cache_data(ttl=60, show_spinner=False)
def get_periodos_disponiveis(limit: int = 24) -> List[str]:
    if not BQ_OK:
        return []
    q = f"""
    SELECT DISTINCT referencia
    FROM `{TABLE_FATURA_ITENS}`
    WHERE referencia IS NOT NULL
    ORDER BY referencia DESC
    LIMIT {int(limit)}
    """
    df = execute_query(q)
    if df is None or df.empty or "referencia" not in df.columns:
        return []
    return df["referencia"].astype(str).tolist()


@st.cache_data(ttl=60, show_spinner=False)
def load_dados_periodo(referencia: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carrega:
    - fatura_itens do período
    - medidores do período (por referencia)
    - clientes (somente das UCs do período)
    """
    if not BQ_OK:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Itens
    q_itens = f"""
    SELECT *
    FROM `{TABLE_FATURA_ITENS}`
    WHERE referencia = @ref
    """
    df_itens = execute_query(q_itens, {"ref": referencia})

    # Medidores
    q_med = f"""
    SELECT *
    FROM `{TABLE_MEDIDORES}`
    WHERE referencia = @ref
    """
    df_med = execute_query(q_med, {"ref": referencia})

    # Clientes (filtra só UCs do período)
    ucs = []
    if df_itens is not None and not df_itens.empty and "unidade_consumidora" in df_itens.columns:
        ucs = (
            df_itens["unidade_consumidora"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    if ucs:
        q_cli = f"""
        SELECT *
        FROM `{TABLE_CLIENTES}`
        WHERE unidade_consumidora IN UNNEST(@ucs)
        """
        df_cli = execute_query(q_cli, {"ucs": ucs})
    else:
        # fallback: vazio
        df_cli = pd.DataFrame()

    return df_itens, df_med, df_cli


def _store_calc_snapshot(payload: Dict[str, Any]) -> None:
    st.session_state["calc_snapshot"] = payload


def _get_calc_snapshot() -> Dict[str, Any]:
    return st.session_state.get("calc_snapshot", {})


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.markdown("### 🔌 ERB Tech")
    st.markdown("---")
    st.markdown("**Versão:** 3.1.0 (calc_engine integrado)")

    st.markdown("---")
    st.markdown("### 🧮 Cálculo de Boletos")

    if not CALC_OK:
        st.error(f"Falha ao importar calc_engine.py: {CALC_IMPORT_ERROR}")
    if not BQ_OK:
        st.warning("BigQuery client não disponível. Ajusta o nome do módulo importado no app.py.")
        if BQ_IMPORT_ERROR:
            st.caption(f"Erro de import: {BQ_IMPORT_ERROR}")

    periodos = get_periodos_disponiveis() if (BQ_OK and CALC_OK) else []
    periodo_sel = st.selectbox(
        "Período (referência)",
        options=periodos if periodos else ["(sem dados)"],
        disabled=not (BQ_OK and CALC_OK and len(periodos) > 0),
    )

    rodar = st.button(
        "Recalcular boletos do período",
        use_container_width=True,
        disabled=not (BQ_OK and CALC_OK and len(periodos) > 0),
    )

    if rodar and periodo_sel and periodo_sel != "(sem dados)":
        with st.spinner("Carregando dados do BigQuery e calculando boletos..."):
            df_itens, df_med, df_cli = load_dados_periodo(periodo_sel)

            if df_itens is None or df_itens.empty:
                st.error("Não encontramos fatura_itens para esse período.")
            else:
                try:
                    res = calculate_boletos(
                        df_itens=df_itens,
                        df_medidores=df_med,
                        df_clientes=df_cli,
                        only_registered_clients=True,
                        only_status_ativo=True,
                    )

                    df_boletos = res.df_boletos.copy()
                    missing = res.missing_clientes or []

                    # guarda no session_state para páginas usarem
                    _store_calc_snapshot({
                        "periodo": periodo_sel,
                        "missing_clientes": missing,
                        "missing_reason": res.missing_reason,
                        "boletos_rows": int(len(df_boletos)) if df_boletos is not None else 0,
                        "boletos_total": float(pd.to_numeric(df_boletos.get("valor_total_boleto"), errors="coerce").fillna(0).sum())
                            if df_boletos is not None and not df_boletos.empty else 0.0,
                    })

                    if missing:
                        st.warning(
                            f"Encontramos {len(missing)} UC(s) sem cadastro ativo em info_clientes. "
                            f"O cálculo foi bloqueado para essas UCs."
                        )
                        st.info("A página **Boletos** deve abrir o formulário de cadastro usando session_state['calc_snapshot'].")
                    else:
                        # grava no BigQuery
                        if df_boletos is None or df_boletos.empty:
                            st.warning("Cálculo retornou vazio. Nada foi gravado.")
                        else:
                            # usamos numero como chave do MERGE (mais seguro)
                            upsert_dataframe(df_boletos, TABLE_BOLETOS, key_column="numero")
                            st.success(f"Boletos calculados e gravados em boletos_calculados ({len(df_boletos)} linhas).")

                except Exception as e:
                    st.error(f"Erro no cálculo: {e}")


# =============================================================================
# Home UI
# =============================================================================

st.markdown('<p class="main-header">⚡ ERB Tech - ACER</p>', unsafe_allow_html=True)
st.markdown("Associação Catarinense de Energias Renováveis")

snapshot = _get_calc_snapshot()

if snapshot:
    missing = snapshot.get("missing_clientes", [])
    st.markdown("#### 📌 Última execução do cálculo")
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Período", snapshot.get("periodo", "-"))
    colB.metric("Linhas calculadas", snapshot.get("boletos_rows", 0))
    colC.metric("Total (R$)", f'{snapshot.get("boletos_total", 0.0):,.2f}'.replace(",", "X").replace(".", ",").replace("X", "."))
    colD.metric("UCs pendentes", len(missing))

    if missing:
        st.markdown(
            '<div class="warn"><b>⚠️ Ação necessária:</b> existem UCs não cadastradas/ativas. '
            'Entre em <b>Boletos</b> para validar e completar o cadastro antes de gravar/calcular para elas.</div>',
            unsafe_allow_html=True
        )
        with st.expander("Ver UCs pendentes"):
            reasons = snapshot.get("missing_reason", {}) or {}
            for uc in missing:
                st.write(f"- **{uc}** — {reasons.get(uc, 'Sem detalhe')}")

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        '<div class="metric-card"><h3>📄 Upload de Faturas</h3><p>Carregue PDFs da CELESC</p></div>',
        unsafe_allow_html=True
    )
    if st.button("Ir para Upload →", key="btn1", use_container_width=True):
        st.switch_page("pages/1_Upload_Faturas.py")

with col2:
    st.markdown(
        '<div class="metric-card"><h3>💰 Boletos</h3><p>Calcule e gere boletos</p></div>',
        unsafe_allow_html=True
    )
    if st.button("Ir para Boletos →", key="btn2", use_container_width=True):
        # a página de boletos pode ler st.session_state["calc_snapshot"]
        st.switch_page("pages/2_Boletos.py")

with col3:
    st.markdown(
        '<div class="metric-card"><h3>📊 Dashboard</h3><p>Métricas gerenciais</p></div>',
        unsafe_allow_html=True
    )
    if st.button("Ir para Dashboard →", key="btn3", use_container_width=True):
        st.switch_page("pages/3_Dashboard.py")

st.caption(
    "Dica: nós podemos rodar o cálculo na home (sidebar) ou diretamente na página Boletos. "
    "Se houver clientes faltantes, a página Boletos deve abrir o formulário de cadastro."
)
