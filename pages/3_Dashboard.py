"""
ERB Tech - Tela 3: Dashboard Gerencial
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))

from utils.bigquery_client import execute_query, TABLE_FATURA_ITENS

st.set_page_config(page_title="Dashboard - ERB Tech", page_icon="📊", layout="wide")
st.title("📊 Dashboard Gerencial")


@st.cache_data(ttl=300)
def carregar_metricas() -> pd.DataFrame:
    query = f"""
    SELECT
        COUNT(DISTINCT unidade_consumidora) AS total_clientes,
        COUNT(DISTINCT referencia) AS total_periodos,
        SUM(CASE WHEN codigo = '0D' THEN quantidade_registrada ELSE 0 END) AS consumo_total,
        SUM(CASE WHEN codigo IN ('0R', '0S') THEN ABS(quantidade_registrada) ELSE 0 END) AS injetada_total,
        SUM(CASE WHEN codigo IN ('0R', '0S') THEN ABS(valor) ELSE 0 END) AS economia_total
    FROM `{TABLE_FATURA_ITENS}`
    """
    return execute_query(query)


@st.cache_data(ttl=300)
def carregar_evolucao() -> pd.DataFrame:
    query = f"""
    SELECT
        referencia,
        COUNT(DISTINCT unidade_consumidora) AS clientes,
        SUM(CASE WHEN codigo = '0D' THEN quantidade_registrada ELSE 0 END) AS consumo,
        SUM(CASE WHEN codigo IN ('0R', '0S') THEN ABS(quantidade_registrada) ELSE 0 END) AS injetada,
        SUM(CASE WHEN codigo IN ('0R', '0S') THEN ABS(valor) ELSE 0 END) AS economia
    FROM `{TABLE_FATURA_ITENS}`
    GROUP BY referencia
    ORDER BY referencia ASC
    """
    return execute_query(query)


@st.cache_data(ttl=300)
def carregar_top_clientes(limite: int = 10) -> pd.DataFrame:
    query = f"""
    SELECT
        nome,
        unidade_consumidora,
        SUM(CASE WHEN codigo = '0D' THEN quantidade_registrada ELSE 0 END) AS consumo_total,
        SUM(CASE WHEN codigo IN ('0R', '0S') THEN ABS(valor) ELSE 0 END) AS economia_total,
        COUNT(DISTINCT referencia) AS meses
    FROM `{TABLE_FATURA_ITENS}`
    WHERE nome IS NOT NULL
    GROUP BY nome, unidade_consumidora
    ORDER BY consumo_total DESC
    LIMIT {int(limite)}
    """
    return execute_query(query)


try:
    df_metricas = carregar_metricas()
    df_evolucao = carregar_evolucao()
    df_top = carregar_top_clientes(10)
    dados_carregados = True
except Exception as e:
    st.error(f"Erro ao carregar dados: {e}")
    dados_carregados = False
    df_metricas = pd.DataFrame()
    df_evolucao = pd.DataFrame()
    df_top = pd.DataFrame()

if dados_carregados and not df_metricas.empty:
    m = df_metricas.iloc[0].to_dict()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Clientes", int(m.get("total_clientes") or 0))
    col2.metric("Períodos", int(m.get("total_periodos") or 0))
    col3.metric("Consumo (kWh)", f"{float(m.get('consumo_total') or 0):,.0f}".replace(",", "X").replace(".", ",").replace("X", "."))
    col4.metric("Injetada (kWh)", f"{float(m.get('injetada_total') or 0):,.0f}".replace(",", "X").replace(".", ",").replace("X", "."))
    col5.metric("Economia (R$)", f"{float(m.get('economia_total') or 0):,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

    st.markdown("---")

    if df_evolucao is not None and not df_evolucao.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ⚡ Consumo x Injetada (kWh)")
            fig_energia = go.Figure()
            fig_energia.add_trace(go.Scatter(x=df_evolucao["referencia"], y=df_evolucao["consumo"],
                                             mode="lines+markers", name="Consumo"))
            fig_energia.add_trace(go.Scatter(x=df_evolucao["referencia"], y=df_evolucao["injetada"],
                                             mode="lines+markers", name="Injetada"))
            fig_energia.update_layout(xaxis_title="Período", yaxis_title="kWh", height=350)
            st.plotly_chart(fig_energia, width="stretch")

        with col2:
            st.markdown("### 💵 Economia Mensal (R$)")
            fig_economia = go.Figure()
            fig_economia.add_trace(go.Bar(x=df_evolucao["referencia"], y=df_evolucao["economia"]))
            fig_economia.update_layout(xaxis_title="Período", yaxis_title="Economia (R$)", height=350)
            st.plotly_chart(fig_economia, width="stretch")

        st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🏆 Top 10 Maiores Clientes (Consumo)")
        if df_top is not None and not df_top.empty:
            df_top_display = df_top.copy()
            df_top_display["nome"] = df_top_display["nome"].astype(str).str[:40]
            st.dataframe(df_top_display, hide_index=True, width="stretch")
        else:
            st.info("Sem dados.")

    with col2:
        st.markdown("### 📊 Distribuição de Consumo (Top 5)")
        if df_top is not None and not df_top.empty:
            fig_pizza = px.pie(df_top.head(5), values="consumo_total", names="nome", hole=0.4)
            fig_pizza.update_layout(height=350)
            st.plotly_chart(fig_pizza, width="stretch")
        else:
            st.info("Sem dados.")

st.markdown("---")
st.markdown("### 📥 Exportar")
if dados_carregados:
    if st.button("💾 Gerar arquivo Excel", width="stretch"):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            if df_evolucao is not None and not df_evolucao.empty:
                df_evolucao.to_excel(writer, sheet_name="Evolução", index=False)
            if df_top is not None and not df_top.empty:
                df_top.to_excel(writer, sheet_name="Top Clientes", index=False)
        output.seek(0)
        st.download_button(
            "⬇️ Download Excel",
            data=output,
            file_name=f"relatorio_acer_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )

st.markdown("---")
st.markdown(
    f"<center>Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M')} | ERB Tech - ACER</center>",
    unsafe_allow_html=True
)
