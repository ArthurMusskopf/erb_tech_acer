"""
ERB Tech - Tela 3: Dashboard Gerencial
VERSÃO FINAL DEFINITIVA
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
import io
import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.bigquery_client import execute_query

st.set_page_config(page_title="Dashboard - ERB Tech", page_icon="📊", layout="wide")
st.title("📊 Dashboard Gerencial")


@st.cache_data(ttl=300)
def carregar_metricas():
    query = """
    SELECT 
        COUNT(DISTINCT unidade_consumidora) as total_clientes,
        COUNT(DISTINCT referencia) as total_periodos,
        SUM(CASE WHEN codigo = '0D' THEN quantidade_registrada ELSE 0 END) as consumo_total,
        SUM(CASE WHEN codigo IN ('0R', '0S') THEN ABS(quantidade_registrada) ELSE 0 END) as injetada_total,
        SUM(CASE WHEN codigo IN ('0R', '0S') THEN ABS(valor) ELSE 0 END) as economia_total
    FROM `football-data-science.erb_tech.fatura_itens`
    """
    return execute_query(query)


@st.cache_data(ttl=300)
def carregar_evolucao():
    query = """
    SELECT 
        referencia,
        COUNT(DISTINCT unidade_consumidora) as clientes,
        SUM(CASE WHEN codigo = '0D' THEN quantidade_registrada ELSE 0 END) as consumo,
        SUM(CASE WHEN codigo IN ('0R', '0S') THEN ABS(quantidade_registrada) ELSE 0 END) as injetada,
        SUM(CASE WHEN codigo IN ('0R', '0S') THEN ABS(valor) ELSE 0 END) as economia
    FROM `football-data-science.erb_tech.fatura_itens`
    GROUP BY referencia
    ORDER BY referencia ASC
    """
    return execute_query(query)


@st.cache_data(ttl=300)
def carregar_top_clientes(limite: int = 10):
    query = f"""
    SELECT 
        nome,
        unidade_consumidora,
        SUM(CASE WHEN codigo = '0D' THEN quantidade_registrada ELSE 0 END) as consumo_total,
        SUM(CASE WHEN codigo IN ('0R', '0S') THEN ABS(valor) ELSE 0 END) as economia_total,
        COUNT(DISTINCT referencia) as meses
    FROM `football-data-science.erb_tech.fatura_itens`
    GROUP BY nome, unidade_consumidora
    ORDER BY consumo_total DESC
    LIMIT {limite}
    """
    return execute_query(query)


try:
    with st.spinner("Carregando dados..."):
        df_metricas = carregar_metricas()
        df_evolucao = carregar_evolucao()
        df_top = carregar_top_clientes()
    dados_carregados = True
except Exception as e:
    st.warning(f"Erro ao carregar dados: {e}")
    dados_carregados = False
    df_metricas = pd.DataFrame([{'total_clientes': 0, 'consumo_total': 0, 'injetada_total': 0, 'economia_total': 0}])

st.markdown("---")

# KPIs
st.markdown("### 🎯 Indicadores Principais")
col1, col2, col3, col4 = st.columns(4)

if not df_metricas.empty:
    m = df_metricas.iloc[0]
    col1.metric("👥 Clientes", f"{int(m.get('total_clientes', 0))}")
    col2.metric("⚡ Consumo Total", f"{float(m.get('consumo_total', 0))/1000:,.0f} MWh")
    col3.metric("🔋 Energia Compensada", f"{float(m.get('injetada_total', 0))/1000:,.0f} MWh")
    col4.metric("💰 Economia Gerada", f"R$ {float(m.get('economia_total', 0)):,.0f}")

st.markdown("---")

if dados_carregados and not df_evolucao.empty:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📈 Evolução do Consumo e Injeção")
        fig_energia = go.Figure()
        fig_energia.add_trace(go.Scatter(x=df_evolucao['referencia'], y=df_evolucao['consumo'], 
                                         mode='lines+markers', name='Consumo', line=dict(color='#FF5722')))
        fig_energia.add_trace(go.Scatter(x=df_evolucao['referencia'], y=df_evolucao['injetada'], 
                                         mode='lines+markers', name='Injetada', line=dict(color='#4CAF50')))
        fig_energia.update_layout(xaxis_title="Período", yaxis_title="kWh", height=350)
        st.plotly_chart(fig_energia, use_container_width=True)

    with col2:
        st.markdown("### 💵 Economia Mensal")
        fig_economia = go.Figure()
        fig_economia.add_trace(go.Bar(x=df_evolucao['referencia'], y=df_evolucao['economia'], marker_color='#1E88E5'))
        fig_economia.update_layout(xaxis_title="Período", yaxis_title="Economia (R$)", height=350)
        st.plotly_chart(fig_economia, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🏆 Top 10 Maiores Clientes")
        if not df_top.empty:
            df_top_display = df_top.copy()
            df_top_display['nome'] = df_top_display['nome'].str[:40]
            st.dataframe(df_top_display, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### 📊 Distribuição de Consumo")
        if not df_top.empty:
            fig_pizza = px.pie(df_top.head(5), values='consumo_total', names='nome', hole=0.4)
            fig_pizza.update_layout(height=350)
            st.plotly_chart(fig_pizza, use_container_width=True)

st.markdown("---")

# Exportação
st.markdown("### 📥 Exportar")
if dados_carregados:
    if st.button("💾 Exportar para Excel", use_container_width=True):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            if not df_evolucao.empty:
                df_evolucao.to_excel(writer, sheet_name='Evolução', index=False)
            if not df_top.empty:
                df_top.to_excel(writer, sheet_name='Top Clientes', index=False)
        output.seek(0)
        st.download_button(
            "⬇️ Download Excel",
            data=output,
            file_name=f"relatorio_acer_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

st.markdown("---")
st.markdown(f"<center>Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M')} | ERB Tech - ACER</center>", unsafe_allow_html=True)
