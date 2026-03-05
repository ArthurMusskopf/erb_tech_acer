"""ERB Tech - Sistema de Gestão de Faturas ACER (Home)"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="ERB Tech - ACER", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.4rem; font-weight: 800; color: #1E88E5; margin-bottom: 0.25rem; }
    .sub { color: #6c757d; margin-top: 0; }
    .card { background: #f8f9fa; border-radius: 12px; padding: 1rem; border-left: 4px solid #1E88E5; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">⚡ ERB Tech - ACER</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Parseamento CELESC → Revisão → Cálculo (fiel ao Excel) → Boleto Sicoob (Sandbox)</div>', unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="card"><h3>📄 Upload + Revisão</h3><p>Carregue PDFs, revise e gere boleto por NF.</p></div>', unsafe_allow_html=True)
    if st.button("Ir para Upload →", key="btn1", width="stretch"):
        st.switch_page("pages/1_Upload_Faturas.py")

with col2:
    st.markdown('<div class="card"><h3>💰 Boletos</h3><p>Selecione uma NF do BigQuery e emita/baixe PDF.</p></div>', unsafe_allow_html=True)
    if st.button("Ir para Boletos →", key="btn2", width="stretch"):
        st.switch_page("pages/2_Boletos.py")

with col3:
    st.markdown('<div class="card"><h3>📊 Dashboard</h3><p>Métricas gerenciais (consumo, injeção, economia).</p></div>', unsafe_allow_html=True)
    if st.button("Ir para Dashboard →", key="btn3", width="stretch"):
        st.switch_page("pages/3_Dashboard.py")
