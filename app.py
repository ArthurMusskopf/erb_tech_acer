"""ERB Tech - Sistema de Gestão de Faturas ACER"""
import streamlit as st

st.set_page_config(page_title="ERB Tech - ACER", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1E88E5; }
    .metric-card { background: #f8f9fa; border-radius: 10px; padding: 1rem; border-left: 4px solid #1E88E5; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🔌 ERB Tech")
    st.markdown("---")
    st.markdown("**Versão:** 3.0.0 Final")

st.markdown('<p class="main-header">⚡ ERB Tech - ACER</p>', unsafe_allow_html=True)
st.markdown("Associação Catarinense de Energias Renováveis")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="metric-card"><h3>📄 Upload de Faturas</h3><p>Carregue PDFs da CELESC</p></div>', unsafe_allow_html=True)
    if st.button("Ir para Upload →", key="btn1", use_container_width=True):
        st.switch_page("pages/1_Upload_Faturas.py")

with col2:
    st.markdown('<div class="metric-card"><h3>💰 Boletos</h3><p>Calcule e gere boletos</p></div>', unsafe_allow_html=True)
    if st.button("Ir para Boletos →", key="btn2", use_container_width=True):
        st.switch_page("pages/2_Boletos.py")

with col3:
    st.markdown('<div class="metric-card"><h3>📊 Dashboard</h3><p>Métricas gerenciais</p></div>', unsafe_allow_html=True)
    if st.button("Ir para Dashboard →", key="btn3", use_container_width=True):
        st.switch_page("pages/3_Dashboard.py")
