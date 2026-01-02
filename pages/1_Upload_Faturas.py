"""
ERB Tech - Tela 1: Upload e Parseamento de Faturas
VERSÃO FINAL DEFINITIVA
"""

import streamlit as st
import pandas as pd
import tempfile
import os
import zipfile
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.pdf_parser import parse_fatura, processar_lote_faturas
from utils.bigquery_client import (
    upsert_dataframe,
    execute_query,
    TABLE_FATURA_ITENS,
    TABLE_MEDIDORES
)

st.set_page_config(page_title="Upload de Faturas - ERB Tech", page_icon="📄", layout="wide")
st.title("📄 Upload e Parseamento de Faturas")

# Session state
if 'faturas_processadas' not in st.session_state:
    st.session_state.faturas_processadas = None

tab_upload, tab_revisao, tab_historico = st.tabs(["📤 Upload", "📝 Revisão", "📜 Histórico"])

with tab_upload:
    st.markdown("### Carregar Faturas")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Arraste PDFs ou arquivo ZIP",
            type=['pdf', 'zip'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} arquivo(s) selecionado(s)")
    
    with col2:
        salvar_auto = st.checkbox("Salvar automaticamente", value=False)
    
    if uploaded_files and st.button("🚀 Processar Faturas", type="primary", use_container_width=True):
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_paths = []
            
            for uploaded_file in uploaded_files:
                if uploaded_file.name.endswith('.zip'):
                    zip_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(zip_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            if file.lower().endswith('.pdf'):
                                pdf_paths.append(os.path.join(root, file))
                else:
                    pdf_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(pdf_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    pdf_paths.append(pdf_path)
            
            if not pdf_paths:
                st.error("Nenhum PDF encontrado!")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(progress, arquivo):
                    progress_bar.progress(progress)
                    status_text.text(f"Processando: {arquivo}")
                
                with st.spinner("Processando..."):
                    resultado = processar_lote_faturas(pdf_paths, update_progress)
                
                progress_bar.progress(1.0)
                status_text.empty()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total", resultado['total'])
                col2.metric("Sucesso", resultado['sucesso'])
                col3.metric("Erros", resultado['erros'])
                
                # Mostrar erros
                erros_lista = [r for r in resultado['resultados'] if not r.sucesso]
                if erros_lista:
                    with st.expander(f"⚠️ {len(erros_lista)} arquivo(s) com erro"):
                        for r in erros_lista:
                            st.error(f"**{r.arquivo}**: {', '.join(r.erros)}")
                
                if not resultado['df_itens'].empty:
                    st.session_state.faturas_processadas = resultado
                    st.success(f"✅ {len(resultado['df_itens'])} itens parseados!")
                    
                    if salvar_auto:
                        try:
                            with st.spinner("Salvando no BigQuery..."):
                                n_itens = upsert_dataframe(resultado['df_itens'], TABLE_FATURA_ITENS, 'id')
                                n_med = upsert_dataframe(resultado['df_medidores'], TABLE_MEDIDORES, 'id') if not resultado['df_medidores'].empty else 0
                            st.success(f"✅ Salvo: {n_itens} itens, {n_med} medidores")
                        except Exception as e:
                            st.error(f"Erro ao salvar: {e}")


with tab_revisao:
    st.markdown("### Revisão dos Dados")
    
    if st.session_state.faturas_processadas is None:
        st.info("📤 Faça o upload de faturas na aba anterior.")
    else:
        resultado = st.session_state.faturas_processadas
        df_itens = resultado['df_itens']
        
        st.markdown(f"**{len(df_itens)} itens** de **{df_itens['unidade_consumidora'].nunique()} faturas**")
        
        # Filtros
        col1, col2 = st.columns(2)
        with col1:
            ucs = ['Todas'] + sorted(df_itens['unidade_consumidora'].dropna().unique().tolist())
            filtro_uc = st.selectbox("Filtrar por UC", ucs)
        with col2:
            periodos = ['Todos'] + sorted(df_itens['referencia'].dropna().unique().tolist(), reverse=True)
            filtro_periodo = st.selectbox("Período", periodos)
        
        df_filtrado = df_itens.copy()
        if filtro_uc != 'Todas':
            df_filtrado = df_filtrado[df_filtrado['unidade_consumidora'] == filtro_uc]
        if filtro_periodo != 'Todos':
            df_filtrado = df_filtrado[df_filtrado['referencia'] == filtro_periodo]
        
        # Exibir dados
        colunas_exibir = ['unidade_consumidora', 'nome', 'referencia', 'codigo', 'descricao', 
                         'quantidade_registrada', 'tarifa', 'valor', 'total_pagar']
        colunas_disponiveis = [c for c in colunas_exibir if c in df_filtrado.columns]
        
        st.dataframe(df_filtrado[colunas_disponiveis], use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Restaurar Original", use_container_width=True):
                st.rerun()
        
        with col2:
            if st.button("✅ Salvar no BigQuery", type="primary", use_container_width=True):
                try:
                    with st.spinner("Salvando..."):
                        n_itens = upsert_dataframe(df_itens, TABLE_FATURA_ITENS, 'id')
                        n_med = 0
                        if not resultado['df_medidores'].empty:
                            n_med = upsert_dataframe(resultado['df_medidores'], TABLE_MEDIDORES, 'id')
                    
                    st.success(f"✅ Salvo: {n_itens} itens, {n_med} medidores")
                    st.balloons()
                    st.session_state.faturas_processadas = None
                    
                except Exception as e:
                    st.error(f"❌ Erro: {e}")
                    st.exception(e)


with tab_historico:
    st.markdown("### Histórico de Dados no BigQuery")
    
    try:
        query = """
        SELECT 
            referencia,
            COUNT(DISTINCT unidade_consumidora) as faturas,
            COUNT(*) as itens,
            SUM(CASE WHEN codigo = '0D' THEN quantidade_registrada ELSE 0 END) as consumo_total
        FROM `football-data-science.erb_tech.fatura_itens`
        GROUP BY referencia
        ORDER BY referencia DESC
        LIMIT 12
        """
        
        df_hist = execute_query(query)
        
        if not df_hist.empty:
            st.dataframe(df_hist, use_container_width=True, hide_index=True)
        else:
            st.warning("Nenhum dado encontrado")
            
    except Exception as e:
        st.warning(f"Não foi possível conectar: {e}")
