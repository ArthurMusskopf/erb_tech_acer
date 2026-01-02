"""
ERB Tech - Tela 2: Cálculo e Validação de Boletos
VERSÃO FINAL DEFINITIVA - Baseada no Excel Formulario_ERB_TechArt.xlsx
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import io
import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.bigquery_client import (
    execute_query,
    get_periodos_disponiveis,
    get_historico_cliente,
    TABLE_FATURA_ITENS,
    TABLE_CLIENTES
)

st.set_page_config(page_title="Boletos - ERB Tech", page_icon="💰", layout="wide")
st.title("💰 Boletos de Cobrança")


def safe_float(val, default=0.0):
    """Converte qualquer valor para float de forma segura"""
    if val is None:
        return default
    if isinstance(val, (pd.Series, np.ndarray)):
        val = val.iloc[0] if len(val) > 0 else default
    if pd.isna(val):
        return default
    try:
        return float(val)
    except:
        return default


def carregar_dados_periodo(periodo: str) -> pd.DataFrame:
    """
    Carrega dados das faturas - SEM colunas duplicadas
    Baseado na estrutura do Excel Calculos_Boleto
    """
    query = f"""
    WITH itens_agregados AS (
        SELECT 
            unidade_consumidora,
            nome,
            referencia,
            vencimento,
            MAX(numero) as nota_fiscal,
            MAX(cnpj) as cnpj,
            MAX(total_pagar) as total_concessionaria,
            -- Consumo
            SUM(CASE WHEN codigo = '0D' THEN quantidade_registrada ELSE 0 END) as consumo_kwh,
            SUM(CASE WHEN codigo = '0D' THEN valor ELSE 0 END) as consumo_te,
            SUM(CASE WHEN codigo = '0E' THEN valor ELSE 0 END) as consumo_tusd,
            -- Injetada (valores já são negativos no banco)
            SUM(CASE WHEN codigo IN ('0R', '0S') THEN ABS(quantidade_registrada) ELSE 0 END) as injetada_kwh,
            SUM(CASE WHEN codigo = '0R' THEN valor ELSE 0 END) as inj_te,
            SUM(CASE WHEN codigo = '0S' THEN valor ELSE 0 END) as inj_tusd,
            -- Bandeiras
            SUM(CASE WHEN codigo = '2L' THEN valor ELSE 0 END) as band_amarela,
            SUM(CASE WHEN codigo = '2M' THEN valor ELSE 0 END) as band_amarela_inj,
            SUM(CASE WHEN codigo = '2U' THEN valor ELSE 0 END) as band_vermelha,
            SUM(CASE WHEN codigo = '2V' THEN valor ELSE 0 END) as band_vermelha_inj
        FROM `{TABLE_FATURA_ITENS}`
        WHERE referencia = '{periodo}'
        GROUP BY unidade_consumidora, nome, referencia, vencimento
    )
    SELECT 
        i.*,
        COALESCE(c.desconto_contratado, 0.15) as desconto,
        COALESCE(c.subvencao, 0) as subvencao_cliente
    FROM itens_agregados i
    LEFT JOIN `{TABLE_CLIENTES}` c 
        ON i.unidade_consumidora = c.unidade_consumidora
    WHERE i.injetada_kwh > 0
    ORDER BY i.nome
    """
    return execute_query(query)


def calcular_boleto(row) -> dict:
    """
    Calcula boleto baseado na lógica do Excel Formulario_ERB_TechArt.xlsx
    Aba: Calculos_Boleto
    """
    # Extrair valores com segurança
    consumo_te = safe_float(row.get('consumo_te'))
    consumo_tusd = safe_float(row.get('consumo_tusd'))
    inj_te = safe_float(row.get('inj_te'))  # Já é negativo
    inj_tusd = safe_float(row.get('inj_tusd'))  # Já é negativo
    
    band_amarela = safe_float(row.get('band_amarela'))
    band_amarela_inj = safe_float(row.get('band_amarela_inj'))
    band_vermelha = safe_float(row.get('band_vermelha'))
    band_vermelha_inj = safe_float(row.get('band_vermelha_inj'))
    
    desconto = safe_float(row.get('desconto'), 0.15)
    subvencao = safe_float(row.get('subvencao_cliente'), 0)
    
    # Cálculos conforme Excel
    # tarifa_cheia = Consumo TE + Consumo TUSD
    tarifa_cheia = consumo_te + consumo_tusd
    
    # tarifa_injetada = Inj TE + Inj TUSD (valores negativos)
    tarifa_injetada = inj_te + inj_tusd
    
    # tarifa_paga_conc = tarifa_cheia + tarifa_injetada
    tarifa_paga_conc = tarifa_cheia + tarifa_injetada
    
    # tarifa_erb = desconto aplicado
    tarifa_erb = tarifa_paga_conc * desconto
    
    # tarifa_bol = valor que vai pro boleto (85% se desconto for 15%)
    tarifa_bol = tarifa_paga_conc * (1 - desconto)
    
    # Bandeiras líquidas
    valor_band_amarela = band_amarela + band_amarela_inj
    valor_band_vermelha = band_vermelha + band_vermelha_inj
    
    # Bandeiras com desconto
    valor_band_amar_desc = valor_band_amarela * (1 - desconto)
    valor_band_vrm_desc = valor_band_vermelha * (1 - desconto)
    
    # Total boleto
    valor_bruto = tarifa_bol + valor_band_amar_desc + valor_band_vrm_desc
    valor_final = max(0, valor_bruto - subvencao)
    
    # Gerar boleto apenas se houver energia injetada e valor > 0
    gerar = valor_final > 0 and tarifa_injetada < 0
    
    return {
        'tarifa_cheia': tarifa_cheia,
        'tarifa_injetada': tarifa_injetada,
        'tarifa_paga_conc': tarifa_paga_conc,
        'tarifa_erb': tarifa_erb,
        'tarifa_bol': tarifa_bol,
        'valor_band_amarela': valor_band_amarela,
        'valor_band_vermelha': valor_band_vermelha,
        'valor_band_amar_desc': valor_band_amar_desc,
        'valor_band_vrm_desc': valor_band_vrm_desc,
        'valor_bruto': valor_bruto,
        'desconto': desconto,
        'subvencao': subvencao,
        'valor_final': valor_final,
        'gerar_boleto': gerar
    }


# =============================================================================
# INTERFACE
# =============================================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    try:
        periodos = get_periodos_disponiveis()
        if not periodos:
            periodos = ["12/2025", "11/2025", "10/2025"]
    except:
        periodos = ["12/2025", "11/2025", "10/2025"]
    periodo = st.selectbox("Período de Referência", periodos)

with col2:
    status_filtro = st.selectbox("Status", ["Todos", "Com boleto", "Sem boleto"])

with col3:
    ordenar_por = st.selectbox("Ordenar por", ["Nome", "Valor (maior)", "Valor (menor)"])

with col4:
    st.markdown("<br>", unsafe_allow_html=True)
    recalcular = st.button("🔄 Carregar/Recalcular")

st.markdown("---")

# Carregar dados
if recalcular or 'boletos_df' not in st.session_state or st.session_state.get('periodo_atual') != periodo:
    try:
        with st.spinner("Carregando dados do BigQuery..."):
            df = carregar_dados_periodo(periodo)
            
            if df.empty:
                st.warning(f"Nenhuma fatura com energia injetada encontrada para {periodo}")
                st.stop()
            
            # Calcular boletos
            calculos = df.apply(calcular_boleto, axis=1)
            calculos_df = pd.DataFrame(calculos.tolist())
            df = pd.concat([df.reset_index(drop=True), calculos_df], axis=1)
            
            st.session_state.boletos_df = df
            st.session_state.periodo_atual = periodo
            
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        st.exception(e)
        st.stop()

# Exibir dados
if 'boletos_df' in st.session_state:
    df = st.session_state.boletos_df.copy()
    
    # Filtros
    if status_filtro == "Com boleto":
        df = df[df['gerar_boleto'] == True]
    elif status_filtro == "Sem boleto":
        df = df[df['gerar_boleto'] == False]
    
    # Ordenação
    if ordenar_por == "Valor (maior)":
        df = df.sort_values('valor_final', ascending=False)
    elif ordenar_por == "Valor (menor)":
        df = df.sort_values('valor_final', ascending=True)
    else:
        df = df.sort_values('nome')
    
    # Resumo
    col1, col2, col3, col4 = st.columns(4)
    
    df_com_boleto = df[df['gerar_boleto'] == True]
    total_boletos = len(df_com_boleto)
    total_valor = safe_float(df_com_boleto['valor_final'].sum())
    total_faturas = len(df)
    economia_total = abs(safe_float(df['tarifa_injetada'].sum()))
    
    col1.metric("Total de Faturas", total_faturas)
    col2.metric("Boletos a Gerar", total_boletos)
    col3.metric("Valor Total", f"R$ {total_valor:,.2f}")
    col4.metric("Economia Total", f"R$ {economia_total:,.2f}")
    
    st.markdown("---")
    st.markdown(f"### 📋 Boletos - {periodo}")
    
    # Lista de boletos
    for idx, row in df.iterrows():
        # Extrair valores com safe_float
        gerar = bool(row.get('gerar_boleto', False))
        valor_final = safe_float(row.get('valor_final'))
        nome = str(row.get('nome', 'N/A'))[:50]
        uc = str(row.get('unidade_consumidora', ''))
        
        status_icon = "🟢" if gerar else "🔴"
        valor_display = f"R$ {valor_final:,.2f}" if gerar else "Sem boleto"
        
        with st.expander(f"{status_icon} **{nome}** | UC: {uc} | **{valor_display}**"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### 📊 Dados do Consumo")
                
                consumo_kwh = safe_float(row.get('consumo_kwh'))
                injetada_kwh = safe_float(row.get('injetada_kwh'))
                
                consumo_df = pd.DataFrame({
                    'Item': ['Consumo Total', 'Energia Injetada', 'Saldo Líquido'],
                    'kWh': [f"{consumo_kwh:,.0f}", f"{injetada_kwh:,.0f}", f"{consumo_kwh - injetada_kwh:,.0f}"]
                })
                st.dataframe(consumo_df, use_container_width=True, hide_index=True)
                
                st.markdown("#### 💵 Memorial de Cálculo")
                
                # Extrair todos os valores com safe_float
                tarifa_cheia = safe_float(row.get('tarifa_cheia'))
                tarifa_injetada = safe_float(row.get('tarifa_injetada'))
                tarifa_paga_conc = safe_float(row.get('tarifa_paga_conc'))
                desconto = safe_float(row.get('desconto'), 0.15)
                tarifa_erb = safe_float(row.get('tarifa_erb'))
                tarifa_bol = safe_float(row.get('tarifa_bol'))
                valor_band_amar_desc = safe_float(row.get('valor_band_amar_desc'))
                valor_band_vrm_desc = safe_float(row.get('valor_band_vrm_desc'))
                valor_bruto = safe_float(row.get('valor_bruto'))
                subvencao = safe_float(row.get('subvencao'))
                
                calculo_df = pd.DataFrame({
                    'Descrição': [
                        'Consumo (TE + TUSD)',
                        'Energia Injetada (crédito)',
                        'Tarifa Paga Concessionária',
                        f'Desconto ERB ({desconto*100:.0f}%)',
                        'Tarifa Boleto (85%)',
                        'Bandeira Amarela c/ desc',
                        'Bandeira Vermelha c/ desc',
                        'Valor Bruto',
                        'Subvenção',
                        '**VALOR FINAL**'
                    ],
                    'Valor (R$)': [
                        f'{tarifa_cheia:,.2f}',
                        f'{tarifa_injetada:,.2f}',
                        f'{tarifa_paga_conc:,.2f}',
                        f'-{tarifa_erb:,.2f}',
                        f'{tarifa_bol:,.2f}',
                        f'{valor_band_amar_desc:,.2f}',
                        f'{valor_band_vrm_desc:,.2f}',
                        f'{valor_bruto:,.2f}',
                        f'-{subvencao:,.2f}',
                        f'**{valor_final:,.2f}**'
                    ]
                })
                st.dataframe(calculo_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("#### 📋 Dados do Cliente")
                nota_fiscal = row.get('nota_fiscal') or 'N/A'
                cnpj = row.get('cnpj') or 'N/A'
                vencimento = row.get('vencimento') or 'N/A'
                
                st.markdown(f"""
                - **Nota Fiscal:** {nota_fiscal}
                - **CNPJ/CPF:** {cnpj}
                - **Vencimento:** {vencimento}
                - **Desconto:** {desconto*100:.0f}%
                - **Subvenção:** R$ {subvencao:,.2f}
                """)
                
                st.markdown("#### 🎯 Ações")
                
                if st.button("📜 Ver Histórico", key=f"hist_{idx}", use_container_width=True):
                    try:
                        hist = get_historico_cliente(uc)
                        if not hist.empty:
                            st.dataframe(hist, use_container_width=True, hide_index=True)
                        else:
                            st.info("Nenhum histórico encontrado")
                    except Exception as e:
                        st.error(f"Erro: {e}")

    # Exportação
    st.markdown("---")
    st.markdown("### ⚡ Ações em Lote")
    
    col1, col2, col3 = st.columns(3)
    
    with col2:
        if st.button("💾 Exportar para Excel", use_container_width=True):
            export_cols = ['unidade_consumidora', 'nome', 'referencia', 'consumo_kwh', 
                          'injetada_kwh', 'tarifa_cheia', 'tarifa_injetada', 
                          'valor_final', 'desconto', 'subvencao']
            export_df = df[[c for c in export_cols if c in df.columns]].copy()
            
            output = io.BytesIO()
            export_df.to_excel(output, index=False)
            output.seek(0)
            
            st.download_button(
                "⬇️ Download Excel",
                data=output,
                file_name=f"boletos_{periodo.replace('/', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

else:
    st.info("👆 Clique em 'Carregar/Recalcular' para carregar os dados.")
