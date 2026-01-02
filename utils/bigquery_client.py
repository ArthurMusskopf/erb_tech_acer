"""
ERB Tech - Cliente BigQuery
VERSÃO FINAL v4 - Corrige tipos STRING para campos de leitura
"""

import streamlit as st
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.oauth2 import service_account
from typing import Optional, List, Dict, Any
import hashlib
from datetime import datetime
import time


# Configurações
PROJECT_ID = "football-data-science"
DATASET_ID = "erb_tech"
LOCATION = "southamerica-east1"

# Tabelas
TABLE_FATURA_ITENS = f"{PROJECT_ID}.{DATASET_ID}.fatura_itens"
TABLE_MEDIDORES = f"{PROJECT_ID}.{DATASET_ID}.medidores_leituras"
TABLE_CLIENTES = f"{PROJECT_ID}.{DATASET_ID}.info_clientes"
TABLE_BOLETOS = f"{PROJECT_ID}.{DATASET_ID}.boletos_calculados"
TABLE_EDIT_LOG = f"{PROJECT_ID}.{DATASET_ID}.edit_log"


@st.cache_resource
def get_bigquery_client() -> bigquery.Client:
    """Cria e retorna um cliente BigQuery"""
    try:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        return bigquery.Client(
            credentials=credentials,
            project=PROJECT_ID,
            location=LOCATION
        )
    except Exception as e:
        st.error(f"Erro ao conectar com BigQuery: {e}")
        raise e


def execute_query(query: str, params: Optional[Dict] = None) -> pd.DataFrame:
    """Executa uma query SQL e retorna um DataFrame"""
    client = get_bigquery_client()
    
    if params:
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(k, "STRING", v) 
                for k, v in params.items()
            ]
        )
        query_job = client.query(query, job_config=job_config)
    else:
        query_job = client.query(query)
    
    return query_job.to_dataframe()


def normalize_for_bigquery(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalização COMPLETA para BigQuery
    - Converte campos de texto para STRING
    - Converte dias para Int64
    - Substitui NaN por None
    """
    df = df.copy()
    
    # IMPORTANTE: Campos que DEVEM ser STRING no BigQuery
    string_cols = [
        "leitura_anterior", "leitura_atual", "proxima_leitura",
        "vencimento", "data_emissao", "referencia",
        "unidade_consumidora", "cliente_numero", "nome", "cnpj", "cep",
        "cidade_uf", "grupo_subgrupo_tensao", "numero", "serie",
        "codigo", "descricao", "unidade", "id",
        # Campos de medidores
        "medidor", "tipo", "posto", "nota_fiscal_numero"
    ]
    
    for c in string_cols:
        if c in df.columns:
            # Converter para string, tratando None e NaN
            df[c] = df[c].apply(lambda x: str(x).strip() if pd.notna(x) and x is not None else None)
            # Substituir strings vazias por None
            df[c] = df[c].replace({"": None, "None": None, "nan": None, "NaN": None})
    
    # Campos inteiros
    if "dias" in df.columns:
        df["dias"] = pd.to_numeric(df["dias"], errors="coerce").astype("Int64")
    
    # Campos numéricos float
    float_cols = [
        "quantidade_registrada", "tarifa", "valor", "pis_valor", 
        "cofins_base", "icms_aliquota", "icms_valor", "tarifa_sem_trib",
        "total_pagar", "constante", "fator", "total_apurado"
    ]
    
    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    
    # Substituir NaN por None para todos os campos
    return df.replace({np.nan: None})


def upsert_dataframe(df: pd.DataFrame, table_id: str, key_column: str = "id") -> int:
    """
    UPSERT via MERGE - IGUAL AO NOTEBOOK ORIGINAL
    """
    if df is None or df.empty:
        return 0
    
    if key_column not in df.columns:
        raise ValueError(f"DataFrame precisa da coluna '{key_column}'.")
    
    client = get_bigquery_client()
    
    # IGUAL AO NOTEBOOK: drop_duplicates com keep="first"
    df_clean = df.drop_duplicates(subset=[key_column], keep="first").copy()
    df_clean = df_clean.dropna(subset=[key_column])
    
    # NORMALIZAÇÃO CRÍTICA
    df_clean = normalize_for_bigquery(df_clean)
    
    if df_clean.empty:
        return 0
    
    # Staging table
    ts = int(time.time())
    staging_table_id = f"{table_id}__staging_{ts}"
    
    try:
        # 1) Load para staging
        load_job = client.load_table_from_dataframe(
            df_clean,
            staging_table_id,
            location=LOCATION,
            job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        )
        load_job.result()
        
        # 2) MERGE
        cols = list(df_clean.columns)
        non_key_cols = [c for c in cols if c != key_column]
        
        def q(c):
            return f"`{c}`"
        
        update_set = ",\n      ".join([f"T.{q(c)} = S.{q(c)}" for c in non_key_cols])
        insert_cols = ", ".join([q(c) for c in cols])
        insert_vals = ", ".join([f"S.{q(c)}" for c in cols])
        
        merge_sql = f"""
        MERGE `{table_id}` T
        USING `{staging_table_id}` S
        ON T.{q(key_column)} = S.{q(key_column)}
        WHEN MATCHED THEN
          UPDATE SET
          {update_set}
        WHEN NOT MATCHED THEN
          INSERT ({insert_cols}) VALUES ({insert_vals})
        """
        
        merge_job = client.query(merge_sql, location=LOCATION)
        merge_job.result()
        
        affected = getattr(merge_job, "num_dml_affected_rows", None)
        return int(affected or len(df_clean))
        
    finally:
        # 3) Limpa staging
        client.delete_table(staging_table_id, not_found_ok=True)


# =============================================================================
# FUNÇÕES ESPECÍFICAS
# =============================================================================

def get_periodos_disponiveis() -> List[str]:
    """Retorna lista de períodos disponíveis"""
    query = f"""
    SELECT DISTINCT referencia
    FROM `{TABLE_FATURA_ITENS}`
    WHERE referencia IS NOT NULL
    ORDER BY referencia DESC
    LIMIT 24
    """
    df = execute_query(query)
    return df['referencia'].tolist() if not df.empty else []


def get_historico_cliente(unidade_consumidora: str, limite: int = 12) -> pd.DataFrame:
    """Busca histórico de faturas de um cliente"""
    query = f"""
    SELECT 
        referencia as periodo,
        SUM(CASE WHEN codigo = '0D' THEN quantidade_registrada ELSE 0 END) as consumo_kwh,
        SUM(CASE WHEN codigo IN ('0R', '0S') THEN ABS(quantidade_registrada) ELSE 0 END) as injetada_kwh,
        MAX(total_pagar) as valor_fatura
    FROM `{TABLE_FATURA_ITENS}`
    WHERE unidade_consumidora = @uc
    GROUP BY referencia
    ORDER BY referencia DESC
    LIMIT {limite}
    """
    return execute_query(query, {"uc": unidade_consumidora})
