"""
ERB Tech - Cliente BigQuery
VERSÃO FINAL v5 - Força schema no staging + garante STRING para campos de leitura
"""

import streamlit as st
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.oauth2 import service_account
from typing import Optional, List, Dict
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
        raise


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


def _ensure_string_dtype(series: pd.Series) -> pd.Series:
    """
    Converte para dtype 'string' do pandas, preservando NULLs como <NA>/None.
    Isso evita o BigQuery inferir INT64 quando só tem números.
    """
    s = series.copy()

    # Troca NaN/None por NA do pandas
    s = s.where(pd.notna(s), pd.NA)

    # Converte para string sem transformar NA em "nan"/"None"
    s = s.astype("string")

    # Trim
    s = s.str.strip()

    # Normaliza vazios para NA
    s = s.replace({"": pd.NA, "None": pd.NA, "nan": pd.NA, "NaN": pd.NA})

    # Converte NA para None (BigQuery python client gosta mais de None do que <NA>)
    return s.astype(object).where(s.notna(), None)


def normalize_for_bigquery(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalização COMPLETA para BigQuery
    - Garante que campos textuais são realmente dtype string/object (evita inferência INT64)
    - Converte dias para Int64
    - Converte floats
    - Substitui NaN por None
    """
    df = df.copy()

    # IMPORTANTE: Campos que DEVEM ser STRING no BigQuery
    string_cols = [
        # Leituras / datas textuais
        "leitura_anterior", "leitura_atual", "proxima_leitura",
        "vencimento", "data_emissao", "referencia",

        # Clientes
        "unidade_consumidora", "cliente_numero", "nome", "cnpj", "cnpj_cpf",
        "cep", "cidade_uf",

        # Fatura/itens
        "grupo_subgrupo_tensao", "numero", "serie",
        "codigo", "descricao", "unidade", "id",

        # Medidores
        "medidor", "tipo", "posto", "nota_fiscal_numero"
    ]

    for c in string_cols:
        if c in df.columns:
            df[c] = _ensure_string_dtype(df[c])

    # Campos inteiros
    if "dias" in df.columns:
        df["dias"] = pd.to_numeric(df["dias"], errors="coerce").astype("Int64")
        df["dias"] = df["dias"].where(df["dias"].notna(), None)

    # Campos numéricos float
    float_cols = [
        "quantidade_registrada", "tarifa", "valor", "pis_valor",
        "cofins_base", "icms_aliquota", "icms_valor", "tarifa_sem_trib",
        "total_pagar", "constante", "fator", "total_apurado",
        "subvencao", "desconto_contratado"
    ]

    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].where(df[c].notna(), None)

    # Qualquer NaN remanescente -> None
    return df.replace({np.nan: None})


def upsert_dataframe(df: pd.DataFrame, table_id: str, key_column: str = "id") -> int:
    """
    UPSERT via MERGE
    FIX v5:
      - staging table é carregada com schema do destino (subconjunto)
      - evita inferência errada (ex.: leitura_anterior virar INT64)
    """
    if df is None or df.empty:
        return 0

    if key_column not in df.columns:
        raise ValueError(f"DataFrame precisa da coluna '{key_column}'.")

    client = get_bigquery_client()

    # Mantém lógica original
    df_clean = df.drop_duplicates(subset=[key_column], keep="first").copy()
    df_clean = df_clean.dropna(subset=[key_column])

    # Normalização crítica
    df_clean = normalize_for_bigquery(df_clean)

    if df_clean.empty:
        return 0

    # Staging table
    ts = int(time.time())
    staging_table_id = f"{table_id}__staging_{ts}"

    # ---- NOVO: Força schema do staging com base no schema real da tabela destino
    dest_table = client.get_table(table_id)
    dest_schema_map = {f.name: f for f in dest_table.schema}

    # Usa somente campos que existem no DataFrame (para não dar mismatch)
    staging_schema = [dest_schema_map[c] for c in df_clean.columns if c in dest_schema_map]

    # Se algum campo do DF não existe na tabela, melhor falhar com mensagem clara
    missing_in_dest = [c for c in df_clean.columns if c not in dest_schema_map]
    if missing_in_dest:
        raise ValueError(
            f"As colunas abaixo existem no DataFrame mas NÃO existem na tabela destino {table_id}: {missing_in_dest}"
        )

    try:
        # 1) Load para staging com schema explícito (sem inferência!)
        job_config = bigquery.LoadJobConfig(
            schema=staging_schema,
            write_disposition="WRITE_TRUNCATE"
        )

        load_job = client.load_table_from_dataframe(
            df_clean,
            staging_table_id,
            location=LOCATION,
            job_config=job_config
        )
        load_job.result()

        # 2) MERGE
        cols = list(df_clean.columns)
        non_key_cols = [c for c in cols if c != key_column]

        def q(c: str) -> str:
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
    return df["referencia"].tolist() if not df.empty else []


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