"""
ERB Tech - Cliente BigQuery
VERSÃO v7 - Query robusta (suporta ARRAY params) + helpers para páginas
"""

from __future__ import annotations

import uuid
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account


# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

PROJECT_ID = "football-data-science"
DATASET_ID = "erb_tech"
LOCATION = "southamerica-east1"

# Tabelas
TABLE_FATURA_ITENS = f"{PROJECT_ID}.{DATASET_ID}.fatura_itens"
TABLE_MEDIDORES = f"{PROJECT_ID}.{DATASET_ID}.medidores_leituras"
TABLE_CLIENTES = f"{PROJECT_ID}.{DATASET_ID}.info_clientes"
TABLE_BOLETOS = f"{PROJECT_ID}.{DATASET_ID}.boletos_calculados"
TABLE_EDIT_LOG = f"{PROJECT_ID}.{DATASET_ID}.edit_log"


# =============================================================================
# CLIENT / QUERY
# =============================================================================

@st.cache_resource
def get_bigquery_client() -> bigquery.Client:
    """Cria e retorna um cliente BigQuery."""
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    return bigquery.Client(credentials=credentials, project=PROJECT_ID, location=LOCATION)


def _infer_bq_param(name: str, value: Any) -> bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter:
    """
    Inferência simples de tipo de parâmetro.
    - listas viram ARRAY<STRING> por padrão
    - bool/int/float/str viram SCALAR
    """
    if isinstance(value, (list, tuple, set)):
        arr = [None if v is None else str(v) for v in value]
        return bigquery.ArrayQueryParameter(name, "STRING", arr)

    if isinstance(value, bool):
        return bigquery.ScalarQueryParameter(name, "BOOL", value)

    if isinstance(value, int):
        return bigquery.ScalarQueryParameter(name, "INT64", value)

    if isinstance(value, float):
        return bigquery.ScalarQueryParameter(name, "FLOAT64", value)

    # default string
    return bigquery.ScalarQueryParameter(name, "STRING", None if value is None else str(value))


def execute_query(query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Executa uma query SQL e retorna um DataFrame. Suporta parâmetros ARRAY."""
    client = get_bigquery_client()

    if params:
        job_config = bigquery.QueryJobConfig(
            query_parameters=[_infer_bq_param(k, v) for k, v in params.items()]
        )
        query_job = client.query(query, job_config=job_config)
    else:
        query_job = client.query(query)

    return query_job.to_dataframe()


# =============================================================================
# NORMALIZAÇÃO / UTILITÁRIOS
# =============================================================================

def _ensure_string_dtype(series: pd.Series) -> pd.Series:
    s = series.copy()
    s = s.where(pd.notna(s), pd.NA)
    s = s.astype("string")
    s = s.str.strip()
    s = s.replace({"": pd.NA, "None": pd.NA, "nan": pd.NA, "NaN": pd.NA})
    return s.astype(object).where(s.notna(), None)


def normalize_for_bigquery(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    string_cols = [
        "leitura_anterior", "leitura_atual", "proxima_leitura",
        "vencimento", "data_emissao", "referencia",
        "unidade_consumidora", "cliente_numero", "nome", "cnpj", "cnpj_cpf",
        "cep", "cidade_uf",
        "grupo_subgrupo_tensao", "classe_modalidade",
        "numero", "serie",
        "codigo", "descricao", "unidade", "id",
        "medidor", "tipo", "posto", "nota_fiscal_numero",
        "status", "check",
    ]

    for c in string_cols:
        if c in df.columns:
            df[c] = _ensure_string_dtype(df[c])

    if "dias" in df.columns:
        df["dias"] = pd.to_numeric(df["dias"], errors="coerce").astype("Int64")
        df["dias"] = df["dias"].where(df["dias"].notna(), None)

    float_cols = [
        "quantidade_registrada", "tarifa", "valor", "pis_valor",
        "cofins_base", "icms_aliquota", "icms_valor", "tarifa_sem_trib",
        "total_pagar", "constante", "fator", "total_apurado",
        "subvencao", "desconto_contratado",
        "custo_disp",
        "tarifa_cheia", "tarifa_injetada", "tarifa_paga_conc",
        "tarifa_erb", "tarifa_bol",
        "valor_band_amarela", "valor_band_vermelha",
        "valor_band_amar_desc", "valor_band_vrm_desc",
        "tarifa_total_boleto", "valor_total_boleto",
    ]

    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].where(df[c].notna(), None)

    int_cols = ["boleto", "n_fases"]
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
            df[c] = df[c].where(df[c].notna(), None)

    return df.replace({np.nan: None})


def _make_key_unique_deterministically(df: pd.DataFrame, key_column: str) -> pd.DataFrame:
    if key_column not in df.columns or df.empty:
        return df

    df = df.copy()
    df[key_column] = _ensure_string_dtype(df[key_column])
    df = df[df[key_column].notna()].copy()
    df = df[df[key_column].astype(str).str.strip() != ""].copy()
    if df.empty:
        return df

    dup_mask = df[key_column].duplicated(keep=False)
    if not dup_mask.any():
        return df

    sort_candidates = [c for c in df.columns if c != key_column]
    df = df.sort_values(by=[key_column] + sort_candidates, kind="mergesort").reset_index(drop=True)
    df["__dup_idx"] = df.groupby(key_column).cumcount()
    needs_suffix = df["__dup_idx"] > 0
    df.loc[needs_suffix, key_column] = (
        df.loc[needs_suffix, key_column].astype(str) + "-" + df.loc[needs_suffix, "__dup_idx"].astype(str)
    )
    df = df.drop(columns=["__dup_idx"])
    df[key_column] = _ensure_string_dtype(df[key_column])
    return df


def upsert_dataframe(df: pd.DataFrame, table_id: str, key_column: str = "id") -> int:
    if df is None or df.empty:
        return 0
    if key_column not in df.columns:
        raise ValueError(f"DataFrame precisa da coluna '{key_column}'.")

    client = get_bigquery_client()

    df_clean = normalize_for_bigquery(df)
    df_clean = _make_key_unique_deterministically(df_clean, key_column)
    if df_clean.empty:
        return 0

    staging_suffix = uuid.uuid4().hex[:12]
    staging_table_id = f"{table_id}__staging_{staging_suffix}"

    dest_table = client.get_table(table_id)
    dest_schema_map = {f.name: f for f in dest_table.schema}

    missing_in_dest = [c for c in df_clean.columns if c not in dest_schema_map]
    if missing_in_dest:
        raise ValueError(
            f"As colunas abaixo existem no DataFrame mas NÃO existem na tabela destino {table_id}: {missing_in_dest}"
        )

    staging_schema = [dest_schema_map[c] for c in df_clean.columns]

    try:
        job_config = bigquery.LoadJobConfig(
            schema=staging_schema,
            write_disposition="WRITE_TRUNCATE",
        )

        load_job = client.load_table_from_dataframe(
            df_clean,
            staging_table_id,
            location=LOCATION,
            job_config=job_config,
        )
        load_job.result()

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
        client.delete_table(staging_table_id, not_found_ok=True)


# =============================================================================
# HELPERS PARA AS PÁGINAS
# =============================================================================

def get_periodos_disponiveis(limit: int = 24) -> List[str]:
    q = f"""
    SELECT DISTINCT referencia
    FROM `{TABLE_FATURA_ITENS}`
    WHERE referencia IS NOT NULL
    ORDER BY referencia DESC
    LIMIT {int(limit)}
    """
    df = execute_query(q)
    return df["referencia"].astype(str).tolist() if df is not None and not df.empty else []


def get_historico_cliente(unidade_consumidora: str, limite: int = 12) -> pd.DataFrame:
    q = f"""
    SELECT
        referencia as periodo,
        SUM(CASE WHEN codigo = '0D' THEN quantidade_registrada ELSE 0 END) as consumo_kwh,
        SUM(CASE WHEN codigo IN ('0R', '0S') THEN ABS(quantidade_registrada) ELSE 0 END) as injetada_kwh,
        MAX(total_pagar) as valor_fatura
    FROM `{TABLE_FATURA_ITENS}`
    WHERE unidade_consumidora = @uc
    GROUP BY referencia
    ORDER BY referencia DESC
    LIMIT {int(limite)}
    """
    return execute_query(q, {"uc": unidade_consumidora})
