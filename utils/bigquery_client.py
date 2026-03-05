"""
ERB Tech - Cliente BigQuery
VERSÃO v8 - Query robusta (suporta ARRAY params) + helpers para páginas
+ Upsert com staging + MERGE
+ Auto-alinhamento de schema (ADD COLUMN quando o DF trouxer colunas novas)
Compatível com Python 3.9 (sem type union "|").
"""

from __future__ import annotations

import uuid
from typing import Optional, List, Dict, Any, Union

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


def _infer_bq_param(
    name: str,
    value: Any
) -> Union[bigquery.ScalarQueryParameter, bigquery.ArrayQueryParameter]:
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
    """
    Normaliza tipos antes de carregar para o BigQuery.
    - strings padronizadas (None no lugar de vazio/NaN)
    - floats/ints coerced
    """
    df = df.copy()

    string_cols = [
        # Fatura
        "leitura_anterior", "leitura_atual", "proxima_leitura",
        "vencimento", "data_emissao", "referencia", "periodo",
        "unidade_consumidora", "cliente_numero", "nome", "cnpj", "cnpj_cpf",
        "cep", "cidade_uf",
        "grupo_subgrupo_tensao", "classe_modalidade",
        "numero", "serie",
        # Itens
        "codigo", "descricao", "unidade", "id",
        # Medidores
        "medidor", "tipo", "posto", "nota_fiscal_numero",
        # Clientes / checks
        "status", "check",
    ]

    for c in string_cols:
        if c in df.columns:
            df[c] = _ensure_string_dtype(df[c])

    if "dias" in df.columns:
        df["dias"] = pd.to_numeric(df["dias"], errors="coerce").astype("Int64")
        df["dias"] = df["dias"].where(df["dias"].notna(), None)

    # Floats (inclui campos do cálculo fiel)
    float_cols = [
        "quantidade_registrada", "tarifa", "valor", "pis_valor",
        "cofins_base", "icms_aliquota", "icms_valor", "tarifa_sem_trib",
        "total_pagar", "constante", "fator", "total_apurado",
        "subvencao", "desconto_contratado", "custo_disp",

        # cálculo (calc_engine)
        "medidores_apurado", "injetada", "med_inj_tusd",
        "tarifa_cheia_trib", "tarifa_cheia_trib2", "tarifa_cheia_trib3",
        "energia_inj_tusd_tarifa", "energia_injet_te_tarifa",
        "tarifa_inj_tusd", "tarifa_inj_te",

        "tarifa_cheia",
        "tarifa_paga_conc",
        "tarifa_erb",
        "tarifa_bol",

        "bandeira_amarela_tarifa",
        "band_am_injet_tarifa",
        "band_vermelha_tarifa",
        "band_vrm_injet_tarifa",

        "valor_band_amarela",
        "valor_band_vermelha",
        "valor_band_amar_desc",
        "valor_band_vrm_desc",

        "tarifa_total_boleto",
        "valor_total_boleto",
    ]

    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].where(df[c].notna(), None)

    # Inteiros
    int_cols = ["boleto", "n_fases", "gerador"]
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
            df[c] = df[c].where(df[c].notna(), None)

    return df.replace({np.nan: None})


def _make_key_unique_deterministically(df: pd.DataFrame, key_column: str) -> pd.DataFrame:
    """
    Garante chave única quando o DF vem com duplicatas no key_column:
    - ordena deterministicamente e adiciona sufixo "-{idx}" a partir da 2a ocorrência.
    """
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


# =============================================================================
# SCHEMA AUTO-FIX (ADD COLUMN)
# =============================================================================

def _infer_bq_type_from_series(s: pd.Series) -> str:
    # Regras simples e seguras p/ v0
    if pd.api.types.is_integer_dtype(s):
        return "INT64"
    if pd.api.types.is_float_dtype(s):
        return "FLOAT64"
    if pd.api.types.is_bool_dtype(s):
        return "BOOL"
    return "STRING"


def ensure_table_columns_from_df(table_id: str, df: pd.DataFrame) -> None:
    """
    Garante que a tabela destino tenha todas as colunas do DF.
    Se faltar coluna, tenta ALTER TABLE ADD COLUMN.
    """
    client = get_bigquery_client()
    dest_table = client.get_table(table_id)
    dest_cols = {f.name for f in dest_table.schema}

    missing = [c for c in df.columns if c not in dest_cols]
    if not missing:
        return

    for c in missing:
        bq_type = _infer_bq_type_from_series(df[c]) if c in df.columns else "STRING"
        ddl = f"ALTER TABLE `{table_id}` ADD COLUMN `{c}` {bq_type}"
        try:
            client.query(ddl, location=LOCATION).result()
        except Exception as e:
            # Se já existe / corrida / etc, ignora. Caso contrário, propaga.
            msg = str(e).lower()
            if "already exists" in msg or "duplicate" in msg or "exists" in msg:
                continue
            raise


# =============================================================================
# UPSERT (STAGING + MERGE)
# =============================================================================

def upsert_dataframe(df: pd.DataFrame, table_id: str, key_column: str = "id") -> int:
    """
    Upsert em BigQuery:
    - normaliza tipos
    - garante schema (adiciona colunas faltantes)
    - carrega em staging
    - MERGE na tabela destino
    """
    if df is None or df.empty:
        return 0
    if key_column not in df.columns:
        raise ValueError(f"DataFrame precisa da coluna '{key_column}'.")

    client = get_bigquery_client()

    df_clean = normalize_for_bigquery(df)
    df_clean = _make_key_unique_deterministically(df_clean, key_column)
    if df_clean.empty:
        return 0

    # garante schema antes de montar schema do staging
    ensure_table_columns_from_df(table_id, df_clean)

    staging_suffix = uuid.uuid4().hex[:12]
    staging_table_id = f"{table_id}__staging_{staging_suffix}"

    dest_table = client.get_table(table_id)
    dest_schema_map = {f.name: f for f in dest_table.schema}

    missing_in_dest = [c for c in df_clean.columns if c not in dest_schema_map]
    if missing_in_dest:
        # Se ainda faltar (permissão/DDL falhou), cai para modo seguro: drop das colunas extras
        df_clean = df_clean[[c for c in df_clean.columns if c in dest_schema_map]].copy()
        if df_clean.empty:
            raise ValueError(
                f"Não foi possível alinhar o schema de {table_id}. "
                f"Colunas faltantes: {missing_in_dest}"
            )

        # Recarrega schema map após drop
        dest_table = client.get_table(table_id)
        dest_schema_map = {f.name: f for f in dest_table.schema}

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
