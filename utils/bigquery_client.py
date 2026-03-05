"""
ERB Tech - Cliente BigQuery
VERSÃO v9 - casting por schema destino + upsert robusto
"""

from __future__ import annotations

import uuid
from typing import Optional, List, Dict, Any, Union

import numpy as np
import pandas as pd
import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account


PROJECT_ID = "football-data-science"
DATASET_ID = "erb_tech"
LOCATION = "southamerica-east1"

TABLE_FATURA_ITENS = f"{PROJECT_ID}.{DATASET_ID}.fatura_itens"
TABLE_MEDIDORES = f"{PROJECT_ID}.{DATASET_ID}.medidores_leituras"
TABLE_CLIENTES = f"{PROJECT_ID}.{DATASET_ID}.info_clientes"
TABLE_BOLETOS = f"{PROJECT_ID}.{DATASET_ID}.boletos_calculados"
TABLE_EDIT_LOG = f"{PROJECT_ID}.{DATASET_ID}.edit_log"


@st.cache_resource
def get_bigquery_client() -> bigquery.Client:
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    return bigquery.Client(credentials=credentials, project=PROJECT_ID, location=LOCATION)


def _infer_bq_param(
    name: str,
    value: Any
) -> Union[bigquery.ScalarQueryParameter, bigquery.ArrayQueryParameter]:
    if isinstance(value, (list, tuple, set)):
        arr = [None if v is None else str(v) for v in value]
        return bigquery.ArrayQueryParameter(name, "STRING", arr)

    if isinstance(value, bool):
        return bigquery.ScalarQueryParameter(name, "BOOL", value)

    if isinstance(value, int):
        return bigquery.ScalarQueryParameter(name, "INT64", value)

    if isinstance(value, float):
        return bigquery.ScalarQueryParameter(name, "FLOAT64", value)

    return bigquery.ScalarQueryParameter(name, "STRING", None if value is None else str(value))


def execute_query(query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    client = get_bigquery_client()

    if params:
        job_config = bigquery.QueryJobConfig(
            query_parameters=[_infer_bq_param(k, v) for k, v in params.items()]
        )
        query_job = client.query(query, job_config=job_config)
    else:
        query_job = client.query(query)

    return query_job.to_dataframe()


def _ensure_string_dtype(series: pd.Series) -> pd.Series:
    s = series.copy()
    s = s.where(pd.notna(s), pd.NA)
    s = s.astype("string")
    s = s.str.strip()
    s = s.replace({"": pd.NA, "None": pd.NA, "nan": pd.NA, "NaN": pd.NA})
    return s.astype(object).where(s.notna(), None)


def normalize_for_bigquery(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # strings comuns
    for c in df.columns:
        if df[c].dtype == object:
            # não força tudo pra string (mantém datas/números se já vierem coerentes),
            # mas limpa strings "nan"/vazio.
            try:
                if pd.api.types.is_string_dtype(df[c]) or df[c].map(lambda x: isinstance(x, str) or x is None).all():
                    df[c] = _ensure_string_dtype(df[c])
            except Exception:
                pass

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


def _infer_bq_type_from_series(s: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(s):
        return "INT64"
    if pd.api.types.is_float_dtype(s):
        return "FLOAT64"
    if pd.api.types.is_bool_dtype(s):
        return "BOOL"
    return "STRING"


def ensure_table_columns_from_df(table_id: str, df: pd.DataFrame) -> None:
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
            msg = str(e).lower()
            if "already exists" in msg or "duplicate" in msg or "exists" in msg:
                continue
            raise


def _cast_df_to_dest_schema(df: pd.DataFrame, dest_schema_map: Dict[str, bigquery.SchemaField]) -> pd.DataFrame:
    """
    Faz cast de cada coluna do df conforme o tipo no BigQuery.
    Evita: custo_disp (INTEGER) chegando como float/string e virando NULL ou quebrando load.
    """
    out = df.copy()

    for col in list(out.columns):
        if col not in dest_schema_map:
            continue

        f = dest_schema_map[col]
        t = (f.field_type or "").upper()

        # NULL safe
        s = out[col]

        try:
            if t in ("STRING", "GEOGRAPHY"):
                out[col] = _ensure_string_dtype(s)

            elif t in ("INT64", "INTEGER"):
                # aceita float/string desde que seja inteiro "de verdade"
                x = pd.to_numeric(s, errors="coerce")
                x = x.round(0)
                out[col] = x.astype("Int64")
                out[col] = out[col].where(out[col].notna(), None)

            elif t in ("FLOAT64", "FLOAT", "NUMERIC", "BIGNUMERIC"):
                x = pd.to_numeric(s, errors="coerce")
                out[col] = x.where(x.notna(), None)

            elif t == "BOOL":
                if pd.api.types.is_bool_dtype(s):
                    out[col] = s.where(pd.notna(s), None)
                else:
                    x = s.astype(str).str.strip().str.lower()
                    out[col] = x.map(lambda v: True if v in ("true", "1", "t", "yes", "y") else (False if v in ("false", "0", "f", "no", "n") else None))

            elif t == "TIMESTAMP":
                x = pd.to_datetime(s, errors="coerce", utc=True)
                # BigQuery aceita datetime; manter como datetime64[ns, UTC]
                out[col] = x.where(x.notna(), None)

            elif t == "DATE":
                x = pd.to_datetime(s, errors="coerce").dt.date
                out[col] = x.where(pd.notna(x), None)

            else:
                # fallback
                out[col] = s.where(pd.notna(s), None)

        except Exception:
            # não quebra por cast, mantém original
            out[col] = s

    return out.replace({np.nan: None})


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

    ensure_table_columns_from_df(table_id, df_clean)

    staging_suffix = uuid.uuid4().hex[:12]
    staging_table_id = f"{table_id}__staging_{staging_suffix}"

    dest_table = client.get_table(table_id)
    dest_schema_map = {f.name: f for f in dest_table.schema}

    # drop cols que não existem (caso DDL tenha falhado)
    missing_in_dest = [c for c in df_clean.columns if c not in dest_schema_map]
    if missing_in_dest:
        df_clean = df_clean[[c for c in df_clean.columns if c in dest_schema_map]].copy()
        if df_clean.empty:
            raise ValueError(
                f"Não foi possível alinhar o schema de {table_id}. "
                f"Colunas faltantes: {missing_in_dest}"
            )
        dest_table = client.get_table(table_id)
        dest_schema_map = {f.name: f for f in dest_table.schema}

    # ✅ cast final conforme schema destino
    df_clean = _cast_df_to_dest_schema(df_clean, dest_schema_map)

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
