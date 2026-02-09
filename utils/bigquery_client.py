"""
ERB Tech - Cliente BigQuery
VERSÃO FINAL v6 - UPSERT robusto (chave única determinística) + staging com UUID
"""

from __future__ import annotations

import uuid
from typing import Optional, List, Dict

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
    """Executa uma query SQL e retorna um DataFrame."""
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


# =============================================================================
# NORMALIZAÇÃO / UTILITÁRIOS
# =============================================================================

def _ensure_string_dtype(series: pd.Series) -> pd.Series:
    """
    Converte para dtype 'string' do pandas, preservando NULLs como <NA>/None.
    Isso evita o BigQuery inferir INT64 quando só tem números (ex.: leituras).
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

    # Converte NA para None (BigQuery python client prefere None ao <NA>)
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


def _make_key_unique_deterministically(df: pd.DataFrame, key_column: str) -> pd.DataFrame:
    """
    Garante que a coluna de chave (ex.: 'id') seja única dentro do DataFrame.

    Por que isso existe?
    - O MERGE do BigQuery exige no máximo 1 linha de origem por chave.
    - O código v5 resolvia isso com drop_duplicates(...), mas isso REMOVE linhas reais
      quando o parse gera IDs repetidos (muito comum quando há mais de um item com o
      mesmo código e a mesma tarifa na fatura).
    - Aqui, em vez de descartar, nós DESAMBIGUAMOS de forma determinística, adicionando
      um sufixo "-<n>" apenas nas duplicatas.

    Observação:
    - Se a sua geração de 'id' já for única, nada muda.
    - Se houver duplicatas, o primeiro registro mantém o id original e os demais viram
      id-1, id-2, ...
    """
    if key_column not in df.columns or df.empty:
        return df

    df = df.copy()
    df[key_column] = _ensure_string_dtype(df[key_column])

    # Remove vazios (não dá para fazer UPSERT sem chave)
    df = df[df[key_column].notna()].copy()
    df = df[df[key_column].astype(str).str.strip() != ""].copy()

    if df.empty:
        return df

    dup_mask = df[key_column].duplicated(keep=False)
    if not dup_mask.any():
        return df

    # Ordenação determinística dentro de cada grupo duplicado
    sort_candidates = [
        "unidade_consumidora", "cliente_numero", "referencia",
        "codigo", "descricao", "unidade",
        "quantidade_registrada", "tarifa", "valor",
        "pis_valor", "icms_aliquota", "icms_valor",
        "vencimento", "data_emissao", "numero", "serie"
    ]
    sort_cols = [c for c in sort_candidates if c in df.columns]

    if sort_cols:
        df = df.sort_values(by=[key_column] + sort_cols, kind="mergesort").reset_index(drop=True)
    else:
        df = df.sort_values(by=[key_column], kind="mergesort").reset_index(drop=True)

    df["__dup_idx"] = df.groupby(key_column).cumcount()

    needs_suffix = df["__dup_idx"] > 0
    df.loc[needs_suffix, key_column] = (
        df.loc[needs_suffix, key_column].astype(str) + "-" + df.loc[needs_suffix, "__dup_idx"].astype(str)
    )

    df = df.drop(columns=["__dup_idx"])
    df[key_column] = _ensure_string_dtype(df[key_column])
    return df


# =============================================================================
# UPSERT
# =============================================================================

def upsert_dataframe(df: pd.DataFrame, table_id: str, key_column: str = "id") -> int:
    """
    UPSERT via MERGE.

    Correções importantes (v6):
      1) NÃO descarta linhas quando o parse gera chaves repetidas:
         - v5 fazia drop_duplicates(subset=[id]), o que silenciosamente removia itens.
         - v6 cria chaves únicas determinísticas para duplicatas (id-1, id-2, ...).
      2) Staging table com UUID (evita colisão entre sessões/execuções no mesmo segundo).
      3) Continua forçando o schema do staging com base no schema real da tabela destino
         (sem inferência, preservando tipos/strings).
    """
    if df is None or df.empty:
        return 0

    if key_column not in df.columns:
        raise ValueError(f"DataFrame precisa da coluna '{key_column}'.")

    client = get_bigquery_client()

    # 1) Normalização crítica (tipos / None)
    df_clean = normalize_for_bigquery(df)

    # 2) Garante chave presente e única (sem perder linhas)
    df_clean = _make_key_unique_deterministically(df_clean, key_column)

    if df_clean.empty:
        return 0

    # 3) Staging table (UUID evita colisão)
    staging_suffix = uuid.uuid4().hex[:12]
    staging_table_id = f"{table_id}__staging_{staging_suffix}"

    # 4) Força schema do staging com base no schema real da tabela destino
    dest_table = client.get_table(table_id)
    dest_schema_map = {f.name: f for f in dest_table.schema}

    missing_in_dest = [c for c in df_clean.columns if c not in dest_schema_map]
    if missing_in_dest:
        raise ValueError(
            f"As colunas abaixo existem no DataFrame mas NÃO existem na tabela destino {table_id}: {missing_in_dest}"
        )

    staging_schema = [dest_schema_map[c] for c in df_clean.columns]

    try:
        # 5) Load para staging com schema explícito (sem inferência)
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

        # 6) MERGE
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
# FUNÇÕES ESPECÍFICAS
# =============================================================================

def get_periodos_disponiveis() -> List[str]:
    """Retorna lista de períodos disponíveis."""
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
    """Busca histórico de faturas de um cliente."""
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
