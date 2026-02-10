"""
ERB Tech - Cliente BigQuery
VERSÃO v7
- Inclui 'classe_modalidade' em fatura_itens (STRING)
- Suporta parâmetros ARRAY no execute_query (p/ checar clientes em lote)
- Inclui n_fases (INT64) e custo_disp (INT64) na normalização quando existirem
- Inclui status como STRING na normalização
- Inclui helpers: checar clientes faltantes + formulário Streamlit de cadastro
"""

from __future__ import annotations

import uuid
from typing import Optional, List, Dict, Any, Iterable

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


def _infer_bq_scalar_type(v: Any) -> str:
    if isinstance(v, bool):
        return "BOOL"
    if isinstance(v, int) and not isinstance(v, bool):
        return "INT64"
    if isinstance(v, float):
        return "FLOAT64"
    return "STRING"


def execute_query(query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Executa SQL e retorna DataFrame.
    v7: suporta params escalares e arrays (IN UNNEST(@param)).
    """
    client = get_bigquery_client()

    if params:
        qps = []
        for k, v in params.items():
            if isinstance(v, (list, tuple, set)):
                vals = list(v)
                # tenta inferir tipo pelo primeiro não-nulo
                non_null = next((x for x in vals if x is not None), None)
                item_type = _infer_bq_scalar_type(non_null) if non_null is not None else "STRING"
                qps.append(bigquery.ArrayQueryParameter(k, item_type, vals))
            else:
                qps.append(bigquery.ScalarQueryParameter(k, _infer_bq_scalar_type(v), v))

        job_config = bigquery.QueryJobConfig(query_parameters=qps)
        query_job = client.query(query, job_config=job_config)
    else:
        query_job = client.query(query)

    return query_job.to_dataframe()


# =============================================================================
# NORMALIZAÇÃO
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

    # Strings (inclui a nova classe_modalidade)
    string_cols = [
        "leitura_anterior", "leitura_atual", "proxima_leitura",
        "vencimento", "data_emissao", "referencia",
        "unidade_consumidora", "cliente_numero", "nome", "cnpj", "cnpj_cpf",
        "cep", "cidade_uf",
        "grupo_subgrupo_tensao", "classe_modalidade",  # <-- NOVO
        "numero", "serie",
        "codigo", "descricao", "unidade", "id",
        "medidor", "tipo", "posto", "nota_fiscal_numero",
        "status",  # <-- usuário
    ]
    for c in string_cols:
        if c in df.columns:
            df[c] = _ensure_string_dtype(df[c])

    # Inteiros (v7: n_fases e custo_disp além de dias)
    int_cols = ["dias", "n_fases", "custo_disp"]
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
            df[c] = df[c].where(df[c].notna(), None)

    # Floats
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

    return df.replace({np.nan: None})


# =============================================================================
# UPSERT (MERGE)
# =============================================================================

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

    sort_candidates = [
        "unidade_consumidora", "cliente_numero", "referencia",
        "codigo", "descricao", "unidade",
        "quantidade_registrada", "tarifa", "valor",
        "pis_valor", "icms_aliquota", "icms_valor",
        "vencimento", "data_emissao", "numero", "serie",
        "classe_modalidade",
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
# DIMENSÃO: info_clientes (checagem + formulário)
# =============================================================================

def infer_n_fases(classe_modalidade: Optional[str]) -> Optional[int]:
    if not classe_modalidade:
        return None
    s = str(classe_modalidade).upper()
    if "TRIF" in s:
        return 3
    if "BIF" in s:
        return 2
    if "MONOF" in s or "MONO" in s:
        return 1
    return None


def infer_custo_disp(n_fases: Optional[int]) -> Optional[int]:
    if n_fases == 3:
        return 100
    if n_fases == 2:
        return 50
    if n_fases == 1:
        return 30
    return None


def get_clientes_existentes_por_uc(ucs: Iterable[str]) -> set:
    ucs = [str(x).strip() for x in ucs if x is not None and str(x).strip()]
    if not ucs:
        return set()

    q = f"""
    SELECT DISTINCT unidade_consumidora
    FROM `{TABLE_CLIENTES}`
    WHERE unidade_consumidora IN UNNEST(@ucs)
    """
    df = execute_query(q, {"ucs": ucs})
    return set(df["unidade_consumidora"].astype(str).tolist()) if not df.empty else set()


def ucs_faltantes_no_cadastro(df_itens: pd.DataFrame) -> List[str]:
    if df_itens is None or df_itens.empty or "unidade_consumidora" not in df_itens.columns:
        return []

    ucs = df_itens["unidade_consumidora"].dropna().astype(str).unique().tolist()
    existentes = get_clientes_existentes_por_uc(ucs)
    faltantes = [uc for uc in ucs if uc not in existentes]
    return sorted(faltantes)


def carregar_header_mais_recente_da_uc(df_itens: pd.DataFrame, uc: str) -> Dict[str, Any]:
    """
    Pega do DataFrame parseado (lote atual) os campos mais úteis para pré-preencher o form.
    """
    if df_itens is None or df_itens.empty:
        return {"unidade_consumidora": uc}

    sub = df_itens[df_itens["unidade_consumidora"].astype(str) == str(uc)]
    if sub.empty:
        return {"unidade_consumidora": uc}

    row = sub.iloc[0].to_dict()
    return {
        "unidade_consumidora": row.get("unidade_consumidora"),
        "cliente_numero": row.get("cliente_numero"),
        "nome": row.get("nome"),
        "cnpj_cpf": row.get("cnpj") or row.get("cnpj_cpf"),
        "cep": row.get("cep"),
        "cidade_uf": row.get("cidade_uf"),
        "grupo_subgrupo_tensao": row.get("grupo_subgrupo_tensao"),
        "classe_modalidade": row.get("classe_modalidade"),
    }


def form_cadastro_cliente(header_prefill: Dict[str, Any]) -> bool:
    """
    Formulário Streamlit para cadastrar/validar cliente em info_clientes.
    Retorna True quando salvou.
    """
    uc = str(header_prefill.get("unidade_consumidora") or "").strip()

    # inferências
    classe_modalidade = header_prefill.get("classe_modalidade")
    n_fases_guess = infer_n_fases(classe_modalidade) or 3
    custo_guess = infer_custo_disp(n_fases_guess) or 100

    with st.form(key=f"form_cliente_{uc}"):
        st.subheader(f"Cadastro do cliente (UC: {uc})")

        unidade_consumidora = st.text_input("Unidade Consumidora (UC)", value=uc)
        cliente_numero = st.text_input("Cliente (número)", value=str(header_prefill.get("cliente_numero") or ""))
        nome = st.text_input("Nome", value=str(header_prefill.get("nome") or ""))
        cnpj_cpf = st.text_input("CPF/CNPJ", value=str(header_prefill.get("cnpj_cpf") or ""))
        cep = st.text_input("CEP", value=str(header_prefill.get("cep") or ""))
        cidade_uf = st.text_input("Cidade/UF", value=str(header_prefill.get("cidade_uf") or ""))

        grupo_subgrupo_tensao = st.text_input(
            "Grupo/Subgrupo Tensão",
            value=str(header_prefill.get("grupo_subgrupo_tensao") or "")
        )

        classe_modalidade_in = st.text_input(
            "Classificação / Modalidade Tarifária / Tipo de Fornecimento",
            value=str(classe_modalidade or "")
        )

        # n_fases e custo_disp
        n_fases = st.selectbox("n_fases (1/2/3)", options=[1, 2, 3], index=[1,2,3].index(n_fases_guess))
        custo_disp = st.number_input("custo_disp (kWh)", min_value=0, step=1, value=int(infer_custo_disp(n_fases) or custo_guess))

        # Atribuídos pelo usuário
        desconto_contratado = st.number_input("desconto_contratado (R$)", value=float(0.0), step=1.0)
        subvencao = st.number_input("subvencao (R$)", value=float(0.0), step=1.0)
        status = st.selectbox("status", options=["ATIVO", "INATIVO", "PENDENTE"], index=0)

        salvar = st.form_submit_button("Salvar cliente")

    if not salvar:
        return False

    # monta DF para upsert
    df_cliente = pd.DataFrame([{
        "unidade_consumidora": unidade_consumidora,
        "cliente_numero": cliente_numero,
        "nome": nome,
        "cnpj_cpf": cnpj_cpf,
        "cep": cep,
        "cidade_uf": cidade_uf,
        "grupo_subgrupo_tensao": grupo_subgrupo_tensao,
        "classe_modalidade": classe_modalidade_in,
        "n_fases": int(n_fases),
        "custo_disp": int(custo_disp),
        "desconto_contratado": float(desconto_contratado),
        "subvencao": float(subvencao),
        "status": status,
    }])

    # IMPORTANTE: a chave da dimensão deve ser UC (ajuste se sua tabela usar outra)
    upsert_dataframe(df_cliente, TABLE_CLIENTES, key_column="unidade_consumidora")
    st.success(f"Cliente UC {unidade_consumidora} salvo em info_clientes.")
    return True
