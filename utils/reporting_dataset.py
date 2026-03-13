# utils/reporting_dataset.py
from __future__ import annotations

from typing import Optional, Dict, Any

import pandas as pd
import streamlit as st

from utils.bigquery_client import (
    execute_query,
    TABLE_FATURAS_WORKFLOW,
    TABLE_BOLETOS,
    TABLE_CLIENTES,
    TABLE_FATURA_ITENS,
)


def _coerce_periodo_sort(df: pd.DataFrame, col: str = "periodo") -> pd.DataFrame:
    """
    Ordena períodos no formato MM/YYYY.
    """
    if df is None or df.empty or col not in df.columns:
        return df

    out = df.copy()
    out["_periodo_dt"] = pd.to_datetime("01/" + out[col].astype(str), format="%d/%m/%Y", errors="coerce")
    out = out.sort_values("_periodo_dt")
    return out.drop(columns=["_periodo_dt"])


@st.cache_data(ttl=300, show_spinner=False)
def load_reporting_fact(
    periodo: Optional[str] = None,
    cliente_numero: Optional[str] = None,
    unidade_consumidora: Optional[str] = None,
) -> pd.DataFrame:
    """
    Base consolidada para report e dashboard.

    Granularidade:
    - 1 linha por NF / UC / período

    Fontes:
    - faturas_workflow: status operacionais
    - boletos_calculados: resultado do cálculo
    - info_clientes: cadastro/contrato
    - fatura_itens: dados parseados/agregados da NF
    """
    where_parts = ["1=1"]
    params: Dict[str, Any] = {}

    if periodo:
        where_parts.append("COALESCE(b.periodo, it.referencia, wf.referencia) = @periodo")
        params["periodo"] = str(periodo)

    if cliente_numero:
        where_parts.append("COALESCE(wf.cliente_numero, it.cliente_numero, b.cliente_numero, cli.cliente_numero) = @cliente_numero")
        params["cliente_numero"] = str(cliente_numero)

    if unidade_consumidora:
        where_parts.append("COALESCE(wf.unidade_consumidora, it.unidade_consumidora, b.unidade_consumidora, cli.unidade_consumidora) = @uc")
        params["uc"] = str(unidade_consumidora)

    where_sql = " AND ".join(where_parts)

    query = f"""
    WITH itens_nf AS (
      SELECT
        numero AS nota_fiscal,
        ANY_VALUE(referencia) AS referencia,
        ANY_VALUE(unidade_consumidora) AS unidade_consumidora,
        ANY_VALUE(cliente_numero) AS cliente_numero,
        ANY_VALUE(nome) AS nome_fatura,
        ANY_VALUE(cnpj) AS cnpj_fatura,
        ANY_VALUE(cep) AS cep_fatura,
        ANY_VALUE(cidade_uf) AS cidade_uf_fatura,
        ANY_VALUE(vencimento) AS vencimento_fatura,
        ANY_VALUE(total_pagar) AS total_pagar_fatura,
        ANY_VALUE(classe_modalidade) AS classe_modalidade_fatura,
        ANY_VALUE(data_emissao) AS data_emissao_fatura,
        SUM(CASE WHEN codigo = '0D' THEN quantidade_registrada ELSE 0 END) AS consumo_kwh_parseado,
        SUM(CASE WHEN codigo IN ('0R', '0S') THEN ABS(quantidade_registrada) ELSE 0 END) AS injetada_kwh_parseada,
        SUM(CASE WHEN codigo = '0R' THEN valor ELSE 0 END) AS inj_te_valor,
        SUM(CASE WHEN codigo = '0S' THEN valor ELSE 0 END) AS inj_tusd_valor,
        SUM(CASE WHEN codigo IN ('2L', '2M', '2U', '2V') THEN valor ELSE 0 END) AS bandeiras_valor_parseado,
        AVG(CASE WHEN codigo IN ('0D', '0E') THEN tarifa_sem_trib ELSE NULL END) AS tarifa_media_sem_trib_parseada
      FROM `{TABLE_FATURA_ITENS}`
      WHERE numero IS NOT NULL
      GROUP BY numero
    )
    SELECT
      wf.id AS workflow_id,
      COALESCE(wf.nota_fiscal, wf.id, b.nota_fiscal, it.nota_fiscal) AS nota_fiscal,
      COALESCE(b.periodo, it.referencia, wf.referencia) AS periodo,
      COALESCE(wf.unidade_consumidora, b.unidade_consumidora, it.unidade_consumidora, cli.unidade_consumidora) AS unidade_consumidora,
      COALESCE(wf.cliente_numero, b.cliente_numero, it.cliente_numero, cli.cliente_numero) AS cliente_numero,
      COALESCE(b.nome, wf.nome, cli.nome, it.nome_fatura) AS nome,
      COALESCE(cli.cnpj_cpf, wf.cnpj_cpf, it.cnpj_fatura, cli.cnpj) AS cnpj_cpf,
      cli.endereco,
      cli.cidade_uf,
      cli.cep,
      cli.email,
      cli.telefone,

      COALESCE(it.data_emissao_fatura, '') AS data_emissao_fatura,
      COALESCE(it.vencimento_fatura, wf.vencimento) AS vencimento,
      COALESCE(it.classe_modalidade_fatura, wf.classe_modalidade) AS classe_modalidade,

      wf.status_parse,
      wf.status_validacao,
      wf.validado_por,
      wf.validado_em,
      wf.status_calculo,
      wf.calculado_em,
      wf.status_emissao,
      wf.emitido_em,
      wf.observacoes,

      cli.desconto_contratado,
      cli.subvencao,
      cli.n_fases,
      cli.custo_disp,
      cli.status AS status_cliente,

      COALESCE(it.total_pagar_fatura, wf.total_pagar) AS valor_concessionaria,
      b.valor_final AS valor_boleto,
      CASE
        WHEN COALESCE(it.total_pagar_fatura, wf.total_pagar) IS NOT NULL AND b.valor_final IS NOT NULL
        THEN COALESCE(it.total_pagar_fatura, wf.total_pagar) - b.valor_final
        ELSE NULL
      END AS economia_calculada,

      -- Parseado
      it.consumo_kwh_parseado,
      it.injetada_kwh_parseada,
      it.bandeiras_valor_parseado,
      it.tarifa_media_sem_trib_parseada,

      -- Calculado
      b.consumo_kwh,
      b.injetada_kwh,
      b.tarifa_cheia,
      b.tarifa_injetada,
      b.tarifa_liquida,
      b.valor_erb,
      b.valor_bandeiras,
      b.check,
      b.medidores_apurado,
      b.injetada,
      b.boleto,
      b.tarifa_cheia_trib,
      b.tarifa_cheia_trib2,
      b.gerador,
      b.energia_inj_tusd,
      b.energia_injet_te,
      b.tarifa_cheia_trib3,
      b.tarifa_inj_tusd,
      b.tarifa_inj_te,
      b.tarifa_paga_conc,
      b.tarifa_erb,

      -- Campos candidatos para o report
      COALESCE(b.tarifa_cheia_trib2, b.tarifa_cheia) AS tarifa_sem_desconto_candidata,
      COALESCE(b.tarifa_erb, b.tarifa_liquida) AS tarifa_com_desconto_candidata,
      COALESCE(b.injetada_kwh, it.injetada_kwh_parseada) AS energia_compensada_kwh,
      COALESCE(b.valor_final, COALESCE(it.total_pagar_fatura, wf.total_pagar)) AS total_a_pagar_report,

      wf.arquivo_nome_original,
      wf.pdf_uri,
      wf.created_at,
      wf.updated_at
    FROM `{TABLE_FATURAS_WORKFLOW}` wf
    LEFT JOIN itens_nf it
      ON wf.id = it.nota_fiscal
    LEFT JOIN `{TABLE_BOLETOS}` b
      ON wf.id = b.id
    LEFT JOIN `{TABLE_CLIENTES}` cli
      ON COALESCE(wf.unidade_consumidora, it.unidade_consumidora, b.unidade_consumidora) = cli.unidade_consumidora
    WHERE {where_sql}
    ORDER BY periodo DESC, cliente_numero, unidade_consumidora, nota_fiscal
    """
    df = execute_query(query, params)
    if df is None:
        return pd.DataFrame()
    return df


@st.cache_data(ttl=300, show_spinner=False)
def load_report_header(
    nota_fiscal: str,
) -> pd.DataFrame:
    """
    Cabeçalho do demonstrativo individual por NF/associado.
    """
    df = load_reporting_fact()
    if df is None or df.empty:
        return pd.DataFrame()

    out = df[df["nota_fiscal"].astype(str) == str(nota_fiscal)].copy()
    if out.empty:
        return pd.DataFrame()

    return out.head(1)


@st.cache_data(ttl=300, show_spinner=False)
def load_report_history(
    cliente_numero: Optional[str] = None,
    unidade_consumidora: Optional[str] = None,
    ano: Optional[int] = None,
) -> pd.DataFrame:
    """
    Histórico mensal para o gráfico/tabela do demonstrativo.
    Pode ser filtrado por cliente ou por UC.
    """
    where_parts = ["1=1"]
    params: Dict[str, Any] = {}

    if cliente_numero:
        where_parts.append("cliente_numero = @cliente_numero")
        params["cliente_numero"] = str(cliente_numero)

    if unidade_consumidora:
        where_parts.append("unidade_consumidora = @uc")
        params["uc"] = str(unidade_consumidora)

    if ano:
        where_parts.append("SUBSTR(periodo, 4, 4) = @ano")
        params["ano"] = str(ano)

    where_sql = " AND ".join(where_parts)

    query = f"""
    WITH base AS (
      SELECT
        COALESCE(b.periodo, wf.referencia) AS periodo,
        COALESCE(b.cliente_numero, wf.cliente_numero) AS cliente_numero,
        COALESCE(b.unidade_consumidora, wf.unidade_consumidora) AS unidade_consumidora,
        COALESCE(b.nome, wf.nome) AS nome,
        COALESCE(b.valor_final, 0) AS valor_boleto,
        COALESCE(wf.total_pagar, 0) AS valor_concessionaria,
        COALESCE(wf.status_emissao, 'nao_emitido') AS status_emissao
      FROM `{TABLE_FATURAS_WORKFLOW}` wf
      LEFT JOIN `{TABLE_BOLETOS}` b
        ON wf.id = b.id
    )
    SELECT
      periodo,
      cliente_numero,
      unidade_consumidora,
      ANY_VALUE(nome) AS nome,
      SUM(valor_concessionaria) AS valor_concessionaria,
      SUM(valor_boleto) AS valor_boleto,
      SUM(valor_concessionaria - valor_boleto) AS economia_calculada,
      MAX(status_emissao) AS status_emissao
    FROM base
    WHERE {where_sql}
    GROUP BY periodo, cliente_numero, unidade_consumidora
    ORDER BY periodo
    """
    df = execute_query(query, params)
    if df is None:
        return pd.DataFrame()

    return _coerce_periodo_sort(df, "periodo")


@st.cache_data(ttl=300, show_spinner=False)
def load_dashboard_drilldown(
    periodo: Optional[str] = None,
    cliente_numero: Optional[str] = None,
    unidade_consumidora: Optional[str] = None,
) -> pd.DataFrame:
    """
    Base resumida para dashboard com drilldown:
    período -> cliente -> UC
    """
    df = load_reporting_fact(
        periodo=periodo,
        cliente_numero=cliente_numero,
        unidade_consumidora=unidade_consumidora,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # padroniza numéricos de interesse
    num_cols = [
        "valor_concessionaria",
        "valor_boleto",
        "economia_calculada",
        "consumo_kwh",
        "injetada_kwh",
        "consumo_kwh_parseado",
        "injetada_kwh_parseada",
        "valor_bandeiras",
        "valor_erb",
    ]
    for c in num_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    grp = (
        out.groupby(
            ["periodo", "cliente_numero", "nome", "unidade_consumidora"],
            dropna=False,
        )
        .agg(
            qtd_faturas=("nota_fiscal", "nunique"),
            valor_concessionaria=("valor_concessionaria", "sum"),
            valor_boleto=("valor_boleto", "sum"),
            economia_calculada=("economia_calculada", "sum"),
            consumo_kwh=("consumo_kwh", "sum"),
            injetada_kwh=("injetada_kwh", "sum"),
            consumo_kwh_parseado=("consumo_kwh_parseado", "sum"),
            injetada_kwh_parseada=("injetada_kwh_parseada", "sum"),
        )
        .reset_index()
    )

    return _coerce_periodo_sort(grp, "periodo")


def build_report_payload(nota_fiscal: str) -> Dict[str, Any]:
    """
    Payload único para a futura geração do demonstrativo por associado.
    """
    header = load_report_header(nota_fiscal)
    if header is None or header.empty:
        return {"header": {}, "historico": pd.DataFrame()}

    row = header.iloc[0].to_dict()

    cliente_numero = str(row.get("cliente_numero") or "")
    uc = str(row.get("unidade_consumidora") or "")
    periodo = str(row.get("periodo") or "")
    ano = None
    if len(periodo) == 7 and "/" in periodo:
        try:
            ano = int(periodo[-4:])
        except Exception:
            ano = None

    historico = load_report_history(
        cliente_numero=cliente_numero if cliente_numero else None,
        unidade_consumidora=uc if uc else None,
        ano=ano,
    )

    return {
        "header": row,
        "historico": historico,
    }
