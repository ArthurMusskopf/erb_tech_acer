# utils/workflow_adapter.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _safe_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def _safe_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("true", "1", "t", "yes", "y"):
        return True
    if s in ("false", "0", "f", "no", "n"):
        return False
    return None


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = pd.to_numeric(x, errors="coerce")
        return None if pd.isna(v) else float(v)
    except Exception:
        return None


def _safe_int(x: Any) -> Optional[int]:
    try:
        v = pd.to_numeric(x, errors="coerce")
        if pd.isna(v):
            return None
        return int(round(float(v), 0))
    except Exception:
        return None


def _safe_date_str(x: Any) -> Optional[str]:
    """
    Mantém DATE em formato YYYY-MM-DD quando já vier assim,
    ou retorna string limpa para o cast do BigQuery converter depois.
    """
    s = _safe_str(x)
    return s


def _first_non_null(df: pd.DataFrame, col: str) -> Optional[Any]:
    if df is None or df.empty or col not in df.columns:
        return None
    series = df[col].dropna()
    if series.empty:
        return None
    return series.iloc[0]


def _build_existing_map(existing_workflow_df: Optional[pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    """
    Indexa workflow existente por NF (id).
    """
    if existing_workflow_df is None or existing_workflow_df.empty:
        return {}

    df = existing_workflow_df.copy()
    if "id" not in df.columns:
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for _, r in df.iterrows():
        nf = _safe_str(r.get("id"))
        if nf:
            out[nf] = r.to_dict()
    return out


def build_workflow_from_parse_results(
    resultados: List[Any],
    *,
    existing_workflow_df: Optional[pd.DataFrame] = None,
    arquivo_hash_by_name: Optional[Dict[str, str]] = None,
    pdf_uri_by_name: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Constrói 1 linha por NF para a tabela faturas_workflow.

    Espera objetos 'ResultadoParsing' do pdf_parser.processar_lote_faturas().
    """
    existing_map = _build_existing_map(existing_workflow_df)
    arquivo_hash_by_name = arquivo_hash_by_name or {}
    pdf_uri_by_name = pdf_uri_by_name or {}

    rows: List[Dict[str, Any]] = []
    now = _now_utc()

    for r in resultados or []:
        header = getattr(r, "header", {}) or {}
        nf = getattr(r, "nf", {}) or {}
        df_itens = getattr(r, "df_itens", pd.DataFrame())
        arquivo = _safe_str(getattr(r, "arquivo", None))
        sucesso = bool(getattr(r, "sucesso", False))
        erros = getattr(r, "erros", []) or []
        alertas = getattr(r, "alertas", []) or []

        nota_fiscal = _safe_str(nf.get("numero")) or _safe_str(_first_non_null(df_itens, "numero"))
        unidade_consumidora = _safe_str(header.get("unidade_consumidora")) or _safe_str(_first_non_null(df_itens, "unidade_consumidora"))
        cliente_numero = _safe_str(header.get("cliente_numero")) or _safe_str(_first_non_null(df_itens, "cliente_numero"))
        nome = _safe_str(header.get("nome")) or _safe_str(_first_non_null(df_itens, "nome"))
        cnpj_cpf = _safe_str(header.get("cnpj_cpf")) or _safe_str(header.get("cnpj")) or _safe_str(_first_non_null(df_itens, "cnpj"))
        referencia = _safe_str(header.get("referencia")) or _safe_str(_first_non_null(df_itens, "referencia"))
        vencimento = _safe_str(header.get("vencimento")) or _safe_str(_first_non_null(df_itens, "vencimento"))
        classe_modalidade = _safe_str(header.get("classe_modalidade")) or _safe_str(_first_non_null(df_itens, "classe_modalidade"))
        grupo_subgrupo_tensao = _safe_str(header.get("grupo_subgrupo_tensao")) or _safe_str(_first_non_null(df_itens, "grupo_subgrupo_tensao"))
        total_pagar = _safe_float(header.get("total_pagar"))
        if total_pagar is None:
            total_pagar = _safe_float(_first_non_null(df_itens, "total_pagar"))

        if not nota_fiscal:
            # sem NF não conseguimos criar workflow consistente
            continue

        existing = existing_map.get(nota_fiscal)
        is_inedita = existing is None
        duplicada_de = None if is_inedita else nota_fiscal

        observacoes_parts: List[str] = []
        if erros:
            observacoes_parts.append("erros: " + " | ".join(str(x) for x in erros))
        if alertas:
            observacoes_parts.append("alertas: " + " | ".join(str(x) for x in alertas))
        observacoes = " || ".join(observacoes_parts) if observacoes_parts else None

        row = {
            "id": nota_fiscal,
            "nota_fiscal": nota_fiscal,
            "unidade_consumidora": unidade_consumidora,
            "cliente_numero": cliente_numero,
            "nome": nome,
            "cnpj_cpf": cnpj_cpf,
            "referencia": referencia,
            "vencimento": vencimento,
            "classe_modalidade": classe_modalidade,
            "grupo_subgrupo_tensao": grupo_subgrupo_tensao,
            "total_pagar": total_pagar,

            "arquivo_nome_original": arquivo,
            "arquivo_hash": _safe_str(arquivo_hash_by_name.get(arquivo)),
            "pdf_uri": _safe_str(pdf_uri_by_name.get(arquivo)),

            "is_inedita": bool(is_inedita),
            "duplicada_de": duplicada_de,

            "status_parse": "parseado" if sucesso else "erro_parse",
            "status_validacao": "pendente",
            "validado_por": None,
            "validado_em": None,

            "status_calculo": "nao_calculada",
            "calculado_em": None,

            "status_emissao": "nao_emitido",
            "emitido_em": None,

            "observacoes": observacoes,

            # preserva created_at se já existia
            "created_at": existing.get("created_at") if existing else now,
            "updated_at": now,
        }

        rows.append(row)

    return pd.DataFrame(rows)


def flatten_sicoob_payload_to_row(
    payload: Dict[str, Any],
    *,
    emission_id: str,
    boleto_calculado_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    nota_fiscal: Optional[str] = None,
    unidade_consumidora: Optional[str] = None,
    cliente_numero: Optional[str] = None,
    nome: Optional[str] = None,
    periodo: Optional[str] = None,
    status_envio: str = "pendente",
    http_status: Optional[int] = None,
    response_json: Optional[Dict[str, Any]] = None,
    linha_digitavel: Optional[str] = None,
    codigo_barras: Optional[str] = None,
    nosso_numero_retorno: Optional[str] = None,
    pdf_gerado: Optional[bool] = None,
    pdf_uri: Optional[str] = None,
) -> pd.DataFrame:
    """
    Achata o payload do POST /boletos do Sicoob para a tabela boletos_emissao_sicoob.

    Guarda todos os campos do exemplo fornecido e os principais campos de retorno.
    """
    p = payload or {}
    pagador = p.get("pagador", {}) or {}
    beneficiario = p.get("beneficiarioFinal", {}) or {}
    msgs = p.get("mensagensInstrucao", []) or []
    rateios = p.get("rateioCreditos", []) or []
    rateio = rateios[0] if rateios else {}

    now = _now_utc()

    row = {
        "id": _safe_str(emission_id),
        "boleto_calculado_id": _safe_str(boleto_calculado_id),
        "workflow_id": _safe_str(workflow_id),
        "nota_fiscal": _safe_str(nota_fiscal),
        "unidade_consumidora": _safe_str(unidade_consumidora),
        "cliente_numero": _safe_str(cliente_numero),
        "nome": _safe_str(nome),
        "periodo": _safe_str(periodo),

        "numeroCliente": _safe_int(p.get("numeroCliente")),
        "codigoModalidade": _safe_int(p.get("codigoModalidade")),
        "numeroContaCorrente": _safe_int(p.get("numeroContaCorrente")),
        "codigoEspecieDocumento": _safe_str(p.get("codigoEspecieDocumento")),
        "dataEmissao": _safe_date_str(p.get("dataEmissao")),
        "nossoNumero": _safe_str(p.get("nossoNumero")),
        "seuNumero": _safe_str(p.get("seuNumero")),
        "identificacaoBoletoEmpresa": _safe_str(p.get("identificacaoBoletoEmpresa")),
        "identificacaoEmissaoBoleto": _safe_int(p.get("identificacaoEmissaoBoleto")),
        "identificacaoDistribuicaoBoleto": _safe_int(p.get("identificacaoDistribuicaoBoleto")),
        "valor": _safe_float(p.get("valor")),
        "dataVencimento": _safe_date_str(p.get("dataVencimento")),
        "dataLimitePagamento": _safe_date_str(p.get("dataLimitePagamento")),
        "valorAbatimento": _safe_float(p.get("valorAbatimento")),

        "tipoDesconto": _safe_int(p.get("tipoDesconto")),
        "dataPrimeiroDesconto": _safe_date_str(p.get("dataPrimeiroDesconto")),
        "valorPrimeiroDesconto": _safe_float(p.get("valorPrimeiroDesconto")),
        "dataSegundoDesconto": _safe_date_str(p.get("dataSegundoDesconto")),
        "valorSegundoDesconto": _safe_float(p.get("valorSegundoDesconto")),
        "dataTerceiroDesconto": _safe_date_str(p.get("dataTerceiroDesconto")),
        "valorTerceiroDesconto": _safe_float(p.get("valorTerceiroDesconto")),

        "tipoMulta": _safe_int(p.get("tipoMulta")),
        "dataMulta": _safe_date_str(p.get("dataMulta")),
        "valorMulta": _safe_float(p.get("valorMulta")),
        "tipoJurosMora": _safe_int(p.get("tipoJurosMora")),
        "dataJurosMora": _safe_date_str(p.get("dataJurosMora")),
        "valorJurosMora": _safe_float(p.get("valorJurosMora")),

        "numeroParcela": _safe_int(p.get("numeroParcela")),
        "aceite": _safe_bool(p.get("aceite")),
        "codigoNegativacao": _safe_int(p.get("codigoNegativacao")),
        "numeroDiasNegativacao": _safe_int(p.get("numeroDiasNegativacao")),
        "codigoProtesto": _safe_int(p.get("codigoProtesto")),
        "numeroDiasProtesto": _safe_int(p.get("numeroDiasProtesto")),

        "pagador_numeroCpfCnpj": _safe_str(pagador.get("numeroCpfCnpj")),
        "pagador_nome": _safe_str(pagador.get("nome")),
        "pagador_endereco": _safe_str(pagador.get("endereco")),
        "pagador_bairro": _safe_str(pagador.get("bairro")),
        "pagador_cidade": _safe_str(pagador.get("cidade")),
        "pagador_cep": _safe_str(pagador.get("cep")),
        "pagador_uf": _safe_str(pagador.get("uf")),
        "pagador_email": _safe_str(pagador.get("email")),

        "beneficiarioFinal_numeroCpfCnpj": _safe_str(beneficiario.get("numeroCpfCnpj")),
        "beneficiarioFinal_nome": _safe_str(beneficiario.get("nome")),

        "mensagemInstrucao_1": _safe_str(msgs[0]) if len(msgs) > 0 else None,
        "mensagemInstrucao_2": _safe_str(msgs[1]) if len(msgs) > 1 else None,
        "mensagemInstrucao_3": _safe_str(msgs[2]) if len(msgs) > 2 else None,
        "mensagemInstrucao_4": _safe_str(msgs[3]) if len(msgs) > 3 else None,
        "mensagemInstrucao_5": _safe_str(msgs[4]) if len(msgs) > 4 else None,

        "gerarPdf": _safe_bool(p.get("gerarPdf")),
        "codigoCadastrarPIX": _safe_int(p.get("codigoCadastrarPIX")),
        "numeroContratoCobranca": _safe_int(p.get("numeroContratoCobranca")),

        "rateio_numeroBanco": _safe_int(rateio.get("numeroBanco")),
        "rateio_numeroAgencia": _safe_int(rateio.get("numeroAgencia")),
        "rateio_numeroContaCorrente": _safe_str(rateio.get("numeroContaCorrente")),
        "rateio_contaPrincipal": _safe_bool(rateio.get("contaPrincipal")),
        "rateio_codigoTipoValorRateio": _safe_int(rateio.get("codigoTipoValorRateio")),
        "rateio_valorRateio": _safe_str(rateio.get("valorRateio")),
        "rateio_codigoTipoCalculoRateio": _safe_int(rateio.get("codigoTipoCalculoRateio")),
        "rateio_numeroCpfCnpjTitular": _safe_str(rateio.get("numeroCpfCnpjTitular")),
        "rateio_nomeTitular": _safe_str(rateio.get("nomeTitular")),
        "rateio_codigoFinalidadeTed": _safe_str(rateio.get("codigoFinalidadeTed")),
        "rateio_codigoTipoContaDestinoTed": _safe_str(rateio.get("codigoTipoContaDestinoTed")),
        "rateio_quantidadeDiasFloat": _safe_int(rateio.get("quantidadeDiasFloat")),
        "rateio_dataFloatCredito": _safe_date_str(rateio.get("dataFloatCredito")),

        "status_envio": _safe_str(status_envio),
        "http_status": _safe_int(http_status),
        "response_json": json.dumps(response_json, ensure_ascii=False) if response_json is not None else None,
        "linha_digitavel": _safe_str(linha_digitavel),
        "codigo_barras": _safe_str(codigo_barras),
        "nosso_numero_retorno": _safe_str(nosso_numero_retorno),
        "pdf_gerado": _safe_bool(pdf_gerado),
        "pdf_uri": _safe_str(pdf_uri),

        "created_at": now,
        "updated_at": now,
    }

    return pd.DataFrame([row])
