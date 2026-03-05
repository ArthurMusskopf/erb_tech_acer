# utils/sicoob_client.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import base64
import re

import requests
import streamlit as st


@dataclass
class SicoobConfig:
    base_url: str
    client_id: str
    access_token: str

    # parâmetros do convênio/contrato
    numero_cliente: int
    codigo_modalidade: int
    numero_contrato_cobranca: int
    numero_conta_corrente: int = 0  # pode ser 0 no exemplo


def _digits(s: str) -> str:
    return re.sub(r"\D+", "", s or "")


def get_sicoob_config_from_secrets() -> SicoobConfig:
    """
    Espera em .streamlit/secrets.toml algo como:

    [sicoob]
    base_url = "https://sandbox.sicoob.com.br/sicoob/sandbox/cobranca-bancaria/v3"
    client_id = "..."
    access_token = "..."
    numero_cliente = 25546454
    codigo_modalidade = 1
    numero_contrato_cobranca = 1
    numero_conta_corrente = 0
    """
    cfg = st.secrets.get("sicoob", None)
    if not cfg:
        raise RuntimeError("Config Sicoob ausente em st.secrets['sicoob'].")

    return SicoobConfig(
        base_url=str(cfg["base_url"]).rstrip("/"),
        client_id=str(cfg["client_id"]),
        access_token=str(cfg["access_token"]),
        numero_cliente=int(cfg["numero_cliente"]),
        codigo_modalidade=int(cfg.get("codigo_modalidade", 1)),
        numero_contrato_cobranca=int(cfg.get("numero_contrato_cobranca", 1)),
        numero_conta_corrente=int(cfg.get("numero_conta_corrente", 0)),
    )


class SicoobCobrancaV3Client:
    def __init__(self, config: SicoobConfig, timeout: int = 30):
        self.cfg = config
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.cfg.access_token}",
            "client_id": self.cfg.client_id,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _raise_for_status(self, resp: requests.Response) -> None:
        if 200 <= resp.status_code < 300:
            return
        try:
            payload = resp.json()
        except Exception:
            payload = resp.text
        raise RuntimeError(f"Sicoob API erro {resp.status_code}: {payload}")

    def create_boleto(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.cfg.base_url}/boletos"
        resp = requests.post(url, json=payload, headers=self._headers(), timeout=self.timeout)
        self._raise_for_status(resp)
        return resp.json()

    def faixa_nosso_numero(self) -> Dict[str, Any]:
        """
        GET /boletos/faixas-nosso-numero (v3) — útil para reservar nosso número.
        """
        url = f"{self.cfg.base_url}/boletos/faixas-nosso-numero"
        params = {
            "numeroCliente": self.cfg.numero_cliente,
            "codigoModalidade": self.cfg.codigo_modalidade,
            "numeroContratoCobranca": self.cfg.numero_contrato_cobranca,
        }
        resp = requests.get(url, params=params, headers=self._headers(), timeout=self.timeout)
        self._raise_for_status(resp)
        return resp.json()

    def segunda_via_pdf(
        self,
        *,
        nosso_numero: Optional[int] = None,
        linha_digitavel: Optional[str] = None,
        codigo_barras: Optional[str] = None,
        gerar_pdf: bool = True,
    ) -> Dict[str, Any]:
        """
        GET /boletos/segunda-via
        Retorna pdfBoleto (base64) no campo resultado.pdfBoleto quando disponível.
        """
        url = f"{self.cfg.base_url}/boletos/segunda-via"
        params: Dict[str, Any] = {
            "numeroCliente": self.cfg.numero_cliente,
            "codigoModalidade": self.cfg.codigo_modalidade,
            "numeroContratoCobranca": self.cfg.numero_contrato_cobranca,
        }
        if nosso_numero is not None:
            params["nossoNumero"] = int(nosso_numero)
        if linha_digitavel:
            params["linhaDigitavel"] = str(linha_digitavel)
        if codigo_barras:
            params["codigoBarras"] = str(codigo_barras)
        if gerar_pdf:
            params["gerarPdf"] = "true"

        resp = requests.get(url, params=params, headers=self._headers(), timeout=self.timeout)
        self._raise_for_status(resp)
        return resp.json()

    @staticmethod
    def decode_pdf_boleto(resp_json: Dict[str, Any]) -> bytes:
        """
        Extrai resultado.pdfBoleto (base64) e devolve bytes do PDF.
        """
        resultado = resp_json.get("resultado", {}) if isinstance(resp_json, dict) else {}
        pdf_b64 = resultado.get("pdfBoleto")
        if not pdf_b64:
            raise RuntimeError("Resposta não trouxe 'resultado.pdfBoleto'.")
        return base64.b64decode(pdf_b64)


def build_boleto_payload_from_row(
    row: Dict[str, Any],
    *,
    cfg: SicoobConfig,
    nosso_numero: int,
    data_emissao: str,
    data_vencimento: str,
    valor: float,
) -> Dict[str, Any]:
    """
    Monta payload mínimo baseado no exemplo do POST /boletos (v3).
    """
    nome = str(row.get("nome") or "PAGADOR")
    cnpj_cpf = _digits(str(row.get("cnpj_cpf") or row.get("cnpj") or ""))
    cep = _digits(str(row.get("cep") or ""))
    cidade_uf = str(row.get("cidade_uf") or "")
    uf = cidade_uf.strip()[-2:] if len(cidade_uf.strip()) >= 2 else "SC"
    cidade = cidade_uf.strip()[:-2].strip() if len(cidade_uf.strip()) > 2 else "Cidade"

    payload = {
        "numeroCliente": cfg.numero_cliente,
        "codigoModalidade": cfg.codigo_modalidade,
        "numeroContaCorrente": cfg.numero_conta_corrente,
        "codigoEspecieDocumento": "DM",
        "dataEmissao": data_emissao,  # YYYY-MM-DD
        "nossoNumero": int(nosso_numero),
        "seuNumero": str(row.get("numero") or nosso_numero),
        "identificacaoBoletoEmpresa": str(row.get("numero") or nosso_numero),
        "identificacaoEmissaoBoleto": 1,
        "identificacaoDistribuicaoBoleto": 1,
        "valor": float(valor),
        "dataVencimento": data_vencimento,         # YYYY-MM-DD
        "dataLimitePagamento": data_vencimento,    # v0: igual ao vencimento
        "valorAbatimento": 0,
        "tipoDesconto": 0,
        "tipoMulta": 0,
        "tipoJurosMora": 0,
        "aceite": True,
        "codigoNegativacao": 0,
        "codigoProtesto": 0,
        "pagador": {
            "numeroCpfCnpj": cnpj_cpf or "12345678900",
            "nome": nome[:80],
            "endereco": "ENDERECO NAO INFORMADO",
            "bairro": "CENTRO",
            "cidade": cidade[:40] or "Cidade",
            "cep": cep or "00000000",
            "uf": uf,
            "email": "nao-informado@acer.local",
        },
        "mensagensInstrucao": [
            f"Referência {row.get('periodo')}",
            f"UC {row.get('unidade_consumidora')}",
            "Cobrança ACER - créditos energia",
        ],
        "gerarPdf": False,
        "numeroContratoCobranca": cfg.numero_contrato_cobranca,
    }
    return payload
