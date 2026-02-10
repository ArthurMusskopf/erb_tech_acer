"""
ERB Tech - Parser de Faturas CELESC
VERSÃO v2 - Inclui parse de "Classificação / Modalidade Tarifária / Tipo de Fornecimento"
e salva em df_itens['classe_modalidade'] (igual ao grupo_subgrupo_tensao).
"""

import re
import pdfplumber
import pandas as pd
import numpy as np
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ResultadoParsing:
    sucesso: bool
    arquivo: str
    header: Dict[str, Any]
    periodo: Dict[str, Any]
    nf: Dict[str, Any]
    df_itens: pd.DataFrame
    df_medidores: pd.DataFrame
    erros: List[str]
    alertas: List[str]


def read_pdf_text(pdf_path: str) -> str:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            pages.append(p.extract_text() or "")
    return "\n".join(pages)


def br2float(s) -> Optional[float]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    try:
        return float(s.replace('.', '').replace(',', '.'))
    except:
        return None


def clean_spaces(s) -> Optional[str]:
    if s is None:
        return None
    return re.sub(r'\s+', ' ', str(s)).strip()


def safe_search(pattern, text, flags=0):
    m = re.search(pattern, text, flags)
    if not m:
        return None
    return (m.group(1) if m.groups() else m.group(0)).strip()


def as_text_or_none(v) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


LABELS = {
    'nome': 'NOME:',
    'cpfcnpj': 'CPF/CNPJ:',
    'endereco': 'ENDERECO:',
    'cep': 'CEP:',
    'cidade': 'CIDADE:',
    'grupo': 'Grupo/Subgrupo Tensão:',
    'cliente': 'Cliente:',
    'unidade_consumidora': 'Unidade Consumidora',
    # Em muitos PDFs esse texto aparece “quebrado”; por isso usamos regex flexível (ver parse_classe_modalidade)
    'classe_modalidade': 'Classificação / Modalidade Tarifária / Tipo de Fornecimento',
}


def extract_between(text, start_label, end_labels, flags=re.S):
    if isinstance(end_labels, str):
        end_labels = [end_labels]
    end_alt = '|'.join(re.escape(lbl) for lbl in end_labels)
    pat = re.escape(start_label) + r'\s*(.*?)\s*(?=(?:' + end_alt + r'))'
    return safe_search(pat, text, flags=flags)


def parse_periodo_leituras(txt: str) -> dict:
    flags = re.I | re.S
    out = {'leitura_anterior': None, 'leitura_atual': None, 'dias': None, 'proxima_leitura': None}

    def first_date_after(label_regex, text, lookahead=220):
        for m in re.finditer(label_regex, text, flags):
            seg = text[m.end(): m.end()+lookahead]
            md = re.search(r'([0-9]{2}/[0-9]{2}/[0-9]{4})', seg)
            if md:
                return md.group(1)
        return None

    la = first_date_after(r'Leit(?:\.|ura)?\s*Anterior', txt)
    lt = first_date_after(r'Leit(?:\.|ura)?\s*Atual', txt)
    pl = first_date_after(r'Pr(?:[óo]x\.?|[óo]xima)\s*Leit(?:\.|ura)?', txt)

    di = None
    for m in re.finditer(r'Dias(?:\s+no\s+per[ií]odo)?', txt, flags):
        seg = txt[m.end(): m.end()+140]
        mi = re.search(r'([0-9]{1,3})\b', seg)
        if mi:
            di = int(mi.group(1))
            break

    if la and lt and pl and isinstance(di, int):
        out.update({'leitura_anterior': la, 'leitura_atual': lt, 'dias': di, 'proxima_leitura': pl})
        return out

    raw_pat = r'([0-9]{2}/[0-9]{2}/[0-9]{4})\s+([0-9]{2}/[0-9]{2}/[0-9]{4})\s+([0-9]{1,3})\s+([0-9]{2}/[0-9]{2}/[0-9]{4})'
    m = re.search(raw_pat, txt, flags=re.I)
    if m:
        out.update({
            'leitura_anterior': la or m.group(1),
            'leitura_atual': lt or m.group(2),
            'dias': di if di is not None else int(m.group(3)),
            'proxima_leitura': pl or m.group(4)
        })
    return out


def parse_classe_modalidade(txt: str) -> Optional[str]:
    """
    Tenta capturar o valor do campo:
    "Classificação / Modalidade Tarifária / Tipo de Fornecimento"

    Estratégia:
    1) Regex pelo rótulo (tolerante a espaços, barras e quebras de linha)
    2) Fallback: primeira linha que contenha MONOFÁSICO/BIFÁSICO/TRIFÁSICO (modo antigo)
    """
    # 1) Busca pelo rótulo (muito comum estar quebrado em várias “colunas” no texto extraído)
    pat = (
        r'Classifica(?:ç|c)ão\s*/\s*Modalidade\s*Tarif[aá]ria\s*/\s*Tipo\s*de\s*Fornecimento'
        r'\s*[:\-]?\s*(?:\n\s*)?([^\n]+)'
    )
    v = safe_search(pat, txt, flags=re.I)
    v = clean_spaces(v)

    # 2) Fallback antigo
    if not v:
        v = safe_search(r'^(.*?(?:MONOF[ÁA]SICO|BIF[ÁA]SICO|TRIF[ÁA]SICO).*)$', txt, flags=re.M | re.I)
        v = clean_spaces(v)

    return v or None


def infer_n_fases(classe_modalidade: Optional[str]) -> Optional[int]:
    if not classe_modalidade:
        return None
    s = clean_spaces(classe_modalidade).upper()

    # tolerância a variações
    if "TRIF" in s:
        return 3
    if "BIF" in s:
        return 2
    if "MONOF" in s or "MONO" in s:
        return 1
    return None


def parse_header(txt: str) -> dict:
    h = {}

    # Campo novo: classificação/modalidade/tipo
    h['classe_modalidade'] = parse_classe_modalidade(txt)
    h['n_fases_parseado'] = infer_n_fases(h.get('classe_modalidade'))

    h['unidade_consumidora'] = safe_search(rf'{LABELS["unidade_consumidora"]}\s*\n\s*0*([0-9]+)', txt) \
        or safe_search(r'\b0*([0-9]{8,})\b', txt)
    h['cliente_numero'] = safe_search(rf'{re.escape(LABELS["cliente"])}\s*([0-9]+)', txt)

    m = re.search(r'(\d{2}/\d{4})\s+(\d{2}/\d{2}/\d{4})\s+R\$?\s*([0-9\.,]+)', txt)
    if m:
        h['referencia'] = m.group(1)
        h['vencimento'] = m.group(2)
        h['total_pagar'] = br2float(m.group(3))

    nome = extract_between(txt, LABELS['nome'], LABELS['cpfcnpj']) or safe_search(r'NOME:\s*(.+)', txt)
    h['nome'] = clean_spaces(nome)
    h['cnpj_cpf'] = safe_search(rf'{re.escape(LABELS["cpfcnpj"])}\s*([^\n]+)', txt)
    ender = extract_between(txt, LABELS['endereco'], LABELS['cep'])
    h['endereco'] = clean_spaces(ender)
    h['cep'] = safe_search(rf'{re.escape(LABELS["cep"])}\s*([0-9\-]+)', txt)
    cid_uf = safe_search(rf'{re.escape(LABELS["cidade"])}\s*([A-ZÇÃÂÉÊÍÓÔÕÚÜ\s]+[A-Z]{{2}})', txt)
    h['cidade_uf'] = clean_spaces(cid_uf)
    h['grupo_subgrupo_tensao'] = safe_search(rf'{re.escape(LABELS["grupo"])}([^\n]+)', txt)

    return h


def parse_nf(txt: str) -> dict:
    m = re.search(r'NOTA FISCAL Nº\s*([0-9]+)\s+SERIE:([0-9]+)\s+DATA EMISSAO:\s*([0-9/]+)', txt, re.I)
    return {'numero': m.group(1), 'serie': m.group(2), 'data_emissao': m.group(3)} if m else {}


def parse_itens(txt: str) -> list:
    itens = []
    for raw in txt.splitlines():
        line = raw.strip()
        if not line.startswith('('):
            continue
        m = re.match(r'^\((\w{1,2})\)\s*(.+?)\s+(KWH|MWH|KVARH|KVAH|UN)\s+(.+)$', line, flags=re.I)
        if not m:
            continue
        code, desc, unit, tail = m.groups()
        nums = [br2float(x) for x in tail.split()]
        item = {'codigo': code, 'descricao': desc.strip(), 'unidade': unit.upper()}
        cols = ['quantidade', 'tarifa', 'valor', 'pis_valor', 'cofins_base', 'icms_aliquota', 'icms_valor', 'tarifa_sem_trib']
        for i, v in enumerate(nums):
            item[cols[i] if i < len(cols) else f'valor_extra_{i-len(cols)+1}'] = v
        itens.append(item)
    return itens


def parse_medidores(txt: str) -> list:
    def is_unico(tok: str) -> bool:
        return tok.strip().lower() in ('único', 'unico')

    medidores = []
    for raw in txt.splitlines():
        line = raw.strip()
        if 'Energia' not in line or not any(is_unico(t) for t in line.split()):
            continue
        parts = line.split()
        if not parts or not re.fullmatch(r'\d+', parts[0] or ''):
            continue
        try:
            unico_idx = next(i for i, p in enumerate(parts) if is_unico(p))
        except StopIteration:
            continue
        tipo = ' '.join(parts[1:unico_idx]).strip() or 'Energia'
        start = unico_idx + 1
        if len(parts) < start + 5:
            continue
        ant, atu, const, fator, tot = parts[start:start+5]
        medidores.append({
            'medidor': parts[0], 'tipo': tipo, 'posto': 'Único',
            'leitura_anterior': br2float(ant), 'leitura_atual': br2float(atu),
            'constante': br2float(const), 'fator': br2float(fator), 'total_apurado': br2float(tot),
        })
    return medidores


def make_item_id(codigo, unidade_consumidora, tarifa, vencimento_str):
    """ID único para item - IGUAL AO NOTEBOOK"""
    if tarifa is None or (isinstance(tarifa, float) and pd.isna(tarifa)):
        tarifa_s = ""
    else:
        tarifa_s = str(tarifa).strip().replace(",", ".")
        tarifa_s = re.sub(r"[^0-9\.]", "", tarifa_s).replace(".", "")

    yymmdd = ""
    if vencimento_str:
        try:
            dt = datetime.strptime(str(vencimento_str).strip(), "%d/%m/%Y")
            yymmdd = dt.strftime("%y%m%d")
        except:
            pass

    return f"{codigo}{unidade_consumidora}{tarifa_s}{yymmdd}"


def parse_fatura(pdf_path: str) -> ResultadoParsing:
    arquivo = Path(pdf_path).name
    erros, alertas = [], []

    try:
        txt = read_pdf_text(pdf_path)
    except Exception as e:
        return ResultadoParsing(False, arquivo, {}, {}, {}, pd.DataFrame(), pd.DataFrame(), [f"Erro: {e}"], [])

    header = parse_header(txt)
    periodo = parse_periodo_leituras(txt)
    nf = parse_nf(txt)
    itens = parse_itens(txt)
    medidores = parse_medidores(txt)

    if not header.get('unidade_consumidora'):
        erros.append("UC não encontrada")
    if not itens:
        alertas.append("Sem itens tarifários")

    # DataFrame de itens
    df_itens = pd.DataFrame(itens) if itens else pd.DataFrame()

    if not df_itens.empty:
        if 'quantidade' in df_itens.columns:
            df_itens = df_itens.rename(columns={'quantidade': 'quantidade_registrada'})

        for c in ['quantidade_registrada', 'tarifa', 'valor', 'pis_valor', 'cofins_base', 'icms_aliquota', 'icms_valor', 'tarifa_sem_trib']:
            if c in df_itens.columns:
                df_itens[c] = pd.to_numeric(df_itens[c], errors='coerce')

        # Campos comuns - IGUAL AO NOTEBOOK
        df_itens['unidade_consumidora'] = header.get('unidade_consumidora')
        df_itens['cliente_numero'] = header.get('cliente_numero')
        df_itens['referencia'] = header.get('referencia')
        df_itens['vencimento'] = header.get('vencimento')
        df_itens['total_pagar'] = header.get('total_pagar')
        df_itens['nome'] = header.get('nome')
        df_itens['cnpj'] = header.get('cnpj_cpf')
        df_itens['cep'] = header.get('cep')
        df_itens['cidade_uf'] = header.get('cidade_uf')
        df_itens['grupo_subgrupo_tensao'] = header.get('grupo_subgrupo_tensao')

        # NOVO: salvar a classificação/modalidade/tipo em fatura_itens
        df_itens['classe_modalidade'] = header.get('classe_modalidade')

        df_itens['leitura_anterior'] = as_text_or_none(periodo.get('leitura_anterior'))
        df_itens['leitura_atual'] = as_text_or_none(periodo.get('leitura_atual'))
        df_itens['dias'] = periodo.get('dias')
        df_itens['proxima_leitura'] = as_text_or_none(periodo.get('proxima_leitura'))
        df_itens['numero'] = nf.get('numero')
        df_itens['serie'] = nf.get('serie')
        df_itens['data_emissao'] = as_text_or_none(nf.get('data_emissao'))

        # ID único
        df_itens['id'] = df_itens.apply(
            lambda r: make_item_id(r.get('codigo'), r.get('unidade_consumidora'), r.get('tarifa'), r.get('vencimento')),
            axis=1
        )

        if 'dias' in df_itens.columns:
            df_itens['dias'] = pd.to_numeric(df_itens['dias'], errors='coerce').astype('Int64')

        # Remover colunas extras (valor_extra_*)
        df_itens = df_itens[[c for c in df_itens.columns if not c.startswith('valor_extra')]]

    # DataFrame de medidores
    df_medidores = pd.DataFrame(medidores) if medidores else pd.DataFrame()

    if not df_medidores.empty:
        df_medidores['unidade_consumidora'] = header.get('unidade_consumidora')
        df_medidores['cliente_numero'] = header.get('cliente_numero')
        df_medidores['referencia'] = header.get('referencia')
        df_medidores['nome'] = header.get('nome')
        df_medidores['nota_fiscal_numero'] = nf.get('numero')

        # ID único para medidores
        df_medidores['id'] = df_medidores.apply(
            lambda r: hashlib.sha256(
                f"{r.get('unidade_consumidora','')}|{r.get('cliente_numero','')}|{r.get('referencia','')}|{r.get('nota_fiscal_numero','')}|{r.get('medidor','')}|{r.get('tipo','')}|{r.get('posto','')}".encode()
            ).hexdigest(),
            axis=1
        )

    return ResultadoParsing(len(erros) == 0, arquivo, header, periodo, nf, df_itens, df_medidores, erros, alertas)


def processar_lote_faturas(pdf_paths: List[str], progress_callback=None) -> Dict[str, Any]:
    resultados = []
    df_itens_all, df_medidores_all = [], []
    total = len(pdf_paths)

    for i, pdf_path in enumerate(pdf_paths):
        resultado = parse_fatura(pdf_path)
        resultados.append(resultado)

        if not resultado.df_itens.empty:
            df_itens_all.append(resultado.df_itens)
        if not resultado.df_medidores.empty:
            df_medidores_all.append(resultado.df_medidores)

        if progress_callback:
            progress_callback((i + 1) / total, resultado.arquivo)

    return {
        'total': total,
        'sucesso': sum(1 for r in resultados if r.sucesso),
        'erros': sum(1 for r in resultados if not r.sucesso),
        'resultados': resultados,
        'df_itens': pd.concat(df_itens_all, ignore_index=True) if df_itens_all else pd.DataFrame(),
        'df_medidores': pd.concat(df_medidores_all, ignore_index=True) if df_medidores_all else pd.DataFrame()
    }
