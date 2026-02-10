# calc_engine.py
# =============================================================================
# ERB Tech - Calculation Engine
# Replica as fórmulas do Excel (Formulario_ERB_TechArt.xlsx -> Calculos_Boleto -> tabela "Calculo")
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math
import re
import unicodedata

import numpy as np
import pandas as pd


# =============================================================================
# Fórmulas extraídas (referência fiel do Excel)
# =============================================================================

FORMULAS_EXCEL_CALCULO: Dict[str, str] = {
    "custo_disp": "_xlfn.XLOOKUP(Calculo[[#This Row],[unidade_consumidora]],Tabela1[unidade_consumidora],Tabela1[custo_disp])",
    "medidores_apurado": 'SUMIFS(medidores_leituras_1[total_apurado],medidores_leituras_1[nota_fiscal_numero],Calculo[[#This Row],[numero]],medidores_leituras_1[tipo],"Energia")-SUMIFS(medidores_leituras_1[total_apurado],medidores_leituras_1[nota_fiscal_numero],Calculo[[#This Row],[numero]],medidores_leituras_1[tipo],"Energia injetada")',
    "injetada": "Calculo[[#This Row],[medidores_apurado]]-Calculo[[#This Row],[custo_disp]]",
    "boleto": "IF(Calculo[[#This Row],[medidores_apurado]]-Calculo[[#This Row],[custo_disp]]<=0,0,1)",
    "desconto_contratado": "_xlfn.XLOOKUP(uc,info_clientes[unidade_consumidora],info_clientes[desconto_contratado])",
    "tarifa_cheia_trib": 'IF(boleto=1,SUMIFS(fatura_itens[tarifa],numero,numero,descricao,"Consumo TE")+SUMIFS(...,"Consumo TUSD"),0)',
    "check": 'IF(tarifa_cheia_trib>10,"Parseamento",IF(tarifa_cheia_trib>1,"Subvenção","Certo"))',
    "subvencao": "_xlfn.XLOOKUP(uc,info_clientes[unidade_consumidora],info_clientes[subvencao])",
    "tarifa_cheia_trib2": 'IF(AND(check<>"Certo",boleto=1),SUMIFS(tarifa,"Consumo TE",quantidade_registrada,"<>"&subvencao)+SUMIFS(...,"Consumo TUSD"...),tarifa_cheia_trib)',
    "gerador": 'IF(_xlfn.XLOOKUP(numero,medidores[nota_fiscal_numero],medidores[tipo],"")="Energia injetada",1,0)',
    "med_inj_tusd": 'boleto*(SUMIFS(qtd,descricao,"Energia Inj. TUSD")-SUMIFS(medidores,total_apurado,tipo,"Energia Injetada"))',
    "Energia Inj. TUSD": "IF(AND(boleto=1,gerador=0,med_inj_tusd<>0),[média ponderada MAXIFS/MINIFS],XLOOKUP(med_inj_tusd&'Energia Inj. TUSD', qtd&desc, tarifa,0))",
    "Energia Injet. TE": "IF(AND(boleto=1,gerador=0,med_inj_tusd<>0),[média ponderada MAXIFS/MINIFS],XLOOKUP(med_inj_tusd&'Energia Injet. TE', qtd&desc, tarifa,0))",
    "tarifa_cheia_trib3": "IF(Energia Inj. TUSD + Energia Injet. TE <>0, soma, 0)",
    "tarifa_inj_tusd": "boleto*IF(tarifa_cheia_trib3=0, MAXIFS(tarifa, descricao='Energia Inj. TUSD'), 0)",
    "tarifa_inj_te": "boleto*IF(tarifa_cheia_trib3=0, MAXIFS(tarifa, descricao='Energia Injet. TE'), 0)",
    "tarifa_cheia": "IF(tarifa_cheia_trib3=0, tarifa_inj_tusd+tarifa_inj_te, tarifa_cheia_trib3)",
    "tarifa_paga_conc": "tarifa_cheia_trib2 + tarifa_cheia",
    "tarifa_erb": "(1-desconto_contratado)*tarifa_cheia_trib2",
    "tarifa_bol": "tarifa_erb - tarifa_paga_conc",
    "Bandeira Amarela": "XLOOKUP(numero&'Bandeira Amarela', numero&descricao, tarifa,0)",
    "Band. Am. Injet.": "boleto*XLOOKUP(numero&'Band. Am. Injet.', numero&descricao, tarifa,0)",
    "valor_band_amarela": "boleto*(Bandeira Amarela + Band. Am. Injet.)",
    "valor_band_amar_desc": "boleto*((1-desconto)*(Bandeira Amarela - valor_band_amarela))",
    "Band. Vermelha": "boleto*XLOOKUP(numero&'Band. Vermelha', numero&descricao, tarifa,0)",
    "Band. Vrm. Injet.": "boleto*XLOOKUP(numero&'Band. Vrm. Injet.', numero&descricao, tarifa,0)",
    "valor_band_vermelha": "boleto*(Band. Vermelha + Band. Vrm. Injet.)",
    "valor_band_vrm_desc": "boleto*((1-desconto)*(Band. Vermelha - valor_band_vermelha))",
    "tarifa_total_boleto": "boleto*valor_band_vrm_desc + valor_band_amar_desc + tarifa_bol",
    "valor_total_boleto": "tarifa_total_boleto * med_inj_tusd",
    "periodo": "XLOOKUP(numero, fatura_itens[numero], fatura_itens[referencia], '')",
    "nome": "XLOOKUP(numero, fatura_itens[numero], fatura_itens[nome], '')",
}


# =============================================================================
# Helpers
# =============================================================================

def _to_str(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return str(x).strip()


def _to_float(x) -> float:
    if x is None:
        return float("nan")
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except:
        return float("nan")


def _norm_text(s: str) -> str:
    """
    Normaliza para comparação robusta:
    - lower
    - remove acentos
    - colapsa espaços
    """
    s = _to_str(s).lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _isclose(a: float, b: float, tol: float = 1e-6) -> bool:
    if np.isnan(a) or np.isnan(b):
        return False
    return abs(a - b) <= tol


def infer_n_fases(classe_modalidade: Optional[str]) -> Optional[int]:
    if not classe_modalidade:
        return None
    s = _norm_text(classe_modalidade).upper()
    if "TRIF" in s:
        return 3
    if "BIF" in s:
        return 2
    if "MONOF" in s or "MONO" in s:
        return 1
    return None


def compute_custo_disp(n_fases: Optional[int]) -> Optional[float]:
    if n_fases == 3:
        return 100.0
    if n_fases == 2:
        return 50.0
    if n_fases == 1:
        return 30.0
    return None


# =============================================================================
# Engine
# =============================================================================

@dataclass
class CalcResult:
    df_boletos: pd.DataFrame
    missing_clientes: List[str]  # lista de UCs não cadastradas
    missing_reason: Dict[str, str]  # UC -> motivo


def _prepare_inputs(
    df_itens: pd.DataFrame,
    df_medidores: pd.DataFrame,
    df_clientes: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dfi = df_itens.copy()
    dfm = df_medidores.copy()
    dfc = df_clientes.copy()

    # strings-chave
    for col in ["numero", "unidade_consumidora", "descricao", "referencia", "nome"]:
        if col in dfi.columns:
            dfi[col] = dfi[col].astype(str)

    for col in ["nota_fiscal_numero", "tipo", "unidade_consumidora"]:
        if col in dfm.columns:
            dfm[col] = dfm[col].astype(str)

    for col in ["unidade_consumidora", "status"]:
        if col in dfc.columns:
            dfc[col] = dfc[col].astype(str)

    # numéricos
    for col in ["tarifa", "quantidade_registrada", "total_apurado"]:
        if col in dfi.columns:
            dfi[col] = pd.to_numeric(dfi[col], errors="coerce")
    if "total_apurado" in dfm.columns:
        dfm["total_apurado"] = pd.to_numeric(dfm["total_apurado"], errors="coerce")

    for col in ["desconto_contratado", "subvencao", "custo_disp", "n_fases"]:
        if col in dfc.columns:
            dfc[col] = pd.to_numeric(dfc[col], errors="coerce")

    # normalizados (comparação)
    if "descricao" in dfi.columns:
        dfi["_desc_norm"] = dfi["descricao"].map(_norm_text)
    else:
        dfi["_desc_norm"] = ""

    if "tipo" in dfm.columns:
        dfm["_tipo_norm"] = dfm["tipo"].map(_norm_text)
    else:
        dfm["_tipo_norm"] = ""

    return dfi, dfm, dfc


def _first_by_numero(dfi: pd.DataFrame, col: str) -> pd.Series:
    """
    Emula XLOOKUP(numero, fatura_itens[numero], fatura_itens[col]) -> primeiro match.
    """
    if "numero" not in dfi.columns or col not in dfi.columns:
        return pd.Series(dtype=object)

    tmp = dfi[["numero", col]].dropna(subset=["numero"])
    # primeiro por ordem original
    tmp = tmp.drop_duplicates(subset=["numero"], keep="first")
    return tmp.set_index("numero")[col]


def _sum_tarifa_by_desc(dfi: pd.DataFrame, desc: str) -> pd.Series:
    dn = _norm_text(desc)
    sub = dfi[dfi["_desc_norm"] == dn]
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.groupby("numero")["tarifa"].sum(min_count=1)


def _sum_qtd_by_desc(dfi: pd.DataFrame, desc: str) -> pd.Series:
    dn = _norm_text(desc)
    sub = dfi[dfi["_desc_norm"] == dn]
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.groupby("numero")["quantidade_registrada"].sum(min_count=1)


def _lookup_tarifa_numero_desc_first(dfi: pd.DataFrame, numero: str, desc: str, default: float = 0.0) -> float:
    dn = _norm_text(desc)
    sub = dfi[(dfi["numero"].astype(str) == str(numero)) & (dfi["_desc_norm"] == dn)]
    if sub.empty:
        return default
    v = sub.iloc[0].get("tarifa", np.nan)
    return default if pd.isna(v) else float(v)


def _max_tarifa_numero_desc(dfi: pd.DataFrame, numero: str, desc: str, default: float = 0.0) -> float:
    dn = _norm_text(desc)
    sub = dfi[(dfi["numero"].astype(str) == str(numero)) & (dfi["_desc_norm"] == dn)]
    if sub.empty:
        return default
    v = sub["tarifa"].max()
    return default if pd.isna(v) else float(v)


def _energia_injetada_tarifa(
    dfi: pd.DataFrame,
    numero: str,
    desc: str,
    med_inj_tusd: float,
    boleto: int,
    gerador: int,
) -> float:
    """
    Replica as colunas:
    - "Energia Inj. TUSD"
    - "Energia Injet. TE"

    Excel (resumo):
    IF(AND(boleto=1, gerador=0, med_inj_tusd<>0),
        média ponderada usando MAXIFS/MINIFS (assumindo 2 linhas),
        XLOOKUP(med_inj_tusd&desc, quantidade&desc, tarifa, 0)
    )
    """
    if boleto != 1:
        return 0.0

    dn = _norm_text(desc)
    sub = dfi[(dfi["numero"].astype(str) == str(numero)) & (dfi["_desc_norm"] == dn)]
    sub = sub.dropna(subset=["tarifa", "quantidade_registrada"])

    if sub.empty:
        return 0.0

    # fallback XLOOKUP por quantidade == med_inj_tusd
    def xlookup_qty():
        if pd.isna(med_inj_tusd) or med_inj_tusd == 0:
            return 0.0
        for _, r in sub.iterrows():
            q = float(r["quantidade_registrada"])
            if _isclose(q, float(med_inj_tusd)):
                return float(r["tarifa"])
        return 0.0

    # condição do IF do Excel
    if gerador == 0 and (not pd.isna(med_inj_tusd)) and float(med_inj_tusd) != 0.0:
        # média ponderada usando apenas max/min (o Excel faz assim)
        # (tar_max*(qty_max/med) + tar_min*(qty_min/med)) / ((qty_max/med)+(qty_min/med))
        if len(sub) == 1:
            return float(sub.iloc[0]["tarifa"])

        # pega max/min por quantidade
        idx_max = sub["quantidade_registrada"].idxmax()
        idx_min = sub["quantidade_registrada"].idxmin()
        rmax = sub.loc[idx_max]
        rmin = sub.loc[idx_min]

        qty_max = float(rmax["quantidade_registrada"])
        qty_min = float(rmin["quantidade_registrada"])
        tar_max = float(rmax["tarifa"])
        tar_min = float(rmin["tarifa"])

        med = float(med_inj_tusd)
        w_max = qty_max / med
        w_min = qty_min / med
        denom = w_max + w_min
        if denom == 0:
            return 0.0
        return (tar_max * w_max + tar_min * w_min) / denom

    # condição falsa -> XLOOKUP
    return xlookup_qty()


def calculate_boletos(
    df_itens: pd.DataFrame,
    df_medidores: pd.DataFrame,
    df_clientes: pd.DataFrame,
    *,
    only_registered_clients: bool = True,
    only_status_ativo: bool = True,
) -> CalcResult:
    """
    Gera o dataframe equivalente ao "Calculos_Boleto" (tabela Calculo).

    Regras:
    - Se only_registered_clients=True: UCs fora de info_clientes são retornadas em missing_clientes e NÃO são calculadas.
    - Se only_status_ativo=True: clientes com status != 'Ativo' são ignorados no cálculo.

    Retorna:
    - df_boletos: colunas em snake_case (compatível com BigQuery)
    - missing_clientes: lista de UCs ausentes
    """
    dfi, dfm, dfc = _prepare_inputs(df_itens, df_medidores, df_clientes)

    required_itens = {"numero", "unidade_consumidora", "descricao", "tarifa", "quantidade_registrada"}
    missing_req = [c for c in required_itens if c not in dfi.columns]
    if missing_req:
        raise ValueError(f"df_itens está sem colunas obrigatórias: {missing_req}")

    required_med = {"nota_fiscal_numero", "tipo", "total_apurado"}
    missing_reqm = [c for c in required_med if c not in dfm.columns]
    if missing_reqm:
        raise ValueError(f"df_medidores está sem colunas obrigatórias: {missing_reqm}")

    required_cli = {"unidade_consumidora", "desconto_contratado", "subvencao", "status", "custo_disp"}
    missing_reqc = [c for c in required_cli if c not in dfc.columns]
    if missing_reqc:
        raise ValueError(f"df_clientes está sem colunas obrigatórias: {missing_reqc}")

    # -----------------------------
    # Base de "boletos" por NF (numero)
    # -----------------------------
    numeros = (
        dfi["numero"]
        .replace(["None", "nan", "NaN", ""], np.nan)
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    base = pd.DataFrame({"numero": numeros}).sort_values("numero").reset_index(drop=True)

    # XLOOKUPs: unidade_consumidora / periodo / nome
    uc_map = _first_by_numero(dfi, "unidade_consumidora")
    periodo_map = _first_by_numero(dfi, "referencia") if "referencia" in dfi.columns else pd.Series(dtype=object)
    nome_map = _first_by_numero(dfi, "nome") if "nome" in dfi.columns else pd.Series(dtype=object)

    base["unidade_consumidora"] = base["numero"].map(uc_map).astype(str)

    # junta clientes
    dfc2 = dfc.copy()
    dfc2["status_norm"] = dfc2["status"].map(_norm_text)

    base = base.merge(
        dfc2[["unidade_consumidora", "desconto_contratado", "subvencao", "status", "status_norm", "custo_disp"]],
        on="unidade_consumidora",
        how="left",
        suffixes=("", "_cli"),
    )

    # Missing clients
    missing_reason: Dict[str, str] = {}
    missing_clientes: List[str] = []

    if only_registered_clients:
        miss = base["custo_disp"].isna() | (base["desconto_contratado"].isna())
        for uc in base.loc[miss, "unidade_consumidora"].astype(str).tolist():
            missing_reason[uc] = "Cliente não cadastrado em info_clientes"
        missing_clientes = sorted(set(base.loc[miss, "unidade_consumidora"].astype(str).tolist()))
        base = base.loc[~miss].copy()

    if only_status_ativo:
        # se não houver status, não bloqueia; se houver e não for ativo, exclui
        mask_inativo = base["status_norm"].notna() & (base["status_norm"] != "ativo")
        if mask_inativo.any():
            for uc in base.loc[mask_inativo, "unidade_consumidora"].astype(str).tolist():
                missing_reason[uc] = "Cliente com status diferente de 'Ativo'"
            base = base.loc[~mask_inativo].copy()

    # período/nome (para exibição)
    base["periodo"] = base["numero"].map(periodo_map)
    base["nome"] = base["numero"].map(nome_map)

    # -----------------------------
    # medidores_apurado (Energia - Energia injetada)
    # -----------------------------
    dfm2 = dfm.copy()
    dfm2["numero"] = dfm2["nota_fiscal_numero"].astype(str)

    # soma por tipo
    g = dfm2.groupby(["numero", "_tipo_norm"])["total_apurado"].sum(min_count=1).unstack()

    energia = g.get(_norm_text("Energia"), pd.Series(dtype=float))
    inj = g.get(_norm_text("Energia injetada"), pd.Series(dtype=float))
    if inj.empty:
        # tolerância a "Energia Injetada"
        inj = g.get(_norm_text("Energia Injetada"), pd.Series(dtype=float))

    base["medidores_apurado"] = base["numero"].map(energia).fillna(0.0) - base["numero"].map(inj).fillna(0.0)

    # -----------------------------
    # injetada / boleto
    # -----------------------------
    base["custo_disp"] = pd.to_numeric(base["custo_disp"], errors="coerce")
    base["injetada"] = base["medidores_apurado"] - base["custo_disp"]
    base["boleto"] = np.where(base["injetada"] <= 0, 0, 1).astype(int)

    # -----------------------------
    # tarifa_cheia_trib (Consumo TE + Consumo TUSD) se boleto=1
    # -----------------------------
    te = _sum_tarifa_by_desc(dfi, "Consumo TE")
    tusd = _sum_tarifa_by_desc(dfi, "Consumo TUSD")
    base["tarifa_cheia_trib"] = base["numero"].map(te).fillna(0.0) + base["numero"].map(tusd).fillna(0.0)
    base.loc[base["boleto"] != 1, "tarifa_cheia_trib"] = 0.0

    # check
    def _check(v: float) -> str:
        if pd.isna(v):
            return ""
        if v > 10:
            return "Parseamento"
        if v > 1:
            return "Subvenção"
        return "Certo"

    base["check"] = base["tarifa_cheia_trib"].map(_check)

    # -----------------------------
    # tarifa_cheia_trib2 (exclui linhas onde quantidade_registrada == subvencao quando check != Certo)
    # -----------------------------
    cons = dfi[dfi["_desc_norm"].isin({_norm_text("Consumo TE"), _norm_text("Consumo TUSD")})].copy()
    cons = cons.merge(base[["numero", "subvencao"]], on="numero", how="left")
    cons["subvencao"] = pd.to_numeric(cons["subvencao"], errors="coerce")

    # critério "<>"&subvencao
    # se subvencao é NaN -> mantém tudo
    cons["_keep"] = cons["subvencao"].isna() | (cons["quantidade_registrada"] != cons["subvencao"])
    cons2 = cons[cons["_keep"]]
    cons_sum_excl = cons2.groupby("numero")["tarifa"].sum(min_count=1)

    base["tarifa_cheia_trib2"] = base["tarifa_cheia_trib"]
    mask_sub = (base["check"] != "Certo") & (base["boleto"] == 1)
    base.loc[mask_sub, "tarifa_cheia_trib2"] = base.loc[mask_sub, "numero"].map(cons_sum_excl).fillna(0.0)

    # -----------------------------
    # gerador
    # (no Excel usa XLOOKUP do primeiro tipo; aqui usamos "existe Energia injetada" -> 1)
    # -----------------------------
    inj_exists = (
        dfm2.groupby("numero")["_tipo_norm"]
        .apply(lambda s: int((_norm_text("Energia injetada") in set(s)) or (_norm_text("Energia Injetada") in set(s))))
    )
    base["gerador"] = base["numero"].map(inj_exists).fillna(0).astype(int)

    # -----------------------------
    # med_inj_tusd
    # boleto * (SUM(qtd Energia Inj. TUSD) - SUM(medidores total_apurado tipo Energia Injetada))
    # -----------------------------
    qtd_inj_tusd = _sum_qtd_by_desc(dfi, "Energia Inj. TUSD")
    inj_med_sum = base["numero"].map(inj).fillna(0.0)
    base["med_inj_tusd"] = base["boleto"] * (base["numero"].map(qtd_inj_tusd).fillna(0.0) - inj_med_sum)

    # -----------------------------
    # Energia Inj. TUSD / Energia Injet. TE (tarifas derivadas)
    # -----------------------------
    energia_inj_tusd_tar = []
    energia_injet_te_tar = []
    for _, r in base.iterrows():
        num = str(r["numero"])
        med = float(r["med_inj_tusd"]) if not pd.isna(r["med_inj_tusd"]) else float("nan")
        energia_inj_tusd_tar.append(
            _energia_injetada_tarifa(dfi, num, "Energia Inj. TUSD", med, int(r["boleto"]), int(r["gerador"]))
        )
        energia_injet_te_tar.append(
            _energia_injetada_tarifa(dfi, num, "Energia Injet. TE", med, int(r["boleto"]), int(r["gerador"]))
        )

    base["energia_inj_tusd_tarifa"] = energia_inj_tusd_tar
    base["energia_injet_te_tarifa"] = energia_injet_te_tar

    base["tarifa_cheia_trib3"] = np.where(
        (base["energia_inj_tusd_tarifa"] + base["energia_injet_te_tarifa"]) != 0,
        base["energia_inj_tusd_tarifa"] + base["energia_injet_te_tarifa"],
        0.0,
    )

    # -----------------------------
    # tarifa_inj_tusd / tarifa_inj_te (fallback quando trib3==0)
    # -----------------------------
    tarifa_inj_tusd = []
    tarifa_inj_te = []
    for _, r in base.iterrows():
        num = str(r["numero"])
        if int(r["boleto"]) != 1:
            tarifa_inj_tusd.append(0.0)
            tarifa_inj_te.append(0.0)
            continue
        if float(r["tarifa_cheia_trib3"]) != 0.0:
            tarifa_inj_tusd.append(0.0)
            tarifa_inj_te.append(0.0)
            continue

        tarifa_inj_tusd.append(_max_tarifa_numero_desc(dfi, num, "Energia Inj. TUSD", default=0.0))
        tarifa_inj_te.append(_max_tarifa_numero_desc(dfi, num, "Energia Injet. TE", default=0.0))

    base["tarifa_inj_tusd"] = tarifa_inj_tusd
    base["tarifa_inj_te"] = tarifa_inj_te

    base["tarifa_cheia"] = np.where(
        base["tarifa_cheia_trib3"] == 0.0,
        base["tarifa_inj_tusd"] + base["tarifa_inj_te"],
        base["tarifa_cheia_trib3"],
    )

    # -----------------------------
    # tarifa_paga_conc / tarifa_erb / tarifa_bol
    # -----------------------------
    base["desconto_contratado"] = pd.to_numeric(base["desconto_contratado"], errors="coerce").fillna(0.0)

    base["tarifa_paga_conc"] = base["tarifa_cheia_trib2"] + base["tarifa_cheia"]
    base["tarifa_erb"] = (1.0 - base["desconto_contratado"]) * base["tarifa_cheia_trib2"]
    base["tarifa_bol"] = base["tarifa_erb"] - base["tarifa_paga_conc"]

    # -----------------------------
    # Bandeiras (XLOOKUP numero&descricao -> tarifa)
    # -----------------------------
    bandeira_amarela = []
    band_am_injet = []
    band_vermelha = []
    band_vrm_injet = []

    for _, r in base.iterrows():
        num = str(r["numero"])
        bol = int(r["boleto"])

        bandeira_amarela.append(_lookup_tarifa_numero_desc_first(dfi, num, "Bandeira Amarela", default=0.0))
        band_am_injet.append(bol * _lookup_tarifa_numero_desc_first(dfi, num, "Band. Am. Injet.", default=0.0))

        band_vermelha.append(bol * _lookup_tarifa_numero_desc_first(dfi, num, "Band. Vermelha", default=0.0))
        band_vrm_injet.append(bol * _lookup_tarifa_numero_desc_first(dfi, num, "Band. Vrm. Injet.", default=0.0))

    base["bandeira_amarela_tarifa"] = bandeira_amarela
    base["band_am_injet_tarifa"] = band_am_injet
    base["band_vermelha_tarifa"] = band_vermelha
    base["band_vrm_injet_tarifa"] = band_vrm_injet

    # -----------------------------
    # valores de bandeira / descontos (como no Excel)
    # -----------------------------
    base["valor_band_amarela"] = base["boleto"] * (base["bandeira_amarela_tarifa"] + base["band_am_injet_tarifa"])
    base["valor_band_amar_desc"] = base["boleto"] * (
        (1.0 - base["desconto_contratado"]) * (base["bandeira_amarela_tarifa"] - base["valor_band_amarela"])
    )

    base["valor_band_vermelha"] = base["boleto"] * (base["band_vermelha_tarifa"] + base["band_vrm_injet_tarifa"])
    base["valor_band_vrm_desc"] = base["boleto"] * (
        (1.0 - base["desconto_contratado"]) * (base["band_vermelha_tarifa"] - base["valor_band_vermelha"])
    )

    # -----------------------------
    # tarifa_total_boleto / valor_total_boleto
    # -----------------------------
    base["tarifa_total_boleto"] = (base["boleto"] * base["valor_band_vrm_desc"]) + base["valor_band_amar_desc"] + base["tarifa_bol"]
    base["valor_total_boleto"] = base["tarifa_total_boleto"] * base["med_inj_tusd"]

    # -----------------------------
    # Saída final (snake_case)
    # -----------------------------
    out_cols = [
        "numero",
        "unidade_consumidora",
        "periodo",
        "nome",
        "custo_disp",
        "medidores_apurado",
        "injetada",
        "boleto",
        "desconto_contratado",
        "check",
        "subvencao",
        "tarifa_cheia_trib",
        "tarifa_cheia_trib2",
        "gerador",
        "med_inj_tusd",
        "energia_inj_tusd_tarifa",
        "energia_injet_te_tarifa",
        "tarifa_cheia_trib3",
        "tarifa_inj_tusd",
        "tarifa_inj_te",
        "tarifa_cheia",
        "tarifa_paga_conc",
        "tarifa_erb",
        "tarifa_bol",
        "bandeira_amarela_tarifa",
        "band_am_injet_tarifa",
        "valor_band_amarela",
        "valor_band_amar_desc",
        "band_vermelha_tarifa",
        "band_vrm_injet_tarifa",
        "valor_band_vermelha",
        "valor_band_vrm_desc",
        "tarifa_total_boleto",
        "valor_total_boleto",
    ]

    df_out = base[out_cols].copy()

    # padroniza NaN -> None (BigQuery friendly)
    df_out = df_out.replace({np.nan: None})

    return CalcResult(
        df_boletos=df_out,
        missing_clientes=missing_clientes,
        missing_reason=missing_reason,
    )
