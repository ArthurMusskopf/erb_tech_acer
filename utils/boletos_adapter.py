# utils/boletos_adapter.py
from __future__ import annotations

from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd


def _to_float(x) -> Optional[float]:
    try:
        v = float(pd.to_numeric(x, errors="coerce"))
        return None if np.isnan(v) else v
    except Exception:
        return None


def _sum_consumo_0d(df_itens: pd.DataFrame) -> Optional[float]:
    if df_itens is None or df_itens.empty:
        return None
    if "codigo" not in df_itens.columns or "quantidade_registrada" not in df_itens.columns:
        return None
    cod = df_itens["codigo"].astype(str).str.strip()
    qtd = pd.to_numeric(df_itens["quantidade_registrada"], errors="coerce").fillna(0.0)
    return float(qtd[cod == "0D"].sum())


def calc_to_boletos_schema(
    df_calc: pd.DataFrame,
    df_itens_nf: pd.DataFrame,
    *,
    status: str = "calculado",
    validado_por: Optional[str] = None,
    validado_em: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Gera um DF com o schema esperado em boletos_calculados:
    - id (REQUIRED) = NF
    - nota_fiscal = NF
    - valor_final = valor_total_boleto
    """
    if df_calc is None or df_calc.empty:
        return pd.DataFrame()

    r = df_calc.iloc[0].to_dict()

    nf = str(r.get("numero") or "")
    now = datetime.utcnow()

    med_inj = _to_float(r.get("med_inj_tusd")) or 0.0

    out: Dict[str, Any] = {
        "id": nf,
        "nota_fiscal": nf,
        "unidade_consumidora": str(r.get("unidade_consumidora") or ""),
        "cliente_numero": str(r.get("cliente_numero") or ""),
        "nome": str(r.get("nome") or ""),
        "periodo": str(r.get("periodo") or ""),

        "consumo_kwh": _sum_consumo_0d(df_itens_nf),
        "injetada_kwh": med_inj,

        "tarifa_cheia": _to_float(r.get("tarifa_cheia_trib2")),
        "tarifa_injetada": _to_float(r.get("tarifa_cheia")),   # pode ser negativa (como no memorial)
        "tarifa_liquida": _to_float(r.get("tarifa_total_boleto")),
        "desconto_contratado": _to_float(r.get("desconto_contratado")),
        "subvencao": _to_float(r.get("subvencao")),

        "valor_erb": (_to_float(r.get("tarifa_erb")) or 0.0) * med_inj,
        "valor_bandeiras": ((_to_float(r.get("valor_band_amar_desc")) or 0.0) + (_to_float(r.get("valor_band_vrm_desc")) or 0.0)) * med_inj,
        "valor_final": _to_float(r.get("valor_total_boleto")),

        "status": status,
        "validado_por": validado_por,
        "validado_em": validado_em,
        "created_at": now,
        "updated_at": now,

        # rastreabilidade extra (já existe no seu schema)
        "custo_disp": _to_float(r.get("custo_disp")),
        "medidores_apurado": _to_float(r.get("medidores_apurado")),
        "injetada": _to_float(r.get("injetada")),
        "boleto": int(_to_float(r.get("boleto")) or 0),
        "tarifa_cheia_trib": _to_float(r.get("tarifa_cheia_trib")),
        "check": str(r.get("check") or ""),
        "tarifa_cheia_trib2": _to_float(r.get("tarifa_cheia_trib2")),
        "gerador": _to_float(r.get("gerador")),
        "energia_inj_tusd": _to_float(r.get("energia_inj_tusd_tarifa")),
        "energia_injet_te": _to_float(r.get("energia_injet_te_tarifa")),
        "tarifa_cheia_trib3": _to_float(r.get("tarifa_cheia_trib3")),
        "tarifa_inj_tusd": _to_float(r.get("tarifa_inj_tusd")),
        "tarifa_inj_te": _to_float(r.get("tarifa_inj_te")),
        "tarifa_paga_conc": _to_float(r.get("tarifa_paga_conc")),
        "tarifa_erb": _to_float(r.get("tarifa_erb")),
    }

    return pd.DataFrame([out])
