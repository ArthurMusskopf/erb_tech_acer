"""
ERB Tech - Tela 2: Boletos (por Fatura)
Objetivo:
- Selecionar uma NF já registrada no BigQuery
- Calcular o boleto (fiel ao Excel) e exibir memorial
- (Opcional) Salvar cálculo em boletos_calculados
- Emitir boleto no Sicoob Sandbox e baixar PDF
"""

from __future__ import annotations

import io
import re
from datetime import datetime, timedelta
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))

from utils.bigquery_client import (
    execute_query,
    upsert_dataframe,
    TABLE_FATURA_ITENS,
    TABLE_MEDIDORES,
    TABLE_CLIENTES,
    TABLE_BOLETOS,
)

from utils.calc_engine import calculate_boletos
from utils.sicoob_client import (
    get_sicoob_config_from_secrets,
    SicoobCobrancaV3Client,
    build_boleto_payload_from_row,
)

st.set_page_config(page_title="Boletos - ERB Tech", page_icon="💰", layout="wide")
st.title("💰 Boletos (por Fatura / NF)")


@st.cache_data(ttl=60, show_spinner=False)
def load_nf_list(limit: int = 200) -> pd.DataFrame:
    q = f"""
    SELECT
      numero,
      referencia,
      unidade_consumidora,
      ANY_VALUE(nome) AS nome,
      ANY_VALUE(vencimento) AS vencimento,
      ANY_VALUE(total_pagar) AS total_pagar
    FROM `{TABLE_FATURA_ITENS}`
    WHERE numero IS NOT NULL
    GROUP BY numero, referencia, unidade_consumidora
    ORDER BY referencia DESC, numero DESC
    LIMIT {int(limit)}
    """
    return execute_query(q)


@st.cache_data(ttl=60, show_spinner=False)
def load_invoice_data(nf: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    q1 = f"SELECT * FROM `{TABLE_FATURA_ITENS}` WHERE numero = @nf"
    q2 = f"SELECT * FROM `{TABLE_MEDIDORES}` WHERE nota_fiscal_numero = @nf"
    return execute_query(q1, {"nf": str(nf)}), execute_query(q2, {"nf": str(nf)})


@st.cache_data(ttl=60, show_spinner=False)
def load_cliente(uc: str) -> pd.DataFrame:
    q = f"SELECT * FROM `{TABLE_CLIENTES}` WHERE unidade_consumidora = @uc LIMIT 1"
    return execute_query(q, {"uc": str(uc)})


def _to_iso_date(v: str) -> str:
    v = str(v or "").strip()
    try:
        return datetime.strptime(v, "%d/%m/%Y").strftime("%Y-%m-%d")
    except Exception:
        try:
            return datetime.strptime(v, "%Y-%m-%d").strftime("%Y-%m-%d")
        except Exception:
            return (datetime.utcnow() + timedelta(days=5)).strftime("%Y-%m-%d")


df_nfs = load_nf_list()
if df_nfs is None or df_nfs.empty:
    st.warning("Nenhuma NF encontrada em BigQuery. Faça upload na Tela 1.")
    st.stop()

df_nfs = df_nfs.copy()
df_nfs["label"] = df_nfs.apply(
    lambda r: f"{r['referencia']} | NF {r['numero']} | UC {r['unidade_consumidora']} | {str(r.get('nome') or '')[:40]}",
    axis=1,
)

nf_label = st.selectbox("Selecione a NF", options=df_nfs["label"].tolist())
nf_sel = str(df_nfs.loc[df_nfs["label"] == nf_label, "numero"].iloc[0])

c1, c2, c3 = st.columns([1, 1, 1])
calc_btn = c1.button("🧮 Calcular", type="primary", width="stretch")
save_btn = c2.button("💾 Salvar cálculo", width="stretch")
emit_btn = c3.button("🏦 Emitir + PDF", width="stretch")

st.markdown("---")

if "calc_nf" not in st.session_state:
    st.session_state.calc_nf = {}

if calc_btn:
    df_it, df_med = load_invoice_data(nf_sel)
    if df_it is None or df_it.empty:
        st.error("NF sem itens.")
    else:
        uc = str(df_it["unidade_consumidora"].dropna().iloc[0]) if "unidade_consumidora" in df_it.columns else ""
        df_cli = load_cliente(uc) if uc else pd.DataFrame()
        if df_cli is None or df_cli.empty:
            st.error(f"UC {uc} não cadastrada em info_clientes. Cadastre na Tela 1.")
        else:
            with st.spinner("Calculando (fiel ao Excel)..."):
                res = calculate_boletos(df_itens=df_it, df_medidores=df_med, df_clientes=df_cli)
            st.session_state.calc_nf[nf_sel] = res.df_boletos.copy()
            st.success("✅ Cálculo concluído.")

df_calc = st.session_state.calc_nf.get(nf_sel)
if df_calc is None or df_calc.empty:
    st.info("Clique em **Calcular** para ver o memorial.")
    st.stop()

st.markdown("### 🧮 Memorial")
st.dataframe(df_calc, hide_index=True, width="stretch")

row = df_calc.iloc[0].to_dict()
valor = float(pd.to_numeric(row.get("valor_total_boleto"), errors="coerce") or 0.0)

buf = io.BytesIO()
df_calc.to_excel(buf, index=False)
buf.seek(0)
st.download_button(
    "⬇️ Exportar memorial (Excel)",
    data=buf,
    file_name=f"memorial_nf_{nf_sel}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    width="stretch",
)

if save_btn:
    try:
        with st.spinner("Salvando em boletos_calculados..."):
            upsert_dataframe(df_calc, TABLE_BOLETOS, key_column="numero")
        st.success("✅ Salvo.")
    except Exception as e:
        st.error(f"Erro ao salvar: {e}")

if emit_btn:
    if valor <= 0:
        st.info("Valor <= 0 indica gerador/remuneração. Emissão de boleto não se aplica.")
    else:
        try:
            cfg = get_sicoob_config_from_secrets()
            sicoob = SicoobCobrancaV3Client(cfg)

            venc_iso = _to_iso_date(str(row.get("vencimento") or ""))
            hoje_iso = datetime.utcnow().strftime("%Y-%m-%d")

            nosso_numero_default = int(re.sub(r"\D+", "", str(nf_sel))[:8] or "0")
            nosso_numero = st.number_input(
                "Nosso número (sandbox)",
                min_value=0,
                value=int(nosso_numero_default),
                step=1,
                key=f"nn_{nf_sel}",
            )

            payload = build_boleto_payload_from_row(
                row,
                cfg=cfg,
                nosso_numero=int(nosso_numero),
                data_emissao=hoje_iso,
                data_vencimento=venc_iso,
                valor=round(float(valor), 2),
            )

            with st.spinner("Criando boleto no Sicoob (sandbox)..."):
                resp_create = sicoob.create_boleto(payload)

            nosso_ret = (resp_create.get("resultado", {}) or {}).get("nossoNumero") or int(nosso_numero)
            st.success(f"✅ Boleto criado. NossoNúmero: {nosso_ret}")

            with st.spinner("Baixando PDF (2ª via)..."):
                resp_pdf = sicoob.segunda_via_pdf(nosso_numero=nosso_ret, gerar_pdf=True)
                pdf_bytes = sicoob.decode_pdf_boleto(resp_pdf)

            st.download_button(
                "⬇️ Download PDF do Boleto",
                data=pdf_bytes,
                file_name=f"boleto_nf_{nf_sel}.pdf",
                mime="application/pdf",
                width="stretch",
            )

            res2 = resp_pdf.get("resultado", {}) if isinstance(resp_pdf, dict) else {}
            if res2.get("linhaDigitavel"):
                st.code(res2["linhaDigitavel"])
            if res2.get("qrCode"):
                st.text_area("QR Code (copia/cola)", value=res2["qrCode"], height=120)

        except Exception as e:
            st.error(f"Falha na emissão/2ª via: {e}")
