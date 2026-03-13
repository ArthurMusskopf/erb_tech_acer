"""
ERB Tech - Tela 3: Dashboard analítico

Objetivo desta versão:
- Usar a base consolidada de reporting_dataset
- Permitir drill down: período -> cliente -> UC
- Exibir KPIs, séries temporais e tabelas analíticas
- Preparar a próxima etapa de demonstrativo/report por associado
"""

from __future__ import annotations

import io
from pathlib import Path
import sys
from typing import Optional

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))

from utils.reporting_dataset import (
    load_reporting_fact,
    load_dashboard_drilldown,
    build_report_payload,
)

st.set_page_config(page_title="Dashboard - ERB Tech", page_icon="📊", layout="wide")
st.title("📊 Dashboard analítico")
st.caption("Drill down: período → cliente → UC")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _fmt_money(v) -> str:
    x = pd.to_numeric(v, errors="coerce")
    if pd.isna(x):
        x = 0.0
    return "R$ " + f"{float(x):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _fmt_int(v) -> str:
    x = pd.to_numeric(v, errors="coerce")
    if pd.isna(x):
        x = 0
    return f"{int(round(float(x), 0)):,}".replace(",", ".")


def _sorted_periodos(values) -> list[str]:
    vals = [str(v) for v in values if str(v).strip()]
    if not vals:
        return []

    df = pd.DataFrame({"periodo": vals})
    df["_ord"] = pd.to_datetime("01/" + df["periodo"], format="%d/%m/%Y", errors="coerce")
    df = df.sort_values("_ord")
    return df["periodo"].drop_duplicates().tolist()


# -----------------------------------------------------------------------------
# Carrega base completa
# -----------------------------------------------------------------------------
df_fact = load_reporting_fact()
if df_fact is None or df_fact.empty:
    st.warning("Nenhuma base consolidada encontrada. Gere uploads, validações e cálculos antes de usar o dashboard.")
    st.stop()

df_fact = df_fact.copy()

for c in [
    "valor_concessionaria",
    "valor_boleto",
    "economia_calculada",
    "consumo_kwh",
    "injetada_kwh",
    "consumo_kwh_parseado",
    "injetada_kwh_parseada",
    "valor_bandeiras",
    "valor_erb",
]:
    if c in df_fact.columns:
        df_fact[c] = pd.to_numeric(df_fact[c], errors="coerce")

for c in ["periodo", "cliente_numero", "nome", "unidade_consumidora", "nota_fiscal"]:
    if c in df_fact.columns:
        df_fact[c] = df_fact[c].fillna("").astype(str)


# -----------------------------------------------------------------------------
# Filtros hierárquicos
# -----------------------------------------------------------------------------
st.markdown("## 🎛️ Filtros")

periodos = _sorted_periodos(df_fact["periodo"].dropna().unique().tolist())

f1, f2, f3 = st.columns(3)

with f1:
    periodo_sel = st.selectbox(
        "Período",
        options=["Todos"] + periodos[::-1],
        index=0,
    )

df_f = df_fact.copy()
if periodo_sel != "Todos":
    df_f = df_f[df_f["periodo"] == periodo_sel].copy()

clientes_opts = (
    df_f[["cliente_numero", "nome"]]
    .drop_duplicates()
    .sort_values(["nome", "cliente_numero"])
)
clientes_labels = ["Todos"] + [
    f"{r['cliente_numero']} | {r['nome']}" for _, r in clientes_opts.iterrows()
]

with f2:
    cliente_label = st.selectbox("Cliente", options=clientes_labels, index=0)

cliente_sel: Optional[str] = None
if cliente_label != "Todos":
    cliente_sel = cliente_label.split(" | ")[0]
    df_f = df_f[df_f["cliente_numero"] == cliente_sel].copy()

ucs_opts = sorted([x for x in df_f["unidade_consumidora"].dropna().unique().tolist() if str(x).strip()])

with f3:
    uc_sel = st.selectbox("UC", options=["Todas"] + ucs_opts, index=0)

if uc_sel != "Todas":
    df_f = df_f[df_f["unidade_consumidora"] == uc_sel].copy()

if df_f.empty:
    st.info("Nenhum dado para os filtros selecionados.")
    st.stop()


# -----------------------------------------------------------------------------
# KPIs
# -----------------------------------------------------------------------------
st.markdown("## 📌 KPIs")

k1, k2, k3, k4, k5, k6 = st.columns(6)

k1.metric("Faturas", _fmt_int(df_f["nota_fiscal"].nunique()))
k2.metric("Clientes", _fmt_int(df_f["cliente_numero"].nunique()))
k3.metric("UCs", _fmt_int(df_f["unidade_consumidora"].nunique()))
k4.metric("Valor concessionária", _fmt_money(df_f["valor_concessionaria"].sum()))
k5.metric("Valor boleto", _fmt_money(df_f["valor_boleto"].sum()))
k6.metric("Economia", _fmt_money(df_f["economia_calculada"].sum()))


# -----------------------------------------------------------------------------
# Base agregada para drilldown
# -----------------------------------------------------------------------------
df_drill = load_dashboard_drilldown(
    periodo=None if periodo_sel == "Todos" else periodo_sel,
    cliente_numero=cliente_sel,
    unidade_consumidora=None if uc_sel == "Todas" else uc_sel,
)

if df_drill is None:
    df_drill = pd.DataFrame()

if not df_drill.empty:
    for c in [
        "valor_concessionaria",
        "valor_boleto",
        "economia_calculada",
        "consumo_kwh",
        "injetada_kwh",
        "consumo_kwh_parseado",
        "injetada_kwh_parseada",
    ]:
        if c in df_drill.columns:
            df_drill[c] = pd.to_numeric(df_drill[c], errors="coerce")


# -----------------------------------------------------------------------------
# Evolução temporal
# -----------------------------------------------------------------------------
st.markdown("## 📈 Evolução temporal")

evol = (
    df_f.groupby("periodo", dropna=False)
    .agg(
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

if not evol.empty:
    evol["_ord"] = pd.to_datetime("01/" + evol["periodo"], format="%d/%m/%Y", errors="coerce")
    evol = evol.sort_values("_ord").drop(columns="_ord")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Valores financeiros por período**")
        st.line_chart(
            evol.set_index("periodo")[["valor_concessionaria", "valor_boleto", "economia_calculada"]]
        )

    with c2:
        st.markdown("**Energia por período**")
        cols_energia = [c for c in ["consumo_kwh", "injetada_kwh", "consumo_kwh_parseado", "injetada_kwh_parseada"] if c in evol.columns]
        st.line_chart(evol.set_index("periodo")[cols_energia])


# -----------------------------------------------------------------------------
# Drilldown analítico
# -----------------------------------------------------------------------------
st.markdown("## 🧭 Drill down analítico")

d1, d2 = st.columns(2)

with d1:
    st.markdown("**Resumo por cliente**")
    resumo_cliente = (
        df_f.groupby(["cliente_numero", "nome"], dropna=False)
        .agg(
            qtd_faturas=("nota_fiscal", "nunique"),
            qtd_ucs=("unidade_consumidora", "nunique"),
            valor_concessionaria=("valor_concessionaria", "sum"),
            valor_boleto=("valor_boleto", "sum"),
            economia_calculada=("economia_calculada", "sum"),
        )
        .reset_index()
        .sort_values("valor_boleto", ascending=False)
    )
    st.dataframe(resumo_cliente, hide_index=True, width="stretch")

with d2:
    st.markdown("**Resumo por UC**")
    resumo_uc = (
        df_f.groupby(["cliente_numero", "nome", "unidade_consumidora"], dropna=False)
        .agg(
            qtd_faturas=("nota_fiscal", "nunique"),
            valor_concessionaria=("valor_concessionaria", "sum"),
            valor_boleto=("valor_boleto", "sum"),
            economia_calculada=("economia_calculada", "sum"),
            consumo_kwh=("consumo_kwh", "sum"),
            injetada_kwh=("injetada_kwh", "sum"),
        )
        .reset_index()
        .sort_values("valor_boleto", ascending=False)
    )
    st.dataframe(resumo_uc, hide_index=True, width="stretch")


# -----------------------------------------------------------------------------
# Base detalhada
# -----------------------------------------------------------------------------
st.markdown("## 🗃️ Base detalhada")

cols_detail = [
    "periodo",
    "cliente_numero",
    "nome",
    "unidade_consumidora",
    "nota_fiscal",
    "valor_concessionaria",
    "valor_boleto",
    "economia_calculada",
    "consumo_kwh",
    "injetada_kwh",
    "status_validacao",
    "status_calculo",
    "status_emissao",
]
cols_detail = [c for c in cols_detail if c in df_f.columns]

st.dataframe(
    df_f[cols_detail].sort_values(["periodo", "cliente_numero", "unidade_consumidora", "nota_fiscal"], ascending=[False, True, True, True]),
    hide_index=True,
    width="stretch",
)

buf_detail = io.BytesIO()
df_f[cols_detail].to_excel(buf_detail, index=False)
buf_detail.seek(0)
st.download_button(
    "⬇️ Exportar base detalhada (Excel)",
    data=buf_detail,
    file_name="base_detalhada_dashboard.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    width="stretch",
)


# -----------------------------------------------------------------------------
# Prévia da próxima etapa: payload do demonstrativo
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("## 🧾 Prévia do demonstrativo por associado")

nfs_report = (
    df_f[["nota_fiscal", "cliente_numero", "nome", "unidade_consumidora", "periodo"]]
    .drop_duplicates()
    .sort_values(["periodo", "cliente_numero", "unidade_consumidora", "nota_fiscal"], ascending=[False, True, True, True])
)

nfs_labels = [
    f"{r['periodo']} | NF {r['nota_fiscal']} | Cliente {r['cliente_numero']} | UC {r['unidade_consumidora']} | {r['nome']}"
    for _, r in nfs_report.iterrows()
]

if nfs_labels:
    nf_report_label = st.selectbox("Selecione uma NF para prévia do demonstrativo", options=nfs_labels)
    nf_report_sel = nf_report_label.split(" | ")[1].replace("NF ", "").strip()

    payload = build_report_payload(nf_report_sel)
    header = payload.get("header", {})
    historico = payload.get("historico", pd.DataFrame())

    if header:
        h1, h2 = st.columns(2)

        with h1:
            st.markdown("**Cabeçalho base do demonstrativo**")
            header_show = pd.DataFrame(
                {
                    "campo": [
                        "nota_fiscal",
                        "periodo",
                        "cliente_numero",
                        "nome",
                        "unidade_consumidora",
                        "cnpj_cpf",
                        "endereco",
                        "cidade_uf",
                        "cep",
                        "data_emissao_fatura",
                        "vencimento",
                        "classe_modalidade",
                        "tarifa_sem_desconto_candidata",
                        "tarifa_com_desconto_candidata",
                        "energia_compensada_kwh",
                        "total_a_pagar_report",
                        "economia_calculada",
                    ],
                    "valor": [
                        header.get("nota_fiscal"),
                        header.get("periodo"),
                        header.get("cliente_numero"),
                        header.get("nome"),
                        header.get("unidade_consumidora"),
                        header.get("cnpj_cpf"),
                        header.get("endereco"),
                        header.get("cidade_uf"),
                        header.get("cep"),
                        header.get("data_emissao_fatura"),
                        header.get("vencimento"),
                        header.get("classe_modalidade"),
                        header.get("tarifa_sem_desconto_candidata"),
                        header.get("tarifa_com_desconto_candidata"),
                        header.get("energia_compensada_kwh"),
                        header.get("total_a_pagar_report"),
                        header.get("economia_calculada"),
                    ],
                }
            )
            st.dataframe(header_show, hide_index=True, width="stretch")

        with h2:
            st.markdown("**Histórico mensal da seleção**")
            if historico is not None and not historico.empty:
                st.dataframe(historico, hide_index=True, width="stretch")
            else:
                st.info("Sem histórico disponível para esta seleção.")
