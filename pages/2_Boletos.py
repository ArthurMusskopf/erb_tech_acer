"""
ERB Tech - Tela 2: Boletos (cálculo fiel + emissão Sicoob Sandbox)
"""

import base64
from datetime import datetime, timedelta
from pathlib import Path
import sys
import io

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))

from utils.bigquery_client import (
    execute_query,
    upsert_dataframe,
    get_periodos_disponiveis,
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
st.title("💰 Boletos de Cobrança (ACER)")


@st.cache_data(ttl=60, show_spinner=False)
def load_boletos_calculados(periodo: str) -> pd.DataFrame:
    q = f"""
    SELECT *
    FROM `{TABLE_BOLETOS}`
    WHERE periodo = @p
    ORDER BY nome
    """
    return execute_query(q, {"p": periodo})


def recalc_and_persist(periodo: str) -> pd.DataFrame:
    with st.spinner("Carregando dados do BigQuery e recalculando..."):
        df_itens = execute_query(f"SELECT * FROM `{TABLE_FATURA_ITENS}` WHERE referencia = @p", {"p": periodo})
        df_med = execute_query(f"SELECT * FROM `{TABLE_MEDIDORES}` WHERE referencia = @p", {"p": periodo})

        # clientes usados no período (somente UCs que aparecem)
        ucs = []
        if df_itens is not None and not df_itens.empty and "unidade_consumidora" in df_itens.columns:
            ucs = df_itens["unidade_consumidora"].dropna().astype(str).unique().tolist()

        if ucs:
            df_cli = execute_query(
                f"SELECT * FROM `{TABLE_CLIENTES}` WHERE unidade_consumidora IN UNNEST(@ucs)",
                {"ucs": ucs},
            )
        else:
            df_cli = pd.DataFrame()

        res = calculate_boletos(
            df_itens=df_itens,
            df_medidores=df_med,
            df_clientes=df_cli,
            only_registered_clients=True,
            only_status_ativo=True,
        )

        if res.missing_clientes:
            st.warning(f"Existem {len(res.missing_clientes)} UC(s) sem cadastro ativo. Cadastre abaixo e recalcule.")
            st.session_state["missing_clientes_boletos"] = res.missing_clientes
            st.session_state["missing_reason_boletos"] = res.missing_reason

        df_out = res.df_boletos.copy()
        if df_out is None or df_out.empty:
            st.warning("Cálculo retornou vazio.")
            return pd.DataFrame()

        upsert_dataframe(df_out, TABLE_BOLETOS, key_column="numero")
        st.success(f"✅ Gravado em {TABLE_BOLETOS}: {len(df_out)} linhas.")
        return df_out


def render_cadastro_clientes(missing: list[str]) -> None:
    st.markdown("### 🧾 Cadastro rápido de UCs pendentes")

    with st.expander("Abrir formulário de cadastro", expanded=True):
        for uc in missing:
            with st.form(f"form_uc_{uc}"):
                st.markdown(f"#### UC **{uc}**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    desconto = st.number_input("Desconto contratado", min_value=0.0, max_value=1.0, value=0.15, step=0.01)
                with col2:
                    subv = st.number_input("Subvenção", min_value=0.0, value=0.0, step=50.0)
                with col3:
                    custo_disp = st.number_input("Custo disponibilidade", min_value=0.0, value=100.0, step=10.0)
                with col4:
                    status = st.selectbox("Status", ["Ativo", "Inativo"], index=0)

                nome = st.text_input("Nome (opcional)", value="")
                cnpj = st.text_input("CNPJ/CPF (opcional)", value="")
                cep = st.text_input("CEP (opcional)", value="")
                cidade_uf = st.text_input("Cidade/UF (opcional)", value="")

                ok = st.form_submit_button("Salvar UC")
                if ok:
                    df = pd.DataFrame([{
                        "unidade_consumidora": str(uc),
                        "desconto_contratado": float(desconto),
                        "subvencao": float(subv),
                        "custo_disp": float(custo_disp),
                        "status": str(status),
                        "nome": nome,
                        "cnpj": cnpj,
                        "cep": cep,
                        "cidade_uf": cidade_uf,
                    }])
                    upsert_dataframe(df, TABLE_CLIENTES, key_column="unidade_consumidora")
                    st.success("✅ UC cadastrada/atualizada.")


# ---------------- UI topo ----------------
colA, colB, colC, colD = st.columns([2, 1, 1, 1])

with colA:
    periodos = get_periodos_disponiveis()
    periodo = st.selectbox("Período de referência", options=periodos if periodos else ["(sem dados)"])
with colB:
    st.markdown("<br>", unsafe_allow_html=True)
    btn_load = st.button("📥 Carregar", use_container_width=True, disabled=(periodo == "(sem dados)"))
with colC:
    st.markdown("<br>", unsafe_allow_html=True)
    btn_recalc = st.button("🧮 Recalcular e gravar", use_container_width=True, disabled=(periodo == "(sem dados)"))
with colD:
    st.markdown("<br>", unsafe_allow_html=True)
    filtro = st.selectbox("Filtro", ["Cobrança (valor>0)", "Geradores (valor<0)", "Todos"], index=0)

st.markdown("---")

# cadastro pendências (se existirem)
missing = st.session_state.get("missing_clientes_boletos", [])
if missing:
    render_cadastro_clientes(missing)

# carregamento
dfb = pd.DataFrame()
if btn_recalc and periodo != "(sem dados)":
    dfb = recalc_and_persist(periodo)
elif btn_load and periodo != "(sem dados)":
    with st.spinner("Carregando boletos calculados..."):
        dfb = load_boletos_calculados(periodo)

if dfb is None or dfb.empty:
    st.info("Carregue um período ou recalcule para gerar a base de boletos.")
    st.stop()

# filtros
dfb["valor_total_boleto_num"] = pd.to_numeric(dfb["valor_total_boleto"], errors="coerce").fillna(0.0)

if filtro == "Cobrança (valor>0)":
    df_show = dfb[dfb["valor_total_boleto_num"] > 0].copy()
elif filtro == "Geradores (valor<0)":
    df_show = dfb[dfb["valor_total_boleto_num"] < 0].copy()
else:
    df_show = dfb.copy()

# resumo
col1, col2, col3, col4 = st.columns(4)
col1.metric("Linhas", len(df_show))
col2.metric("Cobranças (qtd)", int((df_show["valor_total_boleto_num"] > 0).sum()))
col3.metric("Total cobranças (R$)", f"{df_show.loc[df_show['valor_total_boleto_num']>0,'valor_total_boleto_num'].sum():,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
col4.metric("Total geradores (R$)", f"{df_show.loc[df_show['valor_total_boleto_num']<0,'valor_total_boleto_num'].sum():,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

st.markdown("---")

# Sicoob client
try:
    sicoob_cfg = get_sicoob_config_from_secrets()
    sicoob = SicoobCobrancaV3Client(sicoob_cfg)
    sicoob_ok = True
except Exception as e:
    sicoob_ok = False
    st.warning(f"Sicoob não configurado em secrets.toml: {e}")

st.markdown(f"### 📋 Boletos — {periodo}")

for i, r in df_show.sort_values("nome").iterrows():
    nome = str(r.get("nome") or "")[:60]
    uc = str(r.get("unidade_consumidora") or "")
    nf = str(r.get("numero") or "")
    valor = float(r.get("valor_total_boleto_num") or 0.0)

    tag = "🟢 COBRANÇA" if valor > 0 else ("🟣 GERADOR" if valor < 0 else "⚪ 0")
    with st.expander(f"{tag} | {nome} | UC {uc} | NF {nf} | R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")):

        colL, colR = st.columns([2, 1])

        with colL:
            st.markdown("#### 🧾 Memorial (cálculo)")
            cols_show = [
                "medidores_apurado", "custo_disp", "injetada", "med_inj_tusd",
                "tarifa_cheia_trib2", "tarifa_cheia", "tarifa_paga_conc",
                "tarifa_erb", "tarifa_bol",
                "valor_band_amar_desc", "valor_band_vrm_desc",
                "tarifa_total_boleto", "valor_total_boleto",
                "check", "gerador", "boleto",
            ]
            mem = {c: r.get(c) for c in cols_show if c in df_show.columns}
            st.dataframe(pd.DataFrame([mem]).T.rename(columns={0: "valor"}), use_container_width=True)

        with colR:
            st.markdown("#### 👤 Dados p/ boleto")
            st.write(f"**Vencimento (fatura):** {r.get('vencimento')}")
            st.write(f"**CNPJ/CPF:** {r.get('cnpj_cpf')}")
            st.write(f"**CEP:** {r.get('cep')}")
            st.write(f"**Cidade/UF:** {r.get('cidade_uf')}")

            st.markdown("#### 🏦 Sicoob (Sandbox)")
            if valor <= 0:
                st.info("Para valor < 0 (geradores), isso é remuneração. Fluxo de pagamento fica para a próxima fase.")
            else:
                if not sicoob_ok:
                    st.error("Sicoob não configurado.")
                else:
                    # vencimento: tenta converter "DD/MM/YYYY" -> YYYY-MM-DD; senão usa hoje+5
                    venc_raw = str(r.get("vencimento") or "")
                    venc_iso = None
                    try:
                        # casos comuns do parser
                        dt = datetime.strptime(venc_raw, "%d/%m/%Y")
                        venc_iso = dt.strftime("%Y-%m-%d")
                    except Exception:
                        venc_iso = (datetime.utcnow() + timedelta(days=5)).strftime("%Y-%m-%d")

                    hoje_iso = datetime.utcnow().strftime("%Y-%m-%d")

                    nosso_numero = st.number_input(
                        "Nosso número (sandbox)",
                        min_value=0,
                        value=int(re.sub(r"\D+", "", nf)[:8] or "0"),
                        step=1,
                        key=f"nn_{nf}",
                    )

                    if st.button("🚀 Emitir boleto + baixar PDF", key=f"emit_{nf}", use_container_width=True):
                        try:
                            payload = build_boleto_payload_from_row(
                                r.to_dict(),
                                cfg=sicoob_cfg,
                                nosso_numero=int(nosso_numero),
                                data_emissao=hoje_iso,
                                data_vencimento=venc_iso,
                                valor=round(float(valor), 2),
                            )
                            resp_create = sicoob.create_boleto(payload)

                            # tenta extrair nossoNumero retornado; se não vier, usa o enviado
                            nosso_ret = (
                                resp_create.get("resultado", {}).get("nossoNumero")
                                if isinstance(resp_create, dict) else None
                            ) or int(nosso_numero)

                            st.success(f"✅ Boleto criado. NossoNúmero: {nosso_ret}")

                            resp_pdf = sicoob.segunda_via_pdf(nosso_numero=nosso_ret, gerar_pdf=True)
                            pdf_bytes = sicoob.decode_pdf_boleto(resp_pdf)

                            st.download_button(
                                "⬇️ Download PDF do Boleto",
                                data=pdf_bytes,
                                file_name=f"boleto_{nf}.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                            )

                            # mostra linha digitável/qrCode se vierem
                            res2 = resp_pdf.get("resultado", {})
                            if res2.get("linhaDigitavel"):
                                st.code(res2["linhaDigitavel"])
                            if res2.get("qrCode"):
                                st.text_area("QR Code (copia/cola)", value=res2["qrCode"], height=120)

                        except Exception as e:
                            st.error(f"Falha ao emitir/baixar boleto: {e}")

st.markdown("---")
st.markdown("### 📦 Exportação")
buf = io.BytesIO()
df_show.drop(columns=["valor_total_boleto_num"], errors="ignore").to_excel(buf, index=False)
buf.seek(0)
st.download_button(
    "⬇️ Exportar Excel (boletos do filtro)",
    data=buf,
    file_name=f"boletos_{periodo.replace('/','_')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
