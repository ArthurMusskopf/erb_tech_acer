"""
ERB Tech - Tela 1: Upload, Revisão e Emissão de Boleto por Fatura (NF)
Fluxo:
1) Upload e parseamento
2) Revisão (editar itens/medidores/cabeçalho)
3) Validar -> calcular boleto (comparar com Excel)
4) (Opcional) Substituir fatura no BigQuery com os dados revisados
5) Gerar boleto no Sicoob Sandbox
"""

from __future__ import annotations

import io
import os
import sys
import zipfile
import tempfile
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import pandas as pd
import streamlit as st
from google.cloud import bigquery

sys.path.append(str(Path(__file__).parent.parent))

from utils.pdf_parser import processar_lote_faturas, make_item_id
from utils.calc_engine import calculate_boletos
from utils.sicoob_client import (
    get_sicoob_config_from_secrets,
    SicoobCobrancaV3Client,
    build_boleto_payload_from_row,
)
from utils.bigquery_client import (
    upsert_dataframe,
    execute_query,
    get_bigquery_client,
    TABLE_FATURA_ITENS,
    TABLE_MEDIDORES,
    TABLE_CLIENTES,
    TABLE_BOLETOS,
)


st.set_page_config(page_title="Upload de Faturas - ERB Tech", page_icon="📄", layout="wide")
st.title("📄 Upload, Revisão e Emissão de Boleto (por Fatura)")


# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
if "faturas_processadas" not in st.session_state:
    st.session_state.faturas_processadas = None

if "calc_por_nf" not in st.session_state:
    st.session_state.calc_por_nf = {}  # nf -> df_calculo

if "edited_itens" not in st.session_state:
    st.session_state.edited_itens = {}  # nf -> df_itens

if "edited_med" not in st.session_state:
    st.session_state.edited_med = {}  # nf -> df_medidores


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _to_iso_date(v: str) -> str:
    v = str(v or "").strip()
    try:
        return datetime.strptime(v, "%d/%m/%Y").strftime("%Y-%m-%d")
    except Exception:
        try:
            return datetime.strptime(v, "%Y-%m-%d").strftime("%Y-%m-%d")
        except Exception:
            return (datetime.utcnow() + timedelta(days=5)).strftime("%Y-%m-%d")


def _recompute_ids_for_invoice(df_itens: pd.DataFrame, df_med: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Recalcula IDs para evitar inconsistências após edição.
    OBS: como vamos deletar a NF antes de salvar, não há risco de duplicar.
    """
    it = df_itens.copy()
    if "id" in it.columns:
        it["id"] = it.apply(
            lambda r: make_item_id(
                r.get("codigo"),
                r.get("unidade_consumidora"),
                r.get("tarifa"),
                r.get("vencimento"),
            ),
            axis=1,
        )

    med = df_med.copy()
    if not med.empty:
        med["id"] = med.apply(
            lambda r: hashlib.sha256(
                f"{r.get('unidade_consumidora','')}|{r.get('cliente_numero','')}|{r.get('referencia','')}|{r.get('nota_fiscal_numero','')}|{r.get('medidor','')}|{r.get('tipo','')}|{r.get('posto','')}".encode()
            ).hexdigest(),
            axis=1,
        )
    return it, med


def _delete_nf_from_bigquery(nf: str) -> None:
    client = get_bigquery_client()
    cfg = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("nf", "STRING", str(nf))]
    )
    client.query(f"DELETE FROM `{TABLE_FATURA_ITENS}` WHERE numero = @nf", job_config=cfg).result()
    client.query(f"DELETE FROM `{TABLE_MEDIDORES}` WHERE nota_fiscal_numero = @nf", job_config=cfg).result()
    client.query(f"DELETE FROM `{TABLE_BOLETOS}` WHERE numero = @nf", job_config=cfg).result()


def _load_cliente(uc: str) -> pd.DataFrame:
    q = f"SELECT * FROM `{TABLE_CLIENTES}` WHERE unidade_consumidora = @uc LIMIT 1"
    return execute_query(q, {"uc": str(uc)})


def _save_cliente(payload: Dict[str, Any]) -> None:
    df = pd.DataFrame([payload])
    upsert_dataframe(df, TABLE_CLIENTES, key_column="unidade_consumidora")


def _calc_boleto_from_dfs(df_itens_nf: pd.DataFrame, df_med_nf: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    if df_itens_nf is None or df_itens_nf.empty:
        return pd.DataFrame(), "Sem itens para calcular."
    uc = str(df_itens_nf["unidade_consumidora"].dropna().iloc[0]) if "unidade_consumidora" in df_itens_nf.columns else ""
    if not uc:
        return pd.DataFrame(), "UC ausente nos itens."

    df_cli = _load_cliente(uc)
    if df_cli is None or df_cli.empty:
        return pd.DataFrame(), f"UC {uc} não cadastrada em info_clientes."

    res = calculate_boletos(
        df_itens=df_itens_nf,
        df_medidores=df_med_nf if df_med_nf is not None else pd.DataFrame(),
        df_clientes=df_cli,
        only_registered_clients=True,
        only_status_ativo=True,
    )
    if res.df_boletos is None or res.df_boletos.empty:
        return pd.DataFrame(), "Cálculo retornou vazio."
    return res.df_boletos, None


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
tab_upload, tab_revisao, tab_historico = st.tabs(["📤 Upload", "📝 Revisão + Boleto", "📜 Histórico"])

# =========================
# TAB UPLOAD
# =========================
with tab_upload:
    st.markdown("### Carregar Faturas")
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_files = st.file_uploader(
            "Arraste PDFs ou arquivo ZIP",
            type=["pdf", "zip"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} arquivo(s) selecionado(s)")

    with col2:
        salvar_auto = st.checkbox("Salvar automaticamente no BigQuery", value=False)

    if uploaded_files and st.button("🚀 Processar Faturas", type="primary", width="stretch"):
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_paths = []

            for uploaded_file in uploaded_files:
                if uploaded_file.name.lower().endswith(".zip"):
                    zip_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(zip_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(temp_dir)

                    for root, _, files in os.walk(temp_dir):
                        for file in files:
                            if file.lower().endswith(".pdf"):
                                pdf_paths.append(os.path.join(root, file))
                else:
                    pdf_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(pdf_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    pdf_paths.append(pdf_path)

            if not pdf_paths:
                st.error("Nenhum PDF encontrado!")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(progress, arquivo):
                    progress_bar.progress(progress)
                    status_text.text(f"Processando: {arquivo}")

                with st.spinner("Processando..."):
                    resultado = processar_lote_faturas(pdf_paths, update_progress)

                progress_bar.progress(1.0)
                status_text.empty()

                c1, c2, c3 = st.columns(3)
                c1.metric("Total", resultado["total"])
                c2.metric("Sucesso", resultado["sucesso"])
                c3.metric("Erros", resultado["erros"])

                erros_lista = [r for r in resultado["resultados"] if not r.sucesso]
                if erros_lista:
                    with st.expander(f"⚠️ {len(erros_lista)} arquivo(s) com erro"):
                        for r in erros_lista:
                            st.error(f"**{r.arquivo}**: {', '.join(r.erros)}")

                if not resultado["df_itens"].empty:
                    st.session_state.faturas_processadas = resultado
                    st.success(f"✅ {len(resultado['df_itens'])} itens parseados!")

                    if salvar_auto:
                        try:
                            with st.spinner("Salvando no BigQuery..."):
                                n_itens = upsert_dataframe(resultado["df_itens"], TABLE_FATURA_ITENS, "id")
                                n_med = (
                                    upsert_dataframe(resultado["df_medidores"], TABLE_MEDIDORES, "id")
                                    if not resultado["df_medidores"].empty
                                    else 0
                                )
                            st.success(f"✅ Salvo: {n_itens} itens, {n_med} medidores")
                        except Exception as e:
                            st.error(f"Erro ao salvar: {e}")


# =========================
# TAB REVISÃO + BOLETO
# =========================
with tab_revisao:
    st.markdown("### Revisar dados e gerar boleto (1 fatura por vez)")

    if st.session_state.faturas_processadas is None:
        st.info("📤 Faça o upload de faturas na aba anterior.")
    else:
        resultado = st.session_state.faturas_processadas
        df_itens_all = resultado["df_itens"].copy()
        df_med_all = resultado["df_medidores"].copy() if resultado["df_medidores"] is not None else pd.DataFrame()

        nfs = sorted(df_itens_all["numero"].dropna().astype(str).unique().tolist())
        if not nfs:
            st.warning("Não encontramos NFs no lote parseado.")
        else:
            nf_sel = st.selectbox("Selecione a Nota Fiscal (NF) para revisão", options=nfs)

            df_it_nf = df_itens_all[df_itens_all["numero"].astype(str) == str(nf_sel)].copy()
            df_med_nf = df_med_all[df_med_all["nota_fiscal_numero"].astype(str) == str(nf_sel)].copy() if not df_med_all.empty else pd.DataFrame()

            if nf_sel in st.session_state.edited_itens:
                df_it_nf = st.session_state.edited_itens[nf_sel].copy()
            if nf_sel in st.session_state.edited_med:
                df_med_nf = st.session_state.edited_med[nf_sel].copy()

            st.markdown("#### 🧾 Cabeçalho")
            uc0 = str(df_it_nf["unidade_consumidora"].dropna().iloc[0]) if "unidade_consumidora" in df_it_nf.columns and not df_it_nf.empty else ""
            nome0 = str(df_it_nf["nome"].dropna().iloc[0]) if "nome" in df_it_nf.columns and not df_it_nf.empty else ""
            ref0 = str(df_it_nf["referencia"].dropna().iloc[0]) if "referencia" in df_it_nf.columns and not df_it_nf.empty else ""
            venc0 = str(df_it_nf["vencimento"].dropna().iloc[0]) if "vencimento" in df_it_nf.columns and not df_it_nf.empty else ""
            tot0 = float(pd.to_numeric(df_it_nf["total_pagar"], errors="coerce").dropna().iloc[0]) if "total_pagar" in df_it_nf.columns and pd.to_numeric(df_it_nf["total_pagar"], errors="coerce").notna().any() else 0.0

            with st.form(f"header_{nf_sel}"):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    uc = st.text_input("Unidade Consumidora (UC)", value=uc0)
                with c2:
                    nome = st.text_input("Nome", value=nome0)
                with c3:
                    referencia = st.text_input("Referência (MM/AAAA)", value=ref0)
                with c4:
                    vencimento = st.text_input("Vencimento (DD/MM/AAAA)", value=venc0)

                total_pagar = st.number_input("Total a pagar (R$)", min_value=0.0, value=float(tot0), step=10.0)

                apply_header = st.form_submit_button("Aplicar alterações no cabeçalho")
                if apply_header:
                    for col, val in [
                        ("unidade_consumidora", uc),
                        ("nome", nome),
                        ("referencia", referencia),
                        ("vencimento", vencimento),
                        ("total_pagar", total_pagar),
                    ]:
                        if col in df_it_nf.columns:
                            df_it_nf[col] = val
                        if not df_med_nf.empty and col in df_med_nf.columns:
                            df_med_nf[col] = val

                    st.success("Cabeçalho aplicado.")
                    st.session_state.edited_itens[nf_sel] = df_it_nf.copy()
                    st.session_state.edited_med[nf_sel] = df_med_nf.copy()
                    st.rerun()

            st.markdown("---")

            st.markdown("#### ✏️ Itens tarifários (editável)")
            editable_cols = [c for c in ["codigo", "descricao", "unidade", "quantidade_registrada", "tarifa", "valor"] if c in df_it_nf.columns]
            fixed_cols = [c for c in df_it_nf.columns if c not in editable_cols]

            col_cfg = {}
            if "quantidade_registrada" in df_it_nf.columns:
                col_cfg["quantidade_registrada"] = st.column_config.NumberColumn("Quantidade (kWh)", format="%.3f")
            if "tarifa" in df_it_nf.columns:
                col_cfg["tarifa"] = st.column_config.NumberColumn("Tarifa", format="%.8f")
            if "valor" in df_it_nf.columns:
                col_cfg["valor"] = st.column_config.NumberColumn("Valor (R$)", format="%.2f")

            df_it_edit = st.data_editor(
                df_it_nf,
                disabled=fixed_cols,
                column_config=col_cfg,
                hide_index=True,
                width="stretch",
                key=f"edit_itens_{nf_sel}",
            )

            st.markdown("#### ✏️ Medidores (editável)")
            if df_med_nf is None or df_med_nf.empty:
                st.info("Sem tabela de medidores para esta NF.")
                df_med_edit = df_med_nf
            else:
                editable_med_cols = [c for c in ["medidor", "tipo", "posto", "total_apurado"] if c in df_med_nf.columns]
                fixed_med_cols = [c for c in df_med_nf.columns if c not in editable_med_cols]
                med_cfg = {}
                if "total_apurado" in df_med_nf.columns:
                    med_cfg["total_apurado"] = st.column_config.NumberColumn("Total apurado", format="%.3f")

                df_med_edit = st.data_editor(
                    df_med_nf,
                    disabled=fixed_med_cols,
                    column_config=med_cfg,
                    hide_index=True,
                    width="stretch",
                    key=f"edit_med_{nf_sel}",
                )

            st.session_state.edited_itens[nf_sel] = df_it_edit.copy()
            st.session_state.edited_med[nf_sel] = (df_med_edit.copy() if df_med_edit is not None else pd.DataFrame())

            st.markdown("---")

            st.markdown("#### 👤 Cadastro do associado (info_clientes)")
            uc_now = str(df_it_edit["unidade_consumidora"].dropna().iloc[0]) if "unidade_consumidora" in df_it_edit.columns and not df_it_edit.empty else ""
            df_cli = _load_cliente(uc_now) if uc_now else pd.DataFrame()
            if df_cli is None or df_cli.empty:
                st.warning(f"UC **{uc_now}** ainda não cadastrada em **info_clientes**. Cadastre para calcular boleto.")
                with st.form(f"cad_{uc_now}"):
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        desconto = st.number_input("Desconto contratado", min_value=0.0, max_value=1.0, value=0.15, step=0.01)
                    with c2:
                        subv = st.number_input("Subvenção", min_value=0.0, value=0.0, step=50.0)
                    with c3:
                        custo_disp = st.number_input("Custo de disponibilidade", min_value=0.0, value=100.0, step=10.0)
                    with c4:
                        status = st.selectbox("Status", ["Ativo", "Inativo"], index=0)

                    salvar = st.form_submit_button("Salvar cadastro")
                    if salvar:
                        _save_cliente({
                            "unidade_consumidora": uc_now,
                            "desconto_contratado": float(desconto),
                            "subvencao": float(subv),
                            "custo_disp": float(custo_disp),
                            "status": str(status),
                        })
                        st.success("✅ Cadastro salvo. Agora você pode calcular.")
                        st.rerun()
            else:
                st.success(f"✅ UC encontrada: {uc_now} (status: {df_cli.iloc[0].get('status')})")
                st.dataframe(df_cli, hide_index=True, width="stretch")

            st.markdown("---")

            cA, cB, cC, cD = st.columns([1, 1, 1, 1])
            with cA:
                calc_btn = st.button("🧮 Calcular boleto (dados revisados)", type="primary", width="stretch")
            with cB:
                save_rev_btn = st.button("💾 Substituir fatura no BigQuery (revisada)", width="stretch")
            with cC:
                save_calc_btn = st.button("💾 Salvar cálculo no BigQuery", width="stretch")
            with cD:
                emit_btn = st.button("🏦 Emitir boleto Sicoob + PDF", width="stretch")

            if calc_btn:
                with st.spinner("Calculando (fiel ao Excel)..."):
                    df_calc, err = _calc_boleto_from_dfs(df_it_edit, df_med_edit)
                if err:
                    st.error(err)
                else:
                    st.session_state.calc_por_nf[nf_sel] = df_calc.copy()
                    st.success("✅ Cálculo concluído. Confira abaixo.")

            df_calc_show = st.session_state.calc_por_nf.get(nf_sel)
            if df_calc_show is not None and not df_calc_show.empty:
                st.markdown("### 🧮 Resultado do cálculo (memorial)")
                st.dataframe(df_calc_show, hide_index=True, width="stretch")

                row = df_calc_show.iloc[0].to_dict()
                valor_calc = float(pd.to_numeric(row.get("valor_total_boleto"), errors="coerce") or 0.0)
                med_kwh = float(pd.to_numeric(row.get("med_inj_tusd"), errors="coerce") or 0.0)

                st.markdown("#### 🔍 Comparação (opcional) com Excel")
                with st.expander("Abrir comparação"):
                    exp_val = st.number_input("Valor total boleto esperado (Excel)", value=float(valor_calc), step=10.0)
                    exp_med = st.number_input("Med. Inj. TUSD esperado (Excel)", value=float(med_kwh), step=1.0)
                    st.write(f"Δ Valor (calc - excel): **{(valor_calc - exp_val):,.2f}**".replace(",", "X").replace(".", ",").replace("X", "."))
                    st.write(f"Δ kWh (calc - excel): **{(med_kwh - exp_med):,.3f}**".replace(",", "X").replace(".", ",").replace("X", "."))

                buf = io.BytesIO()
                df_calc_show.to_excel(buf, index=False)
                buf.seek(0)
                st.download_button(
                    "⬇️ Exportar memorial (Excel)",
                    data=buf,
                    file_name=f"memorial_nf_{nf_sel}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width="stretch",
                )

            if save_rev_btn:
                try:
                    with st.spinner("Substituindo NF no BigQuery (DELETE + INSERT)..."):
                        it_reid, med_reid = _recompute_ids_for_invoice(df_it_edit, df_med_edit)
                        _delete_nf_from_bigquery(str(nf_sel))
                        n_it = upsert_dataframe(it_reid, TABLE_FATURA_ITENS, "id")
                        n_med = upsert_dataframe(med_reid, TABLE_MEDIDORES, "id") if med_reid is not None and not med_reid.empty else 0
                    st.success(f"✅ NF {nf_sel} substituída no BigQuery: {n_it} itens, {n_med} medidores.")
                except Exception as e:
                    st.error(f"Falha ao substituir no BigQuery: {e}")

            if save_calc_btn:
                df_calc_save = st.session_state.calc_por_nf.get(nf_sel)
                if df_calc_save is None or df_calc_save.empty:
                    st.warning("Calcule primeiro.")
                else:
                    try:
                        with st.spinner("Salvando cálculo em boletos_calculados..."):
                            upsert_dataframe(df_calc_save, TABLE_BOLETOS, key_column="numero")
                        st.success("✅ Cálculo salvo.")
                    except Exception as e:
                        st.error(f"Falha ao salvar cálculo: {e}")

            if emit_btn:
                df_calc_emit = st.session_state.calc_por_nf.get(nf_sel)
                if df_calc_emit is None or df_calc_emit.empty:
                    st.warning("Calcule primeiro.")
                else:
                    row = df_calc_emit.iloc[0].to_dict()
                    valor = float(pd.to_numeric(row.get("valor_total_boleto"), errors="coerce") or 0.0)
                    if valor <= 0:
                        st.info("Valor <= 0 indica gerador/remuneração. Emissão de boleto (cobrança) não se aplica.")
                    else:
                        try:
                            cfg = get_sicoob_config_from_secrets()
                            sicoob = SicoobCobrancaV3Client(cfg)

                            venc_raw = str(row.get("vencimento") or "")
                            venc_iso = _to_iso_date(venc_raw)
                            hoje_iso = datetime.utcnow().strftime("%Y-%m-%d")

                            nosso_numero_default = int(str(nf_sel)[:8]) if str(nf_sel).isdigit() else 0
                            nosso_numero = st.number_input(
                                "Nosso número (sandbox)",
                                min_value=0,
                                value=int(nosso_numero_default),
                                step=1,
                                key=f"nn_emit_{nf_sel}",
                            )

                            payload = build_boleto_payload_from_row(
                                row,
                                cfg=cfg,
                                nosso_numero=int(nosso_numero),
                                data_emissao=hoje_iso,
                                data_vencimento=venc_iso,
                                valor=round(valor, 2),
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


# =========================
# TAB HISTÓRICO
# =========================
with tab_historico:
    st.markdown("### Histórico (BigQuery)")

    try:
        query = f"""
        SELECT
            referencia,
            COUNT(DISTINCT unidade_consumidora) AS faturas,
            COUNT(*) AS itens,
            SUM(CASE WHEN codigo = '0D' THEN quantidade_registrada ELSE 0 END) AS consumo_total
        FROM `{TABLE_FATURA_ITENS}`
        GROUP BY referencia
        ORDER BY referencia DESC
        LIMIT 12
        """
        df_hist = execute_query(query)

        if df_hist is not None and not df_hist.empty:
            st.dataframe(df_hist, hide_index=True, width="stretch")
        else:
            st.warning("Nenhum dado encontrado")

    except Exception as e:
        st.warning(f"Não foi possível conectar: {e}")
