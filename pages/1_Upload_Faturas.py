"""
ERB Tech - Tela 1: Upload + Revisão + Cálculo em lote com bloqueios (info_clientes manual)

v2 (objetivo demo):
- Upload/parsing e gravação BigQuery
- Diagnóstico do lote: NFs inéditas vs já existentes no workflow
- Cadastro manual (dimensão info_clientes) com n_fases e custo_disp INTEGER
- Cálculo por NF e cálculo em lote (somente calculáveis)
- Salva cálculo no BigQuery em boletos_calculados usando schema (id/nota_fiscal/valor_final...)
"""

from __future__ import annotations

import io
import os
import sys
import zipfile
import tempfile
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import pandas as pd
import streamlit as st
from google.cloud import bigquery

sys.path.append(str(Path(__file__).parent.parent))

from utils.pdf_parser import processar_lote_faturas, make_item_id
from utils.calc_engine import calculate_boletos, infer_n_fases, compute_custo_disp
from utils.boletos_adapter import calc_to_boletos_schema
from utils.workflow_adapter import build_workflow_from_parse_results
from utils.bigquery_client import (
    upsert_dataframe,
    execute_query,
    get_bigquery_client,
    TABLE_FATURA_ITENS,
    TABLE_MEDIDORES,
    TABLE_CLIENTES,
    TABLE_BOLETOS,
    TABLE_FATURAS_WORKFLOW,
)

APP_VERSION = "upload_v2_lote_cadastro"

st.set_page_config(page_title="Upload de Faturas - ERB Tech", page_icon="📄", layout="wide")
st.title("📄 Upload, Revisão e Cálculo (por Lote)")
st.caption(f"versão: {APP_VERSION}")


# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
if "faturas_processadas" not in st.session_state:
    st.session_state.faturas_processadas = None
if "calc_por_nf" not in st.session_state:
    st.session_state.calc_por_nf = {}
if "edited_itens" not in st.session_state:
    st.session_state.edited_itens = {}
if "edited_med" not in st.session_state:
    st.session_state.edited_med = {}
if "workflow_lote" not in st.session_state:
    st.session_state.workflow_lote = None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso_date(v: str) -> str:
    v = str(v or "").strip()
    try:
        return datetime.strptime(v, "%d/%m/%Y").strftime("%Y-%m-%d")
    except Exception:
        try:
            return datetime.strptime(v, "%Y-%m-%d").strftime("%Y-%m-%d")
        except Exception:
            return (_now_utc() + timedelta(days=5)).strftime("%Y-%m-%d")


def _num_or_default(x, default: float) -> float:
    v = pd.to_numeric(x, errors="coerce")
    if pd.isna(v):
        return float(default)
    return float(v)


def _int_or_default(x, default: int) -> int:
    v = pd.to_numeric(x, errors="coerce")
    if pd.isna(v):
        return int(default)
    return int(round(float(v), 0))


def _recompute_ids_for_invoice(df_itens: pd.DataFrame, df_med: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    if med is not None and not med.empty:
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
    client.query(f"DELETE FROM `{TABLE_BOLETOS}` WHERE id = @nf OR nota_fiscal = @nf", job_config=cfg).result()


def _load_cliente(uc: str) -> pd.DataFrame:
    q = f"SELECT * FROM `{TABLE_CLIENTES}` WHERE unidade_consumidora = @uc LIMIT 1"
    return execute_query(q, {"uc": str(uc)})


def _load_clientes_for_ucs(ucs: List[str]) -> pd.DataFrame:
    if not ucs:
        return pd.DataFrame()
    q = f"""
    SELECT unidade_consumidora, desconto_contratado, subvencao, status, n_fases, custo_disp
    FROM `{TABLE_CLIENTES}`
    WHERE unidade_consumidora IN UNNEST(@ucs)
    """
    return execute_query(q, {"ucs": ucs})


# -----------------------------------------------------------------------------
# Workflow helpers (Etapa 1)
# -----------------------------------------------------------------------------
def _load_existing_workflow_by_nfs(nfs: List[str]) -> pd.DataFrame:
    """
    Carrega registros já existentes em faturas_workflow para classificar
    inéditas vs já existentes no lote parseado.
    """
    if not nfs:
        return pd.DataFrame()

    q = f"""
    SELECT
        id,
        nota_fiscal,
        unidade_consumidora,
        referencia,
        is_inedita,
        duplicada_de,
        status_parse,
        status_validacao,
        status_calculo,
        status_emissao,
        arquivo_nome_original,
        arquivo_hash,
        pdf_uri,
        created_at,
        updated_at
    FROM `{TABLE_FATURAS_WORKFLOW}`
    WHERE id IN UNNEST(@nfs)
    """
    return execute_query(q, {"nfs": [str(x) for x in nfs if str(x).strip()]})


def _upsert_workflow_status(
    nf: str,
    *,
    status_parse: Optional[str] = None,
    status_validacao: Optional[str] = None,
    validado_por: Optional[str] = None,
    validado_em: Optional[datetime] = None,
    status_calculo: Optional[str] = None,
    calculado_em: Optional[datetime] = None,
    status_emissao: Optional[str] = None,
    emitido_em: Optional[datetime] = None,
    observacoes_append: Optional[str] = None,
) -> None:
    """
    Atualiza a linha da NF em faturas_workflow sem perder os demais campos.
    """
    if not nf:
        return

    q = f"""
    SELECT *
    FROM `{TABLE_FATURAS_WORKFLOW}`
    WHERE id = @nf
    LIMIT 1
    """
    df_current = execute_query(q, {"nf": str(nf)})
    if df_current is None or df_current.empty:
        return

    row = df_current.iloc[0].to_dict()

    if status_parse is not None:
        row["status_parse"] = status_parse

    if status_validacao is not None:
        row["status_validacao"] = status_validacao
        row["validado_por"] = validado_por
        row["validado_em"] = validado_em

    if status_calculo is not None:
        row["status_calculo"] = status_calculo
        row["calculado_em"] = calculado_em

    if status_emissao is not None:
        row["status_emissao"] = status_emissao
        row["emitido_em"] = emitido_em

    if observacoes_append:
        obs_atual = str(row.get("observacoes") or "").strip()
        if obs_atual:
            row["observacoes"] = f"{obs_atual} || {observacoes_append}"
        else:
            row["observacoes"] = observacoes_append

    row["updated_at"] = _now_utc()
    upsert_dataframe(pd.DataFrame([row]), TABLE_FATURAS_WORKFLOW, key_column="id")


def _build_hash_by_filename(uploaded_files) -> Dict[str, str]:
    """
    Gera hash simples do arquivo original enviado pelo usuário.
    Nesta etapa usamos apenas os arquivos do lote atual, sem storage ainda.
    """
    out: Dict[str, str] = {}
    for f in uploaded_files or []:
        try:
            raw = f.getvalue()
            out[f.name] = hashlib.sha256(raw).hexdigest()
        except Exception:
            out[f.name] = ""
    return out


def _summarize_workflow_status(df_workflow: pd.DataFrame) -> Dict[str, int]:
    if df_workflow is None or df_workflow.empty:
        return {
            "total_nf": 0,
            "ineditas": 0,
            "repetidas": 0,
            "parseadas": 0,
            "erro_parse": 0,
        }

    total_nf = len(df_workflow)
    ineditas = int(df_workflow["is_inedita"].fillna(False).astype(bool).sum()) if "is_inedita" in df_workflow.columns else 0
    repetidas = int(total_nf - ineditas)
    parseadas = int((df_workflow.get("status_parse") == "parseado").sum()) if "status_parse" in df_workflow.columns else 0
    erro_parse = int((df_workflow.get("status_parse") == "erro_parse").sum()) if "status_parse" in df_workflow.columns else 0

    return {
        "total_nf": int(total_nf),
        "ineditas": int(ineditas),
        "repetidas": int(repetidas),
        "parseadas": int(parseadas),
        "erro_parse": int(erro_parse),
    }


def _cadastro_motivo(cli_row: Optional[pd.Series]) -> Optional[str]:
    if cli_row is None:
        return "UC não cadastrada"
    if pd.isna(cli_row.get("desconto_contratado")):
        return "desconto_contratado vazio"
    if pd.isna(cli_row.get("custo_disp")):
        return "custo_disp vazio"
    stt = str(cli_row.get("status") or "").strip().lower()
    if not stt:
        return "status vazio"
    if stt != "ativo":
        return f"status '{cli_row.get('status')}'"
    return None


def _calc_boleto_from_dfs(df_itens_nf: pd.DataFrame, df_med_nf: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    if df_itens_nf is None or df_itens_nf.empty:
        return pd.DataFrame(), "Sem itens para calcular."

    uc = ""
    if "unidade_consumidora" in df_itens_nf.columns and df_itens_nf["unidade_consumidora"].notna().any():
        uc = str(df_itens_nf["unidade_consumidora"].dropna().iloc[0]).strip()
    if not uc:
        return pd.DataFrame(), "UC ausente nos itens."

    df_cli = _load_cliente(uc)
    if df_cli is None or df_cli.empty:
        return pd.DataFrame(), f"UC {uc} não cadastrada em info_clientes."

    row = df_cli.iloc[0].to_dict()
    if pd.isna(row.get("desconto_contratado")):
        return pd.DataFrame(), f"Cadastro incompleto (info_clientes): desconto_contratado vazio para UC {uc}"
    if pd.isna(row.get("custo_disp")):
        return pd.DataFrame(), f"Cadastro incompleto (info_clientes): custo_disp vazio para UC {uc}"
    stt = str(row.get("status") or "").strip().lower()
    if not stt:
        return pd.DataFrame(), f"Cadastro incompleto (info_clientes): status vazio para UC {uc}"
    if stt != "ativo":
        return pd.DataFrame(), f"Cliente com status '{row.get('status')}'. Ajuste para 'Ativo' para calcular."

    res = calculate_boletos(
        df_itens=df_itens_nf,
        df_medidores=df_med_nf if df_med_nf is not None else pd.DataFrame(),
        df_clientes=df_cli,
        only_registered_clients=True,
        only_status_ativo=True,
    )

    if res.df_boletos is None or res.df_boletos.empty:
        if getattr(res, "missing_clientes", None):
            return pd.DataFrame(), f"Cálculo filtrou UC(s): {res.missing_clientes} | motivo: {res.missing_reason}"
        return pd.DataFrame(), "Cálculo retornou vazio (sem missing_clientes). Investigar parse/medidores/itens."

    return res.df_boletos, None


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
tab_upload, tab_revisao, tab_historico = st.tabs(["📤 Upload", "🧮 Revisão + Lote", "📜 Histórico"])


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
            arquivo_hash_by_name = _build_hash_by_filename(uploaded_files)

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
                c1.metric("Arquivos no lote", resultado["total"])
                c2.metric("Parseadas com sucesso", resultado["sucesso"])
                c3.metric("Erros de parse", resultado["erros"])

                erros_lista = [r for r in resultado["resultados"] if not r.sucesso]
                if erros_lista:
                    with st.expander(f"⚠️ {len(erros_lista)} arquivo(s) com erro"):
                        for r in erros_lista:
                            st.error(f"**{r.arquivo}**: {', '.join(r.erros)}")

                if not resultado["df_itens"].empty:
                    st.session_state.faturas_processadas = resultado

                    nfs_lote = (
                        resultado["df_itens"]["numero"]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                        if "numero" in resultado["df_itens"].columns
                        else []
                    )

                    existing_workflow_df = _load_existing_workflow_by_nfs(nfs_lote)
                    df_workflow_lote = build_workflow_from_parse_results(
                        resultado["resultados"],
                        existing_workflow_df=existing_workflow_df,
                        arquivo_hash_by_name=arquivo_hash_by_name,
                        pdf_uri_by_name={},
                    )

                    st.session_state["workflow_lote"] = df_workflow_lote.copy()

                    resumo = _summarize_workflow_status(df_workflow_lote)

                    st.markdown("### 📌 Classificação do lote")
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("NFs identificadas", resumo["total_nf"])
                    k2.metric("Inéditas", resumo["ineditas"])
                    k3.metric("Já existentes", resumo["repetidas"])
                    k4.metric("Parse com erro", resumo["erro_parse"])

                    st.markdown("### 🧾 Cabeçalhos do lote (workflow)")
                    if df_workflow_lote is not None and not df_workflow_lote.empty:
                        cols_show = [
                            "id",
                            "unidade_consumidora",
                            "cliente_numero",
                            "nome",
                            "referencia",
                            "vencimento",
                            "total_pagar",
                            "is_inedita",
                            "status_parse",
                            "arquivo_nome_original",
                        ]
                        cols_show = [c for c in cols_show if c in df_workflow_lote.columns]
                        st.dataframe(
                            df_workflow_lote[cols_show].sort_values(["referencia", "id"], ascending=[False, False]),
                            hide_index=True,
                            width="stretch",
                        )

                        repetidas = df_workflow_lote[df_workflow_lote["is_inedita"] == False].copy()
                        if not repetidas.empty:
                            with st.expander("Ver NFs já existentes"):
                                st.dataframe(
                                    repetidas[[c for c in cols_show if c in repetidas.columns]],
                                    hide_index=True,
                                    width="stretch",
                                )

                    st.success(f"✅ {len(resultado['df_itens'])} itens parseados!")

                    if salvar_auto:
                        try:
                            with st.spinner("Salvando no BigQuery..."):
                                n_wf = upsert_dataframe(df_workflow_lote, TABLE_FATURAS_WORKFLOW, "id")

                                nfs_ineditas = (
                                    df_workflow_lote[df_workflow_lote["is_inedita"] == True]["id"]
                                    .dropna()
                                    .astype(str)
                                    .tolist()
                                )

                                if nfs_ineditas:
                                    df_itens_save = resultado["df_itens"][
                                        resultado["df_itens"]["numero"].astype(str).isin(nfs_ineditas)
                                    ].copy()

                                    if resultado["df_medidores"] is not None and not resultado["df_medidores"].empty:
                                        df_med_save = resultado["df_medidores"][
                                            resultado["df_medidores"]["nota_fiscal_numero"].astype(str).isin(nfs_ineditas)
                                        ].copy()
                                    else:
                                        df_med_save = pd.DataFrame()

                                    n_itens = upsert_dataframe(df_itens_save, TABLE_FATURA_ITENS, "id") if not df_itens_save.empty else 0
                                    n_med = upsert_dataframe(df_med_save, TABLE_MEDIDORES, "id") if not df_med_save.empty else 0
                                else:
                                    n_itens = 0
                                    n_med = 0

                        except Exception as e:
                            st.error(f"Erro ao salvar: {e}")
                        else:
                            if resumo["repetidas"] > 0:
                                st.warning(
                                    f"⚠️ Workflow salvo ({n_wf} linhas). "
                                    f"Itens/medidores salvos apenas para NFs inéditas: {resumo['ineditas']} inéditas, {resumo['repetidas']} já existentes."
                                )

                            st.success(
                                f"✅ BigQuery atualizado: workflow={n_wf}, itens={n_itens}, medidores={n_med}"
                            )


# =========================
# TAB REVISÃO + LOTE
# =========================
with tab_revisao:
    st.markdown("### Revisar, cadastrar e calcular (lote do upload)")

    if st.session_state.faturas_processadas is None:
        st.info("📤 Faça o upload de faturas na aba anterior.")
        st.stop()

    resultado = st.session_state.faturas_processadas
    df_itens_all = resultado["df_itens"].copy()
    df_med_all = resultado["df_medidores"].copy() if resultado["df_medidores"] is not None else pd.DataFrame()

    nfs = sorted(df_itens_all["numero"].dropna().astype(str).unique().tolist())
    if not nfs:
        st.warning("Não encontramos NFs no lote parseado.")
        st.stop()

    ucs = (
        sorted(df_itens_all["unidade_consumidora"].dropna().astype(str).unique().tolist())
        if "unidade_consumidora" in df_itens_all.columns
        else []
    )
    df_cli_all = _load_clientes_for_ucs(ucs)

    diag_rows = []
    for nf in nfs:
        df_it_nf_tmp = df_itens_all[df_itens_all["numero"].astype(str) == str(nf)]
        uc_tmp = (
            str(df_it_nf_tmp["unidade_consumidora"].dropna().iloc[0])
            if "unidade_consumidora" in df_it_nf_tmp.columns and df_it_nf_tmp["unidade_consumidora"].notna().any()
            else ""
        )
        hit = (
            df_cli_all[df_cli_all["unidade_consumidora"].astype(str) == uc_tmp]
            if (df_cli_all is not None and not df_cli_all.empty and uc_tmp)
            else pd.DataFrame()
        )
        cli_r = hit.iloc[0] if not hit.empty else None
        motivo = _cadastro_motivo(cli_r)
        diag_rows.append(
            {
                "numero": str(nf),
                "unidade_consumidora": uc_tmp,
                "calculavel": motivo is None,
                "motivo": "" if motivo is None else motivo,
            }
        )

    df_diag = pd.DataFrame(diag_rows)
    bloqueadas = df_diag[~df_diag["calculavel"]]
    ok = df_diag[df_diag["calculavel"]]

    c1, c2, c3 = st.columns(3)
    c1.metric("NFs no lote", len(df_diag))
    c2.metric("Calculáveis", len(ok))
    c3.metric("Bloqueadas (cadastro)", len(bloqueadas))

    if not bloqueadas.empty:
        st.warning("⚠️ Existem NFs bloqueadas por cadastro em info_clientes (manual).")
        with st.expander("Ver NFs bloqueadas"):
            st.dataframe(bloqueadas, hide_index=True, width="stretch")
    else:
        st.success("✅ Todas as NFs do lote têm cadastro mínimo para cálculo.")

    st.markdown("---")

    calc_lote = st.button(
        "🧮 Calcular todas as NFs calculáveis do lote",
        type="primary",
        width="stretch",
        disabled=(len(ok) == 0),
    )
    if calc_lote:
        falhas = []
        for nf in ok["numero"].tolist():
            df_it_nf = df_itens_all[df_itens_all["numero"].astype(str) == str(nf)].copy()
            df_med_nf = (
                df_med_all[df_med_all["nota_fiscal_numero"].astype(str) == str(nf)].copy()
                if not df_med_all.empty and "nota_fiscal_numero" in df_med_all.columns
                else pd.DataFrame()
            )

            df_calc, err = _calc_boleto_from_dfs(df_it_nf, df_med_nf)
            if err:
                falhas.append({"numero": nf, "erro": err})
                _upsert_workflow_status(
                    str(nf),
                    status_calculo="erro_calculo",
                    calculado_em=_now_utc(),
                    observacoes_append=f"erro_calculo_lote: {err}",
                )
            else:
                st.session_state.calc_por_nf[str(nf)] = df_calc.copy()
                _upsert_workflow_status(
                    str(nf),
                    status_calculo="calculada",
                    calculado_em=_now_utc(),
                )

        if falhas:
            st.warning(f"⚠️ {len(falhas)} NF(s) falharam no cálculo do lote.")
            st.dataframe(pd.DataFrame(falhas), hide_index=True, width="stretch")
        st.success("✅ Lote calculado. Resultados disponíveis em 'NF selecionada' abaixo.")

    st.markdown("---")

    nf_sel = st.selectbox("Selecione a NF para conferência", options=nfs)

    df_it_nf = df_itens_all[df_itens_all["numero"].astype(str) == str(nf_sel)].copy()
    df_med_nf = (
        df_med_all[df_med_all["nota_fiscal_numero"].astype(str) == str(nf_sel)].copy()
        if not df_med_all.empty and "nota_fiscal_numero" in df_med_all.columns
        else pd.DataFrame()
    )

    if nf_sel in st.session_state.edited_itens:
        df_it_nf = st.session_state.edited_itens[nf_sel].copy()
    if nf_sel in st.session_state.edited_med:
        df_med_nf = st.session_state.edited_med[nf_sel].copy()

    uc_now = (
        str(df_it_nf["unidade_consumidora"].dropna().iloc[0])
        if "unidade_consumidora" in df_it_nf.columns and df_it_nf["unidade_consumidora"].notna().any()
        else ""
    )

    st.markdown(f"## 👤 Cadastro manual (info_clientes) — UC **{uc_now}**")
    df_cli_one = _load_cliente(uc_now) if uc_now else pd.DataFrame()
    cli_row = df_cli_one.iloc[0] if df_cli_one is not None and not df_cli_one.empty else None

    classe_mod = (
        str(df_it_nf["classe_modalidade"].dropna().iloc[0])
        if "classe_modalidade" in df_it_nf.columns and df_it_nf["classe_modalidade"].notna().any()
        else ""
    )
    n_sug = infer_n_fases(classe_mod) or 3
    custo_sug = int(compute_custo_disp(n_sug) or 100)

    desconto_cur = _num_or_default(cli_row.get("desconto_contratado") if cli_row is not None else None, 0.15)
    subv_cur = _num_or_default(cli_row.get("subvencao") if cli_row is not None else None, 0.0)
    status_cur = str((cli_row.get("status") if cli_row is not None else None) or "Ativo")
    n_cur = _int_or_default(cli_row.get("n_fases") if cli_row is not None else None, int(n_sug))
    custo_cur = _int_or_default(cli_row.get("custo_disp") if cli_row is not None else None, int(custo_sug))

    motivo = _cadastro_motivo(cli_row)
    if motivo is None:
        st.success("✅ Cadastro mínimo OK para cálculo.")
    else:
        st.error(f"Cadastro insuficiente: **{motivo}**")

    with st.form(f"form_cli_{uc_now}"):
        c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
        with c1:
            desconto = st.number_input(
                "desconto_contratado",
                min_value=0.0,
                max_value=1.0,
                value=float(desconto_cur),
                step=0.01,
            )
        with c2:
            subvencao = st.number_input("subvencao", min_value=0.0, value=float(subv_cur), step=50.0)
        with c3:
            n_fases = st.selectbox("n_fases", options=[1, 2, 3], index=[1, 2, 3].index(int(n_cur)))
        with c4:
            custo_disp = st.number_input("custo_disp (INTEGER kWh)", min_value=0, value=int(custo_cur), step=10)
        with c5:
            status = st.selectbox(
                "status",
                options=["Ativo", "Inativo"],
                index=0 if str(status_cur).lower() == "ativo" else 1,
            )

        salvar_cli = st.form_submit_button("💾 Salvar/Atualizar info_clientes")

    if salvar_cli:
        try:
            payload = {
                "unidade_consumidora": uc_now,
                "desconto_contratado": float(desconto),
                "subvencao": float(subvencao),
                "n_fases": int(n_fases),
                "custo_disp": int(custo_disp),
                "status": str(status),
                "updated_at": _now_utc(),
            }
            upsert_dataframe(pd.DataFrame([payload]), TABLE_CLIENTES, key_column="unidade_consumidora")
            _upsert_workflow_status(
                str(nf_sel),
                status_validacao="pendente",
                observacoes_append="cadastro_info_clientes_atualizado",
            )
            st.success("✅ Cadastro salvo. Agora você pode calcular.")
        except Exception as e:
            st.error(f"Falha ao salvar info_clientes: {e}")

    st.markdown("---")

    st.markdown("## ✏️ Itens tarifários (editável)")
    editable_cols = [c for c in ["codigo", "descricao", "unidade", "quantidade_registrada", "tarifa", "valor"] if c in df_it_nf.columns]
    fixed_cols = [c for c in df_it_nf.columns if c not in editable_cols]

    df_it_edit = st.data_editor(
        df_it_nf,
        disabled=fixed_cols,
        hide_index=True,
        width="stretch",
        key=f"edit_it_{nf_sel}",
    )

    st.markdown("## ✏️ Medidores (editável)")
    df_med_edit = df_med_nf
    if df_med_nf is None or df_med_nf.empty:
        st.info("Sem tabela de medidores para esta NF.")
    else:
        editable_med_cols = [c for c in ["medidor", "tipo", "posto", "total_apurado"] if c in df_med_nf.columns]
        fixed_med_cols = [c for c in df_med_nf.columns if c not in editable_med_cols]
        df_med_edit = st.data_editor(
            df_med_nf,
            disabled=fixed_med_cols,
            hide_index=True,
            width="stretch",
            key=f"edit_med_{nf_sel}",
        )

    st.session_state.edited_itens[nf_sel] = df_it_edit.copy()
    st.session_state.edited_med[nf_sel] = (df_med_edit.copy() if df_med_edit is not None else pd.DataFrame())

    st.markdown("---")

    st.markdown("## ✅ Ações da fatura")

    cA, cB, cC, cD = st.columns([1, 1, 1, 1])
    validar_btn = cA.button("✅ Validar dados desta NF", width="stretch")
    calc_btn = cB.button("🧮 Calcular esta NF", type="primary", width="stretch")
    save_calc_btn = cC.button("💾 Salvar cálculo no BigQuery", width="stretch")
    save_rev_btn = cD.button("💾 Substituir fatura no BigQuery (revisada)", width="stretch")

    if validar_btn:
        try:
            _upsert_workflow_status(
                str(nf_sel),
                status_validacao="validada",
                validado_por="app_upload",
                validado_em=_now_utc(),
                observacoes_append="dados_validados_na_tela_1",
            )
            st.success(f"✅ NF {nf_sel} marcada como VALIDADA no workflow.")
        except Exception as e:
            st.error(f"Falha ao validar NF no workflow: {e}")

    if save_rev_btn:
        try:
            with st.spinner("Substituindo NF no BigQuery (DELETE + INSERT)..."):
                it_reid, med_reid = _recompute_ids_for_invoice(df_it_edit, df_med_edit)
                _delete_nf_from_bigquery(str(nf_sel))
                n_it = upsert_dataframe(it_reid, TABLE_FATURA_ITENS, "id")
                n_med = upsert_dataframe(med_reid, TABLE_MEDIDORES, "id") if med_reid is not None and not med_reid.empty else 0

            _upsert_workflow_status(
                str(nf_sel),
                status_validacao="pendente",
                observacoes_append="fatura_revisada_e_substituida_no_bigquery",
            )

            st.success(f"✅ NF {nf_sel} substituída no BigQuery: {n_it} itens, {n_med} medidores.")
            st.info("ℹ️ Como a fatura foi revisada, o status de validação voltou para PENDENTE.")
        except Exception as e:
            st.error(f"Falha ao substituir no BigQuery: {e}")

    if calc_btn:
        df_calc, err = _calc_boleto_from_dfs(df_it_edit, df_med_edit)
        if err:
            _upsert_workflow_status(
                str(nf_sel),
                status_calculo="erro_calculo",
                calculado_em=_now_utc(),
                observacoes_append=f"erro_calculo_individual: {err}",
            )
            st.error(err)
        else:
            st.session_state.calc_por_nf[str(nf_sel)] = df_calc.copy()
            _upsert_workflow_status(
                str(nf_sel),
                status_calculo="calculada",
                calculado_em=_now_utc(),
            )
            st.success("✅ Cálculo concluído.")

    df_calc_show = st.session_state.calc_por_nf.get(str(nf_sel))
    if df_calc_show is not None and not df_calc_show.empty:
        row = df_calc_show.iloc[0].to_dict()
        st.markdown("## ✅ Resultado do cálculo (resumo)")
        st.write(
            f"**Valor total boleto:** R$ {float(pd.to_numeric(row.get('valor_total_boleto'), errors='coerce') or 0.0):,.2f}"
            .replace(",", "X").replace(".", ",").replace("X", ".")
        )
        st.write(
            f"**Base kWh (med_inj_tusd):** {float(pd.to_numeric(row.get('med_inj_tusd'), errors='coerce') or 0.0):,.0f}"
            .replace(",", "X").replace(".", ",").replace("X", ".")
        )
        st.write(f"**Check:** {row.get('check')}")

        with st.expander("Ver memorial completo (tabela)"):
            st.dataframe(df_calc_show, hide_index=True, width="stretch")

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

    if save_calc_btn:
        df_calc_save = st.session_state.calc_por_nf.get(str(nf_sel))
        if df_calc_save is None or df_calc_save.empty:
            st.warning("Calcule primeiro.")
        else:
            try:
                with st.spinner("Salvando cálculo em boletos_calculados (schema correto)..."):
                    df_bq = calc_to_boletos_schema(df_calc_save, df_it_edit, status="calculado")
                    upsert_dataframe(df_bq, TABLE_BOLETOS, key_column="id")

                _upsert_workflow_status(
                    str(nf_sel),
                    status_calculo="calculada",
                    calculado_em=_now_utc(),
                )

                st.success("✅ Cálculo salvo em boletos_calculados.")
            except Exception as e:
                st.error(f"Falha ao salvar cálculo: {e}")


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
