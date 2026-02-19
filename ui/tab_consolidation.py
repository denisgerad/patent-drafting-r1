"""
ui/tab_consolidation.py
Renders Step 2:
  - Q&A document upload
  - Generate Final Draft (consolidation)
  - Download + audit display
"""
from __future__ import annotations

import gc
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st

import config.settings as cfg
from services import document_loader, vector_store, project_manager
from services.export_service import to_docx, to_pdf

logger = logging.getLogger(__name__)


def render(project_path: Optional[str]) -> None:
    st.header("📝 Consolidated Patent Draft")
    _render_qa_uploader(project_path)
    st.divider()
    _render_consolidation_panel(project_path)


# ── Q&A Upload ────────────────────────────────────────────────────────────────

def _render_qa_uploader(project_path: Optional[str]) -> None:
    st.subheader("Upload Q&A Document (your answers to the scrutiny questions)")

    if not st.session_state.get("patent_questions"):
        st.warning("⚠️ Complete Step 1 (generate questions) before uploading answers.")

    meta = st.session_state.get("project_metadata", {})
    if meta.get("qa_uploaded") and st.session_state.qa_processed:
        st.success(f"✅ Q&A loaded: {st.session_state.qa_file}")
        if st.button("🔄 Re-upload Q&A"):
            st.session_state.qa_processed = False
            st.rerun()
        return

    uploaded = st.file_uploader(
        "Upload Q&A answers", type=["pdf", "txt", "docx"], key="qa_uploader_tab2"
    )
    if uploaded and not st.session_state.qa_processed:
        if st.button("🚀 Process Q&A", type="primary", use_container_width=True):
            _process_qa(uploaded, project_path)


def _process_qa(uploaded_file, project_path: Optional[str]) -> None:
    with st.spinner("Processing Q&A document…"):
        file_bytes = uploaded_file.getbuffer()

        if project_path:
            dest = project_manager.save_document(
                project_path, file_bytes, uploaded_file.name, "qa"
            )
            project_manager.update_metadata(project_path, {"qa_uploaded": True})
        else:
            cfg.TEMP_DIR.mkdir(parents=True, exist_ok=True)
            dest = cfg.TEMP_DIR / uploaded_file.name
            dest.write_bytes(file_bytes)

        try:
            chunks = document_loader.load_chunks(dest)
        except Exception as exc:
            st.error(f"❌ Could not load Q&A file: {exc}")
            return

        col_name = (
            vector_store.collection_name_for_project(project_path, "qa")
            if project_path
            else f"qa_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        try:
            col = vector_store.create_collection(chunks, col_name)
        except Exception as exc:
            st.error(f"❌ Q&A vector store failed: {exc}")
            return

        st.session_state.qa_chunks     = chunks
        st.session_state.qa_file       = uploaded_file.name
        st.session_state.qa_content    = "\n\n".join(chunks)
        st.session_state.qa_collection = col
        st.session_state.qa_processed  = True
        gc.collect()

        st.success(f"✅ {uploaded_file.name} – {len(chunks)} chunks indexed")
        st.rerun()


# ── Consolidation ─────────────────────────────────────────────────────────────

def _render_consolidation_panel(project_path: Optional[str]) -> None:
    if not st.session_state.qa_processed:
        st.info("Upload and process your Q&A document above to enable draft generation.")
        return

    if st.session_state.get("final_draft"):
        _render_draft_output(project_path)
        return

    if st.button("✍️ Generate Final Draft (Draft 2)", type="primary"):
        _run_consolidation(project_path)


def _run_consolidation(project_path: Optional[str]) -> None:
    draft1_collection = (
        st.session_state.draft1_collection
        if st.session_state.draft1_processed
        else None
    )

    if not draft1_collection:
        st.error("❌ Draft1 collection not found. Please re-process Draft1 in the sidebar.")
        return

    with st.spinner("🧩 Synthesising Draft 2 from original sheet and Q&A answers…"):
        from workflows.consolidation_workflow import run as consolidate_run
        result = consolidate_run(
            draft1_collection=draft1_collection,
            qa_collection=st.session_state.qa_collection,
            qa_content_override=st.session_state.get("qa_content"),
        )

    if not result.success:
        st.error(f"❌ Consolidation failed: {result.error}")
        st.markdown("""
        **Troubleshooting:**
        - Ensure Ollama is running: `ollama serve`
        - Check model availability: `ollama list`
        - Try a faster model: `ollama pull mistral:7b-instruct-q4_K_M`
        """)
        return

    st.session_state.final_draft         = result.draft2
    st.session_state.final_draft_audit   = result.audit_log
    st.session_state.final_draft_valid   = result.is_valid
    st.session_state.final_draft_missing = result.missing_sentences
    st.session_state.draft2_saved        = False

    # Auto-save to project
    if project_path:
        try:
            d2_path = project_manager.get_file_path(project_path, "draft2")
            d2_path.write_text(result.draft2, encoding="utf-8")
            project_manager.update_metadata(project_path, {"draft2_generated": True})
            st.session_state.draft2_saved = True
        except Exception as exc:
            st.warning(f"Could not save Draft2 to project: {exc}")

    st.rerun()


# ── Draft Output ──────────────────────────────────────────────────────────────

def _render_draft_output(project_path: Optional[str]) -> None:
    if st.session_state.draft2_saved:
        st.success("✅ Draft 2 generated and saved to project!")
    else:
        st.success("✅ Draft 2 generated!")

    # Validation warnings
    if not st.session_state.get("final_draft_valid", True):
        with st.expander("⚠️ Validation issues – some Draft1 sentences may be missing"):
            for m in st.session_state.get("final_draft_missing", [])[:50]:
                st.code(m)

    # Draft preview
    with st.expander("📄 View Draft 2", expanded=True):
        st.markdown(st.session_state.final_draft)

    # Audit log
    if st.session_state.get("final_draft_audit"):
        with st.expander("🔎 Audit Log (Q&A insertions)"):
            st.markdown(st.session_state.final_draft_audit)

    # Download options
    st.subheader("⬇️ Download Draft 2")
    fmt = st.radio("Format:", ("Markdown (.md)", "Word (.docx)", "PDF (.pdf)"), horizontal=True)
    proj_name = (
        st.session_state.project_metadata.get("project_name", "project")
        if st.session_state.project_metadata
        else "project"
    )

    if fmt == "Markdown (.md)":
        st.download_button(
            "📥 Download",
            st.session_state.final_draft,
            file_name=f"Patent_Draft2_{proj_name}.md",
            mime="text/markdown",
        )

    elif fmt == "Word (.docx)":
        base_doc = st.session_state.get("draft1_docx")
        preserve = base_doc is not None
        if preserve:
            st.info("📄 Preserving original DOCX structure (formatting, tables, images)…")
        bio = to_docx(st.session_state.final_draft, base_doc=base_doc, preserve_structure=preserve)
        if bio:
            st.download_button(
                "📥 Download",
                bio,
                file_name=f"Patent_Draft2_{proj_name}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        else:
            st.error("DOCX generation failed.")

    elif fmt == "PDF (.pdf)":
        bio = to_pdf(st.session_state.final_draft)
        if bio:
            st.download_button(
                "📥 Download",
                bio,
                file_name=f"Patent_Draft2_{proj_name}.pdf",
                mime="application/pdf",
            )
        else:
            st.error("PDF generation failed.")

    # Workflow completion
    st.divider()
    st.success("🎉 Workflow complete!")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🆕 Start New Workflow", type="primary", use_container_width=True):
            from ui.session_state import reset_workflow
            reset_workflow(keep_project=True)
            st.rerun()
    with c2:
        if st.button("📂 Select Different Project", use_container_width=True):
            from ui.session_state import reset_workflow
            reset_workflow(keep_project=False)
            st.rerun()

    # Re-generate option
    st.divider()
    if st.button("🔄 Re-generate Draft 2"):
        st.session_state.final_draft       = None
        st.session_state.final_draft_audit = None
        st.session_state.draft2_saved      = False
        st.rerun()
