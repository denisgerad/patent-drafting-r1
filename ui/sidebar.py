"""
ui/sidebar.py
Render the Streamlit sidebar: system status, project management, file upload.
Returns the project_path if a project is active, else None.
"""
from __future__ import annotations

import gc
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st

import config.settings as cfg
from services import document_loader, vector_store, project_manager
from services.ollama_service import OllamaService

logger = logging.getLogger(__name__)

_PATENT_TYPE_DEFAULTS = [
    "Electronics", "Mechanical", "Chemical", "Software",
    "Medical Devices", "Materials", "Other",
]


def render() -> Optional[str]:
    """Render sidebar. Returns current project_path or None."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        _render_system_status()
        st.divider()
        project_path = _render_project_management()
        st.divider()
        _render_draft1_uploader(project_path)
        st.divider()
        _render_session_controls(project_path)

    return project_path


# ── System Status ─────────────────────────────────────────────────────────────

def _render_system_status() -> None:
    st.subheader("🔧 System Status")
    import config.settings as cfg_mod
    from services.cloud_llm_service import provider_name, is_cloud_provider
    provider = cfg_mod.LLM_PROVIDER

    if is_cloud_provider():
        st.success(f"☁️ LLM Provider: **{provider_name()}**")
        # Validate API key is present
        key_present = {
            "azure":  bool(cfg_mod.AZURE_API_KEY),
            "claude": bool(cfg_mod.CLAUDE_API_KEY),
            "openai": bool(cfg_mod.OPENAI_API_KEY),
        }.get(provider, False)
        if key_present:
            st.caption("✅ API key configured")
        else:
            st.error(f"❌ API key missing — set the required env var in .env")
    else:
        ollama = OllamaService()
        if ollama.is_running():
            st.success("✅ Ollama: Running")
            try:
                model = ollama.resolve_model()
                st.caption(f"LLM: `{model}`")
            except Exception:
                st.caption("LLM: (could not resolve)")
        else:
            st.info("ℹ️ Ollama: Stopped (will auto-start when needed)")


# ── Project Management ────────────────────────────────────────────────────────

def _render_project_management() -> Optional[str]:
    st.subheader("📁 Project Management")
    user_id = st.text_input("User ID", value="default_user")

    projects      = project_manager.list_projects(user_id)
    project_names = ["➕ Create New Project"] + [p["name"] for p in projects]

    selected = st.selectbox("Select or Create Project", project_names)

    if selected == "➕ Create New Project":
        return _handle_new_project(user_id)
    else:
        return _handle_existing_project(selected, projects, user_id)


def _handle_new_project(user_id: str) -> Optional[str]:
    name = st.text_input("New Project Name", placeholder="e.g., MVB Railway Module")
    if name and st.button("Create Project", type="primary"):
        path, meta = project_manager.create_project(name, st.session_state.patent_type, user_id)
        st.session_state.current_project   = str(path)
        st.session_state.project_metadata  = meta
        st.success(f"✅ Project '{name}' created!")
        st.rerun()
    return st.session_state.get("current_project")


def _handle_existing_project(
    selected_name: str,
    projects: list,
    user_id: str,
) -> Optional[str]:
    proj = next((p for p in projects if p["name"] == selected_name), None)
    if not proj:
        return None

    project_path = proj["path"]

    # Load project if switching
    if st.session_state.get("current_project") != project_path:
        meta = project_manager.load_project(project_path)
        if meta:
            st.session_state.current_project  = project_path
            st.session_state.project_metadata = meta
            st.session_state.patent_type      = meta.get("patent_type", "Electronics")
            with st.spinner("Restoring project files…"):
                _restore_session(project_path)
            st.success(f"✅ Loaded: {selected_name}")
            _show_project_status(meta)

    if st.session_state.get("current_project"):
        st.caption(f"📂 Current: {selected_name}")

    return project_path


def _show_project_status(meta: dict) -> None:
    flags = {
        "draft1_uploaded":      "✓ Draft1 uploaded",
        "questions_generated":  "✓ Questions generated",
        "qa_uploaded":          "✓ Q&A uploaded",
        "draft2_generated":     "✓ Draft2 generated",
    }
    for key, label in flags.items():
        if meta.get(key):
            st.caption(label)


def _restore_session(project_path: str) -> None:
    """Re-create Chroma collections and reload state from saved files."""
    meta = project_manager.load_project(project_path)
    if not meta:
        return

    # --- Draft1 ---
    if meta.get("draft1_uploaded"):
        draft1_path = project_manager.get_file_path(project_path, "draft1")
        if draft1_path.exists():
            try:
                chunks = document_loader.load_chunks(draft1_path)
                col_name = vector_store.collection_name_for_project(project_path, "draft1")
                col = vector_store.create_collection(chunks, col_name, reuse_if_exists=True)
                st.session_state.draft1_chunks     = chunks
                st.session_state.draft1_file       = draft1_path.name
                st.session_state.draft1_collection  = col
                st.session_state.draft1_processed   = True
                st.session_state.processed          = True
                # Store docx object if applicable
                if draft1_path.suffix.lower() == ".docx":
                    st.session_state.draft1_docx      = document_loader.load_docx_document(draft1_path)
                    st.session_state.draft1_docx_path = str(draft1_path)
            except Exception as exc:
                st.warning(f"Could not restore Draft1: {exc}")

    # --- Questions ---
    if meta.get("questions_generated"):
        qpath = project_manager.get_file_path(project_path, "questions")
        if qpath.exists():
            try:
                import json
                data = json.loads(qpath.read_text(encoding="utf-8"))
                st.session_state.patent_questions = data.get("questions", "")
            except Exception:
                pass

    # --- Q&A ---
    if meta.get("qa_uploaded"):
        qa_path = project_manager.get_file_path(project_path, "qa")
        if qa_path.exists():
            try:
                chunks = document_loader.load_chunks(qa_path)
                col_name = vector_store.collection_name_for_project(project_path, "qa")
                col = vector_store.create_collection(chunks, col_name, reuse_if_exists=True)
                st.session_state.qa_chunks     = chunks
                st.session_state.qa_file       = qa_path.name
                st.session_state.qa_content    = "\n\n".join(chunks)
                st.session_state.qa_collection = col
                st.session_state.qa_processed  = True
            except Exception as exc:
                st.warning(f"Could not restore Q&A: {exc}")

    # --- Draft2 ---
    if meta.get("draft2_generated"):
        d2path = project_manager.get_file_path(project_path, "draft2")
        if d2path.exists():
            try:
                st.session_state.final_draft  = d2path.read_text(encoding="utf-8")
                st.session_state.draft2_saved = True
            except Exception:
                pass

    # --- Classification ---
    if "classification" in meta:
        clf = meta["classification"]
        from workflows.classification_workflow import ClassificationResult
        st.session_state.classification_result = clf.get("auto_classification")
        st.session_state.classification_done   = True
        st.session_state.user_override         = clf.get("user_override", False)
        st.session_state.patent_type           = clf.get("final_domain", "Electronics")

    # --- Always reset readiness report on project load ---
    # The restored project has its own questions (if any) but no cached
    # readiness report — the user must re-run scrutiny to get a fresh score.
    st.session_state.readiness_report = None


# ── Draft1 Uploader ───────────────────────────────────────────────────────────

def _render_draft1_uploader(project_path: Optional[str]) -> None:
    st.subheader("📄 Draft1 Upload (PDF / DOCX)")

    meta = st.session_state.get("project_metadata", {})
    if meta.get("draft1_uploaded") and project_path:
        st.success("✅ Draft1 already uploaded for this project")
        if st.button("🔄 Re-upload Draft1"):
            st.session_state.draft1_processed = False
            st.rerun()

    uploaded = st.file_uploader(
        "Upload patent info sheet", type=["pdf", "docx"], key="draft1_uploader"
    )

    if uploaded and not st.session_state.draft1_processed:
        if st.button("🚀 Process Draft1", type="primary", use_container_width=True):
            _process_draft1(uploaded, project_path)


def _process_draft1(uploaded_file, project_path: Optional[str]) -> None:
    with st.spinner("Processing Draft1…"):
        ollama = OllamaService()
        if ollama.is_running():
            ollama.stop()

        # Save file
        file_bytes = uploaded_file.getbuffer()
        if project_path:
            dest = project_manager.save_document(
                project_path, file_bytes, uploaded_file.name, "draft1"
            )
            project_manager.update_metadata(project_path, {"draft1_uploaded": True})
        else:
            cfg.TEMP_DIR.mkdir(parents=True, exist_ok=True)
            dest = cfg.TEMP_DIR / uploaded_file.name
            dest.write_bytes(file_bytes)

        # Load chunks
        try:
            chunks = document_loader.load_chunks(dest)
        except Exception as exc:
            st.error(f"❌ Could not load file: {exc}")
            return

        # Store DOCX object if applicable
        if dest.suffix.lower() == ".docx":
            st.session_state.draft1_docx      = document_loader.load_docx_document(dest)
            st.session_state.draft1_docx_path = str(dest)

        # Create vector store
        col_name = (
            vector_store.collection_name_for_project(project_path, "draft1")
            if project_path
            else f"draft1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        try:
            col = vector_store.create_collection(chunks, col_name)
        except Exception as exc:
            st.error(f"❌ Vector store failed: {exc}")
            return

        st.session_state.draft1_chunks     = chunks
        st.session_state.draft1_file       = uploaded_file.name
        st.session_state.draft1_collection  = col
        st.session_state.draft1_processed   = True
        st.session_state.processed          = True
        # Clear any stale readiness report from a previous patent —
        # this new document needs its own fresh evaluation
        st.session_state.readiness_report   = None
        st.session_state.patent_questions   = None
        st.session_state.field_of_invention = ""
        st.session_state.mechanism          = ""

        gc.collect()
        st.success(f"✅ {uploaded_file.name} – {len(chunks)} chunks indexed")
        st.rerun()


# ── Session Controls ──────────────────────────────────────────────────────────

def _render_session_controls(project_path: Optional[str]) -> None:
    st.caption("**Session Controls**")
    c1, c2 = st.columns(2)

    with c1:
        if st.button("🆕 Clear Session", use_container_width=True):
            from ui.session_state import reset_workflow
            reset_workflow(keep_project=True)
            st.success("Session cleared – projects preserved.")
            st.rerun()

    with c2:
        if st.button("🔄 Reset All", use_container_width=True,
                     help="Deletes ALL projects, ChromaDB and temp files"):
            vector_store.delete_all_collections()
            import shutil
            for d in (cfg.TEMP_DIR,):
                shutil.rmtree(str(d), ignore_errors=True)
            from ui.session_state import reset_workflow
            reset_workflow(keep_project=False)
            OllamaService().stop()
            st.rerun()
