"""
app.py  –  Entry point for the Patent RAG Streamlit application.

Run with:
    streamlit run app.py
"""
import logging
import sys
from pathlib import Path

# ── Make sub-packages importable when running from repo root ──────────────────
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

# ── Page configuration (must be first Streamlit call) ────────────────────────
st.set_page_config(
    page_title="AI Patent Drafting Tool",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── App modules ───────────────────────────────────────────────────────────────
from ui import session_state, sidebar, tab_scrutiny, tab_consolidation

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    .main-header   { font-size:2.2rem; font-weight:bold; color:#1F77B4; }
    .sub-header    { font-size:1.1rem; color:#666; margin-bottom:1.5rem; }
    .step-card     { background:#f8f9fa; border-radius:.5rem; padding:1rem; margin:.5rem 0; }
</style>
""",
    unsafe_allow_html=True,
)


def main() -> None:
    # 1. Initialise all session state keys
    session_state.init()

    # 2. Render sidebar (returns current project_path or None)
    project_path = sidebar.render()

    # 3. Page header
    st.markdown(
        '<div class="main-header">📝 AI-Powered Patent Drafting Tool</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Automated domain classification · Enablement scrutiny · Document consolidation</div>',
        unsafe_allow_html=True,
    )

    # 4. Main content ─ show landing page until Draft1 is processed
    if not st.session_state.processed:
        _render_landing(project_path)
        return

    # 5. Quick-action bar
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        st.markdown("### Patent Drafting Workflow")
    with c2:
        if st.button("🆕 New Workflow", use_container_width=True):
            session_state.reset_workflow(keep_project=True)
            st.rerun()
    with c3:
        proj_name = (
            st.session_state.project_metadata.get("project_name", "—")
            if st.session_state.project_metadata
            else "—"
        )
        st.info(f"📂 {proj_name}")

    st.divider()

    # 6. Workflow tabs
    tab1, tab2 = st.tabs(["🔍 Step 1: Scrutiny", "📝 Step 2: Consolidation"])
    with tab1:
        tab_scrutiny.render(project_path)
    with tab2:
        tab_consolidation.render(project_path)

    # 7. Footer
    st.divider()
    st.caption(
        "AI-Powered Patent Drafting · Automated Scrutiny & Consolidation · "
        "Powered by CrewAI, Ollama & ChromaDB"
    )


def _render_landing(project_path) -> None:
    """Landing page shown before any document is processed."""
    if project_path:
        meta = st.session_state.project_metadata or {}
        st.info(f"📂 Project selected: **{meta.get('project_name', '—')}**")

        if meta.get("draft1_uploaded") and not st.session_state.draft1_processed:
            if st.button("🚀 Restore Project & Continue Workflow", type="primary"):
                from ui.sidebar import _restore_session
                with st.spinner("Restoring project…"):
                    _restore_session(project_path)
                st.session_state.processed = True
                st.rerun()

    st.divider()
    st.markdown("#### 👈 Upload your **Draft1** patent info sheet in the sidebar to begin")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
**📊 Phase 1 – RAG Setup**
- Load and chunk the patent PDF/DOCX
- Embed and store in ChromaDB
- Memory-efficient (Ollama stopped during this phase)
"""
        )
    with c2:
        st.markdown(
            """
**🤖 Phase 2 – AI Analysis**
- 🔍 Auto-classify patent domain
- ✍️ Manual override option
- Enablement gap analysis (§112)
- Ollama/Mistral LLM
"""
        )
    with c3:
        st.markdown(
            """
**📄 Phase 3 – Consolidation**
- Upload your Q&A answers
- AI merges Draft1 + Q&A → Draft2
- Audit log of all insertions
- Export as MD / DOCX / PDF
"""
        )


if __name__ == "__main__":
    main()
