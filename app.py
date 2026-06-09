"""
Streamlit UI for PatentReviewPipeline
Run: streamlit run app.py
"""

import streamlit as st
from pathlib import Path
import tempfile
import logging

# ── Import your existing RAG helpers ──────────────────────────
from services.vector_store import search, collection_name_for_project
from workflows.rag_workflow import build_two_pass_context

from pipeline import PatentReviewPipeline, Stage

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Patent Review Pipeline",
    page_icon="⚖",
    layout="wide",
)

STATE_DIR = "./patent_states"

# ─────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None

# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.title("⚖ Patent Review Pipeline")
st.caption("Domain-assisted patent drafting — 4-stage review loop")

# ─────────────────────────────────────────────────────────────
# Sidebar — document management
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Document")

    # Existing docs
    state_dir = Path(STATE_DIR)
    state_dir.mkdir(exist_ok=True)
    existing = sorted([p.stem for p in state_dir.glob("*.json")])

    if existing:
        st.subheader("Resume existing")
        selected = st.selectbox("Select document", ["— new —"] + existing)
        if selected != "— new —" and st.button("Load"):
            st.session_state.doc_id   = selected
            st.session_state.pipeline = PatentReviewPipeline(selected, STATE_DIR)
            st.rerun()

    st.divider()
    st.subheader("New document")
    new_id = st.text_input("Document ID", placeholder="e.g. US-2024-001")
    uploaded = st.file_uploader("Upload draft patent (PDF or TXT)", type=["pdf", "txt"])

    if uploaded and new_id and st.button("Start review", type="primary"):
        # Extract text
        if uploaded.type == "application/pdf":
            # Use your existing document_loader if available
            try:
                import pdfplumber
                with pdfplumber.open(uploaded) as pdf:
                    raw_text = "\n\n".join(
                        p.extract_text() or "" for p in pdf.pages
                    )
            except ImportError:
                st.error("Install pdfplumber: pip install pdfplumber")
                st.stop()
        else:
            raw_text = uploaded.read().decode("utf-8", errors="replace")

        pipeline = PatentReviewPipeline(new_id, STATE_DIR)
        pipeline.load_draft(raw_text)
        st.session_state.pipeline = pipeline
        st.session_state.doc_id   = new_id
        st.rerun()

    # Status indicator
    if st.session_state.pipeline:
        st.divider()
        s = st.session_state.pipeline.summary
        st.metric("Stage", s["stage"])
        st.metric("Iterations", s["iterations"])
        if s["go_no_go"]:
            color = "🟢" if s["go_no_go"] == "GO" else "🔴"
            st.metric("Verdict", f"{color} {s['go_no_go']}")
            if s["confidence"] is not None:
                st.progress(s["confidence"], text=f"Confidence {s['confidence']:.0%}")

# ─────────────────────────────────────────────────────────────
# Main panel
# ─────────────────────────────────────────────────────────────
pipeline: PatentReviewPipeline | None = st.session_state.pipeline

if not pipeline:
    st.info("Upload a draft patent document in the sidebar to begin.")
    st.stop()

stage = pipeline.state.stage

# ─────────────────────────────────────────────────────────────
# Step 1
# ─────────────────────────────────────────────────────────────
if stage == Stage.DRAFT_RECEIVED:
    st.header("Step 1 — Model completes draft")
    st.write("The model will read your draft, fill gaps, and mark every change inline.")

    with st.expander("Original draft", expanded=False):
        st.text_area("", pipeline.state.original_draft, height=300, disabled=True)

    domain_type = st.text_input(
        "Domain / product type (optional)",
        placeholder="e.g. flexible_heater_film, optical_coating",
        help="Matched against product_type_checklists.json for targeted RAG",
    )

    if st.button("▶ Run Step 1 — Complete Draft", type="primary"):
        with st.spinner("Model is completing your draft…"):
            # Wire the RAG here:
            rag_context = ""
            field = None
            mechanism = None
            
            if domain_type:
                try:
                    # Try to get existing collection for this domain
                    from services.vector_store import _get_client
                    client = _get_client()
                    collection_name = collection_name_for_project(domain_type, "domain_kb")
                    
                    try:
                        collection = client.get_collection(name=collection_name)
                        if collection.count() > 0:
                            rag_context, field, mechanism = build_two_pass_context(collection, domain_type)
                        else:
                            st.warning(f"Collection '{collection_name}' exists but is empty. Using empty RAG context.")
                    except Exception:
                        st.warning(f"No knowledge base found for '{domain_type}'. Using empty RAG context.")
                except Exception as e:
                    logger.warning(f"Error loading RAG context: {e}")
                    st.warning("Could not load RAG context - proceeding without.")
            
            completed = pipeline.step1_complete_draft(
                rag_context=rag_context,
                domain_type=domain_type,
            )
        st.rerun()

# ─────────────────────────────────────────────────────────────
# Domain markup stage
# ─────────────────────────────────────────────────────────────
elif stage == Stage.DOMAIN_MARKUP:
    st.header("Step 2 — Domain expert review")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Completed draft (with model changes)")
        # Highlight [[ADDED/REVISED]] markers
        display_text = pipeline.state.completed_draft
        st.text_area("", display_text, height=500, disabled=True, key="completed")

        # Change summary
        changes = pipeline.state.completion_diff
        if changes:
            with st.expander(f"📋 {len(changes)} changes made by model"):
                for i, c in enumerate(changes, 1):
                    badge = "🟦 ADDED" if c["type"] == "ADDED" else "🟨 REVISED"
                    st.markdown(f"**{i}. {badge}** — {c['reason']}")

    with col2:
        st.subheader("Your markup")
        st.caption("Strike-outs: prefix line with ~~  |  Questions: prefix with Q:")
        st.caption("Example: ~~This claim is too broad  |  Q: What is the thermal resistance value?")

        markup = st.text_area(
            "Enter your strike-outs and questions here",
            height=400,
            key="markup_input",
            placeholder=(
                "~~Claim 1 line 3: remove 'substantially'\n"
                "Q: What substrate material is used?\n"
                "Q: Is figure 3 referenced in the spec?\n"
                "~~Abstract paragraph 2: remove last sentence"
            )
        )

        if st.button("▶ Submit markup — Run Step 3 Redraft", type="primary", disabled=not markup.strip()):
            pipeline.receive_domain_markup(markup)
            with st.spinner("Model is redrafting based on your markup…"):
                pipeline.step3_redraft()
            st.rerun()

# ─────────────────────────────────────────────────────────────
# Step 4 — Go/No-go
# ─────────────────────────────────────────────────────────────
elif stage == Stage.REDRAFTED:
    st.header("Step 4 — Go / No-Go Assessment")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Redraft")
        st.text_area("", pipeline.state.redraft, height=400, disabled=True)
    with col2:
        st.subheader("Domain markup applied")
        st.text_area("", pipeline.state.domain_markup, height=200, disabled=True)

    if st.button("▶ Run Go/No-Go Assessment", type="primary"):
        with st.spinner("Assessing redraft against domain markup…"):
            result = pipeline.step4_go_no_go()

        verdict    = result.get("verdict", "NO-GO")
        confidence = result.get("confidence", 0.0)
        flagged    = result.get("flagged_items", [])
        rec        = result.get("recommendation", "")

        if verdict == "GO":
            st.success(f"✅ GO — Confidence {confidence:.0%}")
        else:
            st.error(f"🔴 NO-GO — Confidence {confidence:.0%}")

        st.subheader("Recommendation for domain expert")
        st.write(rec)

        if flagged:
            st.subheader(f"⚠ {len(flagged)} unresolved items")
            for item in flagged:
                st.markdown(f"- **{item.get('ref','')}**: {item.get('reason','')}")

        st.info("Returning to domain markup stage for next iteration." if verdict == "NO-GO"
                else "Document closed. Ready for domain sign-off.")
        st.rerun()

# ─────────────────────────────────────────────────────────────
# Closed
# ─────────────────────────────────────────────────────────────
elif stage == Stage.CLOSED:
    st.header("✅ Review Complete")
    st.success(f"Verdict: GO | Confidence: {pipeline.state.confidence:.0%}")
    st.subheader("Final redraft")
    st.text_area("", pipeline.state.redraft, height=500, disabled=True)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇ Download final redraft",
            pipeline.state.redraft,
            file_name=f"{pipeline.state.doc_id}_final.txt",
            mime="text/plain",
        )
    with col2:
        if st.button("Start new iteration"):
            # Reset to DOMAIN_MARKUP for further refinement if needed
            pipeline.state.stage = Stage.DOMAIN_MARKUP
            pipeline.state.domain_markup = ""
            pipeline._save()
            st.rerun()
