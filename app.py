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
    st.caption("Left: what the model changed vs your original. Right: your inline markup editor.")

    # ── CSS for diff rendering and markup editor ──────────────
    st.markdown("""
    <style>
    .diff-container {
        font-family: 'Courier New', monospace;
        font-size: 13px;
        line-height: 1.7;
        background: #0e1117;
        border: 1px solid #2a2d35;
        border-radius: 6px;
        padding: 16px;
        height: 560px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .diff-added   { background: #0d2b0d; color: #4ec94e; border-left: 3px solid #2ea82e; padding-left: 6px; display: block; }
    .diff-removed { background: #2b0d0d; color: #e05c5c; text-decoration: line-through; border-left: 3px solid #a82e2e; padding-left: 6px; display: block; }
    .diff-context { color: #9099a8; display: block; }
    .diff-label   { font-size: 11px; font-weight: bold; border-radius: 3px; padding: 1px 5px; margin-right: 6px; }
    .label-added  { background: #1a4a1a; color: #4ec94e; }
    .label-revised{ background: #2a2a0a; color: #c9c94e; }
    .change-block {
        border: 1px solid #2a2d35;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 10px;
        background: #13151c;
    }
    .change-reason { font-size: 12px; color: #7a7d8a; margin-bottom: 6px; font-style: italic; }
    .markup-legend {
        background: #13151c;
        border: 1px solid #2a2d35;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 12px;
        font-size: 12px;
        color: #9099a8;
    }
    .markup-legend code { background: #1e2130; padding: 1px 5px; border-radius: 3px; color: #c9c94e; }
    </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # ── LEFT: Diff view ───────────────────────────────────────
    with col1:
        st.subheader("Model changes — diff view")

        changes = pipeline.state.completion_diff
        original = pipeline.state.original_draft
        completed = pipeline.state.completed_draft

        if changes:
            # Tab between "visual diff" and "change list"
            tab_diff, tab_list = st.tabs(["📄 Inline diff", "📋 Change list"])

            with tab_diff:
                # Build HTML diff: show original lines, highlight [[ADDED/REVISED]] blocks
                import re as _re
                import html as _html

                def build_diff_html(original_text: str, completed_text: str) -> str:
                    """
                    Render completed_text with [[ADDED/REVISED:...]]...[[/ADDED|/REVISED]]
                    markers as a colour-coded diff against original.
                    Lines present in original but absent in completed shown as removals.
                    """
                    # Split completed into segments: marked changes vs plain text
                    pattern = r"(\[\[(ADDED|REVISED): ([^\]]+)\]\])(.*?)(\[\[/(?:ADDED|REVISED)\]\])"
                    parts = _re.split(pattern, completed_text, flags=_re.DOTALL)

                    html_parts = []
                    i = 0
                    while i < len(parts):
                        chunk = parts[i]
                        if i + 4 < len(parts) and _re.match(r"\[\[(ADDED|REVISED)", chunk):
                            # This is a change marker group from split
                            change_type = parts[i+1]   # ADDED or REVISED
                            reason      = parts[i+2]
                            content     = parts[i+3]
                            label_cls   = "label-added" if change_type == "ADDED" else "label-revised"
                            label_txt   = "＋ ADDED" if change_type == "ADDED" else "✎ REVISED"
                            html_parts.append(
                                f'<span class="diff-added">'
                                f'<span class="diff-label {label_cls}">{label_txt}</span>'
                                f'<em style="font-size:11px;color:#666"> {_html.escape(reason)}</em>\n'
                                f'{_html.escape(content.strip())}'
                                f'</span>'
                            )
                            i += 5
                        else:
                            # Plain context text
                            escaped = _html.escape(chunk)
                            html_parts.append(f'<span class="diff-context">{escaped}</span>')
                            i += 1

                    return "".join(html_parts)

                diff_html = build_diff_html(original, completed)
                st.markdown(
                    f'<div class="diff-container">{diff_html}</div>',
                    unsafe_allow_html=True
                )

            with tab_list:
                for i, c in enumerate(changes, 1):
                    badge_color = "#4ec94e" if c["type"] == "ADDED" else "#c9c94e"
                    badge_label = "＋ ADDED" if c["type"] == "ADDED" else "✎ REVISED"
                    st.markdown(
                        f'<div class="change-block">'
                        f'<div><span style="color:{badge_color};font-weight:bold">{badge_label}</span> '
                        f'<span class="change-reason">— {c["reason"]}</span></div>'
                        f'<div style="font-size:13px;color:#c8d0dc;margin-top:4px">{c["content"][:200]}{"…" if len(c["content"])>200 else ""}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                # Domain verification checkboxes
                st.divider()
                st.markdown("**✓ Verify model changes**")
                st.caption("Check each change you accept. Unchecked items auto-flag for redraft.")

                if "verified_changes" not in st.session_state:
                    st.session_state.verified_changes = {}

                for i, c in enumerate(changes):
                    key = f"verify_{i}"
                    checked = st.checkbox(
                        f"{c['type']} — {c['reason'][:60]}",
                        value=st.session_state.verified_changes.get(key, True),
                        key=key
                    )
                    st.session_state.verified_changes[key] = checked

        else:
            # No [[ADDED/REVISED]] markers found — show raw completed draft
            st.caption("No structured change markers found. Showing completed draft as-is.")
            st.text_area("Completed draft", completed, height=500, disabled=True, key="completed_raw")

    # ── RIGHT: Inline markup editor ───────────────────────────
    with col2:
        st.subheader("Your markup")

        # Legend
        st.markdown("""
        <div class="markup-legend">
        <strong>Markup syntax</strong><br>
        <code>~~text to remove~~</code> &nbsp;→ strikethrough / delete<br>
        <code>++inserted text++</code> &nbsp;&nbsp;→ insert / add<br>
        <code>??your question??</code> &nbsp;→ question for model<br>
        <code>##section note##</code> &nbsp;&nbsp;→ section-level comment<br><br>
        <em>You can also edit freely — the model reads your full markup text.</em>
        </div>
        """, unsafe_allow_html=True)

        # Pre-populate editor with completed draft so domain edits inline
        editor_key = "markup_editor"
        if editor_key not in st.session_state:
            # First load: seed with completed draft
            st.session_state[editor_key] = pipeline.state.completed_draft or ""

        # Reset button
        reset_col, _ = st.columns([1, 3])
        with reset_col:
            if st.button("↺ Reset to completed draft", help="Discard edits and reload model output"):
                st.session_state[editor_key] = pipeline.state.completed_draft or ""
                st.rerun()

        markup_text = st.text_area(
            "Edit inline — strike, insert, comment, question",
            value=st.session_state[editor_key],
            height=460,
            key=editor_key,
            help=(
                "Edit directly in this box.\n"
                "~~wrap deletions like this~~\n"
                "++wrap insertions like this++\n"
                "??wrap questions like this??\n"
                "##wrap section notes like this##"
            ),
        )

        # Live markup summary
        import re as _re2
        deletions  = _re2.findall(r"~~(.+?)~~", markup_text, _re2.DOTALL)
        insertions = _re2.findall(r"\+\+(.+?)\+\+", markup_text, _re2.DOTALL)
        questions  = _re2.findall(r"\?\?(.+?)\?\?", markup_text, _re2.DOTALL)
        comments   = _re2.findall(r"##(.+?)##", markup_text, _re2.DOTALL)

        if any([deletions, insertions, questions, comments]):
            with st.expander(
                f"📊 Markup summary — "
                f"{len(deletions)}✂ {len(insertions)}＋ {len(questions)}? {len(comments)}#",
                expanded=False
            ):
                if deletions:
                    st.markdown("**✂ Deletions**")
                    for d in deletions[:5]:
                        st.markdown(f"- ~~{d[:80]}~~")
                    if len(deletions) > 5:
                        st.caption(f"…and {len(deletions)-5} more")
                if insertions:
                    st.markdown("**＋ Insertions**")
                    for ins in insertions[:5]:
                        st.markdown(f"- `{ins[:80]}`")
                if questions:
                    st.markdown("**? Questions for model**")
                    for q in questions[:5]:
                        st.markdown(f"- {q[:100]}")
                if comments:
                    st.markdown("**# Section notes**")
                    for c in comments[:5]:
                        st.markdown(f"- {c[:100]}")

        # Build rejected-changes note from unchecked verifications
        rejected_notes = ""
        if "verified_changes" in st.session_state and pipeline.state.completion_diff:
            rejected = [
                pipeline.state.completion_diff[i]
                for i, (k, v) in enumerate(st.session_state.verified_changes.items())
                if not v and i < len(pipeline.state.completion_diff)
            ]
            if rejected:
                rejected_notes = "\n\nREJECTED MODEL CHANGES (revert these):\n" + "\n".join(
                    f"- REVERT {r['type']}: {r['reason']}" for r in rejected
                )

        st.divider()
        can_submit = markup_text.strip() and markup_text != pipeline.state.completed_draft
        if st.button(
            "▶ Submit markup — Run Step 3 Redraft",
            type="primary",
            disabled=not can_submit,
            help="Submit your inline edits to the model for redrafting"
        ):
            full_markup = markup_text + rejected_notes
            pipeline.receive_domain_markup(full_markup)
            with st.spinner("Model is redrafting from your markup…"):
                pipeline.step3_redraft()
            # Clear editor state for next iteration
            if editor_key in st.session_state:
                del st.session_state[editor_key]
            if "verified_changes" in st.session_state:
                del st.session_state["verified_changes"]
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
