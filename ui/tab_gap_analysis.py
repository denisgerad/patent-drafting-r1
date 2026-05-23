"""
ui/tab_gap_analysis.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Phase E — New Step 1 UI: Reference Patent Gap Analysis.

Four sub-steps rendered as a progressive flow:

  1a. Reference patent selection (auto candidates + manual entry)
  1b. Gap analysis report (structural / claim / technical / enablement)
  1c. Domain review — HIL gate (RELEVANT / ALREADY COVERED / NOT APPLICABLE)
  1d. Targeted question generation and download

Each sub-step is unlocked after the previous one completes.
Domain expert can navigate back to any step using the step selector.
"""
from __future__ import annotations

import logging
from typing import Optional

import streamlit as st

logger = logging.getLogger(__name__)


# ── Entry point ───────────────────────────────────────────────────────────────

def render(project_path: Optional[str]) -> None:
    """Render the complete Step 1 gap analysis tab."""
    st.header("🔍 Step 1: Reference Patent Gap Analysis")

    if not st.session_state.get("draft1_processed"):
        st.warning("⚠️ Upload and process Draft1 in the sidebar first.")
        return

    # Step navigator
    step = st.session_state.get("gap_analysis_step", "1a")
    _render_step_navigator(step)
    st.divider()

    if step == "1a":
        _render_1a_reference_selection(project_path)
    elif step == "1b":
        _render_1b_gap_report(project_path)
    elif step == "1c":
        _render_1c_hil_review(project_path)
    elif step == "1d":
        _render_1d_questions(project_path)


# ── Step navigator ────────────────────────────────────────────────────────────

def _render_step_navigator(current: str) -> None:
    """Horizontal step indicator with clickable navigation."""
    steps = {
        "1a": "📚 Select References",
        "1b": "🔬 Gap Analysis",
        "1c": "✅ Domain Review",
        "1d": "❓ Questions",
    }
    cols = st.columns(len(steps))

    for col, (key, label) in zip(cols, steps.items()):
        with col:
            is_current   = key == current
            is_available = _step_available(key)
            style = "primary" if is_current else "secondary"
            if is_available:
                if st.button(
                    label,
                    key=f"nav_{key}",
                    type=style,
                    use_container_width=True,
                    disabled=is_current,
                ):
                    st.session_state.gap_analysis_step = key
                    st.rerun()
            else:
                st.button(
                    label,
                    key=f"nav_{key}_disabled",
                    use_container_width=True,
                    disabled=True,
                )


def _step_available(step: str) -> bool:
    """Whether a step can be navigated to."""
    if step == "1a":
        return True
    if step == "1b":
        return bool(st.session_state.get("selected_references"))
    if step == "1c":
        return st.session_state.get("gap_report") is not None
    if step == "1d":
        return st.session_state.get("gap_report_reviewed", False)
    return False


# ══════════════════════════════════════════════════════════════════════════════
# SUB-STEP 1a — Reference patent selection
# ══════════════════════════════════════════════════════════════════════════════

def _render_1a_reference_selection(project_path: Optional[str]) -> None:
    st.subheader("Step 1a: Select Reference Patents")
    st.caption(
        "Select 1-3 granted patents similar to your invention. "
        "The gap analysis compares your draft against these references."
    )

    # ── Field of invention display + edit ─────────────────────────────────────
    # Always try to get a meaningful field (>30 chars with real content).
    # The session state may contain just the heading "Field of the Invention"
    # if the scrutiny workflow ran before this tab. Re-extract if too short.
    stored_field = st.session_state.get("field_of_invention", "")
    if len(stored_field.strip()) < 30:
        extracted = _extract_field_from_document()
        if extracted and len(extracted) > len(stored_field):
            stored_field = extracted
            st.session_state.field_of_invention = extracted

    st.markdown("#### 📄 Field of Invention")
    field = st.text_area(
        "Auto-extracted from your Draft1 document. Edit if incorrect before searching.",
        value=stored_field,
        height=80,
        key="gap_field_edit",
        placeholder=(
            "e.g. The present invention relates to backlight systems for display "
            "devices, particularly for avionic and defence displays requiring "
            "operation in both day mode and NVG compatible mode."
        ),
    )
    if field != stored_field:
        st.session_state.field_of_invention = field

    if len(field.strip()) < 30:
        st.warning(
            "⚠️ The field of invention is too short to search. "
            "Paste the field of invention sentence from your draft document above."
        )
        st.divider()

    st.divider()

    # ── Two panels side by side ───────────────────────────────────────────────
    col_auto, col_manual = st.columns([3, 2])

    with col_auto:
        _render_auto_search_panel(field)

    with col_manual:
        _render_manual_entry_panel()

    # ── Selected references summary ───────────────────────────────────────────
    selected = st.session_state.get("selected_references", [])
    if selected:
        st.divider()
        st.markdown(f"**✅ Selected references ({len(selected)}):**")
        for ref in selected:
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"• **{ref['display_id']}** — {ref['title'][:80]}")
            with col2:
                if st.button("Remove", key=f"remove_{ref['display_id']}"):
                    st.session_state.selected_references = [
                        r for r in selected if r["display_id"] != ref["display_id"]
                    ]
                    st.rerun()

        st.divider()
        if st.button(
            "▶ Run Gap Analysis →",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.gap_analysis_step = "1b"
            st.session_state.gap_report = None   # clear stale report
            st.rerun()
    else:
        st.info("👆 Search for patents above or enter a patent number manually.")


def _render_auto_search_panel(field: str) -> None:
    """Auto-search panel — finds similar granted patents from EPO OPS."""
    st.markdown("#### 🔍 Auto Search")

    field_ok = len(field.strip()) >= 30
    candidates = st.session_state.get("reference_candidates", [])

    col_search, col_explain = st.columns([2, 1])
    with col_search:
        if st.button(
            "🔍 Find Similar Patents",
            use_container_width=True,
            disabled=not field_ok,
            help="Edit the field of invention above if this button is disabled",
        ):
            _run_auto_search(field, with_explanations=False)

    with col_explain:
        if candidates and st.button(
            "💬 Add Explanations",
            use_container_width=True,
            help="Ask local model to explain why each candidate is similar (~30s)",
        ):
            _run_auto_search(field, with_explanations=True)

    # Show any stored search messages (survive the st.rerun())
    if err := st.session_state.pop("_search_error", None):
        st.error(f"Search error: {err}")
    if info := st.session_state.pop("_search_info", None):
        st.info(info)
    if st.session_state.pop("_search_no_results", False):
        st.warning(
            "No patents found for this field of invention on EPO OPS. "
            "Try editing the field text above to use more specific technical terms, "
            "or enter a patent number manually on the right."
        )

    if not candidates:
        st.info("Click 'Find Similar Patents' to search EPO OPS.")
        return

    st.markdown(f"**{len(candidates)} candidates found:**")
    st.caption(
        "💡 **Tip:** For best gap analysis results, select patents with a "
        "**B1 or B2 suffix** (granted patents) rather than **A1** (published "
        "applications). Granted patents have full claims and description available "
        "via EPO OPS. Example: EP3456789**B1** rather than US2014036533**A1**."
    )

    selected_ids = {r["display_id"] for r in st.session_state.get("selected_references", [])}

    for i, cand in enumerate(candidates):
        with st.container():
            col_check, col_info = st.columns([1, 8])
            with col_check:
                is_checked = cand["display_id"] in selected_ids
                checked = st.checkbox(
                    "",
                    value=is_checked,
                    key=f"cand_{cand['display_id']}",
                )
                if checked and not is_checked:
                    _add_to_selected(cand)
                elif not checked and is_checked:
                    _remove_from_selected(cand["display_id"])

            with col_info:
                sim_pct = int(cand.get("similarity_score", 0) * 100)
                st.markdown(
                    f"**{cand['display_id']}** — {cand['title'][:70]}  "
                    f"`{sim_pct}% match`"
                )
                if cand.get("similarity_explanation"):
                    st.caption(f"💬 {cand['similarity_explanation']}")
                elif cand.get("abstract"):
                    st.caption(cand["abstract"][:180] + "…")

                with st.expander("Full abstract"):
                    st.write(cand.get("abstract", "No abstract available."))


def _render_manual_entry_panel() -> None:
    """Manual reference patent panel — PDF upload tab and API number tab."""
    st.markdown("#### 📄 Add Reference Patent")

    tab_pdf, tab_api = st.tabs(["📄 Upload PDF (recommended)", "🌐 Number only (API)"])

    with tab_pdf:
        st.caption(
            "Steps: 1) Go to Espacenet (worldwide.espacenet.com) and find a relevant granted patent. "
            "2) Click Download then Full document to get the PDF. "
            "3) Upload the PDF below and enter the patent number. "
            "4) Click Add Patent."
        )

        uploaded_pdf = st.file_uploader(
            "Patent PDF",
            type=["pdf"],
            key="reference_pdf_upload",
            label_visibility="collapsed",
        )

        if uploaded_pdf:
            st.success(f"Ready: {uploaded_pdf.name}")

        patent_number_pdf = st.text_input(
            "Patent number",
            placeholder="e.g. EP3456789B1  or  US6789921B1",
            key="manual_patent_number_pdf",
        )

        btn_disabled = not (uploaded_pdf and patent_number_pdf.strip())
        if st.button(
            "Add Patent from PDF",
            use_container_width=True,
            disabled=btn_disabled,
            type="primary" if not btn_disabled else "secondary",
            key="btn_add_pdf",
        ):
            _load_and_add_pdf(uploaded_pdf, patent_number_pdf.strip())

    with tab_api:
        st.caption(
            "Enter a patent number to fetch via EPO OPS API. "
            "Full claims available for most EP granted patents (B1/B2). "
            "For US patents, use the PDF tab for complete coverage."
        )

        patent_number_api = st.text_input(
            "Patent number",
            placeholder="e.g. EP3456789B1  or  US6789921B1",
            key="manual_patent_number_api",
        )

        if st.button(
            "Add Patent via API",
            use_container_width=True,
            disabled=not patent_number_api.strip(),
            key="btn_add_api",
        ):
            _fetch_and_add_manual(patent_number_api.strip())

    # Show added references
    selected = st.session_state.get("selected_references", [])
    manual   = [r for r in selected if r.get("source") in ("manual", "pdf")]
    if manual:
        st.markdown("**Added:**")
        for ref in manual:
            src_icon = "PDF" if ref.get("source") == "pdf" else "API"
            quality  = "Full text" if ref.get("has_full_text") else "Abstract only"
            st.markdown(
                f"[{src_icon}] **{ref['display_id']}** - {ref['title'][:50]} ({quality})"
            )


# ── PDF upload action ─────────────────────────────────────────────────────────

def _load_and_add_pdf(uploaded_file, patent_number: str) -> None:
    """
    Load a reference patent from an uploaded PDF file.
    Extracts full text locally — no EPO OPS full-text call needed.
    """
    import io, re
    with st.spinner(f"Loading {patent_number} from PDF..."):
        try:
            from services.document_loader import load_chunks
            from services.patent_retriever import (
                StructuredPatent, _to_epodoc, _epodoc_to_display,
                _parse_claims, _extract_parameters, _extract_figures,
            )
            from services.patent_chunker import _split_sections

            # Read PDF bytes and extract chunks
            pdf_bytes = uploaded_file.read()
            chunks = load_chunks(io.BytesIO(pdf_bytes), filename=uploaded_file.name)
            if not chunks:
                st.error("Could not extract text from PDF. Is it a text-based PDF?")
                return

            full_text = "\n\n".join(chunks)
            sections  = _split_sections(full_text)
            claims_raw  = sections.get("claims_raw", "")
            desc_raw    = sections.get("description_raw", full_text)

            # Fallback claims extraction if _split_sections missed it
            if not claims_raw:
                pat = (r"(?:^|\n)\s*(?:CLAIMS?|What is claimed)"
                       r"[:\s]*\n(.*?)(?=\n\s*(?:ABSTRACT|DESCRIPTION|$))")
                m = re.search(pat, full_text, re.IGNORECASE | re.DOTALL)
                if m:
                    claims_raw = m.group(1).strip()

            # Extract abstract from first section or description start
            abstract = ""
            abstract_match = re.search(
                r"(?:^|\n)\s*ABSTRACT[:\s]*\n(.*?)(?=\n\s*(?:BACKGROUND|CLAIMS?|DESCRIPTION|$))",
                full_text, re.IGNORECASE | re.DOTALL,
            )
            if abstract_match:
                abstract = abstract_match.group(1).strip()[:500]

            # Extract title from first non-empty lines
            first_lines = full_text[:500].split("\n")
            title = next(
                (l.strip() for l in first_lines
                 if len(l.strip()) > 10 and not l.strip().isdigit()),
                patent_number,
            )

            # Build minimal StructuredPatent from local text
            epodoc_id  = _to_epodoc(patent_number) or patent_number
            display_id = _epodoc_to_display(epodoc_id)

            structured = StructuredPatent(
                epodoc_id       = epodoc_id,
                display_id      = display_id,
                title           = title[:200],
                abstract        = abstract,
                grant_date      = "",
                applicant       = "",
                source          = "pdf",
                background      = sections.get("background", ""),
                summary         = sections.get("summary", ""),
                description     = sections.get("description", desc_raw),
                claims          = _parse_claims(claims_raw),
                figures         = _extract_figures(full_text),
                technical_parameters = _extract_parameters(
                    claims_raw + "\n" + desc_raw
                ),
                claims_raw      = claims_raw,
                description_raw = desc_raw,
            )

            # Store in session state
            ref_structured = st.session_state.get("reference_structured", {})
            ref_structured[display_id] = structured.to_dict()
            st.session_state.reference_structured = ref_structured

            # Add to selected references
            cand = {
                "epodoc_id":              epodoc_id,
                "display_id":             display_id,
                "title":                  structured.title,
                "abstract":               abstract,
                "grant_date":             "",
                "applicant":              "",
                "source":                 "pdf",
                "similarity_score":       1.0,
                "similarity_explanation": "Loaded from PDF — full text available",
                "has_full_text":          True,
            }
            _add_to_selected(cand)
            n_claims = len(structured.claims)
            n_params = len(structured.technical_parameters)
            st.success(
                f"Loaded {display_id} from PDF: "
                f"{n_claims} claims, {n_params} parameters extracted."
            )

        except Exception as exc:
            st.error(f"PDF load error: {exc}")
            logger.exception("Reference PDF load error")
    st.rerun()


# ── Auto search actions ───────────────────────────────────────────────────────

def _run_auto_search(field: str, with_explanations: bool) -> None:
    """Call gap_analysis_workflow.search_candidates() and store results."""
    with st.spinner("Searching EPO OPS for similar patents…"):
        try:
            from workflows.gap_analysis_workflow import search_candidates
            result = search_candidates(
                field_of_invention    = field,
                max_results           = 5,
                generate_explanations = with_explanations,
            )
            if result.success:
                candidates = [_candidate_to_dict(c) for c in result.candidates]
                st.session_state.reference_candidates = candidates
                if not candidates:
                    # No results — show helpful message but don't error
                    st.session_state["_search_no_results"] = True
                else:
                    st.session_state.pop("_search_no_results", None)
                if result.error:
                    st.session_state["_search_info"] = result.error
            else:
                st.session_state["_search_error"] = result.error
        except ValueError as exc:
            st.session_state["_search_error"] = (
                f"EPO API not configured. "
                f"Add EPO_CLIENT_ID and EPO_CLIENT_SECRET to your .env file. "
                f"Details: {exc}"
            )
        except Exception as exc:
            st.session_state["_search_error"] = str(exc)
            logger.exception("EPO search error")
    st.rerun()


def _fetch_and_add_manual(patent_number: str) -> None:
    """Fetch a patent by number and add to selected references."""
    with st.spinner(f"Fetching {patent_number}…"):
        try:
            from workflows.gap_analysis_workflow import fetch_manual_candidate
            result = fetch_manual_candidate(patent_number)
            if result.success and result.candidates:
                cand = _candidate_to_dict(result.candidates[0])
                cand["source"] = "manual"
                _add_to_selected(cand)
                st.success(f"Added: {cand['display_id']} — {cand['title'][:60]}")
            else:
                st.error(result.error or f"Could not fetch {patent_number}.")
        except ValueError as exc:
            st.error(f"EPO API not configured: {exc}")
        except Exception as exc:
            st.error(f"Fetch error: {exc}")
    st.rerun()


def _add_to_selected(cand: dict) -> None:
    selected = st.session_state.get("selected_references", [])
    ids = {r["display_id"] for r in selected}
    if cand["display_id"] not in ids:
        if len(selected) >= 3:
            st.warning("Maximum 3 reference patents. Remove one first.")
            return
        st.session_state.selected_references = selected + [cand]


def _remove_from_selected(display_id: str) -> None:
    st.session_state.selected_references = [
        r for r in st.session_state.get("selected_references", [])
        if r["display_id"] != display_id
    ]


def _candidate_to_dict(cand) -> dict:
    """Convert PatentCandidate dataclass to plain dict for session state."""
    return {
        "epodoc_id":              getattr(cand, "epodoc_id", ""),
        "display_id":             getattr(cand, "display_id", ""),
        "title":                  getattr(cand, "title", ""),
        "abstract":               getattr(cand, "abstract", ""),
        "grant_date":             getattr(cand, "grant_date", ""),
        "applicant":              getattr(cand, "applicant", ""),
        "source":                 getattr(cand, "source", "epo"),
        "similarity_score":       getattr(cand, "similarity_score", 0.0),
        "similarity_explanation": getattr(cand, "similarity_explanation", ""),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SUB-STEP 1b — Gap analysis report
# ══════════════════════════════════════════════════════════════════════════════

def _render_1b_gap_report(project_path: Optional[str]) -> None:
    st.subheader("Step 1b: Gap Analysis")

    selected = st.session_state.get("selected_references", [])
    if not selected:
        st.warning("No reference patents selected. Go back to Step 1a.")
        return

    gap_report = st.session_state.get("gap_report")

    # Run gap analysis if not done yet
    if gap_report is None:
        ref_names = ", ".join(r["display_id"] for r in selected)
        st.info(f"Analysing draft against: **{ref_names}**")

        if st.button("▶ Run Gap Analysis", type="primary", use_container_width=True):
            _run_gap_analysis(project_path, selected)
        return

    # ── Display gap report ────────────────────────────────────────────────────
    _render_gap_report_display(gap_report)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Re-run Analysis", use_container_width=True):
            st.session_state.gap_report = None
            st.session_state.gap_report_reviewed = False
            st.rerun()
    with col2:
        if st.button(
            "▶ Proceed to Domain Review →",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.gap_analysis_step = "1c"
            st.rerun()


def _run_gap_analysis(project_path: Optional[str], selected: list[dict]) -> None:
    """Run the gap analysis workflow and store results."""
    collection = st.session_state.get("draft1_collection")
    if not collection:
        st.error("Draft1 collection not found. Please re-upload Draft1.")
        return

    # Convert dicts back to PatentCandidate objects
    from services.patent_retriever import PatentCandidate
    candidates = [
        PatentCandidate(
            epodoc_id  = r["epodoc_id"],
            display_id = r["display_id"],
            title      = r["title"],
            abstract   = r["abstract"],
            grant_date = r.get("grant_date", ""),
            applicant  = r.get("applicant", ""),
            source     = r.get("source", "epo"),
        )
        for r in selected
    ]

    draft_title = (
        st.session_state.get("draft1_file", "Draft document") or "Draft document"
    )

    progress = st.progress(0, text="Fetching reference patent full text…")

    try:
        from workflows.gap_analysis_workflow import run_gap_analysis
        with st.spinner("Running gap analysis…"):
            progress.progress(25, text="Profiling reference patent…")
            result = run_gap_analysis(
                selected_references = candidates,
                draft_collection    = collection,
                draft_title         = draft_title,
            )
        progress.progress(100, text="Done")

        if result.success and result.gap_report:
            st.session_state.gap_report = _gap_report_to_dict(result.gap_report)
            # Store structured refs for question generation
            st.session_state.reference_structured = {
                k: v.to_dict() for k, v in result.structured_refs.items()
            }
            st.session_state.gap_report_reviewed = False
            st.rerun()
        else:
            st.error(f"Gap analysis failed: {result.error}")

    except ValueError as exc:
        st.error(f"EPO API not configured: {exc}")
    except Exception as exc:
        st.error(f"Gap analysis error: {exc}")
        logger.exception("Gap analysis error")


def _render_gap_report_display(gap_report_dict: dict) -> None:
    """Display the gap report with severity colour coding."""
    gaps  = gap_report_dict.get("gaps", [])
    ref   = gap_report_dict.get("reference_id", "")
    title = gap_report_dict.get("reference_title", "")
    words = gap_report_dict.get("draft_word_count", 0)

    st.markdown(f"**Reference:** {ref} — {title}")

    # Show what data quality was available from EPO OPS
    structured = st.session_state.get("reference_structured", {})
    ref_data   = structured.get(ref, {})
    has_claims = bool(ref_data.get("claims_raw", "") or ref_data.get("claims", []))
    has_desc   = bool(ref_data.get("description_raw", "") or ref_data.get("description", ""))

    if has_claims and has_desc:
        st.caption(f"✅ Full text available (claims + description) | Draft: {words} words")
    elif has_claims or has_desc:
        st.caption(f"🟡 Partial text (abstract + {'claims' if has_claims else 'description'}) | Draft: {words} words")
    else:
        st.info(
            "ℹ️ **Abstract-only reference** — EPO OPS returned only the abstract "
            f"for **{ref}**. This is common for US published applications (A1 suffix). "
            "The gap analysis below is based on your **draft structure only** — "
            "it identifies what every patent needs, regardless of the reference. "
            "For richer comparison, add an EP granted patent (B1 suffix) as reference."
        )
    st.caption(f"{len(gaps)} gaps identified")

    if not gaps:
        st.success("✅ No significant gaps found compared to the reference patent.")
        return

    # Summary metrics
    n_critical  = sum(1 for g in gaps if g["severity"] == "CRITICAL")
    n_important = sum(1 for g in gaps if g["severity"] == "IMPORTANT")
    n_optional  = sum(1 for g in gaps if g["severity"] == "OPTIONAL")

    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 Critical",  n_critical)
    c2.metric("🟡 Important", n_important)
    c3.metric("⚪ Optional",  n_optional)

    st.markdown("---")

    # Group gaps by type
    type_order = ["STRUCTURAL", "CLAIM_TOPIC", "TECHNICAL", "ENABLEMENT"]
    type_labels = {
        "STRUCTURAL":  "📋 Structural Gaps (missing sections)",
        "CLAIM_TOPIC": "⚖️ Claim Topic Gaps",
        "TECHNICAL":   "🔧 Technical Parameter Gaps",
        "ENABLEMENT":  "📐 Enablement Gaps (drawings/examples)",
    }

    by_type: dict[str, list] = {}
    for gap in gaps:
        by_type.setdefault(gap["gap_type"], []).append(gap)

    for gtype in type_order:
        if gtype not in by_type:
            continue
        st.markdown(f"### {type_labels.get(gtype, gtype)}")

        for gap in by_type[gtype]:
            _render_gap_card(gap, show_hil=False)


def _render_gap_card(gap: dict, show_hil: bool = False) -> None:
    """Render one gap as an expandable card."""
    sev  = gap.get("severity", "OPTIONAL")
    icon = {"CRITICAL": "🔴", "IMPORTANT": "🟡", "OPTIONAL": "⚪"}.get(sev, "⚪")
    title = gap.get("title", "Gap")
    why   = gap.get("why_it_matters", "")

    with st.expander(f"{icon} **{title}**", expanded=(sev == "CRITICAL")):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**What is missing:**")
            st.write(gap.get("description", ""))
            if why:
                st.markdown("**Why it matters:**")
                st.info(why)

        with col2:
            st.markdown("**Reference patent says:**")
            st.code(gap.get("reference_says", ""), language=None)
            st.markdown("**Draft says:**")
            st.warning(gap.get("draft_says", "Not mentioned."))

        if show_hil:
            _render_hil_controls(gap)


# ══════════════════════════════════════════════════════════════════════════════
# SUB-STEP 1c — HIL domain review
# ══════════════════════════════════════════════════════════════════════════════

def _render_1c_hil_review(project_path: Optional[str]) -> None:
    st.subheader("Step 1c: Domain Review")
    st.info(
        "Review each gap and mark its relevance to this specific invention. "
        "The domain expert's decisions determine which questions are generated."
    )

    gap_report_dict = st.session_state.get("gap_report")
    if not gap_report_dict:
        st.warning("No gap report available. Go back to Step 1b.")
        return

    gaps = gap_report_dict.get("gaps", [])
    if not gaps:
        st.success("✅ No gaps to review.")
        _proceed_to_1d()
        return

    # Invention scope note
    st.markdown("#### Invention scope (optional)")
    scope_note = st.text_area(
        "Describe the core novel contribution in your own words.",
        placeholder=(
            "e.g. 'The novel part is the complementary dot pattern that allows "
            "both edges to illuminate simultaneously without cross-interference. "
            "The backlight sources themselves are not the invention.'"
        ),
        height=80,
        key="invention_scope_note",
    )

    st.divider()

    # Group by type for clearer review
    type_order  = ["STRUCTURAL", "ENABLEMENT", "CLAIM_TOPIC", "TECHNICAL"]
    type_labels = {
        "STRUCTURAL":  "📋 Structural Gaps",
        "CLAIM_TOPIC": "⚖️ Claim Topic Gaps",
        "TECHNICAL":   "🔧 Technical Parameter Gaps",
        "ENABLEMENT":  "📐 Enablement Gaps",
    }

    by_type: dict[str, list] = {}
    for gap in gaps:
        by_type.setdefault(gap["gap_type"], []).append(gap)

    decisions_made = 0
    total_gaps     = len(gaps)

    for gtype in type_order:
        if gtype not in by_type:
            continue
        st.markdown(f"### {type_labels.get(gtype, gtype)}")
        for gap in by_type[gtype]:
            _render_hil_gap_card(gap, gap_report_dict)
            if gap.get("hil_decision", "PENDING") != "PENDING":
                decisions_made += 1

    st.divider()

    # Progress
    pct = decisions_made / total_gaps if total_gaps else 0
    st.progress(pct, text=f"{decisions_made}/{total_gaps} gaps reviewed")

    # Confirm button
    relevant_count = sum(
        1 for g in gaps if g.get("hil_decision") == "RELEVANT"
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            f"✅ Confirm Review — Generate Questions for {relevant_count} Gap(s) →",
            type="primary",
            use_container_width=True,
            disabled=(relevant_count == 0),
        ):
            # Store scope note in report
            gap_report_dict["invention_scope_note"] = scope_note
            st.session_state.gap_report = gap_report_dict
            st.session_state.gap_report_reviewed = True
            st.session_state.gap_analysis_step   = "1d"
            st.session_state.gap_questions        = None  # clear stale
            st.rerun()
    with col2:
        if relevant_count == 0:
            st.warning("Mark at least one gap as RELEVANT to generate questions.")


def _render_hil_gap_card(gap: dict, gap_report_dict: dict) -> None:
    """Render one gap card with HIL decision controls."""
    sev   = gap.get("severity", "OPTIONAL")
    icon  = {"CRITICAL": "🔴", "IMPORTANT": "🟡", "OPTIONAL": "⚪"}.get(sev, "⚪")
    title = gap.get("title", "Gap")
    gid   = gap.get("gap_id", title)
    decision_key  = f"hil_decision_{gid}"
    note_key      = f"hil_note_{gid}"
    current_dec   = gap.get("hil_decision", "PENDING")

    with st.expander(
        f"{icon} {title}",
        expanded=(sev == "CRITICAL" and current_dec == "PENDING"),
    ):
        col1, col2 = st.columns([3, 2])

        with col1:
            st.write(gap.get("description", ""))
            why = gap.get("why_it_matters", "")
            if why:
                st.info(f"**Why it matters:** {why}")

        with col2:
            st.caption("**Reference says:**")
            st.write(gap.get("reference_says", "")[:200])
            st.caption("**Draft says:**")
            st.write(gap.get("draft_says", "Not mentioned.")[:150])

        # HIL decision
        options    = ["PENDING", "RELEVANT", "ALREADY_COVERED", "NOT_APPLICABLE"]
        labels     = {
            "PENDING":          "⬜ Not yet reviewed",
            "RELEVANT":         "✅ Relevant — generate question",
            "ALREADY_COVERED":  "☑️ Already covered by draft",
            "NOT_APPLICABLE":   "🚫 Not applicable to this invention",
        }
        idx = options.index(current_dec) if current_dec in options else 0

        chosen = st.radio(
            "Your decision:",
            options,
            index=idx,
            format_func=lambda x: labels.get(x, x),
            key=decision_key,
            horizontal=True,
        )

        note = st.text_input(
            "Direction note (optional — guides how the question is framed):",
            value=gap.get("hil_note", ""),
            key=note_key,
            placeholder="e.g. 'focus on the dot pattern geometry, not the substrate'",
        )

        # Write back to the gap report dict in session state
        for g in gap_report_dict.get("gaps", []):
            if g.get("gap_id") == gid:
                g["hil_decision"] = chosen
                g["hil_note"]     = note
                break


# ══════════════════════════════════════════════════════════════════════════════
# SUB-STEP 1d — Question generation and download
# ══════════════════════════════════════════════════════════════════════════════

def _render_1d_questions(project_path: Optional[str]) -> None:
    st.subheader("Step 1d: Targeted Questions")

    gap_report_dict = st.session_state.get("gap_report")
    if not gap_report_dict:
        st.warning("No gap report available. Go back to Step 1b.")
        return

    confirmed = [
        g for g in gap_report_dict.get("gaps", [])
        if g.get("hil_decision") == "RELEVANT"
    ]

    if not confirmed:
        st.warning("No gaps marked as RELEVANT. Go back to Step 1c and review.")
        return

    st.info(
        f"Generating questions for **{len(confirmed)} confirmed gap(s)** — "
        f"gaps marked NOT APPLICABLE or ALREADY COVERED are excluded."
    )

    # Show scope note if set
    scope = gap_report_dict.get("invention_scope_note", "")
    if scope:
        st.caption(f"Invention scope note: *{scope}*")

    gap_questions = st.session_state.get("gap_questions")

    if gap_questions is None:
        if st.button(
            "🚀 Generate Targeted Questions",
            type="primary",
            use_container_width=True,
        ):
            _run_question_generation(gap_report_dict, project_path)
        return

    # ── Display questions ─────────────────────────────────────────────────────
    st.success(f"✅ {len(confirmed)} gap(s) addressed with targeted questions.")

    # Mechanism and field banners (same as old scrutiny tab)
    field = st.session_state.get("field_of_invention", "")
    if field:
        st.success(f"🔬 **Field of invention:** {field}")

    st.divider()
    st.markdown(gap_questions)
    st.divider()

    # ── Download options ──────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "📄 Download as TXT",
            data=gap_questions,
            file_name="gap_questions.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with col2:
        try:
            from services.export_service import to_docx
            docx_bio = to_docx(gap_questions)
            st.download_button(
                "📝 Download as DOCX",
                data=docx_bio,
                file_name="gap_questions.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )
        except Exception:
            pass
    with col3:
        if st.button("🔄 Re-generate", use_container_width=True):
            st.session_state.gap_questions = None
            st.rerun()

    # Save to project
    if project_path:
        from services import project_manager
        try:
            project_manager.save_questions(project_path, gap_questions)
        except Exception:
            pass


def _run_question_generation(
    gap_report_dict: dict,
    project_path: Optional[str],
) -> None:
    """Call gap_analysis_workflow.generate_gap_questions() and store result."""
    with st.spinner("Generating targeted questions for confirmed gaps…"):
        try:
            from services.patent_retriever import StructuredPatent
            from services.gap_analyser import (
                GapReport, Gap, GapType, GapSeverity, HILDecision,
            )
            from workflows.gap_analysis_workflow import generate_gap_questions

            # Reconstruct GapReport from dict
            gap_report = _dict_to_gap_report(gap_report_dict)

            # Reconstruct structured refs
            structured_refs = {}
            for sid, sdict in st.session_state.get("reference_structured", {}).items():
                structured_refs[sid] = StructuredPatent.from_dict(sdict)

            field = st.session_state.get("field_of_invention", "")

            result = generate_gap_questions(
                gap_report       = gap_report,
                structured_refs  = structured_refs,
                field_of_invention = field,
            )

            if result.success:
                st.session_state.gap_questions = result.questions
                st.rerun()
            else:
                st.error(f"Question generation failed: {result.error}")

        except Exception as exc:
            st.error(f"Error: {exc}")
            logger.exception("Question generation error")


def _proceed_to_1d() -> None:
    """Jump directly to step 1d."""
    st.session_state.gap_report_reviewed = True
    if st.button("▶ Generate Questions →", type="primary"):
        st.session_state.gap_analysis_step = "1d"
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Serialisation helpers
# (GapReport ↔ dict for session state, which must be JSON-serialisable)
# ══════════════════════════════════════════════════════════════════════════════

def _gap_report_to_dict(report) -> dict:
    """Convert GapReport dataclass to plain dict for session state storage."""
    return {
        "draft_title":     report.draft_title,
        "reference_id":    report.reference_id,
        "reference_title": report.reference_title,
        "draft_word_count": report.draft_word_count,
        "generated_at":    report.generated_at,
        "gaps": [
            {
                "gap_id":        g.gap_id,
                "gap_type":      g.gap_type.value,
                "severity":      g.severity.value,
                "title":         g.title,
                "description":   g.description,
                "reference_says": g.reference_says,
                "draft_says":    g.draft_says,
                "hil_decision":  g.hil_decision.value,
                "hil_note":      g.hil_note,
                "why_it_matters": g.why_it_matters,
                "questions":     g.questions,
            }
            for g in report.gaps
        ],
    }


def _dict_to_gap_report(d: dict):
    """Reconstruct GapReport from session state dict."""
    from services.gap_analyser import (
        GapReport, Gap, GapType, GapSeverity, HILDecision,
    )
    gaps = []
    for g in d.get("gaps", []):
        gaps.append(Gap(
            gap_id         = g["gap_id"],
            gap_type       = GapType(g["gap_type"]),
            severity       = GapSeverity(g["severity"]),
            title          = g["title"],
            description    = g["description"],
            reference_says = g["reference_says"],
            draft_says     = g["draft_says"],
            hil_decision   = HILDecision(g.get("hil_decision", "PENDING")),
            hil_note       = g.get("hil_note", ""),
            why_it_matters = g.get("why_it_matters", ""),
            questions      = g.get("questions", []),
        ))
    return GapReport(
        draft_title      = d["draft_title"],
        reference_id     = d["reference_id"],
        reference_title  = d["reference_title"],
        gaps             = gaps,
        draft_word_count = d.get("draft_word_count", 0),
        generated_at     = d.get("generated_at", ""),
    )


# ── Quick field extraction ────────────────────────────────────────────────────

def _quick_extract_field() -> str:
    """Fast regex extraction of field from draft collection without LLM."""
    return _extract_field_from_document()


def _extract_field_from_document() -> str:
    """
    Extract field of invention from draft document collection.
    Tries all chunks (not just first 3) and multiple regex patterns.
    Validates that the result is meaningful (> 30 chars, not just a heading).
    """
    import re
    try:
        collection = st.session_state.get("draft1_collection")
        if not collection:
            return ""
        result = collection.get(include=["documents"])
        chunks = result.get("documents", [])
        if not chunks:
            return ""

        # Join all chunks for full-document search
        full_text = " ".join(chunks)
        # Normalise whitespace
        full_text = re.sub(r"\s+", " ", full_text)

        # Pattern 1: explicit field heading followed by content
        # Handles: "Field of the Invention The present invention..."
        #          "Field of the Invention: A flexible polymer heater..."
        m = re.search(
            r"field of (?:the )?invention[:\s]+"
            r"((?:the )?present invention[^.]{20,400}\.|[A-Z][^.]{30,400}\.)",
            full_text, re.IGNORECASE,
        )
        if m:
            result_text = m.group(1).strip()
            if len(result_text) > 30:
                return result_text

        # Pattern 2: "The present invention relates to..."
        m = re.search(
            r"((?:the )?present invention relates to[^.]{30,400}\.)",
            full_text, re.IGNORECASE,
        )
        if m:
            return m.group(1).strip()

        # Pattern 3: "invention relates to" anywhere
        m = re.search(
            r"(invention relates to[^.]{30,300}\.)",
            full_text, re.IGNORECASE,
        )
        if m:
            return m.group(1).strip()

        # Pattern 4: "technical field" heading
        m = re.search(
            r"technical field[:\s]+([^.]{30,300}\.)",
            full_text, re.IGNORECASE,
        )
        if m:
            return m.group(1).strip()

        # Pattern 5: fallback — first substantial sentence after "field"
        m = re.search(
            r"field[^a-z][^.]{0,50}([A-Z][^.]{40,300}\.)",
            full_text, re.IGNORECASE,
        )
        if m:
            candidate = m.group(1).strip()
            if len(candidate) > 40:
                return candidate

        return ""
    except Exception as exc:
        logger.warning("Field extraction failed: %s", exc)
        return ""
