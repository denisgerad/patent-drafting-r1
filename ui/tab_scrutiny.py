"""
ui/tab_scrutiny.py
Renders the Step 1 tab:
  1a. Auto-classify domain (with manual override panel)
  1b. Run scrutiny (gap analysis)
"""
from __future__ import annotations

import json
import logging
from typing import Optional

import streamlit as st

import config.settings as cfg
from services import vector_store as vs
from services import project_manager
from services.document_loader import load_chunks

logger = logging.getLogger(__name__)


def _get_available_domains() -> list[str]:
    try:
        data = json.loads(cfg.PATENT_TYPES_JSON.read_text(encoding="utf-8"))
        return list(data.keys())
    except Exception:
        return ["Mechanical", "Electronics", "Software", "Chemical", "Materials", "Medical Devices"]


def render(project_path: Optional[str]) -> None:
    st.header("🔍 Patent Scrutiny & Gap Analysis")

    if not st.session_state.get("draft1_processed"):
        st.warning("⚠️ Please upload and process Draft1 in the sidebar first.")
        return

    _render_classification_panel(project_path)

    if st.session_state.get("classification_done"):
        st.divider()
        _render_scrutiny_panel(project_path)

    if st.session_state.get("patent_questions"):
        _render_questions_output(project_path)


# ── 1a: Classification ────────────────────────────────────────────────────────

def _render_classification_panel(project_path: Optional[str]) -> None:
    st.subheader("Step 1a: Domain Classification")

    if st.session_state.get("classification_done"):
        _render_classification_result(project_path)
        return

    domains = _get_available_domains()
    col1, col2 = st.columns([2, 1])

    with col1:
        st.info(
            "**Auto-classify** uses your Draft1 content to detect the patent domain automatically. "
            "You can review and override the result before proceeding."
        )
        if st.button("🤖 Auto-Classify Domain", type="primary", use_container_width=True):
            _run_auto_classification(project_path, domains)

    with col2:
        st.info("Prefer to set domain manually?")
        with st.expander("✍️ Manual Selection"):
            _render_manual_domain_picker(domains, project_path, label="Confirm Manual Domain")


def _run_auto_classification(project_path: Optional[str], domains: list[str]) -> None:
    """Trigger the classification workflow and store results in session state."""
    collection = st.session_state.get("draft1_collection")
    if not collection:
        st.error("Draft1 collection not found. Please re-process Draft1.")
        return

    with st.spinner("🤖 Auto-classifying patent domain…"):
        from workflows.classification_workflow import run as classify_run
        result = classify_run(collection, timeout=120)

    st.session_state.classification_result = result.to_dict()
    st.session_state.classification_done   = True
    st.session_state.user_override         = False

    if result.success:
        st.session_state.patent_type = result.primary_domain
        if project_path:
            project_manager.update_metadata(project_path, {
                "classification": {
                    "auto_classification":  result.to_dict(),
                    "user_override":        False,
                    "final_domain":         result.primary_domain,
                }
            })
    else:
        st.warning(f"Auto-classification failed ({result.error}). Please select domain manually.")

    st.rerun()


def _render_classification_result(project_path: Optional[str]) -> None:
    """Show the classification result banner + override controls."""
    result_dict = st.session_state.get("classification_result") or {}
    current_domain = st.session_state.patent_type
    is_override    = st.session_state.get("user_override", False)

    confidence = result_dict.get("confidence", 0)
    conf_pct   = f"{confidence:.0%}" if confidence else "N/A"

    if is_override:
        st.success(f"✅ Domain: **{current_domain}** *(manually set)*")
    elif result_dict.get("success", True):
        colour = "green" if confidence >= cfg.CLASSIFIER_CONFIDENCE else "orange"
        st.success(f"✅ Auto-classified: **{current_domain}** — Confidence: :{colour}[{conf_pct}]")
        if result_dict.get("justification"):
            st.caption(f"💡 {result_dict['justification']}")
        if result_dict.get("secondary_domains"):
            st.caption(f"Secondary domains: {', '.join(result_dict['secondary_domains'])}")
    else:
        st.warning(f"⚠️ Auto-classification unavailable – using: **{current_domain}**")

    # Low-confidence warning + explicit override invitation
    if not is_override and confidence and confidence < cfg.CLASSIFIER_CONFIDENCE:
        st.warning(
            f"⚠️ Confidence is low ({conf_pct}). "
            "Please review and override below if the domain is incorrect."
        )

    # Always show override option
    with st.expander("🔄 Override Domain", expanded=(confidence < cfg.CLASSIFIER_CONFIDENCE and not is_override)):
        domains = _get_available_domains()
        _render_manual_domain_picker(
            domains, project_path,
            label="Apply Override",
            default=current_domain,
        )

    if st.button("↩️ Re-run Classification"):
        from ui.session_state import reset_classification
        reset_classification()
        st.rerun()


def _render_manual_domain_picker(
    domains: list[str],
    project_path: Optional[str],
    label: str = "Confirm Domain",
    default: Optional[str] = None,
) -> None:
    current = default or st.session_state.get("patent_type", "Electronics")
    idx = domains.index(current) if current in domains else 0

    chosen = st.selectbox(
        "Select Patent Domain", domains, index=idx, key=f"manual_domain_{label}"
    )
    notes = st.text_area(
        "Additional notes (optional)",
        placeholder="e.g., combines mechanical and electronic subsystems…",
        height=80,
        key=f"notes_{label}",
    )

    if st.button(label, use_container_width=True, key=f"btn_{label}"):
        st.session_state.patent_type          = chosen
        st.session_state.classification_done  = True
        st.session_state.user_override        = True
        st.session_state.classification_in_progress = False
        st.session_state.classification_result = {
            "primary_domain":    chosen,
            "secondary_domains": [],
            "confidence":        1.0,
            "justification":     f"Manual selection. {notes}".strip(),
            "success":           True,
        }
        if project_path:
            project_manager.update_metadata(project_path, {
                "classification": {
                    "auto_classification": None,
                    "user_override":       True,
                    "final_domain":        chosen,
                    "override_reason":     notes or "Manual selection",
                }
            })
        st.success(f"✅ Domain set to **{chosen}**")
        st.rerun()


# ── 1b: Scrutiny ──────────────────────────────────────────────────────────────

def _render_scrutiny_panel(project_path: Optional[str]) -> None:
    st.subheader("Step 1b: Technical Scrutiny")
    st.info(f"Generating enablement questions for **{st.session_state.patent_type}** domain.")

    with st.expander("🪄 Tune Agent Persona (optional)"):
        _render_persona_tuner()

    if st.session_state.patent_questions:
        st.success("✅ Questions already generated for this session.")
        if st.button("🔄 Re-generate Questions"):
            st.session_state.patent_questions  = None
            st.session_state.readiness_report  = None
            st.session_state.field_of_invention = ""
            st.session_state.mechanism          = ""
            st.rerun()
        return

    # ── Data transmission warning for cloud providers ─────────────────────
    import config.settings as _cfg
    from services.cloud_llm_service import is_cloud_provider, provider_name
    if is_cloud_provider():
        st.warning(
            f"⚠️ **Data Privacy Notice**\n\n"
            f"You are using **{provider_name()}** (cloud provider). "
            f"Clicking 'Generate Questions' will send excerpts from your uploaded "
            f"patent document to {provider_name().split('(')[0].strip()} servers "
            f"for processing.\n\n"
            f"**For unpublished / pre-filing patents**, this may constitute prior "
            f"disclosure. Use **Ollama (local)** instead by setting "
            f"`LLM_PROVIDER=ollama` in your `.env` file.\n\n"
            f"Proceed only if your patent is already filed, or if you have a "
            f"Data Processing Agreement with your cloud provider."
        )
        confirmed = st.checkbox(
            "I understand patent content will be sent to a cloud API and I accept this risk",
            key="cloud_data_consent"
        )
        if not confirmed:
            st.info("☝️ Tick the checkbox above to proceed with cloud analysis.")
            return

    if st.button("🚀 Generate Scrutiny Questions", type="primary"):
        _run_scrutiny(project_path)


def _render_persona_tuner() -> None:
    """Optional persona customisation before running scrutiny."""
    try:
        data = json.loads(cfg.PATENT_TYPES_JSON.read_text(encoding="utf-8"))
    except Exception:
        data = {}

    domains = list(data.keys())
    selected = st.selectbox(
        "Tune for domain:", domains,
        index=domains.index(st.session_state.patent_type)
        if st.session_state.patent_type in domains else 0,
        key="persona_domain",
    )
    notes = st.text_input(
        "Expert instructions:",
        placeholder="e.g., Focus on UV-curing cycles and polymer viscosity.",
        key="persona_notes",
    )

    if st.button("Apply Persona", key="apply_persona"):
        from agents.agent_factory import _build_backstory, _load_domain_config
        domain_cfg = _load_domain_config(selected)
        st.session_state.custom_role      = domain_cfg.get("role", f"Patent Specialist ({selected})")
        st.session_state.custom_backstory = _build_backstory(selected, notes, domain_cfg)
        st.session_state.persona_tuned    = True
        st.success(f"Agent persona applied: {st.session_state.custom_role}")


def _run_scrutiny(project_path: Optional[str]) -> None:
    # Clear any stale report from a previous run before starting
    st.session_state.readiness_report   = None
    st.session_state.field_of_invention = ""
    st.session_state.mechanism          = ""
    collection = st.session_state.get("draft1_collection")
    if not collection:
        st.error("Draft1 collection missing. Please re-process Draft1.")
        return

    with st.spinner("🔎 Analysing patent document…"):
        from workflows.scrutiny_workflow import run as scrutiny_run
        result = scrutiny_run(
            collection=collection,
            patent_type=st.session_state.patent_type,
            custom_role=st.session_state.get("custom_role"),
            custom_backstory=st.session_state.get("custom_backstory"),
        )

    if result.agent_log:
        with st.expander("🔍 Agent Processing Log"):
            st.code(result.agent_log, language="text")

    if not result.success:
        st.error(f"❌ Scrutiny failed: {result.error}")
        st.markdown("""
        **Troubleshooting:**
        - Ensure Ollama is running: `ollama serve`
        - Check a model is installed: `ollama list`
        - Try a smaller model: `ollama pull mistral:7b-instruct-q4_K_M`
        """)
        return

    st.session_state.patent_questions    = result.questions
    st.session_state.field_of_invention  = getattr(result, "field_of_invention", "")
    st.session_state.mechanism           = getattr(result, "mechanism", "")
    st.session_state.readiness_report    = getattr(result, "readiness_report", None)
    st.session_state.augmentation_source = getattr(result, "augmentation_source", "none")

    if project_path:
        project_manager.save_questions(project_path, result.questions)

    st.rerun()


# ── Questions Output ──────────────────────────────────────────────────────────

def _render_readiness_report(report) -> None:
    """Render the readiness gate panel with colour-coded verdict."""
    VERDICT_CONFIG = {
        "READY":        ("✅", "success", "Ready for domain review"),
        "BORDERLINE":   ("🟡", "warning", "Borderline — review weak categories before sending"),
        "NOT_READY":    ("❌", "error",   "Not ready — address gaps before domain review"),
        "NO_REFERENCE": ("ℹ️", "info",    "No reference file — section check only"),
    }
    icon, colour, label = VERDICT_CONFIG.get(
        report.verdict, ("❓", "info", report.verdict)
    )

    with st.expander(
        f"{icon} **Readiness Gate: {label}**  "
        f"(overall score: {report.overall_score:.0%})",
        expanded=(report.verdict != "READY"),
    ):
        st.caption(f"Reference: `{report.reference_file_used}` — {report.domain}")
        st.caption(report.verdict_reason)

        # Per-category table
        if report.category_scores:
            st.markdown("**Category breakdown:**")
            for cs in report.category_scores:
                cat_icon  = "✅" if cs.passed else "❌"
                bar_fill  = int(cs.category_score * 10)
                bar       = "█" * bar_fill + "░" * (10 - bar_fill)
                st.markdown(
                    f"{cat_icon} **{cs.name}** &nbsp; "
                    f"`{bar}` {cs.category_score:.0%} &nbsp; "
                    f"_(topic: {cs.topic_coverage:.0%} | "
                    f"depth: {cs.depth_coverage:.0%} | "
                    f"heading: {'✓' if cs.completeness == 1.0 else '✗'})_"
                )
                if cs.missing_key_terms:
                    st.caption(
                        f"   Missing topics: {', '.join(cs.missing_key_terms[:6])}"
                    )
                if cs.missing_depth_markers:
                    st.caption(
                        f"   Missing depth: {', '.join(cs.missing_depth_markers[:4])}"
                    )

        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Score", f"{report.overall_score:.0%}",
                      delta=f"{report.overall_score - report.readiness_threshold:.0%} vs threshold")
        with col2:
            n_pass = len([c for c in report.category_scores if c.passed])
            st.metric("Categories Passed",
                      f"{n_pass}/{len(report.category_scores)}")
        with col3:
            if report.total_depth_markers_expected > 0:
                depth_pct = (report.total_depth_markers_found /
                             report.total_depth_markers_expected)
                st.metric("Depth Markers",
                          f"{report.total_depth_markers_found}/"
                          f"{report.total_depth_markers_expected}",
                          delta=f"{depth_pct:.0%}")

        if report.verdict == "NO_REFERENCE":
            st.info(
                "To enable full readiness scoring for this patent type, "
                "add a reference file: "
                f"domain_reference_questions/{report.product_type}.json  "
                "See domain_reference_questions/README.md for the format."
            )


def _render_questions_output(project_path: Optional[str]) -> None:
    st.divider()
    st.subheader("📋 Generated Scrutiny Questions")

    # Verification banners — show what was extracted before questions
    field     = st.session_state.get("field_of_invention", "")
    mechanism = st.session_state.get("mechanism", "")

    if field:
        st.success(f"🔬 **Field of invention:** {field}")
    else:
        st.warning(
            "⚠️ Field of invention could not be auto-detected. "
            "Add a 'Field of the Invention' section to your Draft1 document."
        )

    if mechanism:
        st.info(
            f"⚙️ **Enabling mechanism identified:** {mechanism}\n\n"
            f"Questions are targeted at this specific technical feature. "
            f"If this is wrong, your Draft1 needs a clearer description "
            f"of the novel component or method."
        )
    else:
        st.warning(
            "⚠️ Enabling mechanism could not be identified. "
            "Questions may be too generic — describe the specific novel "
            "feature in your Draft1 document."
        )

    # Augmentation source badge
    aug_source = st.session_state.get("augmentation_source", "none")
    if aug_source and aug_source != "none":
        badge_map = {
            "bank":        ("🏦", "success", "Expert Bank (Phase 1 — offline, free)"),
            "claude":      ("☁️", "info",    "Claude Augmentation (Phase 2 — cloud, no patent data sent)"),
            "bank+claude": ("🔀", "success", "Bank + Claude Augmentation (hybrid)"),
        }
        icon, colour, label = badge_map.get(aug_source, ("ℹ️", "info", aug_source))
        st.success(
            f"{icon} **Questions augmented by: {label}**\n\n"
            f"The expert knowledge system added questions to address depth gaps. "
            f"No patent document content was sent to any cloud service."
        )

    # ── Readiness gate panel ─────────────────────────────────────────────
    # report is set by the current run. None means either:
    #   (a) No checklist matched this patent type → rater was not called
    #   (b) A new Draft1 was just uploaded (no run yet for this document)
    # In both cases show a clear "no reference" message — never show a
    # stale report from a previous patent.
    report = st.session_state.get("readiness_report")
    if report is not None:
        _render_readiness_report(report)
    else:
        field = st.session_state.get("field_of_invention", "")
        if field:
            # A run completed but no checklist matched this patent type
            st.info(
                "ℹ️ **No readiness reference for this patent type.**\n\n"
                "The questions above were generated but cannot be automatically "
                "scored — no reference file exists for this invention category.\n\n"
                "To enable scoring, add domain expert questions to:\n"
                "`domain_reference_questions/<product_type>.json`\n\n"
                "See `domain_reference_questions/README.md` for the format. "
                "The questions above may still be sent for domain review based "
                "on your own judgement."
            )
        # If field is empty, no run has happened yet — show nothing

    st.divider()
    st.markdown(st.session_state.patent_questions)

    from services.export_service import to_docx
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇️ Download (.txt)",
            st.session_state.patent_questions,
            file_name="Patent_Questions.txt",
            use_container_width=True,
        )
    with c2:
        docx_bio = to_docx(st.session_state.patent_questions)
        if docx_bio:
            st.download_button(
                "⬇️ Download (.docx)",
                docx_bio,
                file_name="Patent_Questions.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )

    st.info("📝 Prepare a Q&A document offline with your answers, then upload it in Step 2.")
