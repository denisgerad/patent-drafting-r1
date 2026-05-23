"""
workflows/gap_analysis_workflow.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Phase D — Gap analysis workflow orchestration.

Orchestrates the four sub-steps of the new Step 1 flow:

  Sub-step 1a: Reference patent selection
    AUTO:   search_candidates(field_of_invention) → ranked candidates
    MANUAL: fetch_by_number(patent_number) → single candidate
    Both paths feed into the same selected_references list.

  Sub-step 1b: Gap analysis
    fetch_structured(patent_number) → StructuredPatent
    profile_reference(patent) → ReferenceProfile
    profile_draft(draft_text)  → DraftProfile
    analyse_gaps(ref, draft)   → GapReport
    enrich_gap_report(report)  → adds why_it_matters via local LLM

  Sub-step 1c: HIL review (UI only — no workflow code)
    Domain expert marks each gap: RELEVANT | ALREADY_COVERED | NOT_APPLICABLE
    Adds direction notes per gap.

  Sub-step 1d: Question generation
    build_gap_questions_task(confirmed_gaps) → CrewAI task
    Local model generates targeted questions for confirmed gaps only.

All LLM calls use the local model. No patent content sent to cloud.
Never raises — all functions return result objects with .success flag.
"""
from __future__ import annotations

import logging
import sys
import threading
from dataclasses import dataclass, field
from io import StringIO
from typing import Optional

from crewai import Crew

import config.settings as cfg
from agents.agent_factory import build_gap_analyst
from services.cloud_llm_service import get_llm, is_cloud_provider, provider_name
from services.gap_analyser import GapReport, Gap, HILDecision, analyse_gaps
from services.ollama_service import OllamaService
from services.patent_chunker import (
    profile_reference, profile_draft,
    draft_text_from_collection,
    reference_summary, draft_summary,
)
from services.patent_retriever import PatentCandidate, PatentRetriever, StructuredPatent
from tasks.task_factory import (
    build_why_it_matters_task,
    build_gap_questions_task,
)

logger = logging.getLogger(__name__)


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class CandidateSearchResult:
    """Result of sub-step 1a auto search."""
    candidates:  list[PatentCandidate]
    success:     bool
    error:       str = ""


@dataclass
class GapAnalysisResult:
    """Result of sub-step 1b gap analysis."""
    gap_report:      Optional[GapReport]
    structured_refs: dict[str, StructuredPatent]  # display_id → StructuredPatent
    agent_log:       str
    success:         bool
    error:           str = ""


@dataclass
class QuestionResult:
    """Result of sub-step 1d question generation."""
    questions:   str
    agent_log:   str
    success:     bool
    gap_count:   int = 0
    error:       str = ""


# ── Sub-step 1a: Candidate search ────────────────────────────────────────────

def search_candidates(
    field_of_invention: str,
    max_results:        int = 5,
    generate_explanations: bool = True,
) -> CandidateSearchResult:
    """
    Auto-search EPO OPS for similar granted patents.
    Returns ranked candidates for display in the reference selector UI.

    If generate_explanations=True, calls the local model to add a
    one-sentence explanation per candidate (adds ~30s for 5 candidates).
    Set to False for a fast initial search without LLM.
    """
    try:
        retriever  = PatentRetriever()
        candidates = retriever.search_similar(
            field_of_invention, max_results=max_results,
        )

        if not candidates:
            return CandidateSearchResult(
                candidates=[],
                success=True,   # not an error — just no results
                error="No similar patents found for this field of invention. "
                      "Try adding a manual patent number.",
            )

        if generate_explanations and candidates:
            llm_fn = _make_llm_generate_fn()
            if llm_fn:
                candidates = retriever.generate_explanations(
                    candidates, field_of_invention, llm_fn
                )

        logger.info("Candidate search: %d results", len(candidates))
        return CandidateSearchResult(candidates=candidates, success=True)

    except Exception as exc:
        logger.error("Candidate search failed: %s", exc)
        return CandidateSearchResult(
            candidates=[], success=False, error=str(exc),
        )


def fetch_manual_candidate(patent_number: str) -> CandidateSearchResult:
    """
    Fetch a single patent by number (manual entry path).
    Returns a single-element candidate list or error.
    """
    try:
        retriever = PatentRetriever()
        candidate = retriever.fetch_by_number(patent_number)
        if not candidate:
            return CandidateSearchResult(
                candidates=[], success=False,
                error=f"Could not fetch patent '{patent_number}'. "
                      f"Check the number format (e.g. EP3456789 or EP3456789B1).",
            )
        logger.info("Manual fetch: %s — %s", candidate.display_id, candidate.title[:60])
        return CandidateSearchResult(candidates=[candidate], success=True)

    except Exception as exc:
        logger.error("Manual patent fetch failed for '%s': %s", patent_number, exc)
        return CandidateSearchResult(
            candidates=[], success=False, error=str(exc),
        )


# ── Sub-step 1b: Gap analysis ─────────────────────────────────────────────────

def run_gap_analysis(
    selected_references: list[PatentCandidate],
    draft_collection,
    draft_title:        str = "Draft document",
    timeout:            int = cfg.CREW_TIMEOUT_SECONDS,
) -> GapAnalysisResult:
    """
    Run the full gap analysis for all selected reference patents.

    For each selected reference:
      1. Fetch structured patent (biblio + claims + description)
      2. Profile the reference patent
      3. Profile the draft document
      4. Run gap analysis
      5. Enrich gaps with why_it_matters via local model

    If multiple references selected, gaps are merged and deduplicated.
    """
    if not selected_references:
        return GapAnalysisResult(
            gap_report=None, structured_refs={},
            agent_log="", success=False,
            error="No reference patents selected.",
        )

    # ── Provider startup ──────────────────────────────────────────────────────
    startup_error = _ensure_llm_ready()
    if startup_error:
        return GapAnalysisResult(
            gap_report=None, structured_refs={},
            agent_log="", success=False,
            error=startup_error,
        )

    # ── Draft text ────────────────────────────────────────────────────────────
    draft_text = draft_text_from_collection(draft_collection)
    if not draft_text.strip():
        return GapAnalysisResult(
            gap_report=None, structured_refs={},
            agent_log="", success=False,
            error="Draft document collection is empty. Upload Draft1 first.",
        )

    draft_profile    = profile_draft(draft_text)
    structured_refs: dict[str, StructuredPatent] = {}
    all_gaps         = []
    agent_log        = ""

    try:
        retriever = PatentRetriever()
    except ValueError as exc:
        return GapAnalysisResult(
            gap_report=None, structured_refs={},
            agent_log="", success=False,
            error=f"EPO API not configured: {exc}",
        )

    # ── Process each reference ────────────────────────────────────────────────
    for candidate in selected_references:
        logger.info("Fetching structured: %s", candidate.display_id)
        structured = retriever.fetch_structured(candidate.display_id)

        if not structured:
            agent_log += (
                f"Warning: could not fetch full text for {candidate.display_id}. "
                f"Using bibliographic data only.\n"
            )
            # Create minimal StructuredPatent from biblio only
            structured = StructuredPatent(
                epodoc_id   = candidate.epodoc_id,
                display_id  = candidate.display_id,
                title       = candidate.title,
                abstract    = candidate.abstract,
                grant_date  = candidate.grant_date,
                applicant   = candidate.applicant,
            )

        structured_refs[candidate.display_id] = structured

        # Profile and analyse
        ref_profile  = profile_reference(structured)
        gap_report_i = analyse_gaps(
            ref_profile, draft_profile,
            draft_title=draft_title,
        )
        all_gaps.extend(gap_report_i.gaps)
        agent_log += f"\n[{candidate.display_id}] {len(gap_report_i.gaps)} gaps found\n"

    # ── Deduplicate gaps across multiple references ───────────────────────────
    if len(selected_references) > 1:
        all_gaps = _deduplicate_gaps(all_gaps)

    # ── Use the first reference for report metadata ───────────────────────────
    primary_ref = selected_references[0]
    from datetime import datetime, timezone
    gap_report = GapReport(
        draft_title     = draft_title,
        reference_id    = primary_ref.display_id,
        reference_title = primary_ref.title,
        gaps            = all_gaps,
        draft_word_count = draft_profile.total_word_count,
        generated_at    = datetime.now(timezone.utc).isoformat(),
    )

    # ── Enrich with why_it_matters (local LLM, one call per gap) ─────────────
    if all_gaps:
        gap_report, enrich_log = _enrich_gap_report(
            gap_report, draft_profile, timeout=timeout
        )
        agent_log += enrich_log

    logger.info(
        "Gap analysis complete: %d gaps, %d critical",
        len(all_gaps), len(gap_report.critical_gaps),
    )
    return GapAnalysisResult(
        gap_report      = gap_report,
        structured_refs = structured_refs,
        agent_log       = agent_log,
        success         = True,
    )


# ── Sub-step 1d: Question generation ─────────────────────────────────────────

def generate_gap_questions(
    gap_report:         GapReport,
    structured_refs:    dict[str, StructuredPatent],
    field_of_invention: str,
    timeout:            int = cfg.CREW_TIMEOUT_SECONDS,
) -> QuestionResult:
    """
    Generate targeted questions for all RELEVANT gaps in the gap report.
    Called after HIL review (sub-step 1c) confirms which gaps matter.

    Uses the local model via CrewAI.
    Never sends patent document content to cloud.
    """
    confirmed = gap_report.confirmed_gaps
    if not confirmed:
        return QuestionResult(
            questions="No gaps confirmed for question generation. "
                      "Mark gaps as RELEVANT in the domain review step.",
            agent_log="",
            success=True,
            gap_count=0,
        )

    startup_error = _ensure_llm_ready()
    if startup_error:
        return QuestionResult(
            questions="", agent_log="", success=False,
            error=startup_error,
        )

    # Build compact summaries for prompt context
    # These contain structure labels only — no patent document content
    ref_summaries = []
    for sid, structured in structured_refs.items():
        ref_prof = profile_reference(structured)
        ref_summaries.append(reference_summary(ref_prof))
    ref_summary_text = "\n\n".join(ref_summaries)

    # Draft summary — no patent content, just structural labels
    draft_sum = (
        f"Draft: {gap_report.draft_word_count} words. "
        f"Reference: {gap_report.reference_id} — {gap_report.reference_title}."
    )

    # Build and run the question generation crew
    gap_analyst = build_gap_analyst()
    task = build_gap_questions_task(
        gap_analyst,
        confirmed_gaps     = confirmed,
        reference_summary  = ref_summary_text,
        draft_summary      = draft_sum,
        field_of_invention = field_of_invention,
    )

    if task is None:
        return QuestionResult(
            questions="", agent_log="", success=False,
            error="Could not build question task — no confirmed gaps.",
        )

    crew        = Crew(agents=[gap_analyst], tasks=[task], verbose=True)
    holder: dict = {"result": None, "error": None}
    log_capture = StringIO()

    def _run():
        old_stdout = sys.stdout
        sys.stdout = log_capture
        try:
            holder["result"] = crew.kickoff()
        except Exception as exc:
            holder["error"] = str(exc)
        finally:
            sys.stdout = old_stdout

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    captured_log = log_capture.getvalue()

    if thread.is_alive():
        return QuestionResult(
            questions="", agent_log=captured_log, success=False,
            error=f"Question generation timed out after {timeout}s.",
        )
    if holder["error"]:
        return QuestionResult(
            questions="", agent_log=captured_log, success=False,
            error=holder["error"],
        )

    raw = str(
        holder["result"].raw
        if hasattr(holder["result"], "raw")
        else holder["result"]
    )

    logger.info("Gap questions generated: %d chars", len(raw))
    return QuestionResult(
        questions  = raw,
        agent_log  = captured_log,
        success    = True,
        gap_count  = len(confirmed),
    )


# ── Gap enrichment (why_it_matters via local LLM) ────────────────────────────

def _enrich_gap_report(
    gap_report: GapReport,
    draft_profile,
    timeout: int = 30,
) -> tuple[GapReport, str]:
    """
    Add why_it_matters explanation to each gap using the local model.
    One LLM call per gap — runs sequentially to avoid overloading Nemo.
    For large reports (> 6 gaps), enriches CRITICAL gaps first.
    """
    log_lines = ["\n[GAP ENRICHMENT]"]
    gap_analyst = build_gap_analyst()

    # Prioritise CRITICAL gaps for enrichment if many gaps
    from services.gap_analyser import GapSeverity
    gaps_to_enrich = gap_report.gaps
    if len(gap_report.gaps) > 6:
        gaps_to_enrich = (
            [g for g in gap_report.gaps if g.severity == GapSeverity.CRITICAL]
            + [g for g in gap_report.gaps if g.severity != GapSeverity.CRITICAL]
        )[:6]  # cap at 6 enriched gaps to keep latency reasonable

    for gap in gaps_to_enrich:
        try:
            task = build_why_it_matters_task(
                gap_analyst,
                gap_title       = gap.title,
                gap_description = gap.description,
                gap_type        = gap.gap_type.value,
                reference_says  = gap.reference_says,
                draft_says      = gap.draft_says,
                field_of_invention = gap_report.draft_title,
            )
            crew   = Crew(agents=[gap_analyst], tasks=[task], verbose=False)
            holder = {"result": None, "error": None}
            log_cap = StringIO()

            def _run(h=holder, lc=log_cap):
                old = sys.stdout; sys.stdout = lc
                try:
                    h["result"] = crew.kickoff()
                except Exception as exc:
                    h["error"] = str(exc)
                finally:
                    sys.stdout = old

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            t.join(timeout=timeout)

            if not t.is_alive() and holder["result"]:
                raw = str(
                    holder["result"].raw
                    if hasattr(holder["result"], "raw")
                    else holder["result"]
                )
                gap.why_it_matters = raw.strip()[:400]
                log_lines.append(f"  Enriched: {gap.title[:50]}")
            else:
                log_lines.append(f"  Skipped (timeout): {gap.title[:50]}")

        except Exception as exc:
            log_lines.append(f"  Error enriching '{gap.title[:40]}': {exc}")

    return gap_report, "\n".join(log_lines)


# ── Gap deduplication ─────────────────────────────────────────────────────────

def _deduplicate_gaps(gaps: list[Gap]) -> list[Gap]:
    """
    Remove duplicate gaps when multiple reference patents are used.
    Two gaps are considered duplicates if their titles are very similar.
    Keep the gap with the richer reference_says text.
    """
    seen_titles: dict[str, Gap] = {}
    for gap in gaps:
        # Normalise title for comparison
        key = gap.title.lower().strip()
        if key not in seen_titles:
            seen_titles[key] = gap
        else:
            # Keep whichever has more reference context
            existing = seen_titles[key]
            if len(gap.reference_says) > len(existing.reference_says):
                seen_titles[key] = gap

    # Preserve original severity ordering
    from services.gap_analyser import GapSeverity
    severity_order = {GapSeverity.CRITICAL: 0, GapSeverity.IMPORTANT: 1, GapSeverity.OPTIONAL: 2}
    result = sorted(seen_titles.values(), key=lambda g: severity_order[g.severity])
    return result


# ── Provider startup helper ───────────────────────────────────────────────────

def _ensure_llm_ready() -> str:
    """
    Ensure the LLM provider is ready. Returns error string or empty string.
    Mirrors the pattern in scrutiny_workflow.run().
    """
    if not is_cloud_provider():
        ollama = OllamaService()
        try:
            ollama.ensure_running()
        except Exception as exc:
            return f"Ollama startup failed: {exc}"
        ok, diag_msg = ollama.diagnose_llm_stack()
        if not ok:
            return diag_msg
    else:
        logger.info("Using cloud provider: %s", provider_name())
    return ""


def _make_llm_generate_fn():
    """
    Return a callable(prompt: str) -> str for the local model.
    Used by PatentRetriever.generate_explanations().
    Returns None if provider not ready.
    """
    try:
        if not is_cloud_provider():
            ollama = OllamaService()
            ollama.ensure_running()
            model = ollama.resolve_model()

            def _ollama_fn(prompt: str) -> str:
                import requests as _req
                resp = _req.post(
                    f"{cfg.OLLAMA_BASE_URL}/api/generate",
                    json={"model": model, "prompt": prompt, "stream": False},
                    timeout=30,
                )
                resp.raise_for_status()
                return resp.json().get("response", "")

            return _ollama_fn
        else:
            # Cloud provider — use the LLM directly via a simple completion
            import anthropic as _ant
            client = _ant.Anthropic(api_key=cfg.CLAUDE_API_KEY)

            def _claude_fn(prompt: str) -> str:
                msg = client.messages.create(
                    model=cfg.CLAUDE_MODEL,
                    max_tokens=200,
                    messages=[{"role": "user", "content": prompt}],
                )
                return msg.content[0].text

            return _claude_fn

    except Exception as exc:
        logger.warning("Could not create LLM generate fn: %s", exc)
        return None
