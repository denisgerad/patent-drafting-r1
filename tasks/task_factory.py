"""
tasks/task_factory.py
Build all CrewAI Task objects.
Pure data – no Streamlit, no side effects.
"""
from __future__ import annotations

import json
import logging
import re
from typing import List, Tuple

from crewai import Task

import config.settings as cfg

logger = logging.getLogger(__name__)


def _load_patent_types() -> dict:
    try:
        return json.loads(cfg.PATENT_TYPES_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _rag_query_for_type(patent_type: str) -> str:
    queries = {
        "Electronics": "invention novelty circuit components voltage signals specifications",
        "Mechanical":  "invention novelty mechanism parts dimensions materials assembly",
        "Chemical":    "invention novelty process composition reaction conditions synthesis",
        "Software":    "invention novelty algorithm method process steps data flow",
        "Medical Devices": "invention novelty device implant clinical parameters biocompatibility",
        "Materials":   "invention novelty material composition microstructure properties synthesis",
    }
    return queries.get(patent_type, "invention novelty technical specifications parameters")


# ── Classification Task ───────────────────────────────────────────────────────

def build_classification_task(classifier, context: str) -> Task:
    """Auto-classify patent domain from Draft1 RAG context."""
    patent_types = _load_patent_types()
    domains_list = "\n    - ".join(patent_types.keys()) if patent_types else (
        "\n    - ".join(["Mechanical", "Electronics", "Software", "Chemical", "Materials", "Medical Devices"])
    )

    description = f"""
Analyse the following patent disclosure and identify the technical domain.

DOCUMENT EXCERPT:
{context}

INSTRUCTIONS:
1. Identify the PRIMARY technical domain relevant to implementation and enablement.
2. Identify up to TWO secondary domains (or leave list empty).
3. Provide a confidence score 0.0–1.0 for the primary domain.
4. Justify using concrete technical indicators from the text (components, processes, materials).

AVAILABLE DOMAINS:
    - {domains_list}

CRITICAL: Respond with ONLY valid JSON (no markdown fences, no preamble):
{{
  "primary_domain": "<domain>",
  "secondary_domains": ["<domain>"],
  "confidence": 0.85,
  "justification": "Single-sentence rationale citing document evidence."
}}
"""
    return Task(
        description=description,
        agent=classifier,
        expected_output="JSON object: primary_domain, secondary_domains, confidence, justification",
    )


def parse_classification_result(raw: str) -> dict:
    """
    Extract JSON from the classifier output.
    Handles cases where the LLM wraps output in markdown fences.
    """
    text = re.sub(r"```(?:json)?", "", raw).strip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try extracting the first {...} block
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    logger.warning("Could not parse classification JSON from: %s", text[:200])
    return {}


# ── Scrutiny Task ─────────────────────────────────────────────────────────────

def build_scrutiny_task(scrutinizer, context: str, patent_type: str) -> Task:
    patent_types = _load_patent_types()
    domain_cfg   = patent_types.get(patent_type, {})
    focus_areas  = domain_cfg.get("focus_areas", [])
    units        = domain_cfg.get("technical_units", [])

    if focus_areas:
        bullets = "\n    ".join(f"- {a}" for a in focus_areas)
        analysis = f"""
    REQUIRED FOCUS AREAS (specific to {patent_type}):
    {bullets}

    For each area demand:
    - Specific numeric values and ranges (not "standard" or "typical")
    - Exact component/material identities
    - Precise operating conditions and limits
    - Step-by-step procedures where applicable"""
    else:
        analysis = """
    REQUIRED CATEGORIES:
    1. SPECIFICATIONS – specific numeric values and ranges.
    2. COMPONENTS/MATERIALS – part numbers, chemical names, grades.
    3. PROCESSES – step-by-step with timing and conditions.
    4. LIMITS – exact failure points and operating boundaries."""

    units_note = (
        f"\n    REQUIRED UNITS: Express all measurements in {', '.join(units)}."
        if units else ""
    )

    description = f"""
Patent Type: {patent_type}

Analyse this Patent Info Sheet:
{context}

Your mission: find the 'missing data' that a domain expert would need to build or replicate this invention.
{analysis}{units_note}

OUTPUT FORMAT:
Provide 5–7 highly technical, direct questions for the inventor.
- Focus on enablement (35 U.S.C. §112): what is needed for someone skilled in the art to replicate this?
- Do NOT ask about cost, manufacturing time, or business considerations.
- Demand precision over generalities.
"""
    return Task(
        description=description,
        agent=scrutinizer,
        expected_output=(
            f"5-7 technical follow-up questions focused on {patent_type} "
            "specifications and enablement requirements."
        ),
    )


# ── Consolidation Task ────────────────────────────────────────────────────────

def build_consolidation_task(consolidator, draft1_context: str, qa_context: str) -> Task:
    description = f"""
You are revising an existing patent disclosure.

STRICT INSTRUCTIONS:
- Do NOT summarise, paraphrase, or rewrite any existing text from Draft 1.
- Preserve ALL original sections, headings, and wording verbatim.
- Your ONLY job is to:
  (a) insert missing technical details from Q&A into the appropriate existing sections,
  (b) expand sections by adding new paragraphs where Q&A adds content,
  (c) add clarifying sentences where technical gaps exist,
  (d) append new subsections if required for §112 enablement.

Editing rules:
- Never delete or shorten existing content.
- If new information fits an existing paragraph, ADD text immediately after it.
- Maintain the same section titles as Draft 1.
- If information doesn't fit anywhere, append under "## Additional Implementation Details".

DOCUMENT 1 — ORIGINAL DRAFT (DO NOT MODIFY):
{draft1_context}

DOCUMENT 2 — INVENTOR Q&A (AUTHORITATIVE NEW FACTS):
{qa_context}

OUTPUT: Draft 1 text PLUS inserted/expanded content.

GUARDRAIL: If any original sentence from Draft 1 is missing in your output, the result is INVALID.

AUDIT: After the draft, append exactly `=== AUDIT LOG ===` on its own line, then a Markdown list:
- Original: "<short Draft1 excerpt>"  => Replaced with: "<new value>" (Source: Q&A)
"""
    return Task(
        description=description,
        agent=consolidator,
        expected_output=(
            "Draft 2 patent disclosure preserving Draft 1 verbatim with Q&A details inserted, "
            "followed by === AUDIT LOG === and a change list."
        ),
    )


# ── Validation Task ───────────────────────────────────────────────────────────

def build_validation_task(validator, user_answers: str) -> Task:
    description = f"""
Review the following inventor answers for technical sufficiency in a patent application:

{user_answers}

RED FLAGS to check:
1. Absence of specific measurements or numeric ranges.
2. Missing step-by-step process descriptions.
3. Vague terms ("standard", "normal", "typical") instead of specific materials/values.

If answers are sufficient → output exactly: PASSED
If insufficient → list required improvements concisely.
"""
    return Task(
        description=description,
        agent=validator,
        expected_output="'PASSED' or a list of required technical improvements.",
    )


# ── Post-processing utilities ─────────────────────────────────────────────────

def verify_draft_inclusion(original_text: str, final_text: str, min_length: int = 20) -> List[str]:
    """Return original sentence fragments absent from final_text."""
    if not original_text or not final_text:
        return []

    missing = []
    for line in original_text.splitlines():
        line = line.strip()
        if not line:
            continue
        for part in re.split(r"(?<=[\.!?])\s+", line):
            part = part.strip()
            if len(part) >= min_length and part not in final_text:
                missing.append(part)
    return missing


def split_audit_log(full_text: str) -> Tuple[str, str]:
    """Split draft text from audit log at the === AUDIT LOG === marker."""
    marker = "=== AUDIT LOG ==="
    if marker in full_text:
        parts = full_text.split(marker, 1)
        return parts[0].strip(), parts[1].strip()
    return full_text, ""
