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

def build_scrutiny_task(
    scrutinizer,
    context: str,
    patent_type: str,
    field_of_invention: str = "",
) -> Task:
    """
    Build a scrutiny task firmly anchored to the document's stated field of invention.

    Root failure mode this fixes
    ─────────────────────────────
    Without an explicit verbatim anchor, the LLM drifts to plausible training-data
    topics for the domain even when the actual invention is something different.
    Example: a flexible polymer heater patent classified as "Electronics" or
    "Optics / Display" was generating LGP dot-pattern questions, because the
    LLM defaulted to its most common association for that domain rather than
    reading what the document actually described.

    Three-guard approach
    ─────────────────────
    1. VERBATIM FIELD ANCHOR  – agent must quote the field/novelty sentence
       verbatim before any analysis. If it cannot find it, it must say so.
    2. HOST-DEVICE SCOPE LOCK – agent is explicitly told NOT to ask about
       the surrounding product/platform the invention is applied to.
    3. DOMAIN ANTI-PATTERNS   – per-domain prohibitions from patent_types.json.
    """
    patent_types  = _load_patent_types()
    domain_cfg    = patent_types.get(patent_type, {})
    focus_areas   = domain_cfg.get("focus_areas", [])
    units         = domain_cfg.get("technical_units", [])
    anti_patterns = domain_cfg.get("anti_patterns", [])

    focus_hint = ""
    if focus_areas:
        bullets = "\n".join(f"    * {a}" for a in focus_areas)
        focus_hint = f"""
DOMAIN LENS ({patent_type}) — apply only where the invention's novelty explicitly touches these:
{bullets}
"""

    prohibition_block = ""
    if anti_patterns:
        ap_bullets = "\n".join(f"  X {ap}" for ap in anti_patterns)
        prohibition_block = f"""
DOMAIN-SPECIFIC PROHIBITIONS:
{ap_bullets}
"""

    units_note = (
        f"\nWhere measurements are required, express them in: {', '.join(units)}."
        if units else ""
    )

    # ── Parse field_of_invention into invention core + application context ──────
    # The field statement often has two parts:
    #   (a) what the invention IS  e.g. "a flexible polymer resistive heater film"
    #   (b) where it is USED       e.g. "applicable to avionics... where condensation
    #                                    or frost formation may occur"
    # We surface both as separate anchors so the LLM generates questions for both.

    import re as _re

    field_core = field_of_invention.strip()
    field_application = ""

    if field_core:
        # Split on common applicability markers
        app_match = _re.search(
            r"(?:applicable to|used in|intended for|for use in|deployed in|"
            r"for applications?|in environments?|where\s+environmental|"
            r"in systems? where)(.*)",
            field_core,
            _re.IGNORECASE | _re.DOTALL,
        )
        if app_match:
            field_application = app_match.group(0).strip()
            # Keep full field for the anchor, extract application sub-text
            field_application = _re.sub(r"\s+", " ", field_application)

    field_anchor_block = ""
    if field_core:
        app_block = ""
        if field_application:
            app_block = f"""
  APPLICATION CONTEXT (from field statement):
    "{field_application}"
  → This MUST generate a dedicated theme: questions about deployment environments,
    environmental qualification standards, and operational use-case requirements.
"""
        field_anchor_block = f"""
PRE-EXTRACTED FIELD OF INVENTION:
  "{field_core}"
{app_block}
MANDATORY: Every theme and every question must be traceable to THIS field statement.
Discard any question that cannot be traced to the invention or its stated applications.
"""

    # Pre-fill FIELD verbatim from Python-extracted value to prevent LLM hallucination.
    # The LLM must NOT re-extract or invent these lines — they are already known.
    prefilled_field_line = (
        f'FIELD (verbatim): "{field_core}"'
        if field_core
        else 'FIELD (verbatim): "NOT FOUND IN DOCUMENT"'
    )
    prefilled_novelty_line = 'NOVELTY (verbatim): "NOT STATED — no formal novelty statement in document"'

    description = f"""
You are a senior patent expert conducting a 35 U.S.C. Section 112 enablement review.
Patent Domain: {patent_type}

=== DOCUMENT TO ANALYSE ===
{context}
===========================
{field_anchor_block}
The field of invention has been pre-extracted from the document. Begin your response
with EXACTLY these two lines (copy them verbatim, do not change them):

{prefilled_field_line}
{prefilled_novelty_line}

Then immediately proceed to generate the enablement questions below.
Do NOT re-extract, re-analyse, or rewrite the FIELD or NOVELTY lines.

STEP 1 — Identify Gaps: Invention Core + Application Context (BOTH required)

PART A — The Invention Itself:
  What technical details are MISSING that a skilled person needs to build
  or replicate the stated invention?
  Ask about the invention's own materials, geometry, fabrication, and performance.
  Do NOT ask about the host device unless the interface is explicitly claimed as novel.

  CORRECT vs WRONG scope:
    Field: "a flexible polymer resistive heater film bonded to an LCD"
    WRONG: "What is the LCD panel resolution?"             <- host device
    WRONG: "What is the LCD backlight voltage?"            <- host device
    CORRECT: "What is the polymer base material, thickness, and glass-transition temperature?"
    CORRECT: "What is the heating element track width, spacing, and sheet resistivity?"
    CORRECT: "What adhesive or bonding method attaches the film to the LCD surface, and what is its thermal conductivity?"

PART B — Application Context (MANDATORY when the field names deployment environments):
  The field statement names specific deployment environments and use conditions
  (e.g. avionics, military, condensation/frost). These environments impose
  additional requirements on the invention. You MUST generate a dedicated theme
  covering:
    - Environmental operating range: temperature range (min/max, rate of change),
      humidity range, altitude, vibration, shock (cite relevant standards, e.g.
      MIL-STD-810, DO-160, IEC 60068)
    - Condensation / frost scenario: at what delta-T does condensation form?
      What surface temperature must the heater maintain to prevent it?
      What is the required heat flux (W/m2) at worst-case ambient?
    - Power and control: how is heater activation triggered? What sensor
      (temperature, humidity, dew-point) controls it? What is the response time?
    - Qualification and safety: what test methods verify the heater meets
      environmental standards? What are the dielectric strength and insulation
      resistance requirements for the deployment environment?
{focus_hint}{prohibition_block}
STEP 2 — Derive 3 to 5 Theme Names
Name themes after the invention's own sub-systems AND deployment context.
REQUIRED themes (always include these two if the field names deployment environments):
  1. One theme covering the invention's physical construction (materials, geometry, fabrication)
  2. One theme named "Environmental Qualification and Operational Requirements" OR similar
     — this covers deployment environments, condensation/frost prevention, and standards

Additional themes derived from the document's own gaps:
  Good: "Heating Element Geometry and Resistivity"
  Good: "Film-to-Display Bonding and Interface Properties"
  Good: "Power Delivery and Thermal Control Logic"
  Bad:  "Specifications"  (too generic)
  Bad:  "Components"      (too generic)

STEP 3 — Write 3 to 5 Questions per Theme
Each question MUST:
  - Be traceable to a specific gap in the stated field/novelty
  - Request drawings, governing equations, numeric ranges, test procedures,
    or comparative performance data
  - NOT ask about the surrounding product unless the patent explicitly claims
    the interface between invention and product as novel
  - NOT use vague phrasing — name the specific element to describe
{units_note}

OUTPUT FORMAT (strict):
First two lines (already provided above — copy them exactly as given):
  FIELD (verbatim): "..."
  NOVELTY (verbatim): "..."

Then blank line, then grouped questions:

[Theme Name 1]
1. [Specific question]
2. [Specific question]
3. [Specific question]

[Theme Name 2]
1. [Specific question]
...
"""

    return Task(
        description=description,
        agent=scrutinizer,
        expected_output=(
            "Two pre-filled header lines (FIELD and NOVELTY verbatim, as provided). "
            f"Then 3-5 named technical theme sections, including one mandatory "
            f"'Environmental Qualification and Operational Requirements' theme when "
            f"the field names deployment environments. Each theme has 3-5 "
            f"enablement questions grounded in the {patent_type} document."
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
