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
        "Electronics":      "invention novelty circuit components voltage signals specifications",
        "Optics / Display": "invention novelty light guide optical structure dot pattern luminance uniformity extraction",
        "Mechanical":       "invention novelty mechanism parts dimensions materials assembly",
        "Chemical":         "invention novelty process composition reaction conditions synthesis",
        "Software":         "invention novelty algorithm method process steps data flow",
        "Medical Devices":  "invention novelty device implant clinical parameters biocompatibility",
        "Materials":        "invention novelty material composition microstructure properties synthesis",
    }
    return queries.get(patent_type, "invention novelty technical specifications parameters claims")


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
    """
    Build a scrutiny task that produces expert-quality, grouped questions.

    Expert vs model question gap analysis
    ──────────────────────────────────────
    The key failure of the old prompt was producing generic circuit-level questions
    (voltage, PCB, EMC) for what was an optics/structural patent. The root causes:

    1. Focus areas in patent_types.json drove the LLM toward its domain's default
       questions rather than what THIS specific invention actually needs.
    2. Flat numbered list format produced shallow, disconnected questions.
    3. No instruction to read the invention's OWN novel claims first.

    This rewrite fixes all three:
    1. The agent must FIRST extract the invention's own novel claims from the text,
       then derive question themes FROM those claims — not from generic domain lists.
    2. Questions are grouped under named technical themes (like expert examples).
    3. Each question must be traceable to a specific gap in the document.
    4. Generic circuit/component questions are explicitly banned unless the
       document's novelty is itself about those circuits/components.
    """
    patent_types = _load_patent_types()
    domain_cfg   = patent_types.get(patent_type, {})
    focus_areas  = domain_cfg.get("focus_areas", [])
    units        = domain_cfg.get("technical_units", [])
    anti_patterns = domain_cfg.get("anti_patterns", [])

    # Build the domain focus hint — used only as a secondary lens, not the primary driver
    focus_hint = ""
    if focus_areas:
        bullets = "\n".join(f"    • {a}" for a in focus_areas)
        focus_hint = f"""
DOMAIN LENS ({patent_type}) — use only where the invention's novelty touches these areas:
{bullets}
"""

    # Novelty-alignment hint: if Primary Novel Feature mentions optical triggers,
    # instruct the reviewer to prioritise optical-structure topics.
    novelty_alignment = ""
    if "optic" in patent_type.lower():
        optics_triggers = [
            "light extraction",
            "waveguide",
            "luminance",
            "dot pattern",
            "optical architecture",
        ]
        triggers = ", ".join(optics_triggers)
        novelty_alignment = f"""
NOVELTY ALIGNMENT (Optics): If the invention's Primary Novel Feature mentions any of: {triggers},
then treat the novelty as explicitly concerning optical structure. Prioritise geometry, light
propagation/extraction models, measurement/validation methods, and fabrication processes when
deriving themes and questions.
"""

        # Optics measurable outputs enforcement
        optics_measurable_block = ""
        if "optic" in patent_type.lower():
          optics_measurable_block = """
    OPTICS MUST PRODUCE (measurable outputs):
    - dimensions (µm, mm)
    - luminance (cd/m²)
    - efficiency (%)
    - angular distribution (°)

    For Optics patent reviews, EACH theme must include at least one question requesting a numerical
    range or measurable parameter (e.g., a dimension range in µm/mm, luminance in cd/m², an
    efficiency percentage, or angular distribution in degrees).
    """

    # Build explicit prohibition block from anti_patterns
    prohibition_block = ""
    if anti_patterns:
        ap_bullets = "\n".join(f"  ✗ {ap}" for ap in anti_patterns)
        prohibition_block = f"""
DOMAIN-SPECIFIC PROHIBITIONS — never ask these for this domain:
{ap_bullets}
"""

    units_note = (
        f"\nWhere measurements are required, express them in: {', '.join(units)}."
        if units else ""
    )

    description = f"""
You are a senior patent expert conducting a technical review for 35 U.S.C. §112 enablement.
Patent Domain: {patent_type}

═══════════════════════════════════════════════════════
DOCUMENT TO ANALYSE:
{context}
═══════════════════════════════════════════════════════

━━━ STEP 1: Extract the Invention's Novel Claims ━━━
Before writing any question, read the document carefully and identify:
  A) What is the PRIMARY novel feature or mechanism of this invention?
     (e.g., "dual-edge light injection into a universal LGP with a complementary dot pattern")
  B) What are the 2–4 SUB-FEATURES that enable or support the primary feature?
  C) What technical details does the document NOT provide about these features?

Only ask questions about gaps in (A), (B), and (C).
Do NOT invent questions about generic domain topics not present in the document.

━━━ STEP 2: Identify 2–4 Technical Themes ━━━
Group the gaps you found into named technical themes that reflect the invention's OWN
structure. Derive theme names FROM the document — do not use generic names like
"Specifications" or "Components". Good theme examples from real expert reviews:
  • "Structural Design & Optical Architecture"
  • "Dot Pattern Geometry & Light Extraction"
  • "Material, Fabrication & Performance Benefits"
  • "Selective Activation & Brightness Control"
{focus_hint}{prohibition_block}
{novelty_alignment}
{optics_measurable_block}
━━━ STEP 3: Write Questions ━━━
For each theme, write 3–5 questions. Each question must:
  ✓ Target a SPECIFIC gap in the document (not a general domain question)
  ✓ Require a precise, technical answer (drawings, formulas, numeric ranges, test data)
  ✓ Ask for one of: drawings/diagrams, governing equations, measured values,
    step-by-step procedures, or comparative performance data
  ✗ NOT ask about cost, schedule, or business considerations
  ✗ NOT ask about circuit-level details (voltage, PCB, EMC) unless the document's
    own novelty is specifically about those circuits
  ✗ NOT use vague phrasing like "describe the process" — be specific about WHAT to describe
{units_note}

Each theme MUST include at least one question that requests a numerical range or measurable parameter.

━━━ OUTPUT FORMAT (strict) ━━━
Produce output with the exact sections and formatting below:

=== NOVELTY SUMMARY ===
Primary Novel Feature:
Supporting Features:
- ...
- ...

=== QUESTIONS ===
[Theme Name 1]
1. [Question targeting a specific document gap]
2. [Question targeting a specific document gap]

[Theme Name 2]
1. [Question targeting a specific document gap]
2. [Question targeting a specific document gap]

...and so on for each theme.

Do not add preamble, do not explain your reasoning outside these sections.
Start directly with the first section heading.
"""

    return Task(
        description=description,
        agent=scrutinizer,
        expected_output=(
            f"A NOVELTY SUMMARY section followed by grouped theme-based QUESTIONS. "
            f"Output must follow the exact headings and format specified in the prompt."
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


# Forbidden-electronics checker for Optics domain
FORBIDDEN_ELECTRONICS = [
    "voltage",
    "current",
    "pcb",
    "emc",
    "impedance",
    "ohm",
]


def check_forbidden_electronics(output_text: str, patent_type: str, forbidden_list: List[str] | None = None) -> dict:
    """Scan a scrutiny output for forbidden electronics keywords when the domain is Optics.

    Returns a dict with:
      - forbidden_found: list of matched forbidden keywords
      - novelty_mentions_electronics: bool (True if Primary Novel Feature mentions electronics)
      - flag: bool (True when forbidden words appear but novelty is not electronics)
      - message: short human-readable explanation

    The check looks for the === NOVELTY SUMMARY === section and inspects the
    `Primary Novel Feature:` line to determine whether the invention's novelty
    is about electronics. Matching is case-insensitive and word-boundary based.
    """
    forbidden = forbidden_list or FORBIDDEN_ELECTRONICS
    text = output_text or ""

    # Only apply this guard for Optics-like domains
    if not patent_type or "optic" not in patent_type.lower():
        return {
            "forbidden_found": [],
            "novelty_mentions_electronics": False,
            "flag": False,
            "message": "Domain not Optics; no electronics guard applied.",
        }

    # Extract NOVELTY SUMMARY primary line if present
    novelty_block = ""
    m = re.search(r"=== NOVELTY SUMMARY ===(.*?)=== QUESTIONS ===", text, re.DOTALL | re.IGNORECASE)
    if m:
        novelty_block = m.group(1)
    else:
        # Fallback: look for the heading and take a small window
        m2 = re.search(r"=== NOVELTY SUMMARY ===(.*)", text, re.DOTALL | re.IGNORECASE)
        if m2:
            novelty_block = m2.group(1)

    primary_text = ""
    if novelty_block:
        pm = re.search(r"Primary Novel Feature:\s*(.*)", novelty_block, re.IGNORECASE)
        if pm:
            primary_text = pm.group(1).strip()

    # Determine whether novelty mentions electronics-related terms
    electronics_terms = ["circuit", "electr", "voltage", "current", "pcb", "emc", "impedance", "ohm"]
    novelty_mentions = False
    for term in electronics_terms:
        if re.search(rf"\b{re.escape(term)}", primary_text, re.IGNORECASE):
            novelty_mentions = True
            break

    # Find forbidden keywords anywhere in the output
    found = []
    for kw in forbidden:
        if re.search(rf"\b{re.escape(kw)}\b", text, re.IGNORECASE):
            found.append(kw)

    flag = False
    message = ""
    if found and not novelty_mentions:
        flag = True
        message = (
            "Forbidden electronics keywords found in Optics output while the Primary Novel Feature "
            "does not mention electronics."
        )
    else:
        message = "No problematic occurrences detected." if not found else "Forbidden keywords found but novelty mentions electronics."

    return {
        "forbidden_found": sorted(set(found)),
        "novelty_mentions_electronics": novelty_mentions,
        "flag": flag,
        "message": message,
    }
