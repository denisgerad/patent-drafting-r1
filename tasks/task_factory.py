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


_CHECKLISTS_PATH = cfg.PATENT_TYPES_JSON.parent / "product_type_checklists.json"


def _load_product_checklists() -> dict:
    try:
        return json.loads(_CHECKLISTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _match_product_checklist(field_of_invention: str) -> list[dict]:
    """
    Match the field_of_invention string against product-type trigger patterns.
    Returns list of matched expert_categories (each is {name, questions}).

    Supports two matching modes per checklist entry:
    - require_all_triggers: true  → ALL triggers must match (AND logic)
    - require_all_triggers: false → ANY trigger matches (OR logic, default)

    This is the KEY mechanism that restores expert-depth questions:
    - field_of_invention provides SCOPE (right topic)
    - product_type_checklists provide DEPTH (right questions for that product type)
    Both are needed. Neither alone is sufficient.
    """
    checklists = _load_product_checklists()
    matched_categories = []
    seen_names = set()

    for key, entry in checklists.items():
        if key.startswith("_"):
            continue
        triggers = entry.get("triggers", [])
        require_all = entry.get("require_all_triggers", False)

        if require_all:
            matched = bool(triggers) and all(
                re.search(t, field_of_invention, re.IGNORECASE) for t in triggers
            )
        else:
            matched = any(
                re.search(t, field_of_invention, re.IGNORECASE) for t in triggers
            )

        if matched:
            logger.info("[CHECKLIST] Matched '%s' for field: %s", key, field_of_invention[:120])
            for cat in entry.get("expert_categories", []):
                if cat["name"] not in seen_names:
                    matched_categories.append(cat)
                    seen_names.add(cat["name"])

    if not matched_categories:
        logger.info("[CHECKLIST] No match — using fallback gap checklist for field: %s", field_of_invention[:120])

    return matched_categories


def _get_fixed_theme_names(field_of_invention: str) -> list[str] | None:
    """Return the fixed_theme_names list for the first matched checklist entry that has one, or None."""
    checklists = _load_product_checklists()
    for key, entry in checklists.items():
        if key.startswith("_"):
            continue
        triggers = entry.get("triggers", [])
        require_all = entry.get("require_all_triggers", False)
        if require_all:
            matched = bool(triggers) and all(
                re.search(t, field_of_invention, re.IGNORECASE) for t in triggers
            )
        else:
            matched = any(
                re.search(t, field_of_invention, re.IGNORECASE) for t in triggers
            )
        if matched and entry.get("fixed_theme_names"):
            return entry["fixed_theme_names"]
    return None


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
    novelty: str = "NOT STATED",
    document_gaps: str = "",
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
        if novelty and novelty != "NOT STATED":
            focus_hint = f"""
⚠ NOVELTY ANCHOR — READ BEFORE GENERATING QUESTIONS:
The extracted novelty of THIS invention is:
  "{novelty}"
Every theme and every question MUST be traceable to this specific novelty.
Do NOT generate questions about components or behaviours outside this scope.
If the checklist above contains questions irrelevant to this novelty, skip them.
DOMAIN LENS ({patent_type}) — apply only where this novelty explicitly touches these:
{bullets}
"""
        else:
            focus_hint = f"""
⚠ NOVELTY ANCHOR — No formal novelty statement was found. Infer the likely novel aspects from the field description and claim structure, and apply these focus areas to those inferred aspects. Do NOT treat absent novelty as permission to ask generic questions — anchor every question to a specific technical gap in THIS document:
DOMAIN LENS ({patent_type}):
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

    # ── Product-type expert checklist (restores depth without losing scope) ───────
    # Match the field_of_invention text against the product-type library.
    # The checklist questions are used as MANDATORY EXAMPLES injected into the
    # prompt — they tell the LLM what a domain expert would always ask about this
    # type of product, independent of what the inventor wrote in the document.
    matched_categories = _match_product_checklist(field_of_invention) if field_of_invention else []

    checklist_block = ""
    if matched_categories:
        lines = [
            "PRODUCT-TYPE EXPERT CHECKLIST",
            "The following categories and example questions represent what a domain expert",
            "ALWAYS asks about this type of invention, regardless of what the document says.",
            "Use these as MANDATORY theme templates. Add them to your output even if the",
            "document does not address them — they represent the GAPS the inventor must fill.",
            "",
        ]
        for cat in matched_categories:
            mandatory_prefix = ""
            if cat.get("mandatory"):
                mandatory_prefix = "\n⚠ MANDATORY — you MUST generate at least one question from this category:\n"
            lines.append(f"{mandatory_prefix}[{cat['name']}]")
            for q in cat["questions"]:
                lines.append(f"  EXAMPLE: {q}")
            lines.append("")
        checklist_block = "\n".join(lines)
    else:
        checklist_block = """
MANDATORY DOCUMENT GAP CHECKLIST (no product-type checklist matched):
For each of the following, ask a question ONLY IF the document does not provide a specific answer:
- Exact numeric performance targets (latency, throughput, accuracy %)
- Specific third-party components, APIs, or protocols named
- Data flow between the primary components described
- Error handling and fallback behaviour
- Hardware or OS platform constraints
Every question must quote or reference a specific passage from the document that reveals the gap.
"""

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
    if novelty and novelty != "NOT STATED":
        prefilled_novelty_line = f'NOVELTY (extracted): "{novelty}"'
    else:
        prefilled_novelty_line = 'NOVELTY (verbatim): "NOT STATED — no formal novelty statement in document"'

    if document_gaps:
        document_gaps_block = f"""
=== DOCUMENT GAP LIST (pre-analysed — mandatory question targets) ===
The following gaps were identified in this specific document.
You MUST generate at least one question per GAP item below.
{document_gaps}
=== END GAP LIST ===
"""
    else:
        document_gaps_block = ""

    # Extract document title from first non-blank line of context
    doc_title = ""
    for line in context.strip().splitlines():
        line = line.strip()
        if line and len(line) > 10:
            doc_title = line
            break

    # Build Step 1 instruction — use fixed theme names if the matched checklist defines them
    _fixed_themes = _get_fixed_theme_names(field_of_invention) if field_of_invention else None
    if _fixed_themes:
        _theme_list = "\n".join(f"  Theme {i + 1}: {name}" for i, name in enumerate(_fixed_themes))
        step1_block = (
            f"Step 1: Generate EXACTLY {len(_fixed_themes)} themes using THESE EXACT NAMES — no substitutions:\n"
            f"{_theme_list}\n"
            f"Do NOT replace any of these with Privacy, Security, Error Handling,\n"
            f"Scalability, or any other topic not in this list.\n"
            f"If the document does not describe a theme's sub-system in detail, still\n"
            f"include the theme and ask what is missing — that absence IS the gap."
        )
    else:
        step1_block = (
            "Step 1: Identify EXACTLY 5 technical themes — one per major sub-system described in the\n"
            "  Field above. Count the distinct sub-systems named in the Field and Novelty lines;\n"
            "  assign one theme to each. If the document does not describe a sub-system in detail,\n"
            "  still create the theme and ask what is missing — that absence IS the gap.\n"
            "  Use concrete sub-system names drawn verbatim or near-verbatim from the Field text."
        )

    description = f""" conducting a technical enablement review under 35 U.S.C. § 112.

╔══════════════════════════════════════════════════════════════════╗
  PATENT UNDER REVIEW : {doc_title}
  FIELD OF INVENTION  : {field_core if field_core else "(see document)"}
  PATENT DOMAIN       : {patent_type}
╚══════════════════════════════════════════════════════════════════╝

DOCUMENT (read every word before generating output):
=== BEGIN DOCUMENT ===
{context}
=== END DOCUMENT ===

⚠ CRITICAL DOMAIN LOCK:
This patent is EXCLUSIVELY about: "{field_core[:180] if field_core else doc_title}"
The word "television" means TV screens and broadcast content — NOT telephone calls.
The word "channels" means communication platforms (WhatsApp, web, voice) — NOT phone lines.
If any theme or question references telephone calls, call duration, or IVR systems,
it is WRONG and must not be generated.
Every theme and every question MUST be about THIS invention and no other.
{prohibition_block}{checklist_block}{document_gaps_block}{focus_hint}
🚫 PROHIBITED QUESTIONS — do not generate any question that:
  • Asks how many epochs, layers, or parameters the model has
  • Asks what training data was used (unless document claims custom training)
  • Asks how often the system is tested, audited, or reviewed
  • Asks what the evaluation metric or benchmark score is
  • Can be answered with only "yes" or "no"
  • Is about general technology category behaviour, not THIS document's gaps
  • Names a specific ML technique (reinforcement learning, active learning,
    transfer learning) unless that exact term appears in the document —
    ask WHAT technique is used instead of naming one
🚫 TECHNOLOGY ASSUMPTION PROHIBITION:
  Never name a specific technology, protocol, provider, or architecture
  in a question unless that exact term appears in the document.
  WRONG: "Does the system use OAuth or OpenID Connect?"
  RIGHT: "Which identity provider or protocol does the system use?"
  WRONG: "Is it an encoder-decoder or transformer architecture?"
  RIGHT: "What is the architecture of the LLM used?"
  WRONG: "Are social media accounts used for authentication?"
  RIGHT: "What identity sources does the system accept?"
🚫 PROHIBITED THEMES — do not generate a theme named or focused on:
  • Security, Privacy, or Data Protection (unless explicitly claimed)
  • Error Handling or Fault Tolerance (unless explicitly claimed)
  • Compliance, Regulation, or Audit
Each question must be answerable only by the inventor — not by a textbook.

RE-READ BEFORE GENERATING:
Field: {field_core[:300] if field_core else doc_title}
Novelty: {novelty[:200] if novelty and novelty != "NOT STATED" else "not stated — infer from field"}

TERM DEFINITIONS FOR THIS DOCUMENT ONLY:
- "multimodal" = text + image + audio processed simultaneously by an LLM
                  NOT "multiphone", NOT voice-only, NOT telephone commands
- "communication channels" = platforms (WhatsApp, web app, REST API)
                              NOT phone lines, NOT voice channels
- "LLM" = large language model for content understanding
           NOT a speech recognition engine

THEME NAMING RULE — MANDATORY:
Each theme name MUST be taken from a component, sub-system, or process
named in the Field or Novelty above.
BANNED theme names: "Theme 1", "Theme Name 1", "Technical Theme",
                    "System Architecture", "Overview"
Use terms drawn verbatim or near-verbatim from: "{field_core[:120] if field_core else doc_title}"
Every theme name must use a term from the Field or Novelty lines above.

TASK — Generate a technical enablement gap analysis for the patent above.

{step1_block}

Step 2: For each theme write EXACTLY 3 to 5 questions (no more, no fewer) that:
  • Reference a specific gap or missing detail in the document above
  • Ask for numeric values, governing equations, test procedures, or drawings
  • Are traceable to a phrase actually present in the document
  • Do NOT ask about surrounding products or platforms unless explicitly claimed
{units_note}
Start your response with EXACTLY these two lines — do not alter them:
{prefilled_field_line}
{prefilled_novelty_line}

Then grouped questions:
[Theme Name 1]
1. ...
2. ...

[Theme Name 2]
1. ...
...
"""

    return Task(
        description=description,
        agent=scrutinizer,
        expected_output=(
            f"Two verbatim header lines (FIELD and NOVELTY as provided), "
            f"then 3-5 named theme sections with 3-5 enablement questions each, "
            f"all grounded in the {patent_type} document content."
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
