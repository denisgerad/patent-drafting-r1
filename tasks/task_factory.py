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
        for trigger in entry.get("triggers", []):
            if re.search(trigger, field_of_invention, re.IGNORECASE):
                for cat in entry.get("expert_categories", []):
                    if cat["name"] not in seen_names:
                        matched_categories.append(cat)
                        seen_names.add(cat["name"])
                break  # one trigger match per product type is enough

    return matched_categories


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
    mechanism: str = "",
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

    # ── Product-type expert checklist (restores depth without losing scope) ───────
    # Match the field_of_invention text against the product-type library.
    # The checklist questions are used as MANDATORY EXAMPLES injected into the
    # prompt — they tell the LLM what a domain expert would always ask about this
    # type of product, independent of what the inventor wrote in the document.
    matched_categories = _match_product_checklist(field_of_invention) if field_of_invention else []

    checklist_block = ""
    if matched_categories:
        # Build the section names list for the hard constraint instruction
        section_names = [f'[{cat["name"]}]' for cat in matched_categories]
        sections_list = "\n  ".join(section_names)

        lines = [
            "MANDATORY OUTPUT STRUCTURE — LOCKED SECTION HEADINGS",
            "═" * 55,
            "Your output MUST contain EXACTLY these section headings,",
            "in this order. This is a hard constraint, not a suggestion:",
            f"  {sections_list}",
            "",
            "RULES:",
            "  - Do NOT add sections not in this list.",
            "  - Do NOT drop any section from this list.",
            "  - Do NOT rename any section.",
            "  - Do NOT merge two sections into one.",
            "  - Each section MUST appear even if the document says nothing",
            "    about it — the absence of information IS the gap to fill.",
            "",
            "FOR EACH SECTION, the following are REQUIRED QUESTION FORMATS.",
            "Your questions must match or exceed this level of specificity.",
            "Do not produce a vaguer version of these questions:",
            "",
        ]
        for cat in matched_categories:
            lines.append(f"[{cat['name']}]")
            for q in cat["questions"]:
                lines.append(f"  REQUIRED FORMAT: {q}")
            lines.append("")
        lines.append("═" * 55)
        checklist_block = "\n".join(lines)

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

    # Supplement mechanism from parameter if workflow extracted one
    # This provides the pre-computed mechanism to the prompt directly,
    # so STEP 0 has a starting point rather than extracting from scratch.
    mechanism_hint = ""
    if mechanism.strip():
        mechanism_hint = f"""
PRE-EXTRACTED MECHANISM (the enabling technical feature):
  "{mechanism.strip()}"
This was identified from the document before the LLM call.
Use this as your MECHANISM line in STEP 0 output — verify it against
the document and refine if you find a more specific description.
"""

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
{app_block}{mechanism_hint}
MANDATORY: Every theme and every question must be traceable to the FIELD
and MECHANISM above. Discard any question that cannot be traced to these.
"""

    description = f"""
You are a senior patent expert conducting a 35 U.S.C. Section 112 enablement review.
Patent Domain: {patent_type}

=== DOCUMENT TO ANALYSE ===
{context}
===========================
{field_anchor_block}
{checklist_block}
MANDATORY STEP 0 — Verbatim Field Extraction
Before writing anything else, extract THREE things from the document:

  (a) FIELD — Copy verbatim the FIELD OF INVENTION sentence.
      If not found: FIELD (verbatim): "NOT FOUND IN DOCUMENT"

  (b) NOVELTY — Copy verbatim the PRIMARY NOVEL CONTRIBUTION sentence IF
      it exists in the document. If absent, write exactly:
      NOVELTY (verbatim): "NOT STATED — invention is [device/method from field]"
      DO NOT invent or infer novelty from your training data.

  (c) MECHANISM — Identify the specific technical feature or component that
      ENABLES the novelty. This is the HOW, not the WHAT.
      Ask yourself: "What physical structure, algorithm step, or technical
      mechanism makes this invention work?"
      Examples of correct mechanism extraction:
        Field: "single universal LGP for dual-edge backlight operation"
          → MECHANISM: "dot pattern / light extraction structure on the LGP surface"
        Field: "flexible polymer heater film bonded to an LCD"
          → MECHANISM: "resistive heating element embedded in the polymer film"
        Field: "time-limited verification codes mapped to identity representation"
          → MECHANISM: "code generation algorithm and identity mapping method"
      If the mechanism is not explicitly stated, write the most specific
      technical noun phrase you can identify from the document.
      DO NOT write the system name or the application — write the component
      or method that does the work.

Label exactly:
  FIELD (verbatim): "..."
  NOVELTY (verbatim): "..."
  MECHANISM: "..."

IMPORTANT: ALL questions about depth and technical detail in STEP 3
must be traceable to the MECHANISM line, not the NOVELTY line.
The MECHANISM is the source of the deep enablement questions.

CRITICAL HALLUCINATION GUARD — READ BEFORE WRITING THE NOVELTY LINE:
The document above is the ONLY permitted source for the NOVELTY line.
Apply this self-check before writing it:

  SELF-CHECK: For every distinct phrase in your proposed NOVELTY sentence,
  ask: "Can I find this phrase, or a close paraphrase, in the document text
  shown above?" If the answer is NO for the majority of phrases, the sentence
  is hallucinated from your training data. Write "NOT STATED" instead.

  KNOWN HALLUCINATION TRIGGERS to watch for — these are topics that an LLM
  commonly invents when it cannot find a novelty statement:
    * Any authentication method NOT mentioned in the document
      (biometrics, fingerprint, facial recognition, voice recognition)
    * Any optical metric NOT in the document
      (luminance uniformity %, optical extraction efficiency, LGP dot pattern)
    * Any performance improvement percentage NOT quoted from the document
    * Any material, component, or standard NOT named in the document

  RULE: If in doubt, write "NOT STATED". A wrong NOVELTY line causes every
  downstream question to be about the wrong invention. "NOT STATED" is safe.

STEP 1 — Identify Gaps in the Stated Invention

PART A — The Invention Itself:
  Ask ONLY about the MECHANISM identified in STEP 0.
  What technical details about that mechanism are missing that a skilled
  person needs to build or replicate it?

  SCOPE RULE — TWO LEVELS:

  Level 1 (Field scope): Do NOT ask about anything not named in the FIELD line.
    If field says "verification codes mapped to identity representation"
      → do NOT ask about biometrics or facial recognition
    If field says "heater film bonded to LCD"
      → do NOT ask about LCD resolution or panel brightness

  Level 2 (Mechanism scope): Do NOT ask about the system AROUND the mechanism.
    The mechanism is the novel part. Everything else is context.
    Ask about the mechanism's own properties — not the things it connects to.
    Examples:
      MECHANISM: "dot pattern on LGP surface"
        CORRECT: dot geometry, spacing, depth, density gradient, governing formula
        WRONG:   backlight source power, LCD panel size, thermal management,
                 mode switching timing — these surround the LGP, they are not it
      MECHANISM: "resistive heating element in polymer film"
        CORRECT: element geometry, sheet resistance, activation temperature
        WRONG:   LCD resolution, LCD backlight voltage, display connectors
      MECHANISM: "TOTP code generation algorithm"
        CORRECT: algorithm inputs, time window, key derivation, code length
        WRONG:   UI design, server hardware, network topology

PART B — Application Context (ONLY include if the field statement explicitly
  names deployment environments or use conditions):
  If the field statement says where or when the invention is used, generate
  one dedicated theme covering:
    - Operational conditions imposed by those specific environments
      (use the exact environment names from the field statement)
    - Qualification standards relevant to those environments
    - Performance requirements driven by those use conditions
  Do NOT invent deployment environments not stated in the field.
{focus_hint}{prohibition_block}
STEP 2 — Derive 3 to 5 Theme Names
Name themes after the invention's own sub-systems — derived from the FIELD line.
  Good theme naming rule: take the key nouns from the field statement and make
  them into theme names. For example:
    Field: "time-limited verification codes mapped to an identity representation"
      → "Verification Code Generation & Lifecycle"
      → "Identity Representation & Mapping"
      → "Security Model & Threat Analysis"
      → "Integration & Deployment Requirements"
    Field: "flexible polymer heater film bonded to an LCD"
      → "Polymer Film Composition & Form Factor"
      → "Heating Element Geometry & Power"
      → "Film Bonding & Interface Method"
      → "Environmental Qualification & Operating Range"
  Bad theme names (too generic for any invention):
    "Specifications", "Components", "System Design", "Technical Details"
  If the field statement explicitly names deployment environments, include
  one theme covering operational requirements for those environments.

STEP 3 — Write 3 to 5 Questions per Theme
Each question MUST:
  - Be traceable to a specific property or gap of the MECHANISM (from STEP 0)
  - Request ONE of: drawings/diagrams, governing equations, numeric values
    with units, fabrication procedures, or measured performance comparisons
  - Name the specific parameter, component, or condition being asked about
    (never use open-ended phrasing like "describe the design" or "explain how")
  - Match or exceed the specificity level of the REQUIRED FORMAT questions above

DEPTH TEST — before writing each question, ask:
  "Could an inventor answer this with a single number, formula, drawing,
   or yes/no + specification?" If yes, the question has the right depth.
  "Would any inventor of any product be able to answer this?" If yes,
   the question is too generic — name the specific element.

EXAMPLES of wrong vs right depth:
  WRONG: "What material is the light guide made of?"
  RIGHT: "What is the polymer substrate, refractive index (at 550nm),
          and thickness (mm) of the light guide plate?"

  WRONG: "How does the pattern ensure uniform illumination?"
  RIGHT: "Provide the dot size variation (µm) from edge to centre and
          the mathematical relationship between dot density and extraction
          efficiency — is there a governing formula or lookup table?"

  WRONG: "How is the backlight source connected?"
  RIGHT: [DO NOT ASK — backlight source is not the MECHANISM]
{units_note}

OUTPUT FORMAT (strict):
First three lines:
  FIELD (verbatim): "..."
  NOVELTY (verbatim): "..."
  MECHANISM: "..."

Then blank line, then grouped questions using ONLY the locked section
headings from the MANDATORY OUTPUT STRUCTURE above (if checklist exists),
or theme names derived from the MECHANISM (if no checklist):

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
            "Verbatim FIELD and NOVELTY lines (NOVELTY written as 'NOT STATED' if absent). "
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
