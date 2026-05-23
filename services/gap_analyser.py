"""
services/gap_analyser.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Phase C — Gap analysis engine.

Compares DraftProfile against ReferenceProfile across three passes:

  Pass 1 — Structural gaps (section-level)
    Sections present in reference, absent or thin in draft.

  Pass 2 — Claim topic gaps
    Topics in reference independent claims not mentioned in draft.

  Pass 3 — Technical parameter gaps
    Numeric parameters in reference not mentioned in draft.

Output: GapReport — structured list of gaps with severity, explanation
template, and a "what the reference says" excerpt for each.

The LLM (Phase D workflow) uses GapReport to:
  - Generate "why it matters" explanations per gap
  - Generate targeted questions for confirmed gaps (Step 1d)

Domain expert (Step 1c HIL) reviews GapReport and marks each gap:
  RELEVANT | ALREADY_COVERED | NOT_APPLICABLE

Design: deterministic Python only. No LLM calls. Fast (< 1 second).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from services.patent_chunker import ReferenceProfile, DraftProfile

log = logging.getLogger(__name__)


# ── Severity levels ───────────────────────────────────────────────────────────

class GapSeverity(str, Enum):
    CRITICAL  = "CRITICAL"    # likely examiner rejection
    IMPORTANT = "IMPORTANT"   # weakens the application
    OPTIONAL  = "OPTIONAL"    # good to have, not blocking


class GapType(str, Enum):
    STRUCTURAL  = "STRUCTURAL"   # section missing entirely
    CLAIM_TOPIC = "CLAIM_TOPIC"  # claimed concept not addressed
    TECHNICAL   = "TECHNICAL"    # numeric parameter not specified
    ENABLEMENT  = "ENABLEMENT"   # drawings/examples not provided


class HILDecision(str, Enum):
    PENDING           = "PENDING"
    RELEVANT          = "RELEVANT"
    ALREADY_COVERED   = "ALREADY_COVERED"
    NOT_APPLICABLE    = "NOT_APPLICABLE"


# ── Gap data structures ───────────────────────────────────────────────────────

@dataclass
class Gap:
    """One identified gap between draft and reference patent."""
    gap_id:        str          # unique ID e.g. "struct_001"
    gap_type:      GapType
    severity:      GapSeverity
    title:         str          # short display title e.g. "Missing: Detailed Description"
    description:   str          # what is missing and why it matters (template)
    reference_says: str         # what the reference patent says about this topic
    draft_says:    str          # what (little) the draft says, or "Not mentioned"

    # Set by domain expert in HIL review (Step 1c)
    hil_decision:  HILDecision  = HILDecision.PENDING
    hil_note:      str          = ""  # domain expert direction note

    # Set by LLM in Phase D
    why_it_matters: str = ""    # LLM-generated explanation
    questions:     list[str]    = field(default_factory=list)  # targeted questions


@dataclass
class GapReport:
    """
    Complete gap analysis report comparing one draft against one reference.
    Input to the HIL review UI (Step 1c) and question generation (Step 1d).
    """
    draft_title:      str
    reference_id:     str
    reference_title:  str
    gaps:             list[Gap]
    draft_word_count: int
    generated_at:     str = ""

    @property
    def critical_gaps(self) -> list[Gap]:
        return [g for g in self.gaps if g.severity == GapSeverity.CRITICAL]

    @property
    def confirmed_gaps(self) -> list[Gap]:
        return [g for g in self.gaps if g.hil_decision == HILDecision.RELEVANT]

    @property
    def pending_gaps(self) -> list[Gap]:
        return [g for g in self.gaps if g.hil_decision == HILDecision.PENDING]

    @property
    def readiness_summary(self) -> str:
        n_critical  = len(self.critical_gaps)
        n_total     = len(self.gaps)
        n_confirmed = len(self.confirmed_gaps)
        if n_critical == 0:
            return f"No critical gaps. {n_total} total gaps identified."
        return (f"{n_critical} critical gap(s) of {n_total} total. "
                f"{n_confirmed} confirmed by domain review.")

    def gaps_by_type(self) -> dict[str, list[Gap]]:
        result: dict[str, list[Gap]] = {}
        for g in self.gaps:
            key = g.gap_type.value
            result.setdefault(key, []).append(g)
        return result


# ── Main gap analysis function ────────────────────────────────────────────────

def analyse_gaps(
    reference: ReferenceProfile,
    draft: DraftProfile,
    draft_title: str = "Draft document",
) -> GapReport:
    """
    Compare draft against reference and produce a structured GapReport.

    Pass 0 — Draft-only structural check (always runs, regardless of reference quality)
             Flags sections every patent needs that are absent from the draft.

    Pass 1 — Reference comparison: sections present in reference, absent in draft
    Pass 2 — Claim topic gaps
    Pass 3 — Technical parameter gaps
    Pass 4 — Enablement gaps (drawings)

    Pass 0 ensures useful gaps are reported even when the reference patent
    has limited full-text (e.g. US published applications via EPO OPS).
    """
    from datetime import datetime, timezone
    log.info(
        "Gap analysis: reference=%s, draft=%d words",
        reference.display_id, draft.total_word_count,
    )

    gaps: list[Gap] = []
    counter = [0]

    def next_id(prefix: str) -> str:
        counter[0] += 1
        return f"{prefix}_{counter[0]:03d}"

    # ══════════════════════════════════════════════════════════════════════════
    # PASS 0 — Draft-only structural check (always runs)
    # Every patent application needs these sections regardless of reference.
    # ══════════════════════════════════════════════════════════════════════════
    mandatory_sections = {
        "Detailed Description": (
            GapSeverity.CRITICAL,
            "The Detailed Description is mandatory in every patent application. "
            "It must describe the invention in enough detail that a skilled person "
            "can make and use it (35 U.S.C. §112). This is the largest and most "
            "important section of a patent. The draft does not have one.",
            "All granted patents have a Detailed Description section describing "
            "the invention's structure, components, operation, and preferred embodiments.",
        ),
        "Claims": (
            GapSeverity.CRITICAL,
            "Claims define the legal scope of the patent — what competitors cannot "
            "copy without a licence. A patent application without claims cannot be "
            "examined or granted. Claims must be drafted by a patent attorney.",
            "All granted patents have at least one independent claim defining "
            "the novel contribution of the invention.",
        ),
        "Abstract": (
            GapSeverity.IMPORTANT,
            "The Abstract is required by patent offices worldwide. It summarises "
            "the invention in 150 words or fewer and appears in patent databases. "
            "Examiners and prior art searchers use it to assess relevance.",
            "All granted patents have a concise abstract summarising the invention.",
        ),
        "Background of the Invention": (
            GapSeverity.IMPORTANT,
            "The Background section explains the technical problem being solved "
            "and distinguishes the invention from prior art. It supports the "
            "argument for novelty and non-obviousness.",
            "Granted patents include a Background section describing prior art "
            "limitations that the invention overcomes.",
        ),
    }

    for section_name, (severity, description, reference_says) in mandatory_sections.items():
        draft_section = draft.sections.get(section_name)
        if not draft_section or not draft_section.present:
            gaps.append(Gap(
                gap_id        = next_id("mandatory"),
                gap_type      = GapType.STRUCTURAL,
                severity      = severity,
                title         = f"Missing: {section_name}",
                description   = description,
                reference_says = reference_says,
                draft_says    = "Not present in draft.",
            ))

    # Check figures separately
    if not draft.has_figures_mentioned:
        gaps.append(Gap(
            gap_id        = next_id("mandatory"),
            gap_type      = GapType.ENABLEMENT,
            severity      = GapSeverity.CRITICAL,
            title         = "Missing: Technical Drawings / Figures",
            description   = (
                "Patent applications for physical inventions almost always require "
                "technical drawings. Drawings are referenced throughout the Detailed "
                "Description and Claims. Without drawings, the examiner cannot "
                "verify that the invention is fully described."
            ),
            reference_says = (
                "Granted patents for physical devices include cross-section drawings, "
                "exploded views, and schematic diagrams showing the invention's structure."
            ),
            draft_says = "No figures or drawings mentioned in draft.",
        ))

    # Flag if draft is very thin overall
    if draft.total_word_count < 200:
        gaps.append(Gap(
            gap_id       = next_id("mandatory"),
            gap_type     = GapType.STRUCTURAL,
            severity     = GapSeverity.CRITICAL,
            title        = f"Draft is very thin ({draft.total_word_count} words)",
            description  = (
                f"The draft document contains only {draft.total_word_count} words. "
                f"A patent application typically requires 2,000-10,000+ words to "
                f"fully describe the invention. This draft needs significant expansion "
                f"before it can form the basis of a patent application."
            ),
            reference_says = (
                "A complete patent application typically has 2,000-15,000 words "
                "covering all sections from Abstract through Claims."
            ),
            draft_says = f"Draft: {draft.total_word_count} words total.",
        ))

    reference_has_fulltext = (
        reference.has_description or reference.has_claims
    )
    log.info(
        "Pass 0 complete: %d mandatory gaps | reference_has_fulltext=%s",
        len(gaps), reference_has_fulltext,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PASS 1 — Reference comparison: sections present in reference, absent in draft
    # Only runs when the reference patent has full text available.
    # ══════════════════════════════════════════════════════════════════════════
    if not reference_has_fulltext:
        log.info("Pass 1-4 skipped: reference patent has no full text (abstract only)")
        # Still run Pass 5 even when reference has no full text
        checklist_gaps = _analyse_technical_gaps(draft, reference.display_id)
        gaps.extend(checklist_gaps)
        log.info("Pass 5 (early path): %d checklist gaps", len(checklist_gaps))
        severity_order = {GapSeverity.CRITICAL: 0, GapSeverity.IMPORTANT: 1, GapSeverity.OPTIONAL: 2}
        gaps.sort(key=lambda g: severity_order[g.severity])
        return GapReport(
            draft_title      = draft_title,
            reference_id     = reference.display_id,
            reference_title  = reference.title,
            gaps             = gaps,
            draft_word_count = draft.total_word_count,
            generated_at     = datetime.now(timezone.utc).isoformat(),
        )

    # Section importance hierarchy for patents
    section_severity = {
        "Detailed Description":           GapSeverity.CRITICAL,
        "Claims":                         GapSeverity.CRITICAL,
        "Abstract":                       GapSeverity.IMPORTANT,
        "Background of the Invention":    GapSeverity.IMPORTANT,
        "Summary of the Invention":       GapSeverity.IMPORTANT,
        "Brief Description of Drawings":  GapSeverity.IMPORTANT,
    }

    # ══════════════════════════════════════════════════════════════════════════
    # PASS 1 — Structural gaps
    # ══════════════════════════════════════════════════════════════════════════

    # Section importance hierarchy for patents
    section_severity = {
        "Detailed Description":           GapSeverity.CRITICAL,
        "Claims":                         GapSeverity.CRITICAL,
        "Abstract":                       GapSeverity.IMPORTANT,
        "Background of the Invention":    GapSeverity.IMPORTANT,
        "Summary of the Invention":       GapSeverity.IMPORTANT,
        "Brief Description of Drawings":  GapSeverity.IMPORTANT,
    }

    for section_name, ref_section in reference.sections.items():
        if not ref_section.present:
            # Reference doesn't have it either — skip
            continue

        draft_section = draft.sections.get(section_name)
        draft_present = draft_section is not None and draft_section.present

        if not draft_present:
            severity = section_severity.get(section_name, GapSeverity.OPTIONAL)

            # Determine what the reference says
            if section_name == "Claims":
                ref_excerpt = (
                    f"Reference has {reference.claims.total_count} claims "
                    f"({reference.claims.independent_count} independent). "
                    f"Sample: '{reference.claims.sample_claim[:200]}'"
                )
            elif ref_section.excerpt:
                ref_excerpt = f"Reference section begins: '{ref_section.excerpt[:200]}'"
            else:
                ref_excerpt = f"Reference has {ref_section.word_count} words in this section."

            # Determine what draft says (if anything)
            if draft_section and draft_section.excerpt:
                draft_excerpt = f"Draft mentions: '{draft_section.excerpt[:150]}'"
            else:
                draft_excerpt = "Not present in draft."

            description = _structural_gap_description(
                section_name, ref_section, reference, draft
            )

            gaps.append(Gap(
                gap_id        = next_id("struct"),
                gap_type      = GapType.STRUCTURAL,
                severity      = severity,
                title         = f"Missing section: {section_name}",
                description   = description,
                reference_says = ref_excerpt,
                draft_says    = draft_excerpt,
            ))

        elif section_name == "Detailed Description":
            # Check if draft's description is significantly thinner than reference
            ref_words   = ref_section.word_count
            draft_words = draft_section.word_count if draft_section else 0
            if ref_words > 0 and draft_words < ref_words * 0.25:
                gaps.append(Gap(
                    gap_id       = next_id("struct"),
                    gap_type     = GapType.STRUCTURAL,
                    severity     = GapSeverity.CRITICAL,
                    title        = "Thin Detailed Description",
                    description  = (
                        f"The draft's Detailed Description is significantly shorter than the "
                        f"reference patent ({draft_words} vs {ref_words} words — "
                        f"less than 25% of reference depth). Patent examiners require a "
                        f"description that enables a skilled person to reproduce the invention."
                    ),
                    reference_says = (
                        f"Reference Detailed Description: {ref_words} words covering "
                        f"{len(ref_section.subsections)} subsections. "
                        f"Excerpt: '{ref_section.excerpt[:200]}'"
                    ),
                    draft_says = (
                        f"Draft description: {draft_words} words. "
                        f"Excerpt: '{draft_section.excerpt[:150]}'"
                        if draft_section else "Minimal description found."
                    ),
                ))

    # ══════════════════════════════════════════════════════════════════════════
    # PASS 2 — Claim topic gaps
    # ══════════════════════════════════════════════════════════════════════════

    if reference.claims.independent_count > 0:
        draft_text_lower = " ".join(draft.mentioned_topics).lower()
        # Also check draft section text
        for section in draft.sections.values():
            draft_text_lower += " " + section.excerpt.lower()

        missing_topics: list[str] = []
        for topic in reference.claims.independent_topics:
            # Fuzzy match: check if key words from topic appear in draft
            topic_words = [w for w in topic.lower().split() if len(w) > 3]
            if not topic_words:
                continue
            matched = any(w in draft_text_lower for w in topic_words)
            if not matched:
                missing_topics.append(topic)

        if missing_topics:
            # Group into one gap or split into individual gaps
            # For readability: one gap per missing topic if <= 5, grouped if more
            if len(missing_topics) <= 5:
                for topic in missing_topics:
                    gaps.append(Gap(
                        gap_id       = next_id("claim"),
                        gap_type     = GapType.CLAIM_TOPIC,
                        severity     = GapSeverity.IMPORTANT,
                        title        = f"Claim topic not addressed: {topic}",
                        description  = (
                            f"The reference patent's independent claims address '{topic}' "
                            f"but the draft document does not mention this concept. "
                            f"If this is part of the novel contribution, it must be "
                            f"disclosed in the Detailed Description before it can be claimed."
                        ),
                        reference_says = (
                            f"Reference independent claim covers: '{topic}'. "
                            f"Sample claim: '{reference.claims.sample_claim[:200]}'"
                        ),
                        draft_says = "Not found in draft.",
                    ))
            else:
                # Group all missing topics into one gap
                gaps.append(Gap(
                    gap_id       = next_id("claim"),
                    gap_type     = GapType.CLAIM_TOPIC,
                    severity     = GapSeverity.IMPORTANT,
                    title        = f"{len(missing_topics)} claim topics not addressed",
                    description  = (
                        f"The reference patent's independent claims address topics that "
                        f"the draft does not mention: {', '.join(missing_topics[:8])}. "
                        f"These concepts must be disclosed and described before they "
                        f"can be included in patent claims."
                    ),
                    reference_says = (
                        f"Reference independent claims cover: "
                        f"{', '.join(missing_topics[:8])}."
                    ),
                    draft_says = "None of these topics found in draft.",
                ))

    elif reference.has_claims:
        # Has claims but couldn't parse them — flag as generic gap
        if not draft.has_claims_section:
            gaps.append(Gap(
                gap_id       = next_id("claim"),
                gap_type     = GapType.CLAIM_TOPIC,
                severity     = GapSeverity.CRITICAL,
                title        = "No claims section in draft",
                description  = (
                    "The reference patent has a claims section defining the legal "
                    "scope of the invention. The draft has no claims section. "
                    "Claims must be drafted before filing."
                ),
                reference_says = (
                    f"Reference has {reference.claims.total_count} claims. "
                    f"Sample: '{reference.claims.sample_claim[:200]}'"
                ),
                draft_says = "No claims section detected in draft.",
            ))

    # ══════════════════════════════════════════════════════════════════════════
    # PASS 3 — Technical parameter gaps
    # ══════════════════════════════════════════════════════════════════════════

    if reference.parameters:
        # Check which reference parameters the draft mentions
        draft_full_text = " ".join(
            s.excerpt for s in draft.sections.values()
        ).lower()
        # Also include mentioned topics
        draft_full_text += " " + " ".join(draft.mentioned_topics)

        missing_params: list[str] = []
        for param in reference.parameters:
            # Check if parameter name key words appear in draft
            name_words = [w for w in param.name.lower().split() if len(w) > 3]
            if not name_words:
                continue
            found = any(w in draft_full_text for w in name_words)
            if not found:
                missing_params.append(f"{param.name} ({param.value} {param.unit})")

        if missing_params:
            severity = (
                GapSeverity.CRITICAL if len(missing_params) >= 3
                else GapSeverity.IMPORTANT
            )
            gaps.append(Gap(
                gap_id       = next_id("param"),
                gap_type     = GapType.TECHNICAL,
                severity     = severity,
                title        = f"{len(missing_params)} technical parameter(s) unspecified",
                description  = (
                    f"The reference patent specifies numeric values for technical "
                    f"parameters that the draft does not mention. Without these values, "
                    f"an examiner may reject the application for lack of enablement "
                    f"under 35 U.S.C. §112."
                ),
                reference_says = (
                    f"Reference specifies: {'; '.join(missing_params[:6])}."
                ),
                draft_says = (
                    "Draft mentions general concepts but no specific numeric values "
                    "for these parameters."
                    if draft.parameters_found
                    else "No numeric technical parameters found in draft."
                ),
            ))

    # ══════════════════════════════════════════════════════════════════════════
    # PASS 4 — Enablement gaps (drawings, examples)
    # ══════════════════════════════════════════════════════════════════════════

    if reference.has_figures and not draft.has_figures_mentioned:
        gaps.append(Gap(
            gap_id       = next_id("enable"),
            gap_type     = GapType.ENABLEMENT,
            severity     = GapSeverity.CRITICAL,
            title        = "No drawings or figures",
            description  = (
                "The reference patent includes technical drawings that are "
                "referenced throughout the Detailed Description. The draft has "
                "no figures. Patent applications for physical inventions almost "
                "always require drawings for enablement."
            ),
            reference_says = (
                f"Reference includes: {', '.join(reference.figures[:5])}."
                if reference.figures
                else "Reference has figures referenced in description."
            ),
            draft_says = "No figures or drawings mentioned or attached.",
        ))

    # ══════════════════════════════════════════════════════════════════════════
    # PASS 5 — Checklist-driven technical content gaps (always runs)
    # Uses product-type checklist to identify specific technical information
    # missing from the draft. Works without reference full-text.
    # ══════════════════════════════════════════════════════════════════════════
    checklist_gaps = _analyse_technical_gaps(draft, reference.display_id)
    gaps.extend(checklist_gaps)
    log.info("Pass 5 complete: %d checklist technical gaps", len(checklist_gaps))

    # Sort: CRITICAL first, then IMPORTANT, then OPTIONAL
    severity_order = {
        GapSeverity.CRITICAL:  0,
        GapSeverity.IMPORTANT: 1,
        GapSeverity.OPTIONAL:  2,
    }
    gaps.sort(key=lambda g: severity_order[g.severity])

    report = GapReport(
        draft_title      = draft_title,
        reference_id     = reference.display_id,
        reference_title  = reference.title,
        gaps             = gaps,
        draft_word_count = draft.total_word_count,
        generated_at     = datetime.now(timezone.utc).isoformat(),
    )

    log.info(
        "Gap analysis complete: %d gaps (%d critical, %d important)",
        len(gaps),
        len(report.critical_gaps),
        len([g for g in gaps if g.severity == GapSeverity.IMPORTANT]),
    )
    return report


# ── Gap description templates ─────────────────────────────────────────────────

def _structural_gap_description(
    section_name: str,
    ref_section,
    reference: ReferenceProfile,
    draft: DraftProfile,
) -> str:
    """Generate a clear description of why a structural gap matters."""

    templates = {
        "Detailed Description": (
            "The Detailed Description is the most critical section of a patent "
            "application. It must describe the invention in sufficient detail to "
            "enable a person skilled in the field to make and use the invention "
            "(35 U.S.C. §112). The reference patent's Detailed Description has "
            f"{ref_section.word_count} words. The draft lacks this section."
        ),
        "Claims": (
            "Claims define the legal scope of the patent — what competitors cannot "
            "make, use, or sell without a licence. A patent application without "
            "claims cannot be examined. The reference patent has "
            f"{reference.claims.total_count} claims covering "
            f"{', '.join(reference.claims.independent_topics[:4])}."
        ),
        "Abstract": (
            "The Abstract is required by the patent office and appears in patent "
            "databases. It must concisely summarise the invention in 150 words or "
            "fewer. Examiners and searchers use the abstract to assess relevance."
        ),
        "Background of the Invention": (
            "The Background section describes the technical problem being solved and "
            "distinguishes the invention from prior art. It establishes context for "
            "the examiner and supports the argument for novelty."
        ),
        "Summary of the Invention": (
            "The Summary provides a brief, high-level description of what the "
            "invention does. It bridges the Background and Detailed Description "
            "and often mirrors the independent claims."
        ),
        "Brief Description of Drawings": (
            "This section lists each figure and what it shows. It is required when "
            "drawings are submitted. Without it, the examiner cannot cross-reference "
            "the description text to the figures."
        ),
    }

    return templates.get(
        section_name,
        f"The reference patent includes a '{section_name}' section "
        f"({ref_section.word_count} words). This section is absent from the draft."
    )


# ── Pass 5: Checklist-driven technical gap analysis ──────────────────────────

def _analyse_technical_gaps(
    draft: "DraftProfile",
    reference_id: str,
) -> list[Gap]:
    """
    Pass 5 — Checklist-driven technical content gaps.

    Matches the draft's field_of_invention against product-type checklists
    to identify the expected technical topics for this invention type.
    For each expected topic that is absent from the draft, creates a
    technical gap entry with a specific description and targeted question.

    Works entirely from the existing product_type_checklists.json.
    No LLM call. No reference patent full-text required.
    """
    import json, re
    from pathlib import Path

    gaps: list[Gap] = []
    counter_base = 500  # offset to avoid ID collision with Passes 0-4

    # ── Load checklist ────────────────────────────────────────────────────────
    checklist_path = Path(__file__).parent.parent / "product_type_checklists.json"
    try:
        checklists = json.loads(checklist_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not load checklists for Pass 5: %s", exc)
        return []

    # ── Match product type from draft field_of_invention ─────────────────────
    field = draft.field_of_invention or ""
    # Also search draft topic mentions
    topic_text = field + " " + " ".join(draft.mentioned_topics)

    matched_key  = None
    matched_entry = None
    for key, entry in checklists.items():
        if key.startswith("_"):
            continue
        for trigger in entry.get("triggers", []):
            if re.search(trigger, topic_text, re.IGNORECASE):
                matched_key   = key
                matched_entry = entry
                break
        if matched_key:
            break

    if not matched_entry:
        log.info("Pass 5: no checklist match for field '%s'", field[:80])
        return []

    log.info("Pass 5: matched checklist '%s'", matched_key)

    # ── Get anti-patterns (topics NOT to ask about) ───────────────────────────
    anti_patterns = matched_entry.get("anti_patterns", [])

    # Build set of prohibited topic keywords from anti-patterns
    prohibited_keywords: set[str] = set()
    for ap in anti_patterns:
        # Extract key nouns from anti-pattern text
        words = re.findall(r"[a-z][a-z\-]{3,}", ap.lower())
        # Only use words that appear after "NOT" or "not"
        not_match = re.search(r"not\s+(?:ask\s+about\s+)?(.{10,60}?)(?:\s*—|\s*$)", ap, re.IGNORECASE)
        if not_match:
            prohibited_keywords.update(
                re.findall(r"[a-z][a-z\-]{3,}", not_match.group(1).lower())
            )

    # ── Draft full text for coverage checking ─────────────────────────────────
    # Exclude field_of_invention: the title contains key nouns (identity,
    # codes, system) that would falsely mark topics as covered.
    # Only check section body text and mentioned topics.
    draft_full = (
        " ".join(draft.mentioned_topics) + " "
        + " ".join(s.excerpt for s in draft.sections.values())
    ).lower()
    if len(draft_full.strip()) < 100 and draft.novelty_statement:
        draft_full += " " + draft.novelty_statement.lower()

    # ── Score each expert category ────────────────────────────────────────────
    categories = matched_entry.get("expert_categories", [])
    gap_counter = counter_base

    for cat in categories:
        cat_name  = cat.get("name", "")
        questions = cat.get("questions", [])

        if not questions:
            continue

        # Check which questions in this category are addressed by the draft
        addressed_count = 0
        missing_topics:  list[str] = []

        for q in questions:
            # Extract the key technical subject of this question
            # (the specific noun/parameter being asked about)
            subject = _extract_question_subject(q)
            if not subject:
                continue

            # Check if any word from the subject appears in draft
            subject_words = [w for w in subject.lower().split()
                             if len(w) > 3 and w not in prohibited_keywords]
            if not subject_words:
                continue

            found = any(w in draft_full for w in subject_words)
            if found:
                addressed_count += 1
            else:
                missing_topics.append(subject)

        # If more than half the category topics are missing → create a gap
        if not missing_topics:
            continue

        coverage_pct = addressed_count / len(questions) if questions else 1.0

        # Severity: CRITICAL if core invention mechanism category, IMPORTANT otherwise
        core_categories = {
            "Structural Design & Optical Architecture",
            "Heating Element Geometry & Resistivity",
            "Organic Emissive Stack & Device Architecture",
            "Physical Structure & Geometry",
            "Switching Topology & Operating Point",
            "Algorithm Specification & Completeness",
        }
        severity = (
            GapSeverity.CRITICAL
            if cat_name in core_categories or coverage_pct < 0.2
            else GapSeverity.IMPORTANT
        )

        # Build the gap title and description
        gap_counter += 1
        missing_display = missing_topics[:4]
        remaining = len(missing_topics) - len(missing_display)
        missing_str = ", ".join(missing_display)
        if remaining > 0:
            missing_str += f" (and {remaining} more)"

        # Get the first unanswered question as the depth example
        depth_example = ""
        for q in questions:
            subj = _extract_question_subject(q)
            if subj and subj in missing_topics[:1]:
                depth_example = q[:200]
                break
        if not depth_example and questions:
            depth_example = questions[0][:200]

        gaps.append(Gap(
            gap_id        = f"tech_{gap_counter:03d}",
            gap_type      = GapType.TECHNICAL,
            severity      = severity,
            title         = f"Missing technical detail: {cat_name}",
            description   = (
                f"The draft does not provide the technical information needed "
                f"to write the '{cat_name}' section of the Detailed Description. "
                f"Specifically missing: **{missing_str}**. "
                f"Without this, a patent attorney cannot draft claims covering "
                f"the invention's novel {cat_name.lower()} aspects."
            ),
            reference_says = (
                "Expert reference questions for this category require:\n"
                + "\n".join("• " + q[:120] for q in questions[:3])
            ),
            draft_says = (
                f"Draft covers {addressed_count}/{len(questions)} topics "
                f"({coverage_pct:.0%}) in this category. "
                f"Missing: {missing_str}."
            ),
        ))

    log.info(
        "Pass 5: %d technical gaps from checklist '%s'",
        len(gaps), matched_key,
    )
    return gaps


def _extract_question_subject(question: str) -> str:
    """
    Extract the core technical subject from a checklist question.
    Returns the most specific noun phrase — what the question is asking about.

    Examples:
      "What is the sheet resistivity (Ω/sq) at 20°C?"
        → "sheet resistivity"
      "Provide the dot size variation (µm) from edge to centre"
        → "dot size variation"
      "Describe the fabrication method used to realise the extraction pattern"
        → "fabrication method extraction pattern"
    """
    import re

    q = question.lower()

    # Pattern 1: explicit technical term before a unit in parentheses
    m = re.search(
        r"((?:[a-z][a-z\-]+\s+){1,3}(?:[a-z][a-z\-]+))\s*\([^)]*(?:µm|mm|nm|°c|ω|%|gpa|mpa|v|a|w|hz|cd|lm)\)",
        q
    )
    if m:
        return m.group(1).strip()

    # Pattern 2: specific technical noun phrases
    m = re.search(
        r"((?:dot|layer|substrate|element|coating|pattern|structure|circuit|"
        r"material|geometry|thickness|resistance|conductivity|temperature|"
        r"efficiency|fabrication|refractive|luminance|density|distribution|"
        r"encapsulation|bonding|adhesive|winding|topology|algorithm|"
        r"mobility|lifetime|degradation|uniformity)\s*(?:[a-z\-]+ ){0,2})",
        q
    )
    if m:
        return m.group(1).strip()

    # Pattern 3: first meaningful noun phrase after question word
    m = re.search(
        r"(?:what|provide|describe|specify|explain|identify|clarify)\s+"
        r"(?:is|are|the|a|an)?\s*"
        r"((?:[a-z][a-z\-]+\s+){1,3}[a-z][a-z\-]+)",
        q
    )
    if m:
        candidate = m.group(1).strip()
        # Filter out generic phrases
        generic = {"complete layer", "following items", "minimum bend", "maximum continuous"}
        if candidate not in generic and len(candidate) > 5:
            return candidate

    # Fallback: first 4 content words
    stop = {"what","is","are","the","a","an","provide","describe","specify",
            "explain","identify","clarify","how","does","can","for","in","of",
            "with","at","from","to","and","or","if","this","that","which"}
    words = [w for w in re.findall(r"[a-z][a-z\-]{2,}", q) if w not in stop]
    return " ".join(words[:3]) if words else ""
