"""
services/patent_chunker.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Phase B — Smart section extraction for gap analysis.

Takes two inputs:
  1. StructuredPatent  — reference granted patent (from patent_retriever.py)
  2. draft_text        — full text of inventor's draft document

Produces two outputs:
  1. ReferenceProfile  — what sections, claims, and parameters a granted
                         patent has (the target the draft must match)
  2. DraftProfile      — what sections and topics the draft document covers

The gap_analyser.py (Phase C) compares these two profiles to produce
the structured gap report.

Design principle:
  Deterministic Python only. No LLM calls. Fast.
  The LLM is only used later (Phase D) to generate the "why it matters"
  explanation and the targeted questions for confirmed gaps.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from services.patent_retriever import StructuredPatent, PatentClaim, TechnicalParameter

log = logging.getLogger(__name__)


# ── Output data structures ────────────────────────────────────────────────────

@dataclass
class SectionProfile:
    """Presence and depth of one section in a document."""
    name:         str
    present:      bool
    word_count:   int
    excerpt:      str    # first 200 chars for display in gap report
    subsections:  list[str] = field(default_factory=list)


@dataclass
class ClaimProfile:
    """Summary of the claims in a patent document."""
    total_count:         int
    independent_count:   int
    dependent_count:     int
    independent_topics:  list[str]   # topics from independent claims
    all_topics:          list[str]   # deduplicated across all claims
    sample_claim:        str         # text of first independent claim


@dataclass
class ReferenceProfile:
    """
    Complete profile of a reference granted patent.
    This is the target structure the draft must match.
    """
    epodoc_id:    str
    display_id:   str
    title:        str

    # Sections
    sections:     dict[str, SectionProfile]   # section_name → SectionProfile
    # Claims
    claims:       ClaimProfile
    # Technical parameters found in reference
    parameters:   list[TechnicalParameter]
    # Figures referenced
    figures:      list[str]

    # Derived metrics for gap analysis
    has_background:   bool = False
    has_summary:      bool = False
    has_description:  bool = False
    has_claims:       bool = False
    has_figures:      bool = False
    description_depth: str = "thin"  # "thin" | "moderate" | "detailed"

    @property
    def section_names(self) -> list[str]:
        return [k for k, v in self.sections.items() if v.present]

    @property
    def technical_terms(self) -> list[str]:
        """All significant technical terms from claims and description."""
        terms = set()
        for p in self.parameters:
            terms.add(p.name)
        for t in self.claims.all_topics:
            terms.add(t)
        return sorted(terms)


@dataclass
class DraftProfile:
    """
    Profile of the inventor's draft document.
    Compared against ReferenceProfile to find gaps.
    """
    # Sections detected in draft
    sections:          dict[str, SectionProfile]
    # What topics are mentioned
    mentioned_topics:  list[str]
    # What technical parameters appear (with values)
    parameters_found:  list[str]   # parameter names (no values required)
    # Has claims section
    has_claims_section: bool = False
    # Has any figures/drawings mentioned
    has_figures_mentioned: bool = False
    # Word count
    total_word_count:  int = 0
    # Detected field of invention (verbatim)
    field_of_invention: str = ""
    # Detected novelty statement
    novelty_statement:  str = ""


# ── Reference patent profiler ─────────────────────────────────────────────────

def profile_reference(patent: StructuredPatent) -> ReferenceProfile:
    """
    Build a ReferenceProfile from a StructuredPatent.
    Analyses what sections, claims, and parameters the granted patent has.
    This defines the target the draft must match.
    """
    log.info("Profiling reference patent: %s", patent.epodoc_id)

    # ── Section profiles ──────────────────────────────────────────────────────
    sections: dict[str, SectionProfile] = {}

    # Abstract
    sections["Abstract"] = _section_profile(
        "Abstract", patent.abstract,
        required_words=30,
    )

    # Background
    sections["Background of the Invention"] = _section_profile(
        "Background of the Invention", patent.background,
        required_words=50,
    )

    # Summary
    sections["Summary of the Invention"] = _section_profile(
        "Summary of the Invention", patent.summary,
        required_words=50,
    )

    # Detailed Description — most important section
    desc_text = patent.description or patent.description_raw
    sections["Detailed Description"] = _section_profile(
        "Detailed Description", desc_text,
        required_words=100,
    )

    # Claims — assessed from parsed claims list
    claims_present = len(patent.claims) > 0 or bool(patent.claims_raw.strip())
    sections["Claims"] = SectionProfile(
        name       = "Claims",
        present    = claims_present,
        word_count = len(patent.claims_raw.split()) if patent.claims_raw else 0,
        excerpt    = patent.claims_raw[:200] if patent.claims_raw else "",
    )

    # Figures
    figures_present = len(patent.figures) > 0
    sections["Brief Description of Drawings"] = SectionProfile(
        name       = "Brief Description of Drawings",
        present    = figures_present,
        word_count = len(patent.figures) * 10,   # approximate
        excerpt    = "; ".join(patent.figures[:3]),
    )

    # ── Claims profile ────────────────────────────────────────────────────────
    independent = [c for c in patent.claims if c.claim_type == "independent"]
    dependent   = [c for c in patent.claims if c.claim_type == "dependent"]

    all_topics: list[str] = []
    for c in patent.claims:
        all_topics.extend(c.topics)
    all_topics = list(dict.fromkeys(all_topics))[:20]

    ind_topics: list[str] = []
    for c in independent:
        ind_topics.extend(c.topics)
    ind_topics = list(dict.fromkeys(ind_topics))[:12]

    sample_claim = independent[0].text[:300] if independent else (
        patent.claims[0].text[:300] if patent.claims else patent.claims_raw[:300]
    )

    claims_profile = ClaimProfile(
        total_count        = len(patent.claims),
        independent_count  = len(independent),
        dependent_count    = len(dependent),
        independent_topics = ind_topics,
        all_topics         = all_topics,
        sample_claim       = sample_claim,
    )

    # ── Depth assessment ──────────────────────────────────────────────────────
    desc_words = sections["Detailed Description"].word_count
    if desc_words >= 500:
        depth = "detailed"
    elif desc_words >= 150:
        depth = "moderate"
    else:
        depth = "thin"

    profile = ReferenceProfile(
        epodoc_id   = patent.epodoc_id,
        display_id  = patent.display_id,
        title       = patent.title,
        sections    = sections,
        claims      = claims_profile,
        parameters  = patent.technical_parameters,
        figures     = patent.figures,
        has_background  = sections["Background of the Invention"].present,
        has_summary     = sections["Summary of the Invention"].present,
        has_description = sections["Detailed Description"].present,
        has_claims      = claims_present,
        has_figures     = figures_present,
        description_depth = depth,
    )

    log.info(
        "Reference profile: %d sections present, %d claims, %d params, depth=%s",
        len(profile.section_names), len(patent.claims),
        len(patent.technical_parameters), depth,
    )
    return profile


# ── Draft document profiler ───────────────────────────────────────────────────

def profile_draft(draft_text: str) -> DraftProfile:
    """
    Build a DraftProfile from the inventor's draft document full text.

    The draft is typically an information sheet — 1-4 pages, often lacking
    standard patent sections. We detect what IS there rather than assuming
    standard structure.
    """
    log.info("Profiling draft document (%d chars)", len(draft_text))

    text_lower = draft_text.lower()
    words      = draft_text.split()
    total_words = len(words)

    # ── Section detection ─────────────────────────────────────────────────────
    sections: dict[str, SectionProfile] = {}

    # Standard patent sections — check if any heading-like text indicates presence
    section_patterns = {
        "Abstract": [
            r"\babstract\b", r"\bsummary\b",
        ],
        "Background of the Invention": [
            r"\bbackground\b", r"\bprior art\b", r"\bproblem\b",
            r"\bexisting solution", r"\bcurrent approach",
        ],
        "Summary of the Invention": [
            r"\bsummary\b", r"\bthe invention\b", r"\bthe present invention\b",
            r"\bnovel\b", r"\binvention relates\b",
        ],
        "Detailed Description": [
            r"\bdetailed\b", r"\bdescription\b", r"\bembodiment\b",
            r"\bpreferred\b", r"\bspecification\b",
        ],
        "Claims": [
            r"\bclaim\b", r"\bclaimed\b", r"\bclaims?\s*\:",
            r"\bwhat is claimed\b",
        ],
        "Brief Description of Drawings": [
            r"\bfigure\b", r"\bfig\.\b", r"\bdrawing\b",
            r"\bschematic\b", r"\billustrat",
        ],
    }

    for section_name, patterns in section_patterns.items():
        matches = any(re.search(p, text_lower) for p in patterns)
        if matches:
            # Find the approximate location and extract context
            for p in patterns:
                m = re.search(p, text_lower)
                if m:
                    start = max(0, m.start() - 20)
                    end   = min(len(draft_text), m.end() + 200)
                    excerpt = draft_text[start:end].strip()
                    break
            else:
                excerpt = ""
            # Count words in vicinity (rough)
            word_count = total_words // max(len(section_patterns), 1)
        else:
            excerpt    = ""
            word_count = 0

        sections[section_name] = SectionProfile(
            name       = section_name,
            present    = matches,
            word_count = word_count if matches else 0,
            excerpt    = excerpt[:200],
        )

    # ── Topic detection ───────────────────────────────────────────────────────
    technical_nouns = _extract_technical_nouns(draft_text)

    # ── Parameter detection ───────────────────────────────────────────────────
    # Look for any numeric values with units — indicates technical specificity
    param_patterns = [
        r"\d+\s*(?:µm|mm|nm|cm|°C|K|GPa|MPa|kPa|Ω|ohm|W|V|A|Hz|kHz|MHz|%|ppm)",
        r"\d+\s*to\s*\d+",
        r"\d+\.\d+",
    ]
    params_found = []
    for pattern in param_patterns:
        if re.search(pattern, draft_text, re.IGNORECASE):
            params_found.append(pattern)

    # ── Field and novelty detection ───────────────────────────────────────────
    field_match = re.search(
        r"(?:field of (?:the )?invention|technical field|invention relates)[:\s]+([^\n]{20,300})",
        draft_text, re.IGNORECASE,
    )
    field_of_invention = field_match.group(1).strip() if field_match else ""

    novelty_match = re.search(
        r"(?:novel|novelty|the invention|unique|innovative|the present invention)[:\s]+([^\n]{20,300})",
        draft_text, re.IGNORECASE,
    )
    novelty_statement = novelty_match.group(1).strip() if novelty_match else ""

    # ── Special flags ─────────────────────────────────────────────────────────
    has_claims = any(
        re.search(p, text_lower)
        for p in [r"\bclaim\s+\d+", r"\bwhat is claimed\b", r"\bclaims?\s*\:"]
    )
    has_figures = any(
        re.search(p, text_lower)
        for p in [r"\bfig\.\s*\d+", r"\bfigure\s+\d+", r"\battached drawing"]
    )

    profile = DraftProfile(
        sections           = sections,
        mentioned_topics   = technical_nouns,
        parameters_found   = params_found,
        has_claims_section = has_claims,
        has_figures_mentioned = has_figures,
        total_word_count   = total_words,
        field_of_invention = field_of_invention,
        novelty_statement  = novelty_statement,
    )

    present_sections = [k for k, v in sections.items() if v.present]
    log.info(
        "Draft profile: %d words, sections=%s, has_claims=%s, has_figures=%s",
        total_words, present_sections, has_claims, has_figures,
    )
    return profile


# ── Helper: retrieve full draft text from ChromaDB collection ─────────────────

def draft_text_from_collection(collection) -> str:
    """
    Retrieve all chunks from a ChromaDB collection and join into full text.
    Used to get the complete draft document text for profiling.
    """
    try:
        result = collection.get(include=["documents"])
        chunks = result.get("documents", [])
        if not chunks:
            return ""
        return "\n\n".join(chunks)
    except Exception as exc:
        log.error("Could not retrieve draft text from collection: %s", exc)
        return ""


# ── Internal helpers ──────────────────────────────────────────────────────────

def _section_profile(
    name: str,
    text: str,
    required_words: int = 50,
) -> SectionProfile:
    """Build a SectionProfile from a section's text content."""
    if not text or not text.strip():
        return SectionProfile(name=name, present=False, word_count=0, excerpt="")

    words      = len(text.split())
    present    = words >= required_words
    excerpt    = text[:200].strip()
    subsections = _detect_subsections(text)

    return SectionProfile(
        name        = name,
        present     = present,
        word_count  = words,
        excerpt     = excerpt,
        subsections = subsections,
    )


def _detect_subsections(text: str) -> list[str]:
    """Detect subsection headings within a section."""
    # Look for numbered or titled subsections
    matches = re.findall(
        r"^(?:\d+\.\d+\s+|[A-Z][A-Z\s]{3,20}\n)",
        text, re.MULTILINE,
    )
    return [m.strip() for m in matches[:5]]


def _extract_technical_nouns(text: str) -> list[str]:
    """
    Extract significant technical noun phrases from draft text.
    Used to assess topic coverage for gap analysis.
    """
    stop_words = {
        "the", "a", "an", "and", "or", "of", "in", "on", "at", "to",
        "for", "is", "are", "be", "by", "with", "as", "that", "this",
        "which", "when", "where", "how", "what", "such", "may", "can",
        "will", "shall", "should", "have", "has", "been", "being",
        "present", "invention", "according", "described", "provided",
    }

    # Extract multi-word technical phrases (more reliable than single words)
    phrases = re.findall(
        r"\b([a-z][a-z\-]+ (?:[a-z][a-z\-]+ )?(?:film|layer|element|"
        r"substrate|coating|material|pattern|structure|circuit|device|"
        r"method|system|component|geometry|distribution|density|"
        r"composition|thickness|resistance|conductivity|temperature|"
        r"efficiency|assembly|interface|surface|bonding|adhesive|"
        r"electrode|emitter|driver|controller|module|array|stack))\b",
        text.lower(),
    )

    # Also extract single capitalised technical words (e.g. ITO, OLED, NVG)
    acronyms = re.findall(r"\b[A-Z]{2,6}\b", text)

    # Combine and deduplicate
    all_terms = phrases + [a.lower() for a in acronyms]
    seen, unique = set(), []
    for t in all_terms:
        if t not in seen and t not in stop_words:
            seen.add(t)
            unique.append(t)

    return unique[:30]


# ── Utility: summarise profiles for LLM context injection ────────────────────

def reference_summary(ref: ReferenceProfile) -> str:
    """
    Compact text summary of reference profile for injection into LLM prompt.
    Keeps it under ~500 tokens.
    """
    lines = [
        f"Reference patent: {ref.display_id} — {ref.title}",
        f"Sections present: {', '.join(ref.section_names)}",
        f"Claims: {ref.claims.independent_count} independent, "
        f"{ref.claims.dependent_count} dependent",
    ]
    if ref.claims.independent_topics:
        lines.append(f"Claim topics: {', '.join(ref.claims.independent_topics[:8])}")
    if ref.parameters:
        param_strs = [f"{p.name} ({p.value} {p.unit})" for p in ref.parameters[:6]]
        lines.append(f"Technical parameters: {'; '.join(param_strs)}")
    if ref.figures:
        lines.append(f"Figures: {', '.join(ref.figures[:4])}")
    return "\n".join(lines)


def draft_summary(draft: DraftProfile) -> str:
    """
    Compact text summary of draft profile for injection into LLM prompt.
    """
    present = [k for k, v in draft.sections.items() if v.present]
    missing = [k for k, v in draft.sections.items() if not v.present]
    lines = [
        f"Draft: {draft.total_word_count} words",
        f"Sections present: {', '.join(present) if present else 'none detected'}",
        f"Sections absent:  {', '.join(missing) if missing else 'none'}",
        f"Has claims: {'yes' if draft.has_claims_section else 'no'}",
        f"Has figures: {'yes' if draft.has_figures_mentioned else 'no'}",
    ]
    if draft.mentioned_topics:
        lines.append(f"Topics mentioned: {', '.join(draft.mentioned_topics[:10])}")
    if draft.field_of_invention:
        lines.append(f"Field: {draft.field_of_invention[:120]}")
    return "\n".join(lines)


# ── Public aliases for patent_retriever functions (used by PDF upload path) ───
# These wrap the private helpers in patent_retriever so patent_chunker.py
# can expose them without circular imports.

def _split_sections(text: str) -> dict:
    """Split patent full text into sections. Public alias for PDF upload path."""
    import re
    sections = {
        "background":    "",
        "summary":       "",
        "description":   "",
        "claims_raw":    "",
        "description_raw": "",
    }

    # Claims section
    m = re.search(
        r"\n\s*(?:CLAIMS?|What is claimed(?:\s+is)?)\s*[:\n](.*?)(?=\n\s*(?:ABSTRACT|$))",
        text, re.IGNORECASE | re.DOTALL,
    )
    if m:
        sections["claims_raw"] = m.group(1).strip()

    # Background
    m = re.search(
        r"\n\s*BACKGROUND(?:\s+OF\s+THE\s+INVENTION)?\s*[:\n](.*?)"
        r"(?=\n\s*(?:SUMMARY|BRIEF|DETAILED|CLAIMS?|$))",
        text, re.IGNORECASE | re.DOTALL,
    )
    if m:
        sections["background"] = m.group(1).strip()[:4000]

    # Summary
    m = re.search(
        r"\n\s*SUMMARY(?:\s+OF\s+THE\s+INVENTION)?\s*[:\n](.*?)"
        r"(?=\n\s*(?:BRIEF|DETAILED|CLAIMS?|$))",
        text, re.IGNORECASE | re.DOTALL,
    )
    if m:
        sections["summary"] = m.group(1).strip()[:3000]

    # Detailed description
    m = re.search(
        r"\n\s*DETAILED\s+DESCRIPTION(?:\s+OF(?:\s+THE)?(?:\s+PREFERRED)?\s+EMBODIMENTS?)?\s*[:\n](.*?)"
        r"(?=\n\s*(?:CLAIMS?|ABSTRACT|$))",
        text, re.IGNORECASE | re.DOTALL,
    )
    if m:
        sections["description"] = m.group(1).strip()[:8000]

    sections["description_raw"] = sections["description"] or text[:8000]
    return sections


def _parse_claims_from_text(claims_raw: str) -> list:
    """Parse claims text into PatentClaim list. Public alias for PDF path."""
    from services.patent_retriever import _parse_claims
    return _parse_claims(claims_raw)


def _extract_parameters_from_text(text: str) -> list:
    """Extract technical parameters from text. Public alias for PDF path."""
    from services.patent_retriever import _extract_parameters
    return _extract_parameters(text)


def _extract_figures_from_text(text: str) -> list:
    """Extract figure references from text. Public alias for PDF path."""
    from services.patent_retriever import _extract_figures
    return _extract_figures(text)


def profile_from_chunks(chunks: list[str], patent_number: str) -> "ReferenceProfile":
    """
    Build a ReferenceProfile directly from text chunks (PDF upload path).
    Used when a reference patent is loaded from a local PDF rather than
    fetched via EPO OPS API.

    This is the key function that makes Passes 1-4 of gap_analyser work
    with a locally loaded PDF.
    """
    from services.patent_retriever import StructuredPatent
    full_text = "\n\n".join(chunks)
    sections  = _split_sections(full_text)

    import re
    # Extract title from first non-empty lines
    first_lines = full_text[:500].split("\n")
    title = next(
        (l.strip() for l in first_lines
         if len(l.strip()) > 10 and not l.strip().isdigit()),
        patent_number,
    )

    # Build minimal StructuredPatent from local text
    from services.patent_retriever import _to_epodoc, _epodoc_to_display
    epodoc_id  = _to_epodoc(patent_number) or patent_number
    display_id = _epodoc_to_display(epodoc_id)

    structured = StructuredPatent(
        epodoc_id       = epodoc_id,
        display_id      = display_id,
        title           = title[:200],
        abstract        = "",
        grant_date      = "",
        applicant       = "",
        background      = sections.get("background", ""),
        summary         = sections.get("summary", ""),
        description     = sections.get("description", ""),
        claims          = _parse_claims_from_text(sections.get("claims_raw", "")),
        figures         = _extract_figures_from_text(full_text),
        technical_parameters = _extract_parameters_from_text(
            sections.get("claims_raw", "") + "\n" + sections.get("description", "")
        ),
        claims_raw      = sections.get("claims_raw", ""),
        description_raw = sections.get("description_raw", ""),
    )
    return profile_reference(structured)
