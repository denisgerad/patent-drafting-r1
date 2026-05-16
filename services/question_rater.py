"""
services/question_rater.py
Programmatic readiness gate for generated patent scrutiny questions.

Scores model-generated questions against domain expert reference questions
on three independent dimensions:

  1. CATEGORY COMPLETENESS — are all required section headings present?
  2. TOPIC COVERAGE        — do key technical terms appear in the output?
  3. DEPTH MARKERS         — are specific depth indicators (units, drawings,
                             formulas, comparatives) present?

No LLM call is made. This is entirely deterministic Python — fast,
objective, and free of self-consistency bias.

The output is a ReadinessReport which the UI uses to show:
  ✅ READY     — score ≥ threshold in all dimensions
  🟡 BORDERLINE — overall passes but one category is weak
  ❌ NOT READY  — specific gaps listed so inventor knows what to fix
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Path to domain reference question files
_REF_DIR = Path(__file__).parent.parent / "domain_reference_questions"


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class CategoryScore:
    name: str
    weight: float
    category_score: float          # 0.0 – 1.0 weighted average of 3 dimensions
    completeness: float            # was the section heading present?
    topic_coverage: float          # fraction of key terms found
    depth_coverage: float          # fraction of depth markers found
    missing_key_terms: list[str]   = field(default_factory=list)
    missing_depth_markers: list[str] = field(default_factory=list)
    passed: bool = False


@dataclass
class ReadinessReport:
    product_type: str
    domain: str
    overall_score: float           # 0.0 – 1.0
    readiness_threshold: float
    category_threshold: float
    category_scores: list[CategoryScore]
    verdict: str                   # "READY" | "BORDERLINE" | "NOT_READY" | "NO_REFERENCE"
    verdict_reason: str
    reference_file_used: str
    total_model_questions: int
    total_depth_markers_found: int
    total_depth_markers_expected: int

    @property
    def is_ready(self) -> bool:
        return self.verdict == "READY"

    @property
    def failed_categories(self) -> list[CategoryScore]:
        return [c for c in self.category_scores if not c.passed]

    def summary_lines(self) -> list[str]:
        """Human-readable summary for UI display."""
        lines = [
            f"Overall score: {self.overall_score:.0%}  "
            f"(threshold: {self.readiness_threshold:.0%})",
            f"Verdict: {self.verdict} — {self.verdict_reason}",
            "",
        ]
        for cs in self.category_scores:
            icon = "✅" if cs.passed else "❌"
            lines.append(
                f"{icon} [{cs.name}]  "
                f"score={cs.category_score:.0%}  "
                f"(coverage={cs.topic_coverage:.0%}  "
                f"depth={cs.depth_coverage:.0%}  "
                f"present={'yes' if cs.completeness == 1.0 else 'NO'})"
            )
            if cs.missing_key_terms:
                lines.append(
                    f"   Missing terms: {', '.join(cs.missing_key_terms[:5])}"
                )
            if cs.missing_depth_markers:
                lines.append(
                    f"   Missing depth: {', '.join(cs.missing_depth_markers[:4])}"
                )
        return lines


# ── Public API ────────────────────────────────────────────────────────────────

def rate(
    model_questions: str,
    product_type: str,
    checklist_categories: Optional[list[dict]] = None,
) -> ReadinessReport:
    """
    Score model_questions against reference questions for product_type.

    Parameters
    ----------
    model_questions     : raw text output from the scrutiny crew
    product_type        : key matching domain_reference_questions/*.json
                          (e.g. "light_guide_plate", "power_electronics")
    checklist_categories: optional list of checklist category dicts —
                          used as fallback if no reference file exists

    Returns
    -------
    ReadinessReport with verdict and per-category breakdown
    """
    ref_data = _load_reference(product_type)

    if ref_data is None:
        # No reference file — use checklist as lightweight fallback
        return _rate_against_checklist(
            model_questions, product_type, checklist_categories
        )

    return _rate_against_reference(model_questions, ref_data)


def load_reference_file(product_type: str) -> Optional[dict]:
    """Return the raw reference JSON for a product type, or None."""
    return _load_reference(product_type)


def list_available_references() -> list[str]:
    """Return list of product types that have reference files."""
    if not _REF_DIR.exists():
        return []
    return [f.stem for f in _REF_DIR.glob("*.json")]


# ── Reference file loading ────────────────────────────────────────────────────

def _load_reference(product_type: str) -> Optional[dict]:
    ref_path = _REF_DIR / f"{product_type}.json"
    if not ref_path.exists():
        logger.info(
            "No reference file for '%s' at %s", product_type, ref_path
        )
        return None
    try:
        data = json.loads(ref_path.read_text(encoding="utf-8"))
        logger.info("Loaded reference: %s", ref_path.name)
        return data
    except Exception as exc:
        logger.error("Could not load reference file %s: %s", ref_path, exc)
        return None


# ── Main scoring engine ───────────────────────────────────────────────────────

def _rate_against_reference(
    model_questions: str,
    ref_data: dict,
) -> ReadinessReport:
    """Full scoring against a reference JSON file."""
    product_type        = ref_data.get("product_type", "unknown")
    domain              = ref_data.get("domain", "unknown")
    readiness_threshold = float(ref_data.get("readiness_threshold", 0.65))
    category_threshold  = float(ref_data.get("category_threshold", 0.50))
    categories          = ref_data.get("categories", [])

    model_lower = model_questions.lower()

    # Count total model questions
    total_model_questions = len(re.findall(r"^\s*\d+\.", model_questions, re.MULTILINE))

    # Count total depth markers found in the whole output
    all_depth_markers_expected = sum(
        len(c.get("depth_markers", [])) for c in categories
    )
    all_depth_markers_found = sum(
        1 for c in categories
        for dm in c.get("depth_markers", [])
        if dm.lower() in model_lower
    )

    # Score each category
    category_scores = []
    weighted_sum    = 0.0
    total_weight    = 0.0

    for cat in categories:
        cs = _score_category(cat, model_questions, model_lower, category_threshold)
        category_scores.append(cs)
        weighted_sum  += cs.category_score * cs.weight
        total_weight  += cs.weight

    overall = weighted_sum / total_weight if total_weight > 0 else 0.0

    # Determine verdict
    all_cats_passed = all(cs.passed for cs in category_scores)
    any_cat_failed  = any(not cs.passed for cs in category_scores)

    if overall >= readiness_threshold and all_cats_passed:
        verdict = "READY"
        reason  = (
            f"All {len(categories)} categories scored above "
            f"{category_threshold:.0%} and overall score "
            f"{overall:.0%} ≥ {readiness_threshold:.0%}."
        )
    elif overall >= readiness_threshold and any_cat_failed:
        verdict = "BORDERLINE"
        failed  = [cs.name for cs in category_scores if not cs.passed]
        reason  = (
            f"Overall {overall:.0%} passes, but these categories "
            f"are below {category_threshold:.0%}: {', '.join(failed)}."
        )
    else:
        verdict = "NOT_READY"
        failed  = [cs.name for cs in category_scores if not cs.passed]
        reason  = (
            f"Overall score {overall:.0%} is below {readiness_threshold:.0%}. "
            f"Weak categories: {', '.join(failed) if failed else 'overall depth insufficient'}."
        )

    return ReadinessReport(
        product_type=product_type,
        domain=domain,
        overall_score=overall,
        readiness_threshold=readiness_threshold,
        category_threshold=category_threshold,
        category_scores=category_scores,
        verdict=verdict,
        verdict_reason=reason,
        reference_file_used=f"{product_type}.json",
        total_model_questions=total_model_questions,
        total_depth_markers_found=all_depth_markers_found,
        total_depth_markers_expected=all_depth_markers_expected,
    )


def _score_category(
    cat: dict,
    model_questions: str,
    model_lower: str,
    category_threshold: float,
) -> CategoryScore:
    """Score one category on three dimensions."""
    name   = cat.get("name", "")
    weight = float(cat.get("weight", 1.0))
    key_terms     = [t.lower() for t in cat.get("key_terms", [])]
    depth_markers = [d.lower() for d in cat.get("depth_markers", [])]

    # ── Dimension 1: Category completeness ──────────────────────────────────
    # Was the section heading present in the output?
    # Check for the section name (case-insensitive, allow partial match for
    # cases where the model slightly rephrases the heading)
    name_words = name.lower().split()
    # Require at least 60% of the heading words to appear together
    heading_found = False
    if name.lower() in model_lower:
        heading_found = True
    else:
        # Fuzzy: check key words of heading
        key_name_words = [w for w in name_words if len(w) > 3]
        if key_name_words:
            matches = sum(1 for w in key_name_words if w in model_lower)
            heading_found = (matches / len(key_name_words)) >= 0.6
    completeness = 1.0 if heading_found else 0.0

    # ── Dimension 2: Topic keyword coverage ─────────────────────────────────
    if key_terms:
        found_terms   = [t for t in key_terms if t in model_lower]
        missing_terms = [t for t in key_terms if t not in model_lower]
        topic_cov     = len(found_terms) / len(key_terms)
    else:
        found_terms   = []
        missing_terms = []
        topic_cov     = 1.0  # no terms to check — pass by default

    # ── Dimension 3: Depth marker coverage ──────────────────────────────────
    if depth_markers:
        found_dm   = [d for d in depth_markers if d in model_lower]
        missing_dm = [d for d in depth_markers if d not in model_lower]
        depth_cov  = len(found_dm) / len(depth_markers)
    else:
        found_dm   = []
        missing_dm = []
        depth_cov  = 1.0

    # ── Weighted category score ──────────────────────────────────────────────
    # Weights per dimension: completeness 25%, topic 40%, depth 35%
    category_score = (
        0.25 * completeness
        + 0.40 * topic_cov
        + 0.35 * depth_cov
    )

    passed = category_score >= category_threshold

    return CategoryScore(
        name=name,
        weight=weight,
        category_score=category_score,
        completeness=completeness,
        topic_coverage=topic_cov,
        depth_coverage=depth_cov,
        missing_key_terms=missing_terms[:8],    # cap for display
        missing_depth_markers=missing_dm[:6],
        passed=passed,
    )


# ── Checklist fallback (when no reference file exists) ────────────────────────

def _rate_against_checklist(
    model_questions: str,
    product_type: str,
    checklist_categories: Optional[list[dict]],
) -> ReadinessReport:
    """
    Lightweight fallback when no reference JSON exists.
    Checks only category completeness using checklist section names.
    Returns a NO_REFERENCE verdict with a note to add reference questions.
    """
    model_lower = model_questions.lower()

    if not checklist_categories:
        return ReadinessReport(
            product_type=product_type,
            domain=product_type,
            overall_score=0.0,
            readiness_threshold=0.65,
            category_threshold=0.50,
            category_scores=[],
            verdict="NO_REFERENCE",
            verdict_reason=(
                f"No reference question file found for '{product_type}'. "
                f"Add domain expert questions to "
                f"domain_reference_questions/{product_type}.json "
                f"to enable readiness scoring."
            ),
            reference_file_used="(none)",
            total_model_questions=0,
            total_depth_markers_found=0,
            total_depth_markers_expected=0,
        )

    scores = []
    for cat in checklist_categories:
        name     = cat.get("name", "")
        name_lower = name.lower()
        present  = name_lower in model_lower
        scores.append(CategoryScore(
            name=name,
            weight=1.0,
            category_score=1.0 if present else 0.0,
            completeness=1.0 if present else 0.0,
            topic_coverage=0.0,
            depth_coverage=0.0,
            missing_key_terms=[],
            missing_depth_markers=[],
            passed=present,
        ))

    n_passed  = sum(1 for s in scores if s.passed)
    overall   = n_passed / len(scores) if scores else 0.0
    all_pass  = all(s.passed for s in scores)

    verdict = "BORDERLINE" if all_pass else "NOT_READY"
    reason  = (
        f"No reference file — section heading check only. "
        f"{n_passed}/{len(scores)} sections present. "
        f"Add domain_reference_questions/{product_type}.json for full scoring."
    )

    return ReadinessReport(
        product_type=product_type,
        domain=product_type,
        overall_score=overall,
        readiness_threshold=0.65,
        category_threshold=0.50,
        category_scores=scores,
        verdict=verdict,
        verdict_reason=reason,
        reference_file_used="(checklist fallback)",
        total_model_questions=len(re.findall(r"^\s*\d+\.", model_questions, re.MULTILINE)),
        total_depth_markers_found=0,
        total_depth_markers_expected=0,
    )
