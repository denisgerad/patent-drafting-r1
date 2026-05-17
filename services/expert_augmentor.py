"""
services/expert_augmentor.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PHASE 2 (ONLINE, per-run) — Called ONLY when the readiness gate scores
below threshold AND the Phase 1 expert bank alone is insufficient.

WHAT IS SENT TO CLAUDE:
  - Product type label         ("flexible_heater_film")
  - Mechanism description      ("resistive heating element in polymer film")
  - Weak category names        (["Fabrication & Process", "Materials"])
  - Missing term labels        (["injection moulding", "refractive index"])

WHAT IS NEVER SENT TO CLAUDE:
  - Patent document content
  - Inventor details
  - Any proprietary specifications or measurements

The patent document stays on the local machine. Claude only sees
structural labels describing what is missing — not what was written.

Circuit breaker: if Claude is unavailable or call fails, the function
returns an empty list and logs a warning. The workflow continues with
whatever the local model and expert bank produced.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class AugmentationResult:
    """Result of one Phase 2 augmentation call."""
    questions_by_category: dict[str, list[str]]  # category_name → [questions]
    source: str                                   # "claude" | "bank" | "none"
    model_used: str
    total_added: int
    error: str = ""

    @property
    def success(self) -> bool:
        return not self.error and self.total_added > 0

    def all_questions(self) -> list[str]:
        q = []
        for qs in self.questions_by_category.values():
            q.extend(qs)
        return q


# ── Bank loader ───────────────────────────────────────────────────────────────

def load_expert_bank(product_type: str) -> Optional[dict]:
    """
    Load the approved expert bank for a product type.
    Returns None if no bank exists or bank is not yet approved.
    """
    bank_path = _ROOT / "domain_reference_questions" / f"{product_type}_bank.json"
    if not bank_path.exists():
        return None
    try:
        bank = json.loads(bank_path.read_text(encoding="utf-8"))
        if bank.get("review_status") != "APPROVED":
            logger.info(
                "Expert bank for '%s' exists but is PENDING_EXPERT_REVIEW — skipping.",
                product_type,
            )
            return None
        return bank
    except Exception as exc:
        logger.warning("Could not load expert bank for '%s': %s", product_type, exc)
        return None


def get_bank_questions_for_categories(
    bank: dict,
    category_names: list[str],
) -> dict[str, list[str]]:
    """
    Extract questions from the bank for specific weak categories.
    Matches category names case-insensitively.
    """
    result = {}
    bank_cats = {c["name"].lower(): c for c in bank.get("categories", [])}

    for name in category_names:
        # Try exact match first, then partial match
        matched = bank_cats.get(name.lower())
        if not matched:
            for k, v in bank_cats.items():
                if name.lower() in k or k in name.lower():
                    matched = v
                    break
        if matched:
            result[matched["name"]] = matched.get("questions", [])

    return result


# ── Phase 2: Claude augmentation ─────────────────────────────────────────────

def augment_with_claude(
    product_type: str,
    mechanism: str,
    weak_categories: list[str],
    missing_terms: list[str],
    existing_questions: list[str],
    model: str = "claude-sonnet-4-20250514",
) -> AugmentationResult:
    """
    Call Claude to generate deeper questions for weak categories.

    IMPORTANT: This function sends ONLY structural labels to Claude.
    The patent document content is never included.

    Parameters
    ----------
    product_type      : e.g. "flexible_heater_film"
    mechanism         : e.g. "resistive heating element in polymer film"
    weak_categories   : category names that scored below threshold
    missing_terms     : specific technical terms missing from model output
    existing_questions: list of questions already generated (to avoid duplicates)
    """
    if not weak_categories:
        return AugmentationResult({}, "none", "", 0)

    # Validate API key is set before attempting call
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return AugmentationResult(
            {}, "none", "", 0,
            error="ANTHROPIC_API_KEY not set — skipping cloud augmentation"
        )

    prompt = _build_augmentation_prompt(
        product_type, mechanism, weak_categories, missing_terms, existing_questions
    )

    logger.info(
        "Phase 2 augmentation: product_type=%s weak_cats=%s",
        product_type, weak_categories,
    )

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text
    except Exception as exc:
        logger.warning("Claude augmentation call failed: %s — continuing without it", exc)
        return AugmentationResult({}, "none", "", 0, error=str(exc))

    # Parse response
    try:
        import re
        clean = re.sub(r"```(?:json)?", "", raw).strip("`").strip()
        parsed = json.loads(clean)
    except Exception:
        try:
            import re
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            parsed = json.loads(m.group()) if m else {}
        except Exception as exc:
            return AugmentationResult({}, "claude", model, 0, error=f"Parse failed: {exc}")

    # Extract questions by category
    questions_by_cat: dict[str, list[str]] = {}
    for cat in parsed.get("augmentation", []):
        name = cat.get("category", "")
        qs   = cat.get("questions", [])
        if name and qs:
            questions_by_cat[name] = qs

    total = sum(len(v) for v in questions_by_cat.values())
    logger.info("Phase 2 augmentation: %d questions added across %d categories",
                total, len(questions_by_cat))

    return AugmentationResult(
        questions_by_category=questions_by_cat,
        source="claude",
        model_used=model,
        total_added=total,
    )


def _build_augmentation_prompt(
    product_type: str,
    mechanism: str,
    weak_categories: list[str],
    missing_terms: list[str],
    existing_questions: list[str],
) -> str:
    """
    Build compact augmentation prompt.
    Contains ONLY labels — no patent document content.
    """
    cats_text  = "\n".join(f"  - {c}" for c in weak_categories)
    terms_text = ", ".join(missing_terms[:15]) if missing_terms else "(see categories above)"
    existing_sample = "\n".join(f"  - {q}" for q in existing_questions[:8])

    return f"""You are a senior patent analyst generating expert enablement questions.

INVENTION CATEGORY: {product_type.replace("_", " ").title()}
ENABLING MECHANISM: {mechanism}

CONTEXT: A patent scrutiny system has already generated questions for this invention
but these specific categories scored below the required depth threshold:

WEAK CATEGORIES (need deeper questions):
{cats_text}

MISSING TECHNICAL TERMS (these topics are absent from existing questions):
  {terms_text}

EXISTING QUESTIONS ALREADY GENERATED (do not duplicate these):
{existing_sample}

TASK: For each weak category listed above, generate 3-5 additional expert questions
that specifically target the missing terms and technical depth gaps.

Each question must:
- Name the specific parameter, component, or condition
- Specify measurement units
- Request drawings, formulas, test data, or comparative performance data
- Be answerable with a number, drawing, formula, or step-by-step procedure

OUTPUT: Return ONLY valid JSON, no explanation:
{{
  "augmentation": [
    {{
      "category": "exact category name from weak categories list",
      "questions": ["question 1", "question 2", "question 3"]
    }}
  ]
}}
"""


# ── Main entry point ──────────────────────────────────────────────────────────

def get_augmentation(
    product_type: str,
    mechanism: str,
    readiness_report,
    existing_questions: str,
    enable_cloud: bool = True,
) -> AugmentationResult:
    """
    Main entry point called by scrutiny_workflow.

    Strategy:
    1. Try Phase 1 bank for weak categories (free, offline)
    2. If bank insufficient and cloud enabled, try Phase 2 Claude call
    3. Return combined result with source label

    Parameters
    ----------
    product_type      : matched product type from checklist
    mechanism         : extracted mechanism string
    readiness_report  : ReadinessReport from question_rater
    existing_questions: full text of model-generated questions
    enable_cloud      : if False, only use bank (Phase 1 only)
    """
    if not product_type:
        return AugmentationResult({}, "none", "", 0, error="No product type matched")

    weak_cats    = [c.name for c in readiness_report.failed_categories]
    missing_terms = []
    for c in readiness_report.failed_categories:
        missing_terms.extend(c.missing_key_terms[:3])
        missing_terms.extend(c.missing_depth_markers[:2])

    if not weak_cats:
        return AugmentationResult({}, "none", "", 0)

    # ── Phase 1: Try expert bank first ────────────────────────────────────────
    bank = load_expert_bank(product_type)
    bank_questions: dict[str, list[str]] = {}

    if bank:
        bank_questions = get_bank_questions_for_categories(bank, weak_cats)
        bank_total = sum(len(v) for v in bank_questions.values())
        logger.info(
            "Phase 1 bank: %d questions from %d categories",
            bank_total, len(bank_questions),
        )
        if bank_total > 0 and len(bank_questions) >= len(weak_cats):
            # Bank covers all weak categories — no cloud call needed
            return AugmentationResult(
                questions_by_category=bank_questions,
                source="bank",
                model_used="expert_bank",
                total_added=bank_total,
            )
    else:
        logger.info("No approved expert bank for '%s'", product_type)

    # ── Phase 2: Cloud augmentation for remaining gaps ────────────────────────
    if not enable_cloud:
        return AugmentationResult(
            questions_by_category=bank_questions,
            source="bank",
            model_used="expert_bank",
            total_added=sum(len(v) for v in bank_questions.values()),
        )

    # Only send category names and missing term labels — no patent content
    existing_q_list = [
        line.strip().lstrip("0123456789. ")
        for line in existing_questions.splitlines()
        if line.strip() and line.strip()[0].isdigit()
    ]

    cloud_result = augment_with_claude(
        product_type=product_type,
        mechanism=mechanism,
        weak_categories=weak_cats,
        missing_terms=missing_terms,
        existing_questions=existing_q_list,
    )

    # Merge bank + cloud results
    merged = {**bank_questions}
    for cat, qs in cloud_result.questions_by_category.items():
        if cat in merged:
            merged[cat].extend(qs)
        else:
            merged[cat] = qs

    total = sum(len(v) for v in merged.values())
    source = "bank+claude" if (bank_questions and cloud_result.success) else \
             "claude" if cloud_result.success else "bank"

    return AugmentationResult(
        questions_by_category=merged,
        source=source,
        model_used=cloud_result.model_used or "expert_bank",
        total_added=total,
        error=cloud_result.error,
    )
