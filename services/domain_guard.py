"""
services/domain_guard.py
Domain-specific guard utilities for post-scrutiny validation.
"""
from __future__ import annotations

from typing import List, Dict

FORBIDDEN_ELECTRONICS: List[str] = [
    "voltage",
    "current",
    "pcb",
    "emc",
    "impedance",
    "ohm",
    "ampere",
]


def check_forbidden_electronics(text: str) -> List[str]:
    """Return a list of forbidden-electronics terms present in `text`.

    Matching is case-insensitive and simple substring-based to be robust
    against minor tokenisation differences in LLM output.
    """
    violations: List[str] = []
    if not text:
        return violations
    lower = text.lower()
    for term in FORBIDDEN_ELECTRONICS:
        if term in lower:
            violations.append(term)
    return violations


def check_domain_violation(output: str, patent_type: str) -> Dict[str, object]:
    """Check `output` for domain-specific violations based on `patent_type`.

    Currently enforces forbidden-electronics terms for the "Optics / Display" domain.
    Returns a dict with keys: `status` ("OK" or "VIOLATION") and `violations` (list).
    """
    if patent_type == "Optics / Display":
        violations = check_forbidden_electronics(output)
        if violations:
            return {"status": "VIOLATION", "violations": violations}

    return {"status": "OK", "violations": []}
