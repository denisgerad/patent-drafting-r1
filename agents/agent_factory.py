"""
agents/agent_factory.py
Build all CrewAI agents.  No Streamlit, no global state.

LLM resolution is delegated entirely to services/cloud_llm_service.py.
Set LLM_PROVIDER in your .env to switch between Ollama, Azure, Claude, or OpenAI.
No changes needed here when switching providers.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from crewai import Agent

import config.settings as cfg
from services.cloud_llm_service import get_llm

logger = logging.getLogger(__name__)


# ── Domain config helpers ─────────────────────────────────────────────────────

def _load_domain_config(patent_type: str) -> dict:
    try:
        data = json.loads(cfg.PATENT_TYPES_JSON.read_text(encoding="utf-8"))
        return data.get(patent_type, {})
    except Exception:
        return {}


def _build_backstory(
    patent_type: str,
    user_notes: str = "",
    domain_cfg: Optional[dict] = None,
) -> str:
    if domain_cfg is None:
        domain_cfg = _load_domain_config(patent_type)

    focus  = ", ".join(domain_cfg.get("focus_areas", []))
    units  = ", ".join(domain_cfg.get("technical_units", []))
    role   = domain_cfg.get("role", f"Patent Specialist ({patent_type})")
    parts  = [f"{role} with deep expertise in {patent_type}."]
    if focus:
        parts.append(f"Focus areas include: {focus}.")
    if units:
        parts.append(f"Requires precise measurements in: {units}.")
    if user_notes:
        parts.append(f"User notes: {user_notes}.")
    return " ".join(parts[:3])


# ── Agent builders ────────────────────────────────────────────────────────────
# Each builder calls get_llm() internally.
# The caller never needs to know which provider is active.

def build_scrutinizer(
    patent_type: str,
    custom_role: Optional[str] = None,
    custom_backstory: Optional[str] = None,
) -> Agent:
    domain_cfg = _load_domain_config(patent_type)
    role       = custom_role or domain_cfg.get("role", "Patent Enablement Specialist")

    default_backstory = (
        f"You are a senior {patent_type} patent expert with 20+ years of hands-on "
        f"engineering and patent prosecution experience. "
        f"You have reviewed hundreds of patent applications and know exactly what "
        f"information an examiner will demand for Section 112 enablement. "
        f"Your first instinct is always to identify what is NOVEL about this specific "
        f"invention before asking any questions. You group your questions by the "
        f"invention's own technical sub-systems, not by a generic checklist. "
        f"You always demand drawings, governing equations, or measured performance data "
        f"rather than prose descriptions."
    )
    backstory = custom_backstory or default_backstory

    return Agent(
        role=role,
        goal=(
            "Read the patent disclosure, identify its specific novel claims and features, "
            "then produce grouped technical questions that expose exactly what information "
            "is missing for 35 U.S.C. Section 112 enablement — grounded in the document's "
            "own novelty, not in generic domain checklists."
        ),
        backstory=backstory,
        llm=get_llm(),
        verbose=True,
    )


def build_consolidator() -> Agent:
    return Agent(
        role="Technical Integration Specialist",
        goal="Incorporate every specific technical detail from the Q&A into Draft 1.",
        backstory=(
            "You are a meticulous patent engineer. Your job is NOT to summarise. "
            "Expand Draft 1 by injecting precise data points from the Q&A. "
            "If Draft 1 says 'thin layer' and Q&A says '5 microns', replace it. "
            "Preserve the original professional tone while maximising technical density."
        ),
        llm=get_llm(),
        verbose=True,
    )


def build_classifier() -> Agent:
    return Agent(
        role="Patent Classification Analyst",
        goal="Identify the most appropriate technical domain for this invention.",
        backstory=(
            "You are a patent classification expert across Mechanical, Electronics, "
            "Software, Chemical, Materials, and Medical Devices domains. "
            "You base decisions solely on technical content (components, processes, "
            "materials, algorithms) and output a concise JSON object with your reasoning."
        ),
        llm=get_llm(),
        verbose=False,
    )


def build_validator() -> Agent:
    return Agent(
        role="Technical Quality Auditor",
        goal="Ensure inventor responses are technically sufficient for patent drafting.",
        backstory=(
            "You are a strict technical editor. Reject answers that are vague, "
            "non-numeric, or overly brief. Require specific units (mm, microns, C) "
            "and step-by-step process details."
        ),
        llm=get_llm(),
        verbose=True,
    )
