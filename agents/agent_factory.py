"""
agents/agent_factory.py
Build all CrewAI agents.  No Streamlit, no global state.
Call build_*() and pass the result to the task factories.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from crewai import Agent

import config.settings as cfg
from services.ollama_service import OllamaService

logger = logging.getLogger(__name__)


def _load_domain_config(patent_type: str) -> dict:
    """Load domain entry from patent_types.json. Returns empty dict on failure."""
    try:
        data = json.loads(cfg.PATENT_TYPES_JSON.read_text(encoding="utf-8"))
        return data.get(patent_type, {})
    except Exception:
        return {}


def _build_backstory(patent_type: str, user_notes: str = "", domain_cfg: Optional[dict] = None) -> str:
    if domain_cfg is None:
        domain_cfg = _load_domain_config(patent_type)

    focus   = ", ".join(domain_cfg.get("focus_areas", []))
    units   = ", ".join(domain_cfg.get("technical_units", []))
    role    = domain_cfg.get("role", f"Patent Specialist ({patent_type})")
    parts   = [f"{role} with deep expertise in {patent_type}."]
    if focus:
        parts.append(f"Focus areas include: {focus}.")
    if units:
        parts.append(f"Requires precise measurements in: {units}.")
    if user_notes:
        parts.append(f"User notes: {user_notes}.")
    return " ".join(parts[:3])


# ── Scrutinizer ───────────────────────────────────────────────────────────────

def build_scrutinizer(
    patent_type: str,
    llm_model: str,
    custom_role: Optional[str] = None,
    custom_backstory: Optional[str] = None,
) -> Agent:
    domain_cfg = _load_domain_config(patent_type)
    role      = custom_role      or domain_cfg.get("role", "Patent Enablement Specialist")
    backstory = custom_backstory or _build_backstory(patent_type, "", domain_cfg)

    return Agent(
        role=role,
        goal="Identify the technical parameters required to meet the 35 U.S.C. 112 'Enablement' standard.",
        backstory=backstory,
        llm=llm_model,
        verbose=True,
    )


# ── Consolidator ──────────────────────────────────────────────────────────────

def build_consolidator(llm_model: str) -> Agent:
    return Agent(
        role="Technical Integration Specialist",
        goal="Incorporate every specific technical detail from the Q&A into Draft 1.",
        backstory=(
            "You are a meticulous patent engineer. Your job is NOT to summarise. "
            "Expand Draft 1 by injecting precise data points from the Q&A. "
            "If Draft 1 says 'thin layer' and Q&A says '5 microns', replace it. "
            "Preserve the original professional tone while maximising technical density."
        ),
        llm=llm_model,
        verbose=True,
    )


# ── Classifier ────────────────────────────────────────────────────────────────

def build_classifier(llm_model: str) -> Agent:
    return Agent(
        role="Patent Classification Analyst",
        goal="Identify the most appropriate technical domain for this invention.",
        backstory=(
            "You are a patent classification expert across Mechanical, Electronics, "
            "Software, Chemical, Materials, and Medical Devices domains. "
            "You base decisions on technical content only (components, processes, materials, algorithms) "
            "and output a concise JSON object with your reasoning."
        ),
        llm=llm_model,
        verbose=False,
    )


# ── Validator ─────────────────────────────────────────────────────────────────

def build_validator(llm_model: str) -> Agent:
    return Agent(
        role="Technical Quality Auditor",
        goal="Ensure inventor responses are technically sufficient for patent drafting.",
        backstory=(
            "You are a strict technical editor. Reject answers that are vague, "
            "non-numeric, or overly brief. Require specific units (mm, microns, °C) "
            "and step-by-step process details."
        ),
        llm=llm_model,
        verbose=True,
    )


# ── Convenience: resolve model once and build all agents ─────────────────────

def build_patent_agents(patent_type: str, **kwargs) -> dict:
    """
    Resolve model, ensure Ollama is running, build scrutinizer + consolidator.
    Returns dict with keys: llm_model, scrutinizer, consolidator.
    Raises OllamaNotAvailableError / NoModelsFoundError on failure.
    """
    ollama = OllamaService()
    ollama.ensure_running()
    llm_model = ollama.resolve_model()

    return {
        "llm_model":   llm_model,
        "scrutinizer": build_scrutinizer(patent_type, llm_model, **kwargs),
        "consolidator": build_consolidator(llm_model),
    }
