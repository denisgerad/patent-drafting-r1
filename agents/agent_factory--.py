"""
agents/agent_factory.py
Build all CrewAI agents.  No Streamlit, no global state.
Call build_*() and pass the result to the task factories.

LLM COMPATIBILITY
-----------------
CrewAI ≥ 0.80 requires a proper LLM object (backed by litellm) rather than
a raw "ollama/model-name" string.  `make_llm()` handles this transparently:

  - crewai ≥ 0.80  →  uses crewai.LLM(model=..., base_url=...)
  - crewai < 0.80  →  falls back to plain string (old behaviour)

You must also have litellm installed:
    pip install litellm

The model string passed to LLM should use the litellm Ollama prefix:
    "ollama/mistral:7b-instruct-q4_K_M"
combined with base_url="http://localhost:11434".
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from crewai import Agent

import config.settings as cfg
from services.ollama_service import OllamaService

logger = logging.getLogger(__name__)


# ── LLM factory ──────────────────────────────────────────────────────────────

def make_llm(model_string: str) -> Any:
    """
    Return an LLM object compatible with the installed CrewAI version.

    Parameters
    ----------
    model_string : str
        Model resolved by OllamaService.resolve_model(), e.g.
        "ollama/mistral:7b-instruct-q4_K_M"

    Compatibility matrix
    --------------------
    crewai ≥ 0.80   →  crewai.LLM(model=..., base_url=...) via litellm
    crewai < 0.80   →  plain string passthrough (old behaviour)
    fallback        →  ChatOllama from langchain_ollama or langchain_community
    """
    base_url = cfg.OLLAMA_BASE_URL

    # ── Strategy 1: crewai.LLM (crewai ≥ 0.80 + litellm installed) ──────────
    try:
        from crewai import LLM  # available from crewai 0.80+
        llm = LLM(model=model_string, base_url=base_url)
        logger.info("Using crewai.LLM with model=%s base_url=%s", model_string, base_url)
        return llm
    except ImportError:
        pass  # crewai < 0.80 – try next strategy
    except Exception as exc:
        logger.warning("crewai.LLM construction failed (%s) – trying fallbacks", exc)

    # ── Strategy 2: langchain_ollama (preferred langchain integration) ────────
    try:
        from langchain_ollama import ChatOllama
        # Strip "ollama/" prefix for langchain
        bare_model = model_string.replace("ollama/", "")
        llm = ChatOllama(model=bare_model, base_url=base_url)
        logger.info("Using langchain_ollama.ChatOllama with model=%s", bare_model)
        return llm
    except ImportError:
        pass

    # ── Strategy 3: langchain_community (older langchain) ────────────────────
    try:
        from langchain_community.chat_models import ChatOllama as ChatOllamaCommunity
        bare_model = model_string.replace("ollama/", "")
        llm = ChatOllamaCommunity(model=bare_model, base_url=base_url)
        logger.info("Using langchain_community.ChatOllama with model=%s", bare_model)
        return llm
    except ImportError:
        pass

    # ── Strategy 4: plain string (crewai < 0.80 fallback) ────────────────────
    logger.warning(
        "All LLM adapters failed – falling back to plain string '%s'. "
        "If you see LiteLLM errors, run: pip install litellm",
        model_string,
    )
    return model_string


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

def build_scrutinizer(
    patent_type: str,
    model_string: str,
    custom_role: Optional[str] = None,
    custom_backstory: Optional[str] = None,
) -> Agent:
    domain_cfg = _load_domain_config(patent_type)
    role      = custom_role      or domain_cfg.get("role", "Patent Enablement Specialist")
    backstory = custom_backstory or _build_backstory(patent_type, "", domain_cfg)

    return Agent(
        role=role,
        goal="Identify technical parameters required to meet the 35 U.S.C. §112 'Enablement' standard.",
        backstory=backstory,
        llm=make_llm(model_string),
        verbose=True,
    )


def build_consolidator(model_string: str) -> Agent:
    return Agent(
        role="Technical Integration Specialist",
        goal="Incorporate every specific technical detail from the Q&A into Draft 1.",
        backstory=(
            "You are a meticulous patent engineer. Your job is NOT to summarise. "
            "Expand Draft 1 by injecting precise data points from the Q&A. "
            "If Draft 1 says 'thin layer' and Q&A says '5 microns', replace it. "
            "Preserve the original professional tone while maximising technical density."
        ),
        llm=make_llm(model_string),
        verbose=True,
    )


def build_classifier(model_string: str) -> Agent:
    return Agent(
        role="Patent Classification Analyst",
        goal="Identify the most appropriate technical domain for this invention.",
        backstory=(
            "You are a patent classification expert across Mechanical, Electronics, "
            "Software, Chemical, Materials, and Medical Devices domains. "
            "You base decisions solely on technical content (components, processes, "
            "materials, algorithms) and output a concise JSON object with your reasoning."
        ),
        llm=make_llm(model_string),
        verbose=False,
    )


def build_validator(model_string: str) -> Agent:
    return Agent(
        role="Technical Quality Auditor",
        goal="Ensure inventor responses are technically sufficient for patent drafting.",
        backstory=(
            "You are a strict technical editor. Reject answers that are vague, "
            "non-numeric, or overly brief. Require specific units (mm, microns, °C) "
            "and step-by-step process details."
        ),
        llm=make_llm(model_string),
        verbose=True,
    )


# ── Convenience builder ───────────────────────────────────────────────────────

def build_patent_agents(patent_type: str, **kwargs) -> dict:
    """
    Resolve Ollama model, ensure server is running, build scrutinizer + consolidator.
    Returns dict: {llm_model, scrutinizer, consolidator}.
    """
    ollama = OllamaService()
    ollama.ensure_running()
    model_string = ollama.resolve_model()

    return {
        "llm_model":    model_string,
        "scrutinizer":  build_scrutinizer(patent_type, model_string, **kwargs),
        "consolidator": build_consolidator(model_string),
    }
