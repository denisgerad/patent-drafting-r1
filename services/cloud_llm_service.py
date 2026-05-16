"""
services/cloud_llm_service.py
Unified LLM provider service.

Replaces the Ollama-only dependency with a provider-agnostic interface.
The LLM_PROVIDER environment variable selects the backend at startup:

  LLM_PROVIDER=ollama   → local Ollama (existing behaviour, default)
  LLM_PROVIDER=azure    → Azure OpenAI Service
  LLM_PROVIDER=claude   → Anthropic Claude API
  LLM_PROVIDER=openai   → OpenAI API direct

All providers return a crewai.LLM object (backed by litellm) that can be
passed directly to crewai.Agent(llm=...).

NOTHING in tasks/, workflows/, or ui/ changes — only this service and
agents/agent_factory.py (which calls get_llm()) need to know about providers.
"""
from __future__ import annotations

import logging
from typing import Any

import config.settings as cfg

logger = logging.getLogger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────

def get_llm() -> Any:
    """
    Return a crewai.LLM object for the configured provider.

    Call this once per workflow run. The returned object is passed to
    agents via Agent(llm=get_llm()).

    For cloud providers, writes an audit entry to llm_audit.log so there
    is a local record of every time patent content was sent externally.

    Raises
    ------
    ValueError  if LLM_PROVIDER is unrecognised or required env vars are missing.
    ImportError if crewai or litellm are not installed.
    """
    provider = cfg.LLM_PROVIDER

    if provider == "azure":
        llm = _make_azure_llm()
    elif provider == "claude":
        llm = _make_claude_llm()
    elif provider == "openai":
        llm = _make_openai_llm()
    elif provider == "ollama":
        return _make_ollama_llm()   # local — no audit needed
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{provider}'. "
            "Valid values: ollama, azure, claude, openai"
        )

    # Write audit entry for every cloud LLM call
    _write_audit_entry(provider)
    return llm


def _write_audit_entry(provider: str) -> None:
    """
    Append one line to llm_audit.log recording that patent content
    was transmitted to a cloud provider API.
    Format: ISO-8601 timestamp | provider | model/deployment
    """
    import datetime, pathlib
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    model_info = {
        "azure":  f"azure/{cfg.AZURE_DEPLOYMENT_NAME}",
        "claude": cfg.CLAUDE_MODEL,
        "openai": cfg.OPENAI_MODEL,
    }.get(provider, provider)

    log_path = pathlib.Path("llm_audit.log")
    entry = f"{timestamp} | provider={provider} | model={model_info} | DATA SENT TO CLOUD API\n"
    try:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(entry)
        logger.info("Audit entry written: %s", entry.strip())
    except Exception as exc:
        logger.warning("Could not write audit log: %s", exc)


def provider_name() -> str:
    """Human-readable name of the active provider."""
    names = {
        "azure":  f"Azure OpenAI ({cfg.AZURE_DEPLOYMENT_NAME})",
        "claude": f"Anthropic Claude ({cfg.CLAUDE_MODEL})",
        "openai": f"OpenAI ({cfg.OPENAI_MODEL})",
        "ollama": "Ollama (local)",
    }
    return names.get(cfg.LLM_PROVIDER, cfg.LLM_PROVIDER)


def is_cloud_provider() -> bool:
    """True when the provider requires an internet connection and API key."""
    return cfg.LLM_PROVIDER in ("azure", "claude", "openai")


# ── Provider implementations ──────────────────────────────────────────────────

def _require_crewai_llm():
    """Import crewai.LLM or raise a clear error."""
    try:
        from crewai import LLM
        return LLM
    except ImportError as exc:
        raise ImportError(
            "crewai.LLM is not available. "
            "Install: pip install crewai litellm"
        ) from exc


def _make_azure_llm():
    """
    Build an LLM object for Azure OpenAI.

    Required env vars (.env):
        AZURE_OPENAI_API_KEY      your Azure resource API key
        AZURE_OPENAI_ENDPOINT     https://YOUR-RESOURCE.openai.azure.com/
        AZURE_OPENAI_DEPLOYMENT   your deployment name (e.g. gpt-4o)
        AZURE_OPENAI_API_VERSION  (optional, default 2024-02-01)

    litellm model string format: "azure/<deployment_name>"
    """
    _validate(
        ("AZURE_OPENAI_API_KEY",  cfg.AZURE_API_KEY),
        ("AZURE_OPENAI_ENDPOINT", cfg.AZURE_API_BASE),
        ("AZURE_OPENAI_DEPLOYMENT", cfg.AZURE_DEPLOYMENT_NAME),
        provider="azure",
    )

    LLM = _require_crewai_llm()

    model_string = f"azure/{cfg.AZURE_DEPLOYMENT_NAME}"

    llm = LLM(
        model=model_string,
        api_key=cfg.AZURE_API_KEY,
        base_url=cfg.AZURE_API_BASE,
        api_version=cfg.AZURE_API_VERSION,
    )

    logger.info(
        "Azure OpenAI LLM: deployment=%s endpoint=%s version=%s",
        cfg.AZURE_DEPLOYMENT_NAME, cfg.AZURE_API_BASE, cfg.AZURE_API_VERSION,
    )
    return llm


def _make_claude_llm():
    """
    Build an LLM object for Anthropic Claude.

    Required env vars (.env):
        ANTHROPIC_API_KEY   your Anthropic API key (sk-ant-...)
        CLAUDE_MODEL        model ID (default claude-sonnet-4-20250514)

    Available models:
        claude-opus-4-20250514          most capable, slower, higher cost
        claude-sonnet-4-20250514        balanced capability and speed (recommended)
        claude-haiku-4-5-20251001      fastest, lowest cost

    litellm model string format: "anthropic/<model-id>"
    """
    _validate(
        ("ANTHROPIC_API_KEY", cfg.CLAUDE_API_KEY),
        provider="claude",
    )

    LLM = _require_crewai_llm()

    model_string = f"anthropic/{cfg.CLAUDE_MODEL}"

    llm = LLM(
        model=model_string,
        api_key=cfg.CLAUDE_API_KEY,
    )

    logger.info("Anthropic Claude LLM: model=%s", cfg.CLAUDE_MODEL)
    return llm


def _make_openai_llm():
    """
    Build an LLM object for OpenAI direct.

    Required env vars:
        OPENAI_API_KEY   your OpenAI API key (sk-...)
        OPENAI_MODEL     model name (default gpt-4o)
    """
    _validate(
        ("OPENAI_API_KEY", cfg.OPENAI_API_KEY),
        provider="openai",
    )

    LLM = _require_crewai_llm()

    llm = LLM(
        model=cfg.OPENAI_MODEL,
        api_key=cfg.OPENAI_API_KEY,
    )

    logger.info("OpenAI LLM: model=%s", cfg.OPENAI_MODEL)
    return llm


def _make_ollama_llm():
    """
    Build an LLM object for local Ollama.
    Resolves model automatically from running Ollama instance,
    or uses OLLAMA_MODEL env var if set.
    """
    # Import here to avoid circular dependency
    from services.ollama_service import OllamaService

    ollama = OllamaService()
    ollama.ensure_running()
    model_string = ollama.resolve_model()

    LLM = _require_crewai_llm()

    llm = LLM(
        model=model_string,
        base_url=cfg.OLLAMA_BASE_URL,
    )

    logger.info("Ollama LLM: model=%s", model_string)
    return llm


# ── Validation helper ─────────────────────────────────────────────────────────

def _validate(*name_value_pairs: tuple[str, str], provider: str) -> None:
    """Raise ValueError with a clear message if any required env var is empty."""
    missing = [name for name, value in name_value_pairs if not value]
    if missing:
        raise ValueError(
            f"LLM_PROVIDER='{provider}' requires the following env vars "
            f"(missing or empty): {missing}. "
            f"Set them in your .env file."
        )
