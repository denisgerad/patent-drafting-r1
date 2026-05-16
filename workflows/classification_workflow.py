"""
workflows/classification_workflow.py
Encapsulates the full domain auto-classification flow:
  1. RAG search over Draft1
  2. Run classifier agent
  3. Parse + validate JSON result
  4. Return structured ClassificationResult

The UI layer calls run() and decides whether to show the override panel.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import List, Optional

from crewai import Crew

import config.settings as cfg
from agents.agent_factory import build_classifier
from services.cloud_llm_service import get_llm, is_cloud_provider
from services.ollama_service import OllamaService
from services.vector_store import search
from tasks.task_factory import build_classification_task, parse_classification_result

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    primary_domain: str
    secondary_domains: List[str] = field(default_factory=list)
    confidence: float = 0.0
    justification: str = ""
    success: bool = True
    error: str = ""

    @property
    def is_high_confidence(self) -> bool:
        return self.confidence >= cfg.CLASSIFIER_CONFIDENCE

    def to_dict(self) -> dict:
        return {
            "primary_domain":   self.primary_domain,
            "secondary_domains": self.secondary_domains,
            "confidence":       self.confidence,
            "justification":    self.justification,
            "success":          self.success,
            "error":            self.error,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ClassificationResult":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def run(collection, timeout: int = 120) -> ClassificationResult:
    """
    Run auto-classification using an Ollama-backed classifier agent.

    Parameters
    ----------
    collection : chromadb.Collection  –  Draft1 vector store
    timeout    : seconds before we give up and return a failure result

    Returns
    -------
    ClassificationResult – always returns (never raises); check .success
    """
    # 1. Ensure LLM is available
    if not is_cloud_provider():
        ollama = OllamaService()
        try:
            ollama.ensure_running()
        except Exception as exc:
            logger.error("Ollama unavailable for classification: %s", exc)
            return ClassificationResult(
                primary_domain="Electronics",
                success=False,
                error=f"Ollama unavailable: {exc}",
            )

    # 2. RAG context for classifier
    context = search(
        collection,
        "technical components materials processes algorithms systems methods implementation",
        n_results=cfg.CLASSIFIER_RAG_RESULTS,
    )

    # 3. Build agent + task
    classifier = build_classifier()
    task       = build_classification_task(classifier, context)
    crew       = Crew(agents=[classifier], tasks=[task], verbose=False)

    # 4. Run with timeout
    holder: dict = {"result": None, "error": None}

    def _run():
        try:
            holder["result"] = crew.kickoff()
        except Exception as exc:
            holder["error"] = str(exc)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        return ClassificationResult(
            primary_domain="Electronics",
            success=False,
            error=f"Classification timed out after {timeout}s.",
        )

    if holder["error"]:
        return ClassificationResult(
            primary_domain="Electronics",
            success=False,
            error=holder["error"],
        )

    # 5. Parse JSON from agent output
    raw = str(
        holder["result"].raw
        if hasattr(holder["result"], "raw")
        else holder["result"]
    )
    parsed = parse_classification_result(raw)

    if not parsed or "primary_domain" not in parsed:
        return ClassificationResult(
            primary_domain="Electronics",
            success=False,
            error=f"Could not parse classifier output: {raw[:300]}",
        )

    return ClassificationResult(
        primary_domain   = parsed.get("primary_domain", "Electronics"),
        secondary_domains= parsed.get("secondary_domains", []),
        confidence       = float(parsed.get("confidence", 0.0)),
        justification    = parsed.get("justification", ""),
        success          = True,
    )
