"""
workflows/consolidation_workflow.py
Run the Consolidation (Draft 2 generation) crew.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import List, Optional

from crewai import Crew

import config.settings as cfg
from agents.agent_factory import build_consolidator
from services.cloud_llm_service import get_llm, is_cloud_provider
from services.ollama_service import OllamaService
from services.vector_store import search
from tasks.task_factory import (
    build_consolidation_task,
    split_audit_log,
    verify_draft_inclusion,
)

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationResult:
    draft2: str
    audit_log: str
    missing_sentences: List[str]
    is_valid: bool
    success: bool
    error: str = ""


def run(
    draft1_collection,
    qa_collection,
    qa_content_override: Optional[str] = None,
    timeout: int = cfg.CREW_TIMEOUT_SECONDS,
) -> ConsolidationResult:
    """
    Run consolidation crew and return Draft 2.
    Never raises – check .success.

    Parameters
    ----------
    draft1_collection    : chromadb.Collection for the original patent sheet
    qa_collection        : chromadb.Collection for the Q&A answers
    qa_content_override  : if provided, use this raw text instead of RAG search
                           (more reliable for short Q&A docs)
    """
    ollama = OllamaService()
    try:
        ollama.ensure_running()
        llm_model = ollama.resolve_model()
    except Exception as exc:
        return ConsolidationResult(
            draft2="", audit_log="", missing_sentences=[],
            is_valid=False, success=False, error=str(exc),
        )

    # ── Build contexts ──────────────────────────────────────────────────────
    draft1_context = ""
    try:
        draft1_context = search(
            draft1_collection,
            "technical invention layers process materials",
            n_results=15,
        )
    except Exception as exc:
        logger.warning("Draft1 RAG search failed: %s", exc)

    qa_context = qa_content_override or ""
    if not qa_context and qa_collection:
        try:
            qa_context = search(
                qa_collection,
                "answers technical parameters thickness curing materials",
                n_results=5,
            )
        except Exception as exc:
            logger.warning("Q&A RAG search failed: %s", exc)

    if not draft1_context and not qa_context:
        return ConsolidationResult(
            draft2="", audit_log="", missing_sentences=[],
            is_valid=False, success=False,
            error="No content found in Draft1 or Q&A collections.",
        )

    # ── Build crew ──────────────────────────────────────────────────────────
    consolidator = build_consolidator(llm_model)
    task = build_consolidation_task(consolidator, draft1_context, qa_context)
    crew = Crew(agents=[consolidator], tasks=[task], verbose=True)

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
        return ConsolidationResult(
            draft2="", audit_log="", missing_sentences=[],
            is_valid=False, success=False,
            error=f"Consolidation timed out after {timeout}s.",
        )

    if holder["error"]:
        return ConsolidationResult(
            draft2="", audit_log="", missing_sentences=[],
            is_valid=False, success=False, error=holder["error"],
        )

    raw = str(
        holder["result"].raw
        if hasattr(holder["result"], "raw")
        else holder["result"]
    )

    draft2, audit_log = split_audit_log(raw)
    missing = verify_draft_inclusion(draft1_context, draft2)

    return ConsolidationResult(
        draft2=draft2,
        audit_log=audit_log,
        missing_sentences=missing,
        is_valid=len(missing) == 0,
        success=True,
    )
