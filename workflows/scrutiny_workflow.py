"""
workflows/scrutiny_workflow.py
Run the Scrutiny (gap-analysis) crew with timeout and verbose log capture.
"""
from __future__ import annotations

import logging
import sys
import threading
from dataclasses import dataclass
from io import StringIO
from typing import Optional

from crewai import Crew

import config.settings as cfg
from agents.agent_factory import build_scrutinizer
from services.ollama_service import OllamaService
from services.vector_store import search
from tasks.task_factory import build_scrutiny_task, _rag_query_for_type

logger = logging.getLogger(__name__)


@dataclass
class ScrutinyResult:
    questions: str
    agent_log: str
    success: bool
    error: str = ""


def run(
    collection,
    patent_type: str,
    custom_role: Optional[str] = None,
    custom_backstory: Optional[str] = None,
    timeout: int = cfg.CREW_TIMEOUT_SECONDS,
) -> ScrutinyResult:
    """
    Run scrutiny crew and return questions + verbose log.
    Never raises – check .success.
    """
    ollama = OllamaService()
    try:
        ollama.ensure_running()
        llm_model = ollama.resolve_model()
    except Exception as exc:
        return ScrutinyResult(questions="", agent_log="", success=False, error=str(exc))

    # Verify model can generate text
    ok, msg = ollama.test_generation(llm_model)
    if not ok:
        return ScrutinyResult(
            questions="", agent_log="", success=False,
            error=f"Model generation test failed: {msg}",
        )

    rag_query = _rag_query_for_type(patent_type)
    context   = search(collection, rag_query, n_results=5)

    scrutinizer = build_scrutinizer(
        patent_type, llm_model,
        custom_role=custom_role,
        custom_backstory=custom_backstory,
    )
    task = build_scrutiny_task(scrutinizer, context, patent_type)
    crew = Crew(agents=[scrutinizer], tasks=[task], verbose=True)

    holder: dict = {"result": None, "error": None}
    log_capture = StringIO()

    def _run():
        old_stdout = sys.stdout
        sys.stdout = log_capture
        try:
            holder["result"] = crew.kickoff()
        except Exception as exc:
            holder["error"] = str(exc)
        finally:
            sys.stdout = old_stdout

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    captured_log = log_capture.getvalue()

    if thread.is_alive():
        return ScrutinyResult(
            questions="", agent_log=captured_log, success=False,
            error=f"Scrutiny timed out after {timeout}s.",
        )

    if holder["error"]:
        return ScrutinyResult(
            questions="", agent_log=captured_log, success=False,
            error=holder["error"],
        )

    raw = str(
        holder["result"].raw
        if hasattr(holder["result"], "raw")
        else holder["result"]
    )
    return ScrutinyResult(questions=raw, agent_log=captured_log, success=True)
