"""
workflows/scrutiny_workflow.py
Run the Scrutiny (gap-analysis) crew with timeout and verbose log capture.

2-Pass RAG strategy (fixes hallucinated invention scope)
─────────────────────────────────────────────────────────
Pass 1 — BROAD, neutral query: pulls abstract / field-of-invention / title chunks
          → extract_field_of_invention() parses out the stated field text
Pass 2 — TARGETED query: uses the extracted field name to pull relevant
          technical detail chunks

Both passes' contexts are concatenated and passed to the task.
The extracted field string is also injected into the task as `field_of_invention`
so the LLM must quote it verbatim before generating any questions.
"""
from __future__ import annotations

import logging
import re
import sys
import threading
from dataclasses import dataclass
from io import StringIO
from typing import Optional, Tuple

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
    field_of_invention: str = ""   # what was extracted from the document
    error: str = ""


# ── Field extraction helpers ──────────────────────────────────────────────────

def extract_field_of_invention(context: str) -> str:
    """
    Extract the COMPLETE field-of-invention block from RAG context, including
    the application/deployment context that follows the core invention statement.

    Strategy
    --------
    1. Locate the "Field of the Invention" heading (or equivalent).
    2. Capture everything from that heading up to the next major section heading
       (Background, Summary, Description, Claims, etc.) — this preserves the
       full "applicable to... where conditions may cause..." sentence.
    3. Fall back to shorter pattern matches if the full block is not found.

    Returns the extracted text (may be multiple sentences), or empty string.
    """
    # Strategy 1 — capture full paragraph under "Field of the Invention" heading
    full_block = re.search(
        r"(?:field of (?:the )?invention|technical field)"
        r"[:\s\n]+"
        r"(.*?)"            # capture everything after the heading
        r"(?=\n\s*\n|"   # until blank line
        r"background|summary|description|claims|brief|drawings|"
        r"detailed description|objects? of|prior art)",
        context,
        re.IGNORECASE | re.DOTALL,
    )
    if full_block:
        field = re.sub(r"\s+", " ", full_block.group(1).strip())
        if len(field) > 15:
            logger.info("Extracted full field block: %s", field[:200])
            return field

    # Strategy 2 — "present invention relates to / is applicable to" multi-sentence
    multi = re.search(
        r"(?:present invention|invention relates?|invention is)[^.]{0,200}"
        r"(?:applicable to|used in|for use in|intended for)[^.]{0,400}\.",
        context,
        re.IGNORECASE | re.DOTALL,
    )
    if multi:
        field = re.sub(r"\s+", " ", multi.group(0).strip())
        if len(field) > 15:
            logger.info("Extracted multi-sentence field: %s", field[:200])
            return field

    # Strategy 3 — shorter single-sentence fallbacks
    short_patterns = [
        r"(?:relates? to|directed to|pertains? to)[:\s]+([^\n]{20,400})",
        r"(?:present invention)[:\s]+([^\n]{20,400})",
        r"(?:abstract|summary)[:\s\n]+([^\n]{20,400})",
    ]
    for pattern in short_patterns:
        m = re.search(pattern, context, re.IGNORECASE | re.DOTALL)
        if m:
            field = re.sub(r"\s+", " ", m.group(1).strip())
            if len(field) > 15:
                logger.info("Extracted short field: %s", field[:120])
                return field

    logger.warning("Could not extract field of invention from context")
    return ""


def build_two_pass_context(collection, patent_type: str) -> Tuple[str, str]:
    """
    Two-pass RAG retrieval.

    Returns
    -------
    (combined_context, field_of_invention)
      combined_context   : Pass1 + Pass2 chunks joined, deduplicated
      field_of_invention : extracted field sentence (may be empty)
    """
    # Pass 1 — broad, invention-neutral query to capture abstract/field section
    pass1_query = (
        "field of invention technical field abstract summary novel contribution "
        "relates to present invention directed to"
    )
    pass1_context = search(collection, pass1_query, n_results=8)

    # Extract field of invention from pass 1
    field = extract_field_of_invention(pass1_context)

    # Pass 2 — targeted query: use the field text if found, else fall back to domain query
    if field:
        # Use the first ~60 chars of the field as a targeted query
        pass2_query = field[:60]
    else:
        pass2_query = _rag_query_for_type(patent_type)

    pass2_context = search(collection, pass2_query, n_results=8)

    # Deduplicate: combine, split on double newline, unique paragraphs, rejoin
    combined = pass1_context + "\n\n" + pass2_context
    seen = set()
    deduped_parts = []
    for para in combined.split("\n\n"):
        key = para.strip()
        if key and key not in seen:
            seen.add(key)
            deduped_parts.append(key)

    combined_context = "\n\n".join(deduped_parts)
    logger.info(
        "2-pass RAG: %d pass1 + %d pass2 chunks → %d unique paragraphs",
        len(pass1_context.split("\n\n")),
        len(pass2_context.split("\n\n")),
        len(deduped_parts),
    )
    return combined_context, field


# ── Main workflow ─────────────────────────────────────────────────────────────

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
        model_string = ollama.resolve_model()
    except Exception as exc:
        return ScrutinyResult(questions="", agent_log="", success=False, error=str(exc))

    # Pre-flight: check litellm / crewai.LLM stack
    ok, diag_msg = ollama.diagnose_llm_stack()
    if not ok:
        return ScrutinyResult(questions="", agent_log="", success=False, error=diag_msg)

    # Verify model can generate text
    ok, msg = ollama.test_generation(model_string)
    if not ok:
        return ScrutinyResult(
            questions="", agent_log="", success=False,
            error=f"Model generation test failed: {msg}",
        )

    # 2-pass RAG — captures field of invention before domain-specific chunks
    context, field_of_invention = build_two_pass_context(collection, patent_type)

    scrutinizer = build_scrutinizer(
        patent_type, model_string,
        custom_role=custom_role,
        custom_backstory=custom_backstory,
    )
    task = build_scrutiny_task(
        scrutinizer,
        context,
        patent_type,
        field_of_invention=field_of_invention,
    )
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
    return ScrutinyResult(
        questions=raw,
        agent_log=captured_log,
        success=True,
        field_of_invention=field_of_invention,
    )
