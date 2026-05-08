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
from services.ollama_service import OllamaService, get_model_string
from services.vector_store import search
from tasks.task_factory import build_scrutiny_task, _rag_query_for_type

logger = logging.getLogger(__name__)


@dataclass
class ScrutinyResult:
    questions: str
    agent_log: str
    success: bool
    field_of_invention: str = ""   # what was extracted from the document
    novelty: str = ""              # novelty statement extracted from the document
    error: str = ""


# ── Post-processing filter ───────────────────────────────────────────────────

PROHIBITED_PATTERNS = [
    r"(?i)how\s+(?:are|is|was|were)\s+.{0,40}(?:train|trained|fine.?tun)",
    r"(?i)training\s+data",
    r"(?i)how\s+many\s+epochs",
    r"(?i)beyond\s+those\s+specified",
    r"(?i)comply\s+with\s+.{0,30}(?:GDPR|regulation|standard|compliance)",
    r"(?i)(?:audit|review)\s+.{0,30}(?:system|process|compliance)",
    r"(?i)user\s+preferences\s+and\s+customiz",
    # Multimodal/telephone hallucination suppression
    r"(?i)multiphone",
    r"(?i)\bvoice\s+command",
    r"(?i)speech\s+recogni[sz]",
    r"(?i)how\s+many\s+(?:voice\s+)?commands\s+(?:are\s+)?supported",
    # Diagram/flowchart/pseudocode requests cannot be answered in Q&A
    # Broad pattern: allows words between verb and diagram type (e.g. "sequence diagram")
    r"(?i)(?:provide|show|include|give).{0,40}(?:diagram|flowchart|pseudocode|sequence diagram|state diagram)",
    # ML technique hallucination suppression
    r"(?i)(?:reinforcement|active|transfer|federated)\s+learning\s+mechanism",
    r"(?i)any\s+(?:reinforcement|active|unsupervised)\s+learning",
    # Compliance / privacy-regulation questions
    r"(?i)compliance\s+with\s+.{0,40}(?:GDPR|CCPA|HIPAA|privacy\s+law|regulation)",
    r"(?i)privacy\s+settings\s+.{0,30}(?:offer|provide|allow)\s+(?:user|to)",
    # Layer/parameter count questions (model internals — not inventor-answerable)
    r"(?i)how\s+many\s+layers",
]


def filter_prohibited_questions(raw_output: str) -> tuple[str, list[str]]:
    """
    Removes numbered question lines containing prohibited patterns.
    Returns (cleaned_output, list_of_removed_lines).
    """
    lines = raw_output.split("\n")
    clean_lines = []
    removed = []

    for line in lines:
        stripped = line.strip()
        # Only filter numbered question lines (start with digit + dot/paren)
        if re.match(r"^\d+[\.)]", stripped):
            if any(re.search(p, stripped) for p in PROHIBITED_PATTERNS):
                removed.append(stripped)
                continue
        clean_lines.append(line)

    return "\n".join(clean_lines), removed


def clean_crew_output(raw: str) -> str:
    """
    Strips echoed task instructions from LLM output.
    Keeps everything from the FIELD line or first [Theme] bracket onward.
    """
    markers = [
        r"FIELD\s*\(verbatim\)",
        r"\[Theme",
    ]
    for marker in markers:
        match = re.search(marker, raw, re.IGNORECASE)
        if match:
            return raw[match.start():].strip()
    return raw.strip()


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


def extract_novelty(text: str) -> str:
    """
    Extracts a novelty statement from common patent section patterns.
    Returns 'NOT STATED' if no formal novelty language is found.
    """
    patterns = [
        # Summary section capture
        r"SUMMARY OF (?:THE )?INVENTION[\s\S]{0,200}?((?:The (?:present )?invention "
        r"(?:provides|relates to|is directed to|comprises|enables|allows|discloses)[^.]{20,300})\. )",
        # Explicit novelty language
        r"((?:The (?:present )?invention (?:provides|is directed to|comprises|enables|discloses))"
        r"[^.]{20,300}\.)",
        # "In accordance with" pattern
        r"(In accordance with (?:the present )?invention[^.]{20,300}\.)",
        # "A novel / An improved / A new" pattern
        r"((?:A novel|An improved|A new)[^.]{20,300}(?:system|method|apparatus|device)[^.]{0,200}\.)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            novelty = match.group(1).strip()
            if len(novelty) > 30:
                logger.info("Extracted novelty: %s", novelty[:120])
                return novelty
    # Fallback: grab the most specific "More particularly" sentence
    # (common in patents that skip formal novelty language)
    fallback_pattern = r"(More particularly[^.]{20,400}\.)"
    match = re.search(fallback_pattern, text, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        if len(candidate) > 40:
            logger.info("Extracted novelty (fallback): %s", candidate[:120])
            return candidate

    logger.info("Could not extract novelty statement from context")
    return "NOT STATED"


def extract_technical_assertions(context: str, model_string: str) -> str:
    """
    Lightweight pre-pass: calls the LLM directly to identify VAGUE or ABSENT
    technical assertions in the document. Returns a formatted CLAIM/GAP string
    for injection into the scrutiny task prompt.
    Silent fail — returns "" on any error so the main scrutiny task is unaffected.
    """
    import requests as _requests

    api_model = model_string.replace("ollama/", "").replace("mistral/", "")
    prompt = (
        "You are a patent enablement analyst. Read this patent draft and "
        "identify technical assertions that are VAGUE (described without mechanism) "
        "or ABSENT (implied but never described).\n\n"
        "Output ONLY items where the document makes a claim but does not explain HOW.\n"
        "Format strictly as:\n"
        "CLAIM: <what the document asserts>\n"
        "GAP: <what mechanism/value/protocol is missing>\n\n"
        "Maximum 8 items. Do not explain or add commentary.\n\n"
        "Patent draft:\n"
        + context[:4000]
    )

    # Mistral API path
    if cfg.LLM_PROVIDER == "mistral_api":
        if not cfg.MISTRAL_API_KEY:
            logger.warning("[ASSERTIONS] MISTRAL_API_KEY not set — skipping pre-pass")
            return ""
        try:
            r = _requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {cfg.MISTRAL_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": api_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 800,
                },
                timeout=60,
            )
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"]
                if "CLAIM:" in content and "GAP:" in content:
                    logger.info("Extracted %d chars of document gaps (Mistral API)", len(content))
                    return content.strip()
            return ""
        except Exception as exc:
            logger.warning("[ASSERTIONS] Mistral API pre-pass failed: %s", exc)
            return ""

    # Ollama path
    try:
        r = _requests.post(
            f"{cfg.OLLAMA_BASE_URL}/api/generate",
            json={"model": api_model, "prompt": prompt, "stream": False},
            timeout=cfg.OLLAMA_GEN_TIMEOUT,
        )
        if r.status_code == 200:
            content = r.json().get("response", "")
            if "CLAIM:" in content and "GAP:" in content:
                logger.info("Extracted %d chars of document gaps", len(content))
                return content.strip()
    except Exception as exc:
        logger.warning("extract_technical_assertions failed (non-fatal): %s", exc)
    return ""


def build_two_pass_context(collection, patent_type: str) -> Tuple[str, str, str]:
    """
    Two-pass RAG retrieval.

    Returns
    -------
    (combined_context, field_of_invention, novelty)
      combined_context   : Pass1 + Pass2 chunks joined, deduplicated
      field_of_invention : extracted field sentence (may be empty)
      novelty            : extracted novelty statement (may be 'NOT STATED')
    """
    # Pass 1 — broad, invention-neutral query to capture abstract/field section
    pass1_query = (
        "field of invention technical field abstract summary novel contribution "
        "relates to present invention directed to"
    )
    pass1_context = search(collection, pass1_query, n_results=8)

    # Extract field of invention and novelty from pass 1
    field = extract_field_of_invention(pass1_context)
    novelty = extract_novelty(pass1_context)

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
    return combined_context, field, novelty


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
        if cfg.LLM_PROVIDER == "mistral_api":
            if not cfg.MISTRAL_API_KEY:
                return ScrutinyResult(
                    questions="", agent_log="", success=False,
                    error="MISTRAL_API_KEY is not set in .env",
                )
            model_string = get_model_string()
            logger.info("Using Mistral API model: %s", model_string)
        else:
            ollama.ensure_running()
            model_string = ollama.resolve_model()
    except Exception as exc:
        return ScrutinyResult(questions="", agent_log="", success=False, error=str(exc))

    # Pre-flight LLM stack check (Ollama only)
    if cfg.LLM_PROVIDER != "mistral_api":
        ok, diag_msg = ollama.diagnose_llm_stack()
        if not ok:
            return ScrutinyResult(questions="", agent_log="", success=False, error=diag_msg)

        ok, msg = ollama.test_generation(model_string)
        if not ok:
            return ScrutinyResult(
                questions="", agent_log="", success=False,
                error=f"Model generation test failed: {msg}",
            )

    # 2-pass RAG — captures field of invention before domain-specific chunks
    context, field_of_invention, novelty = build_two_pass_context(collection, patent_type)

    # Pre-pass: extract vague/absent technical assertions for mandatory question targeting
    document_gaps = extract_technical_assertions(context, model_string)

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
        novelty=novelty,
        document_gaps=document_gaps,
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

    cleaned, removed = filter_prohibited_questions(raw)
    cleaned = clean_crew_output(cleaned)
    if removed:
        logger.warning(
            "[FILTER] Removed %d prohibited question(s):\n%s",
            len(removed),
            "\n".join(f"  - {q}" for q in removed),
        )

    return ScrutinyResult(
        questions=cleaned,
        agent_log=captured_log,
        success=True,
        field_of_invention=field_of_invention,
        novelty=novelty,
    )
