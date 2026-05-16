"""
workflows/scrutiny_workflow.py
Run the Scrutiny (gap-analysis) crew with timeout, verbose log capture,
post-processing hallucination validation, and readiness rating.

3-Pass RAG strategy
────────────────────
Pass 1 — broad neutral query    → field / abstract chunks
Pass 2 — targeted query (priority order):
          (a) extracted field text       (most precise)
          (b) checklist product-type query (overrides user domain selection)
          (c) domain generic query        (last resort)
Pass 3 — checklist-specific query when different from pass 2

This fixes the bug where selecting "Electronics" or "Optics/Display" for
a heater film patent caused the wrong RAG chunks to be retrieved, because
the domain-generic query pulled LCD circuit chunks instead of heater film
property chunks. The checklist match on field_of_invention now determines
the pass-2 query, making it independent of the user's domain selection.

Mechanism extraction
─────────────────────
Three-strategy fallback:
  1. Document keyword search (most reliable)
  2. Checklist-driven hint (new — works when document is thin)
  3. Field statement word extraction (last resort)

Post-processing
───────────────
_validate_novelty_line() checks the NOVELTY line with 4-gram overlap.
_postprocess_questions() applies the validator and logs corrections.
rate_questions() scores output against domain reference questions.
"""
from __future__ import annotations

import json
import logging
import re
import sys
import threading
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Optional, Tuple

from crewai import Crew

import config.settings as cfg
from agents.agent_factory import build_scrutinizer
from services.cloud_llm_service import get_llm, is_cloud_provider, provider_name
from services.ollama_service import OllamaService
from services.question_rater import rate as rate_questions
from services.vector_store import search
from tasks.task_factory import build_scrutiny_task, _rag_query_for_type

logger = logging.getLogger(__name__)

_CHECKLISTS_PATH = Path(__file__).parent.parent / "product_type_checklists.json"


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class ScrutinyResult:
    questions: str
    agent_log: str
    success: bool
    field_of_invention: str = ""
    mechanism: str = ""
    readiness_report: Optional[object] = None
    error: str = ""


# ── Checklist-driven RAG queries ──────────────────────────────────────────────
# More precise than generic domain queries. Keyed by product type.
# These ensure relevant chunks are retrieved regardless of domain selection.

_CHECKLIST_RAG_QUERIES: dict[str, str] = {
    "flexible_heater_film":       "polymer film heating element resistive layer adhesive bonding thermal activation",
    "light_guide_plate":          "dot pattern extraction structure light guide uniformity dual-edge luminance",
    "led_array_pcb":              "LED array interleaved day NVG PCB activation switching uniform illumination",
    "optical_coating":            "coating layer refractive index deposition transmittance haze thin film",
    "power_electronics":          "switching converter topology magnetic core winding gate drive efficiency",
    "digital_fpga_asic":          "clock domain FPGA logic pipeline timing resource utilisation RTL",
    "sensor_signal_acquisition":  "sensor ADC front-end noise calibration sampling bandwidth",
    "wireless_communication":     "RF antenna modulation link budget protocol transceiver sensitivity",
    "motor_actuator_drive":       "motor FOC commutation current loop PWM drive torque",
    "embedded_firmware":          "RTOS firmware task scheduler interrupt bootloader watchdog safety",
    "algorithm_method_patent":    "algorithm pseudocode input output complexity steps termination",
    "oled_display":               "organic emissive layer OLED stack encapsulation flexible substrate TFT drive circuit lifetime",
    "machine_learning_ai":        "neural network training dataset model architecture inference layers",
    "communication_protocol":     "message format state machine protocol packet handshake reliability",
    "database_data_structure":    "schema index query transaction database storage consistency",
    "identity_verification_system": "verification code token identity mapping cryptographic algorithm",
}

# Mechanism hints — used when document context is too thin for keyword extraction
_MECHANISM_HINTS: dict[str, str] = {
    "flexible_heater_film":       "resistive heating element embedded in flexible polymer film",
    "light_guide_plate":          "dot pattern / light extraction structure on LGP surface",
    "led_array_pcb":              "interleaved day/NVG LED array on single PCB",
    "optical_coating":            "thin film optical layer stack",
    "power_electronics":          "switching power converter topology with magnetic components",
    "digital_fpga_asic":          "digital logic architecture with clock domain management",
    "sensor_signal_acquisition":  "analog front-end and ADC signal chain",
    "wireless_communication":     "RF transceiver and antenna design",
    "motor_actuator_drive":       "motor drive inverter with closed-loop control",
    "embedded_firmware":          "RTOS task scheduler and firmware architecture",
    "algorithm_method_patent":    "computational algorithm with defined input-output mapping",
    "oled_display":               "organic emissive stack with flexible substrate and encapsulation barrier",
    "machine_learning_ai":        "neural network model architecture and training process",
    "communication_protocol":     "message format and state machine protocol",
    "database_data_structure":    "data schema and index structure",
    "identity_verification_system": "time-limited code generation algorithm and identity mapping",
}

# Ordered from most specific to most generic
_MECHANISM_KEYWORDS = [
    r"dot pattern", r"extraction pattern", r"extraction structure",
    r"microstructure", r"diffusion structure", r"scattering structure",
    r"heating element", r"resistive element", r"heater track",
    r"resistive film", r"heater film",
    r"algorithm", r"generation method", r"mapping method",
    r"circuit topology", r"switching topology", r"control loop",
    r"sensor element", r"transducer", r"active layer",
    r"coating layer", r"thin film", r"active region",
    r"novel (?:structure|pattern|method|algorithm|arrangement|configuration)",
    r"characterized by", r"comprising a",
]


# ── Helper functions ──────────────────────────────────────────────────────────

def _match_checklist_type(field_of_invention: str) -> str:
    """
    Match field_of_invention against checklist triggers.
    Returns the matched product type key or empty string.
    Centralised so both RAG query and mechanism extractor use the same match.
    """
    if not field_of_invention:
        return ""
    try:
        checklists = json.loads(_CHECKLISTS_PATH.read_text(encoding="utf-8"))
        for key, entry in checklists.items():
            if key.startswith("_"):
                continue
            for trigger in entry.get("triggers", []):
                if re.search(trigger, field_of_invention, re.IGNORECASE):
                    return key
    except Exception:
        pass
    return ""


def extract_field_of_invention(context: str) -> str:
    """Extract the complete field-of-invention block including application context."""
    # Strategy 1 — full block under heading
    full_block = re.search(
        r"(?:field of (?:the )?invention|technical field)"
        r"[:\s\n]+(.*?)"
        r"(?=\n\s*\n|background|summary|description|claims|brief|"
        r"drawings|detailed description|objects? of|prior art)",
        context, re.IGNORECASE | re.DOTALL,
    )
    if full_block:
        field = re.sub(r"\s+", " ", full_block.group(1).strip())
        if len(field) > 15:
            return field

    # Strategy 2 — multi-sentence with applicability
    multi = re.search(
        r"(?:present invention|invention relates?|invention is)[^.]{0,200}"
        r"(?:applicable to|used in|for use in|intended for)[^.]{0,400}\.",
        context, re.IGNORECASE | re.DOTALL,
    )
    if multi:
        field = re.sub(r"\s+", " ", multi.group(0).strip())
        if len(field) > 15:
            return field

    # Strategy 3 — short fallbacks
    for pattern in [
        r"(?:relates? to|directed to|pertains? to)[:\s]+([^\n]{20,400})",
        r"(?:present invention)[:\s]+([^\n]{20,400})",
        r"(?:abstract|summary)[:\s\n]+([^\n]{20,400})",
    ]:
        m = re.search(pattern, context, re.IGNORECASE | re.DOTALL)
        if m:
            field = re.sub(r"\s+", " ", m.group(1).strip())
            if len(field) > 15:
                return field

    return ""


def extract_mechanism(
    context: str,
    field_of_invention: str,
    matched_product_type: str = "",
) -> str:
    """
    Extract the enabling mechanism — the HOW behind the WHAT.

    Strategy 1: document keyword search (most reliable)
    Strategy 2: checklist-driven hint (works when doc context is thin)
    Strategy 3: field statement word extraction (last resort)
    """
    combined = (context + " " + field_of_invention).lower()

    for kw in _MECHANISM_KEYWORDS:
        m = re.search(kw + r"[^.]{0,120}", combined, re.IGNORECASE)
        if m:
            mechanism = re.sub(r"\s+", " ", m.group(0).strip())
            if len(mechanism) > 80:
                mechanism = mechanism[:80].rsplit(" ", 1)[0]
            logger.info("Mechanism via keyword '%s': %s", kw, mechanism)
            return mechanism

    if matched_product_type and matched_product_type in _MECHANISM_HINTS:
        hint = _MECHANISM_HINTS[matched_product_type]
        logger.info("Mechanism via checklist hint for '%s': %s", matched_product_type, hint)
        return hint

    generic_words = {
        "the", "a", "an", "present", "invention", "relates", "to", "for",
        "of", "in", "and", "or", "that", "which", "is", "are", "be",
        "system", "device", "method", "apparatus", "arrangement",
    }
    if field_of_invention:
        words = field_of_invention.split()
        technical = [w for w in words if w.lower() not in generic_words and len(w) > 3]
        if technical:
            fallback = " ".join(technical[-4:])
            logger.info("Mechanism via field fallback: %s", fallback)
            return fallback

    return ""


# ── Hallucination validator ───────────────────────────────────────────────────

def _validate_novelty_line(novelty_text: str, raw_document_context: str) -> Tuple[str, bool]:
    NOT_STATED = "NOT STATED — no explicit novelty sentence found in document"
    THRESHOLD  = 0.15

    if not novelty_text or "NOT STATED" in novelty_text.upper():
        return novelty_text, False

    doc_lower = re.sub(r"\s+", " ", raw_document_context.lower())
    nov_lower = re.sub(r"\s+", " ", novelty_text.lower())

    for b in [
        "the present invention", "the invention", "according to the invention",
        "in one embodiment", "the system", "the method", "the device",
        "a novel approach", "based on the concept of",
    ]:
        nov_lower = nov_lower.replace(b, "")

    words = nov_lower.split()
    if len(words) < 4:
        return novelty_text, False

    ngrams = [" ".join(words[i:i+4]) for i in range(len(words) - 3)]
    if not ngrams:
        return novelty_text, False

    found    = sum(1 for ng in ngrams if ng in doc_lower)
    coverage = found / len(ngrams)

    logger.info("Novelty n-gram coverage: %.0f%% (%d/%d)", coverage * 100, found, len(ngrams))

    if coverage < THRESHOLD:
        logger.warning("NOVELTY REJECTED (%.0f%% < %.0f%%): %s",
                       coverage * 100, THRESHOLD * 100, novelty_text[:200])
        return NOT_STATED, True

    return novelty_text, False


def _postprocess_questions(raw_output: str, raw_document_context: str) -> Tuple[str, list]:
    corrections = []
    novelty_match = re.search(
        r"(NOVELTY\s*\(verbatim\)\s*:\s*[\"']?)([^\"'\n]+)([\"']?)",
        raw_output, re.IGNORECASE,
    )
    if novelty_match:
        novelty_text = novelty_match.group(2).strip()
        validated, was_corrected = _validate_novelty_line(novelty_text, raw_document_context)
        if was_corrected:
            raw_output = raw_output.replace(
                novelty_match.group(0),
                f'NOVELTY (verbatim): "{validated}"',
                1,
            )
            corrections.append(
                f"NOVELTY auto-corrected: < 15% n-gram overlap with document. "
                f"Original: \"{novelty_text[:120]}\""
            )
    return raw_output, corrections


# ── 3-pass RAG ────────────────────────────────────────────────────────────────

def build_two_pass_context(collection, patent_type: str) -> Tuple[str, str, str]:
    """
    3-pass RAG retrieval. Returns (combined_context, field_of_invention, mechanism).

    Pass 1 — broad neutral query (field / abstract)
    Pass 2 — targeted: field text → checklist query → domain query
    Pass 3 — checklist-specific if different from pass 2
    """
    pass1_context = search(
        collection,
        "field of invention technical field abstract summary novel contribution "
        "relates to present invention directed to",
        n_results=8,
    )

    field = extract_field_of_invention(pass1_context)
    matched_product_type = _match_checklist_type(field)

    # Pass 2 query — checklist overrides domain selection when available
    if field:
        pass2_query = field[:60]
    elif matched_product_type and matched_product_type in _CHECKLIST_RAG_QUERIES:
        pass2_query = _CHECKLIST_RAG_QUERIES[matched_product_type]
        logger.info("Pass-2 override: checklist query for '%s' (domain was '%s')",
                    matched_product_type, patent_type)
    else:
        pass2_query = _rag_query_for_type(patent_type)

    pass2_context = search(collection, pass2_query, n_results=8)

    # Pass 3 — checklist-specific query (only if different from pass 2)
    pass3_context = ""
    if matched_product_type and matched_product_type in _CHECKLIST_RAG_QUERIES:
        checklist_query = _CHECKLIST_RAG_QUERIES[matched_product_type]
        if checklist_query != pass2_query:
            pass3_context = search(collection, checklist_query, n_results=5)

    # Deduplicate
    seen: set = set()
    deduped = []
    for para in (pass1_context + "\n\n" + pass2_context + "\n\n" + pass3_context).split("\n\n"):
        key = para.strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(key)

    combined_context = "\n\n".join(deduped)
    mechanism = extract_mechanism(combined_context, field, matched_product_type)

    logger.info(
        "3-pass RAG: %d paragraphs | field='%s' | type='%s' | mechanism='%s'",
        len(deduped),
        field[:60] if field else "(not found)",
        matched_product_type or "(none)",
        mechanism[:50] if mechanism else "(not found)",
    )
    return combined_context, field, mechanism


# ── Main workflow ─────────────────────────────────────────────────────────────

def run(
    collection,
    patent_type: str,
    custom_role: Optional[str] = None,
    custom_backstory: Optional[str] = None,
    timeout: int = cfg.CREW_TIMEOUT_SECONDS,
) -> ScrutinyResult:
    """Run scrutiny crew. Never raises — check .success."""

    # Provider startup
    if not is_cloud_provider():
        ollama = OllamaService()
        try:
            ollama.ensure_running()
        except Exception as exc:
            return ScrutinyResult(questions="", agent_log="", success=False,
                                  error=f"Ollama startup failed: {exc}")
        ok, diag_msg = ollama.diagnose_llm_stack()
        if not ok:
            return ScrutinyResult(questions="", agent_log="", success=False, error=diag_msg)
    else:
        logger.info("Using cloud provider: %s", provider_name())

    # 3-pass RAG
    context, field_of_invention, mechanism = build_two_pass_context(collection, patent_type)

    scrutinizer = build_scrutinizer(patent_type, custom_role=custom_role,
                                    custom_backstory=custom_backstory)
    task = build_scrutiny_task(
        scrutinizer, context, patent_type,
        field_of_invention=field_of_invention,
        mechanism=mechanism,
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
        return ScrutinyResult(questions="", agent_log=captured_log, success=False,
                              error=f"Scrutiny timed out after {timeout}s.")
    if holder["error"]:
        return ScrutinyResult(questions="", agent_log=captured_log, success=False,
                              error=holder["error"])

    raw = str(holder["result"].raw if hasattr(holder["result"], "raw") else holder["result"])

    # Post-process: validate NOVELTY line
    corrected_raw, corrections = _postprocess_questions(raw, context)
    if corrections:
        for c in corrections:
            logger.warning("Post-processing: %s", c)
        captured_log += "\n\n[POST-PROCESSING CORRECTIONS]\n" + "\n".join(corrections)

    # Readiness rating — use _match_checklist_type result directly
    matched_type = _match_checklist_type(field_of_invention) if field_of_invention else None
    readiness_report = None
    if matched_type and corrected_raw:
        try:
            readiness_report = rate_questions(corrected_raw, matched_type)
            logger.info("Readiness: %s (%.0f%%) type=%s",
                        readiness_report.verdict,
                        readiness_report.overall_score * 100,
                        matched_type)
        except Exception as exc:
            logger.warning("Readiness rating failed: %s", exc)

    return ScrutinyResult(
        questions=corrected_raw,
        agent_log=captured_log,
        success=True,
        field_of_invention=field_of_invention,
        mechanism=mechanism,
        readiness_report=readiness_report,
    )
