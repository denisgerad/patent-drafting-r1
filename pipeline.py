"""
PatentReviewPipeline
====================
Replaces CrewAI with a simple 4-state machine.
Reuses your existing two-pass RAG context builder and product_type_checklists.

States:
  DRAFT_RECEIVED  → step1_complete_draft()
  DOMAIN_MARKUP   → step3_redraft()          (domain edits happen outside)
  REDRAFTED       → step4_go_no_go()
  CLOSED

All LLM calls go through Ollama (NeMo by default, 7B fallback).
Document state persists to a JSON sidecar so the loop survives restarts.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional
import requests


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

OLLAMA_BASE = "http://localhost:11434"
PRIMARY_MODEL   = "mistral-nemo"        # Will match mistral-nemo:* tags
FALLBACK_MODEL  = "mistral"             # Will match mistral:7b-instruct or similar

UNKNOWN_TAG = "[REQUIRES INVENTOR INPUT]"   # model must use this, never invent


# ─────────────────────────────────────────────────────────────
# Ollama helper
# ─────────────────────────────────────────────────────────────

def _get_available_models() -> list[str]:
    """Fetch list of available models from Ollama."""
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        r.raise_for_status()
        models = r.json().get("models", [])
        return [m.get("name", "") for m in models if m.get("name")]
    except Exception as e:
        raise RuntimeError(f"Failed to list Ollama models: {e}")


def _find_model(keyword: str, available_models: list[str]) -> Optional[str]:
    """Find first model matching keyword (case-insensitive prefix match)."""
    keyword_lower = keyword.lower()
    for model in available_models:
        if model.lower().startswith(keyword_lower):
            return model
    return None


def _ollama_generate(prompt: str, system: str = "", timeout: int = 120) -> str:
    """Call Ollama /api/generate. Falls back to 7B if NeMo unavailable."""
    # Get available models
    try:
        available = _get_available_models()
        if not available:
            raise RuntimeError("No models available in Ollama.")
    except Exception as e:
        raise RuntimeError(f"Could not connect to Ollama: {e}")
    
    # Try primary model (NeMo)
    model = _find_model(PRIMARY_MODEL, available)
    if not model:
        # Try fallback (mistral 7B)
        model = _find_model(FALLBACK_MODEL, available)
    
    if not model:
        available_str = ", ".join(available)
        raise RuntimeError(
            f"Neither '{PRIMARY_MODEL}' nor '{FALLBACK_MODEL}' found in Ollama.\n"
            f"Available models: {available_str}"
        )
    
    payload = {
        "model":  model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 8192},
    }
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json=payload,
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Ollama not running — start with: ollama serve")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Ollama error with model '{model}': {e}")


# ─────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────

class Stage(str, Enum):
    DRAFT_RECEIVED = "DRAFT_RECEIVED"
    DOMAIN_MARKUP  = "DOMAIN_MARKUP"
    REDRAFTED      = "REDRAFTED"
    CLOSED         = "CLOSED"


@dataclass
class DocumentState:
    """Single source of truth for one patent review cycle."""
    doc_id:          str
    stage:           Stage           = Stage.DRAFT_RECEIVED

    original_draft:  str             = ""   # raw text of user upload
    completed_draft: str             = ""   # Step 1 output
    completion_diff: list[dict]      = field(default_factory=list)  # [{section, original, added, reason}]

    domain_markup:   str             = ""   # domain's strike-outs + questions (plain text)
    redraft:         str             = ""   # Step 3 output

    go_no_go:        Optional[str]   = None  # "GO" | "NO-GO"
    go_no_go_notes:  str             = ""
    confidence:      Optional[float] = None  # 0.0–1.0

    rag_context:     str             = ""   # cached from two-pass RAG
    domain_type:     str             = ""   # e.g. "flexible_heater_film"
    iterations:      int             = 0

    def save(self, path: Path) -> None:
        d = asdict(self)
        d["stage"] = self.stage.value
        path.write_text(json.dumps(d, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "DocumentState":
        d = json.loads(path.read_text(encoding="utf-8"))
        d["stage"] = Stage(d["stage"])
        return cls(**d)


# ─────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────

_SYSTEM_BASE = f"""You are a senior patent attorney and technical drafter.
You work ONLY from the document text provided.
If information is genuinely missing or uncertain, you MUST write exactly: {UNKNOWN_TAG}
Never invent technical specifications, measurements, materials, or claim scope.
"""

def _build_step1_prompt(draft: str, rag_context: str, domain_type: str) -> str:
    domain_hint = f"\nDomain context (from reference corpus):\n{rag_context}\n" if rag_context else ""
    type_hint   = f"\nProduct type identified: {domain_type}\n" if domain_type else ""
    return f"""
{domain_hint}{type_hint}

## USER DRAFT PATENT DOCUMENT
{draft}

## YOUR TASK — Step 1: Complete and Annotate

Review the draft above and produce an IMPROVED VERSION that:

1. Fills gaps in the specification with technically grounded language
   — if you cannot fill a gap from the draft text or domain context, write {UNKNOWN_TAG}
2. Strengthens claim language where it is vague or overbroad
3. Adds missing standard patent sections (Brief Description of Drawings,
   Summary of Invention, etc.) if absent
4. Marks EVERY change you make with this exact format inline:
   [[ADDED: <one-line reason>]]  ... added text ...  [[/ADDED]]
   [[REVISED: <one-line reason>]]  ... revised text ...  [[/REVISED]]

Do NOT remove any original text — only add or revise.
Output the full improved document with all change markers.
"""

def _build_step3_prompt(completed_draft: str, domain_markup: str) -> str:
    return f"""
## COMPLETED DRAFT (after Step 1)
{completed_draft}

## DOMAIN EXPERT MARKUP
The domain expert has reviewed the above and provided the following
strike-outs and questions (plain text representation):
{domain_markup}

## YOUR TASK — Step 3: Redraft

Produce a CLEAN REDRAFT that:
1. Removes all struck-out content indicated by the domain expert
2. Resolves each domain question where the answer can be derived
   from the existing document — otherwise write {UNKNOWN_TAG}
3. Marks every change with:
   [[RESOLVED: <domain question or strike-out ref>]] ... new text ... [[/RESOLVED]]
   [[FLAGGED: <question that cannot be resolved>]] {UNKNOWN_TAG} [[/FLAGGED]]

Output the full redrafted document with change markers.
"""

def _build_step4_prompt(completed_draft: str, redraft: str, domain_markup: str) -> str:
    return f"""
## ORIGINAL COMPLETED DRAFT
{completed_draft}

## DOMAIN MARKUP
{domain_markup}

## REDRAFT
{redraft}

## YOUR TASK — Step 4: Go / No-Go Assessment

Compare the redraft against the domain markup and assess readiness
for the next domain review cycle.

Respond in this EXACT JSON format (no markdown fences):
{{
  "verdict": "GO" or "NO-GO",
  "confidence": <float 0.0-1.0>,
  "resolved_count": <int>,
  "unresolved_count": <int>,
  "flagged_items": [
    {{"ref": "<domain question or section>", "reason": "<why unresolved>"}}
  ],
  "recommendation": "<one paragraph for domain expert>"
}}

GO means: all domain questions resolved or legitimately flagged as {UNKNOWN_TAG}.
NO-GO means: open questions remain that the model could have resolved but did not,
             or the redraft introduced new inconsistencies.
"""


# ─────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────

class PatentReviewPipeline:
    """
    Usage:
        pipeline = PatentReviewPipeline(doc_id="US-2024-001", state_dir="./states")

        # Step 1
        pipeline.load_draft(raw_text)
        completed = pipeline.step1_complete_draft(rag_context, domain_type)

        # (domain reviews completed draft, produces markup string)
        pipeline.receive_domain_markup(markup_text)

        # Step 3
        redraft = pipeline.step3_redraft()

        # Step 4
        result = pipeline.step4_go_no_go()
        print(result["verdict"], result["confidence"])
    """

    def __init__(self, doc_id: str, state_dir: str = "./patent_states"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self.state_dir / f"{doc_id}.json"

        if self._state_path.exists():
            self.state = DocumentState.load(self._state_path)
            print(f"[Pipeline] Resumed: {doc_id} @ {self.state.stage.value}")
        else:
            self.state = DocumentState(doc_id=doc_id)
            print(f"[Pipeline] New document: {doc_id}")

    # ── Ingestion ──────────────────────────────────────────────

    def load_draft(self, raw_text: str) -> None:
        """Call once when user uploads their draft."""
        assert self.state.stage == Stage.DRAFT_RECEIVED, \
            f"Cannot load draft in stage {self.state.stage}"
        self.state.original_draft = raw_text
        self._save()

    # ── Step 1 ─────────────────────────────────────────────────

    def step1_complete_draft(
        self,
        rag_context: str = "",
        domain_type: str = "",
    ) -> str:
        """
        Model completes the draft and marks every change.
        Returns the completed draft text.
        Plug in your existing build_two_pass_context() here for rag_context.
        """
        assert self.state.stage == Stage.DRAFT_RECEIVED, \
            f"step1 requires DRAFT_RECEIVED, got {self.state.stage}"
        assert self.state.original_draft, "Load draft first"

        self.state.rag_context  = rag_context
        self.state.domain_type  = domain_type

        print("[Step 1] Generating completed draft…")
        t0 = time.time()

        prompt = _build_step1_prompt(
            self.state.original_draft, rag_context, domain_type
        )
        result = _ollama_generate(prompt, system=_SYSTEM_BASE)

        self.state.completed_draft = result
        self.state.completion_diff = self._extract_changes(result)
        self.state.stage           = Stage.DOMAIN_MARKUP
        self.state.iterations     += 1
        self._save()

        print(f"[Step 1] Done in {time.time()-t0:.1f}s — "
              f"{len(self.state.completion_diff)} changes marked")
        return result

    # ── Domain markup receipt ──────────────────────────────────

    def receive_domain_markup(self, markup_text: str) -> None:
        """
        Call after domain expert has reviewed completed_draft
        and provided their strike-outs + questions as plain text.
        """
        assert self.state.stage == Stage.DOMAIN_MARKUP, \
            f"receive_domain_markup requires DOMAIN_MARKUP, got {self.state.stage}"
        self.state.domain_markup = markup_text
        self._save()
        print(f"[Domain] Markup received ({len(markup_text)} chars)")

    # ── Step 3 ─────────────────────────────────────────────────

    def step3_redraft(self) -> str:
        """
        Model redrafts based on domain markup.
        Returns the redrafted document text.
        """
        assert self.state.stage == Stage.DOMAIN_MARKUP, \
            f"step3 requires DOMAIN_MARKUP, got {self.state.stage}"
        assert self.state.domain_markup, "Receive domain markup first"

        print("[Step 3] Generating redraft from domain markup…")
        t0 = time.time()

        prompt = _build_step3_prompt(
            self.state.completed_draft,
            self.state.domain_markup,
        )
        result = _ollama_generate(prompt, system=_SYSTEM_BASE)

        self.state.redraft = result
        self.state.stage   = Stage.REDRAFTED
        self._save()

        print(f"[Step 3] Done in {time.time()-t0:.1f}s")
        return result

    # ── Step 4 ─────────────────────────────────────────────────

    def step4_go_no_go(self) -> dict:
        """
        Model assesses whether the redraft resolves all domain markup.
        Returns dict with verdict, confidence, flagged_items, recommendation.
        """
        assert self.state.stage == Stage.REDRAFTED, \
            f"step4 requires REDRAFTED, got {self.state.stage}"

        print("[Step 4] Running go/no-go assessment…")
        t0 = time.time()

        prompt = _build_step4_prompt(
            self.state.completed_draft,
            self.state.redraft,
            self.state.domain_markup,
        )
        raw = _ollama_generate(prompt, system=_SYSTEM_BASE)
        result = self._parse_json_response(raw)

        self.state.go_no_go       = result.get("verdict", "NO-GO")
        self.state.go_no_go_notes = result.get("recommendation", "")
        self.state.confidence     = result.get("confidence", 0.0)

        # If GO → close; if NO-GO → loop back for another domain round
        if self.state.go_no_go == "GO":
            self.state.stage = Stage.CLOSED
        else:
            self.state.stage = Stage.DOMAIN_MARKUP   # reset for next iteration
            self.state.domain_markup = ""            # clear for fresh markup

        self._save()

        print(f"[Step 4] {self.state.go_no_go} "
              f"(confidence={self.state.confidence:.2f}) "
              f"in {time.time()-t0:.1f}s")
        return result

    # ── Helpers ────────────────────────────────────────────────

    def _extract_changes(self, text: str) -> list[dict]:
        """Parse [[ADDED/REVISED: reason]] ... [[/ADDED|/REVISED]] markers."""
        pattern = r"\[\[(ADDED|REVISED): ([^\]]+)\]\](.*?)\[\[/(?:ADDED|REVISED)\]\]"
        matches = re.findall(pattern, text, re.DOTALL)
        return [
            {"type": m[0], "reason": m[1].strip(), "content": m[2].strip()}
            for m in matches
        ]

    def _parse_json_response(self, raw: str) -> dict:
        """Safely parse JSON from LLM response, stripping markdown fences."""
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            # Fallback: extract what we can
            verdict = "GO" if '"verdict": "GO"' in raw else "NO-GO"
            conf_match = re.search(r'"confidence":\s*([\d.]+)', raw)
            return {
                "verdict": verdict,
                "confidence": float(conf_match.group(1)) if conf_match else 0.5,
                "flagged_items": [],
                "recommendation": raw[:500],
            }

    def _save(self) -> None:
        self.state.save(self._state_path)

    # ── Convenience ────────────────────────────────────────────

    @property
    def summary(self) -> dict:
        s = self.state
        return {
            "doc_id":     s.doc_id,
            "stage":      s.stage.value,
            "iterations": s.iterations,
            "changes_marked": len(s.completion_diff),
            "go_no_go":   s.go_no_go,
            "confidence": s.confidence,
        }
