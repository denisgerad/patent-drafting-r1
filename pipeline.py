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

OLLAMA_TIMEOUT   = 600          # 10 min — large docs on local hardware need this
MAX_DRAFT_CHARS  = 6000         # truncate input to avoid context overflow on 7B
MAX_TOKENS_OUT   = 2048         # cap output length so model doesn't hang generating


# ─────────────────────────────────────────────────────────────
# Scaffold loader — reads product_type_checklists.json
# ─────────────────────────────────────────────────────────────

def load_scaffold(domain_type: str, checklists_path: str = "product_type_checklists.json") -> dict:
    """
    Load technical scaffold for a given domain_type from checklists.
    Returns dict with:
      - expert_categories: list of {category, questions[]}
      - technical_requirements: key specs the model MUST address
    Returns empty dict if file missing or domain_type not found.
    """
    try:
        p = Path(checklists_path)
        if not p.exists():
            p = Path("..") / checklists_path
        if not p.exists():
            return {}
        data = json.loads(p.read_text(encoding="utf-8"))
        entry = data.get(domain_type, {})
        return entry if entry and not domain_type.startswith("_") else {}
    except Exception as e:
        print(f"[Scaffold] Could not load {checklists_path}: {e}")
        return {}


def build_scaffold_prompt_block(scaffold: dict) -> str:
    """
    Convert scaffold expert_categories into a structured prompt block
    the model uses to drive technical augmentation, not just reformatting.
    """
    if not scaffold:
        return ""

    cats = scaffold.get("expert_categories", [])
    tech_reqs = scaffold.get("technical_requirements", [])

    lines = ["## DOMAIN TECHNICAL SCAFFOLD"]
    lines.append("The following domain knowledge defines what a complete patent in this")
    lines.append("technology area MUST address. Use this to AUGMENT the draft with")
    lines.append("technically grounded content — do not merely reformat existing text.")
    lines.append("")

    if tech_reqs:
        lines.append("### Mandatory Technical Requirements")
        lines.append("Every complete patent in this domain must specify:")
        for req in tech_reqs:
            lines.append(f"  - {req}")
        lines.append("")

    if cats:
        lines.append("### Expert Category Checklist")
        lines.append("For each category, check the draft and either:")
        lines.append("  (a) confirm present — no action needed")
        lines.append("  (b) augment with domain-grounded content using [[ADDED: <reason>]]")
        lines.append("  (c) flag as inventor-required using [[FLAGGED: <question>]] " + UNKNOWN_TAG)
        lines.append("")
        for cat in cats:
            cat_name = cat.get("category", "General")
            questions = cat.get("questions", [])
            lines.append(f"**{cat_name}**")
            for q in questions:
                lines.append(f"  - {q}")
            lines.append("")

    return "\n".join(lines)


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
    inventor_queries: list[dict]     = field(default_factory=list)  # from Step 1 query sheet

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


def _ollama_generate(prompt: str, system: str = "", timeout: int = OLLAMA_TIMEOUT) -> str:
    """
    Call Ollama /api/generate with streaming enabled.
    Streaming prevents ReadTimeout on long generations — the connection
    stays alive as tokens arrive rather than waiting for the full response.
    Falls back to 7B if NeMo unavailable.
    """
    # Get available models
    try:
        available = _get_available_models()
        if not available:
            raise RuntimeError("No models available in Ollama.")
    except Exception as e:
        raise RuntimeError(f"Could not connect to Ollama: {e}")
    
    # Try primary model (NeMo), then fallback (mistral 7B)
    for keyword in [PRIMARY_MODEL, FALLBACK_MODEL]:
        model = _find_model(keyword, available)
        if not model:
            continue
        
        payload = {
            "model":  model,
            "prompt": prompt,
            "system": system,
            "stream": True,   # streaming keeps connection alive — fixes ReadTimeout
            "options": {
                "temperature": 0.2,
                "num_ctx":     8192,
                "num_predict": MAX_TOKENS_OUT,  # cap output length
            },
        }
        try:
            r = requests.post(
                f"{OLLAMA_BASE}/api/generate",
                json=payload,
                timeout=timeout,
                stream=True,
            )
            r.raise_for_status()

            # Accumulate streamed token chunks
            chunks = []
            for line in r.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        chunks.append(data.get("response", ""))
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
            return "".join(chunks).strip()

        except requests.exceptions.ConnectionError:
            raise RuntimeError("Ollama not running — start with: ollama serve")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Ollama error with model '{model}': {e}")
        except requests.exceptions.ReadTimeout:
            # If streaming times out, try fallback model before giving up
            print(f"[Ollama] Timeout with {model}, trying fallback…")
            continue

    available_str = ", ".join(available)
    raise RuntimeError(
        f"Neither '{PRIMARY_MODEL}' nor '{FALLBACK_MODEL}' found in Ollama.\n"
        f"Available models: {available_str}"
    )


# ─────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────

_SYSTEM_BASE = f"""You are a senior patent attorney and technical drafter.
You work ONLY from the document text provided.
If information is genuinely missing or uncertain, you MUST write exactly: {UNKNOWN_TAG}
Never invent technical specifications, measurements, materials, or claim scope.
"""

def _build_step1_prompt(draft: str, rag_context: str, domain_type: str, scaffold: dict = None) -> str:
    """
    Builds the Step 1 augmentation prompt.
    scaffold dict (from product_type_checklists.json) is the primary driver
    of technical content — the model works through each expert category
    explicitly rather than doing open-ended gap detection.
    """
    rag_block       = f"## REFERENCE CORPUS CONTEXT\n{rag_context}\n" if rag_context else ""
    scaffold_block  = build_scaffold_prompt_block(scaffold or {})
    type_hint       = f"Product type: {domain_type}" if domain_type else "Product type: not specified"

    return f"""
{rag_block}
{scaffold_block}

## USER DRAFT PATENT DOCUMENT
({type_hint})

{draft}

## YOUR TASK — Step 1: Technical Augmentation

Your primary goal is TECHNICAL AUGMENTATION, not reformatting.
Work through each category in the Domain Technical Scaffold above.
For each one, assess what the draft says and what is missing.

RULES:
1. Every addition must be technically specific — cite materials, values,
   mechanisms, tolerances, or process steps from the scaffold or RAG context.
   Generic filler ("the device may include...") is NOT acceptable.
2. If a scaffold question cannot be answered from the draft or context,
   mark it as a flagged inventor question:
   [[FLAGGED: <exact scaffold question>]] {UNKNOWN_TAG} [[/FLAGGED]]
3. Mark every augmentation with:
   [[ADDED: <scaffold category — reason>]] ... technical text ... [[/ADDED]]
4. Mark every revision with:
   [[REVISED: <what was wrong — scaffold category>]] ... revised text ... [[/REVISED]]
5. Do NOT remove original text. Do NOT reformat without adding substance.
6. After the augmented document, append a section:

---
## INVENTOR QUERY SHEET
List every [[FLAGGED]] item as a numbered question for the inventor.
Format each as:
  Q<N>. [Category]: <question>
  Context: <quote the relevant draft passage, max 30 words>
---

Output: full augmented document with markers, then the Inventor Query Sheet.
"""


def extract_inventor_query_sheet(completed_text: str) -> list[dict]:
    """
    Parse the Inventor Query Sheet appended by the model after Step 1.
    Returns list of {number, category, question, context}.
    """
    # Find the query sheet section
    sheet_match = re.search(
        r"## INVENTOR QUERY SHEET(.+?)(?:---|\Z)", completed_text, re.DOTALL
    )
    if not sheet_match:
        # Fallback: extract [[FLAGGED:...]] markers directly
        flagged = re.findall(
            r"\[\[FLAGGED: ([^\]]+)\]\]", completed_text
        )
        return [{"number": i+1, "category": "", "question": q, "context": ""}
                for i, q in enumerate(flagged)]

    sheet_text = sheet_match.group(1)
    pattern = r"Q(\d+)\.\s*\[([^\]]+)\]:\s*(.+?)\n\s*Context:\s*(.+?)(?=Q\d+\.|$)"
    matches = re.findall(pattern, sheet_text, re.DOTALL)

    if matches:
        return [
            {
                "number":   int(m[0]),
                "category": m[1].strip(),
                "question": m[2].strip(),
                "context":  m[3].strip(),
            }
            for m in matches
        ]

    # Simpler fallback: just extract Q<N>. lines
    lines = [l.strip() for l in sheet_text.split("\n") if re.match(r"Q\d+\.", l.strip())]
    return [{"number": i+1, "category": "", "question": l, "context": ""}
            for i, l in enumerate(lines)]

def _build_step3_prompt(completed_draft: str, domain_markup: str) -> str:
    return f"""
## COMPLETED DRAFT (after Step 1)
{completed_draft}

## DOMAIN EXPERT MARKUP
The domain expert edited the completed draft inline using this syntax:
  ~~deleted text~~        → remove this text entirely
  ++inserted text++       → insert this text at that position
  ??question text??       → answer this question inline if possible, else {UNKNOWN_TAG}
  ##section comment##     → apply this guidance to the surrounding section
  REVERT <TYPE>: <reason> → undo a model change from Step 1

The marked-up document is:
{domain_markup}

## YOUR TASK — Step 3: Redraft

Produce a CLEAN REDRAFT that:
1. Deletes all ~~...~~ spans
2. Inserts all ++...++ spans at their marked positions
3. Answers all ??...?? questions inline — if unanswerable write {UNKNOWN_TAG}
4. Applies ##...## section guidance to improve the surrounding paragraph
5. Reverts any REVERT instructions back to original wording
6. Marks every resulting change with:
   [[RESOLVED: <brief ref>]] ... new text ... [[/RESOLVED]]
   [[FLAGGED: <question>]] {UNKNOWN_TAG} [[/FLAGGED]]

Output ONLY the clean redrafted document. Do not include the markup symbols.
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

    def reset(self) -> None:
        """Reset pipeline to DRAFT_RECEIVED stage for new document."""
        self.state = DocumentState(doc_id=self.state.doc_id)
        self._save()
        print(f"[Pipeline] Reset: {self.state.doc_id} → DRAFT_RECEIVED")

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

        # Truncate draft if oversized — 7B struggles beyond ~6k chars input
        draft = self.state.original_draft
        if len(draft) > MAX_DRAFT_CHARS:
            print(f"[Step 1] Draft truncated {len(draft)}→{MAX_DRAFT_CHARS} chars for context window")
            draft = draft[:MAX_DRAFT_CHARS] + "\n\n[... DRAFT TRUNCATED FOR CONTEXT WINDOW ...]"

        print("[Step 1] Generating completed draft…")
        t0 = time.time()

        # Load domain scaffold to drive augmentation
        scaffold = load_scaffold(domain_type)
        if scaffold:
            print(f"[Step 1] Scaffold loaded for '{domain_type}': "
                  f"{len(scaffold.get('expert_categories',[]))} categories")
        else:
            print(f"[Step 1] No scaffold for '{domain_type}' — augmenting from draft only")

        prompt = _build_step1_prompt(draft, rag_context, domain_type, scaffold)
        result = _ollama_generate(prompt, system=_SYSTEM_BASE)

        self.state.completed_draft = result
        self.state.completion_diff = self._extract_changes(result)
        self.state.inventor_queries = extract_inventor_query_sheet(result)
        self.state.stage           = Stage.DOMAIN_MARKUP
        self.state.iterations     += 1
        self._save()

        print(f"[Step 1] Done in {time.time()-t0:.1f}s — "
              f"{len(self.state.completion_diff)} changes marked, "
              f"{len(self.state.inventor_queries)} inventor queries")
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
        result = _ollama_generate(prompt, system=_SYSTEM_BASE, timeout=OLLAMA_TIMEOUT)

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
        raw = _ollama_generate(prompt, system=_SYSTEM_BASE, timeout=OLLAMA_TIMEOUT)
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
