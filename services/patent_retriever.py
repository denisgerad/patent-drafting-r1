"""
services/patent_retriever.py - Phase A patent retrieval for gap analysis.
Wraps services/epo_client.py (copied from patent-search-automation).
"""
from __future__ import annotations
import json, logging, re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from services.epo_client import EPOClient

log = logging.getLogger(__name__)
_CACHE_DIR = Path(__file__).parent.parent / "data" / "reference_patents"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class PatentCandidate:
    epodoc_id: str; display_id: str; title: str; abstract: str
    grant_date: str; applicant: str; source: str = "epo"
    similarity_score: float = 0.0; similarity_explanation: str = ""

@dataclass
class PatentClaim:
    number: int; claim_type: str; text: str
    depends_on: Optional[int] = None; topics: list[str] = field(default_factory=list)

@dataclass
class TechnicalParameter:
    name: str; value: str; unit: str; section: str; context: str

@dataclass
class StructuredPatent:
    epodoc_id: str; display_id: str; title: str; abstract: str
    grant_date: str; applicant: str; source: str = "epo"
    background: str = ""; summary: str = ""; description: str = ""
    claims: list[PatentClaim] = field(default_factory=list)
    figures: list[str] = field(default_factory=list)
    technical_parameters: list[TechnicalParameter] = field(default_factory=list)
    claims_raw: str = ""; description_raw: str = ""; fetch_timestamp: str = ""

    def to_dict(self): return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["claims"] = [PatentClaim(**c) for c in d.get("claims", [])]
        d["technical_parameters"] = [TechnicalParameter(**p) for p in d.get("technical_parameters", [])]
        return cls(**d)


class PatentRetriever:
    """Main API for patent retrieval. Wraps EPOClient with search, ranking, caching."""

    def __init__(self):
        self._epo = EPOClient()

    def search_similar(self, field_of_invention: str, max_results: int = 5) -> list[PatentCandidate]:
        """Find similar granted patents from EPO OPS using tiered CQL search."""
        terms = _extract_terms(field_of_invention)
        if not terms:
            return []
        seen, candidates = set(), []
        for cql in _build_cql_tiers(terms):
            if len(candidates) >= max_results * 3:
                break
            for epodoc_id in self._epo.search(cql, max_results=25):
                if epodoc_id in seen:
                    continue
                seen.add(epodoc_id)
                biblio = self._epo.fetch_biblio(epodoc_id)
                if not biblio or biblio["abstract"] == "NO ABSTRACT FOUND":
                    continue
                candidates.append(PatentCandidate(
                    epodoc_id=epodoc_id, display_id=_epodoc_to_display(epodoc_id),
                    title=biblio["title"], abstract=biblio["abstract"],
                    grant_date="", applicant="", source="epo",
                ))
        return _rank_by_overlap(candidates, terms)[:max_results]

    def fetch_by_number(self, patent_number: str) -> Optional[PatentCandidate]:
        """Fetch single patent by user-entered number. Caches locally."""
        epodoc_id = _to_epodoc(patent_number)
        if not epodoc_id:
            return None
        cached = _load_cache(epodoc_id, "biblio")
        if cached:
            return PatentCandidate(**cached)
        biblio = self._epo.fetch_biblio(epodoc_id)
        if not biblio:
            return None
        c = PatentCandidate(epodoc_id=epodoc_id, display_id=_epodoc_to_display(epodoc_id),
                            title=biblio["title"], abstract=biblio["abstract"],
                            grant_date="", applicant="", source="epo")
        _save_cache(epodoc_id, "biblio", {
            "epodoc_id": c.epodoc_id, "display_id": c.display_id,
            "title": c.title, "abstract": c.abstract,
            "grant_date": c.grant_date, "applicant": c.applicant, "source": c.source,
        })
        return c

    def fetch_structured(self, patent_number: str) -> Optional[StructuredPatent]:
        """Fetch full structured patent: biblio + claims + description. Cached."""
        epodoc_id = _to_epodoc(patent_number)
        if not epodoc_id:
            return None
        cached = _load_cache(epodoc_id, "structured")
        if cached:
            return StructuredPatent.from_dict(cached)
        candidate = self.fetch_by_number(patent_number)
        if not candidate:
            return None
        claims_raw      = self._epo.fetch_claims(epodoc_id)
        description_raw = self._epo.fetch_description(epodoc_id)
        from datetime import datetime, timezone
        s = StructuredPatent(
            epodoc_id=epodoc_id, display_id=_epodoc_to_display(epodoc_id),
            title=candidate.title, abstract=candidate.abstract,
            grant_date=candidate.grant_date, applicant=candidate.applicant,
            **_split_sections(description_raw),
            claims=_parse_claims(claims_raw),
            figures=_extract_figures(description_raw),
            technical_parameters=_extract_parameters(claims_raw + "\n" + description_raw),
            claims_raw=claims_raw, description_raw=description_raw,
            fetch_timestamp=datetime.now(timezone.utc).isoformat(),
        )
        _save_cache(epodoc_id, "structured", s.to_dict())
        log.info("Structured: %s — %d claims, %d params", epodoc_id, len(s.claims), len(s.technical_parameters))
        return s

    def generate_explanations(self, candidates, field_of_invention, llm_fn):
        """Add one-sentence LLM explanation per candidate (local model only)."""
        for c in candidates:
            try:
                prompt = (f"In exactly one sentence starting with 'This patent', explain why "
                          f"this patent is similar to: '{field_of_invention[:180]}'\n"
                          f"Patent: {c.title}\nAbstract: {c.abstract[:300]}")
                raw = llm_fn(prompt).strip()
                c.similarity_explanation = raw.split(".")[0].strip() + "."
            except Exception as exc:
                log.warning("Explanation failed %s: %s", c.epodoc_id, exc)
                c.similarity_explanation = f"Similar patent ({c.similarity_score:.0%} overlap)"
        return candidates


def _to_epodoc(patent_number: str) -> str:
    pn = patent_number.strip().upper().replace(" ", "")
    if re.match(r"^[A-Z]{2}\.\d+\.[A-Z]\d*$", pn):
        return pn
    m = re.match(r"^([A-Z]{2})(\d+)([A-Z]\d*)?$", pn)
    if m:
        return f"{m.group(1)}.{m.group(2)}.{m.group(3) or 'B1'}"
    return ""

def _epodoc_to_display(epodoc_id: str) -> str:
    return epodoc_id.replace(".", "")

def _extract_terms(field: str) -> list[str]:
    stop = {"the","a","an","and","or","of","in","on","at","to","for","is","are",
            "be","by","with","as","that","this","both","either","when","where",
            "which","such","may","can","its","their","from","between","during",
            "using","present","invention","relates","particularly","more",
            "specifically","capable","providing","having","comprising",
            "system","method","device","apparatus","single","universal"}
    words = re.findall(r"\b[a-z][a-z\-]{2,}\b", field.lower())
    terms = [w for w in words if w not in stop]
    seen, unique = set(), []
    for t in sorted(terms, key=len, reverse=True):
        if t not in seen:
            seen.add(t); unique.append(t)
    return unique[:10]

def _build_cql_tiers(terms: list[str]) -> list[str]:
    def q(t): return f'(ti="{t}" OR ab="{t}")' if " " in t else f"(ti={t} OR ab={t})"
    tiers = []
    if len(terms) >= 3: tiers.append(" AND ".join(q(t) for t in terms[:4]))
    if len(terms) >= 2: tiers.append(" AND ".join(q(t) for t in terms[:2]))
    tiers.append(q(terms[0]))
    if len(terms) >= 2: tiers.append(" OR ".join(q(t) for t in terms[:3]))
    return tiers

def _rank_by_overlap(candidates, terms):
    for c in candidates:
        target = (c.title + " " + c.abstract).lower()
        hits = sum(1 for t in terms if t in target)
        title_hits = sum(1 for t in terms if t in c.title.lower())
        c.similarity_score = (hits + title_hits * 0.5) / max(len(terms), 1)
    return sorted(candidates, key=lambda c: c.similarity_score, reverse=True)

def _parse_claims(claims_raw: str) -> list[PatentClaim]:
    if not claims_raw.strip(): return []
    blocks = re.split(r"\n\s*(?:claim\s+)?(\d+)\s*\.\s*", claims_raw, flags=re.IGNORECASE)
    claims, i = [], 1
    while i < len(blocks) - 1:
        try:
            num, text = int(blocks[i]), blocks[i+1].strip()
            dep = re.search(r"(?:according to|of|as in)\s+claim\s+(\d+)", text, re.IGNORECASE)
            depends_on = int(dep.group(1)) if dep else None
            claims.append(PatentClaim(num, "dependent" if depends_on else "independent", text, depends_on, []))
        except (ValueError, IndexError): pass
        i += 2
    return claims

def _extract_parameters(text: str) -> list[TechnicalParameter]:
    patterns = [
        (r"(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*(Ω/sq|ohm/sq)", "sheet resistivity"),
        (r"(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*(µm|microns?)",  "thickness"),
        (r"(-?\d+)\s*(?:to|-)\s*(-?\d+)\s*°C",                             "temperature range"),
        (r"(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*(GPa|MPa)",     "mechanical property"),
        (r"(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*(nm)",           "optical dimension"),
        (r"(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*(mm|cm)",        "dimension"),
        (r"(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*(W/cm²|W/m²)",  "power density"),
    ]
    results, seen = [], set()
    for pattern, name in patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            g = m.groups()
            value, unit = f"{g[0]}-{g[1]}", (g[-1] if len(g) >= 3 else "")
            if (value, unit) not in seen:
                seen.add((value, unit))
                ctx = text[max(0, m.start()-50):m.end()+50].replace("\n"," ")
                results.append(TechnicalParameter(name, value, unit, "claims_or_description", ctx.strip()))
    return results[:25]

def _extract_figures(description: str) -> list[str]:
    figs = re.findall(r"FIG(?:URE|S?)\.?\s*(\d+[A-Za-z]?)\s*[–\-]?\s*([^.\n]{10,80})?",
                      description, re.IGNORECASE)
    result = []
    for num, cap in figs:
        cap = cap.strip() if cap else ""
        result.append(f"FIG. {num}" + (f" — {cap}" if cap else ""))
    return list(dict.fromkeys(result))[:15]

def _split_sections(description_raw: str) -> dict[str, str]:
    sections = {"background": "", "summary": "", "description": ""}
    for key, pat in [
        ("background",  r"(?:background|prior art)[^\n]*\n(.*?)(?=\n\s*(?:summary|brief|detailed)|\Z)"),
        ("summary",     r"(?:summary of the invention|summary)[^\n]*\n(.*?)(?=\n\s*(?:detailed|brief description)|\Z)"),
        ("description", r"(?:detailed description|description of.*?embodiment)[^\n]*\n(.*?)(?=\n\s*(?:claims|what is claimed)|\Z)"),
    ]:
        m = re.search(pat, description_raw, re.IGNORECASE | re.DOTALL)
        if m: sections[key] = m.group(1).strip()[:6000]
    if not any(sections.values()):
        sections["description"] = description_raw[:8000]
    return sections

def _load_cache(epodoc_id: str, cache_type: str):
    safe = epodoc_id.replace(".", "_")
    path = _CACHE_DIR / f"{safe}_{cache_type}.json"
    if path.exists():
        try: return json.loads(path.read_text())
        except: pass
    return None

def _save_cache(epodoc_id: str, cache_type: str, data: dict):
    safe = epodoc_id.replace(".", "_")
    path = _CACHE_DIR / f"{safe}_{cache_type}.json"
    try: path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception as exc: log.warning("Cache write failed: %s", exc)
