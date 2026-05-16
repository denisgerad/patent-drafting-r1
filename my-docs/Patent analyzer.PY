"""
Patent Readiness Analyzer — Local Edition
==========================================
Runs entirely on your machine. No data leaves your network.
Uses Ollama (local LLM) for analysis.

SETUP:
  1. Install Ollama → https://ollama.com/download
  2. Pull your model:
       ollama pull mistral          (if using Mistral 7B)
       ollama pull llama3           (alternative)
  3. Install Python deps:
       pip install flask requests
  4. Run:
       python patent_analyzer.py
  5. Open browser → http://localhost:5000

CONFIG:
  Change OLLAMA_MODEL below to match whichever model you have pulled.
"""

from flask import Flask, request, jsonify, render_template_string
import requests
import json
import re

# ── Configuration ─────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "mistral"          # change to "llama3", "phi3", etc.
FLASK_PORT      = 5000
# ──────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a senior patent analyst with 20 years of experience 
reviewing patent applications before domain expert review. Your job is to 
evaluate a patent draft and return a structured JSON assessment. 
Be precise, technical, and critical. Do not be lenient."""

def build_analysis_prompt(draft_text: str) -> str:
    return f"""Analyze the following patent draft and score it across 7 dimensions.

PATENT DRAFT:
---
{draft_text[:6000]}
---

Return ONLY a valid JSON object (no markdown, no explanation) with this exact structure:

{{
  "scores": {{
    "completeness": {{
      "score": <0-100>,
      "note": "<one sentence: what is present and what is missing>"
    }},
    "technical_depth": {{
      "score": <0-100>,
      "note": "<one sentence: are there specific numbers, algorithms, protocols, or are descriptions vague?>"
    }},
    "claimability": {{
      "score": <0-100>,
      "note": "<one sentence: are there clear novel elements that can form patent claims?>"
    }},
    "ambiguity": {{
      "score": <0-100>,
      "note": "<one sentence: score HIGH if language is clear and precise, LOW if vague. Specific examples of ambiguous phrases if any.>"
    }},
    "hallucination_risk": {{
      "score": <0-100>,
      "note": "<one sentence: score HIGH if all claims are grounded and verifiable, LOW if there are unsupported assertions. Score is inverted — 100 means no risk.>"
    }},
    "implementation_detail": {{
      "score": <0-100>,
      "note": "<one sentence: could an engineer build this from the description alone?>"
    }},
    "terminology_quality": {{
      "score": <0-100>,
      "note": "<one sentence: is correct patent and technical terminology used consistently?>"
    }}
  }},
  "overall_score": <weighted average, integer 0-100>,
  "decision": "<one of: ready_for_review | needs_clarification | poor_quality>",
  "decision_rationale": "<two sentences explaining the decision>",
  "top_issues": [
    "<most critical issue to fix before review>",
    "<second most critical issue>",
    "<third most critical issue>"
  ]
}}

Scoring guidance:
- completeness: 100 = all sections (Field, Background, Summary, Claims, Detailed Description, Abstract) present and substantive
- technical_depth: 100 = specific parameters, latency numbers, algorithms, protocols named; 0 = all hand-wavy prose
- claimability: 100 = clear novel invention with defensible boundaries; 0 = describes a concept with no claim-worthy elements
- ambiguity: 100 = every term precisely defined; 0 = vague language throughout ("a system", "some mechanism")  
- hallucination_risk: 100 = all assertions are grounded; 0 = many unsupported or unverifiable claims
- implementation_detail: 100 = complete enough to implement; 0 = no engineer could build this from the description
- terminology_quality: 100 = professional patent language throughout; 0 = colloquial / incorrect terminology

Decision thresholds:
- overall >= 72 → ready_for_review
- overall 48–71 → needs_clarification  
- overall < 48 → poor_quality"""


# ── Ollama call ───────────────────────────────────────────────────────────────
def analyze_with_ollama(draft_text: str) -> dict:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_analysis_prompt(draft_text)}
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,   # low temp for consistent structured output
            "num_predict": 1200
        }
    }
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=120
    )
    resp.raise_for_status()
    content = resp.json()["message"]["content"].strip()

    # Strip markdown code fences if the model adds them
    content = re.sub(r"^```[a-z]*\n?", "", content)
    content = re.sub(r"\n?```$", "", content)

    return json.loads(content)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    draft = (data or {}).get("draft", "").strip()
    if not draft:
        return jsonify({"error": "No draft text provided"}), 400
    if len(draft) < 100:
        return jsonify({"error": "Draft is too short to analyze meaningfully"}), 400

    try:
        result = analyze_with_ollama(draft)
        return jsonify(result)
    except requests.exceptions.ConnectionError:
        return jsonify({
            "error": "Cannot connect to Ollama. Make sure it is running: ollama serve"
        }), 503
    except json.JSONDecodeError as e:
        return jsonify({
            "error": f"Model returned non-JSON output. Try a larger model. Detail: {e}"
        }), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/models")
def list_models():
    """Returns models currently available in Ollama."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        return jsonify({"models": models, "current": OLLAMA_MODEL})
    except Exception:
        return jsonify({"models": [], "current": OLLAMA_MODEL,
                        "error": "Ollama not reachable"})


# ── HTML UI ───────────────────────────────────────────────────────────────────
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Patent Readiness Analyzer</title>
<style>
  :root {
    --bg:       #0e0f11;
    --surface:  #16181c;
    --border:   #2a2d33;
    --text:     #e8e9eb;
    --muted:    #6b7280;
    --accent:   #f0a500;
    --green:    #22c55e;
    --amber:    #f59e0b;
    --red:      #ef4444;
    --blue:     #3b82f6;
    --radius:   8px;
    --font:     'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    font-size: 13px;
    line-height: 1.6;
    min-height: 100vh;
  }
  .shell {
    max-width: 820px;
    margin: 0 auto;
    padding: 2rem 1.5rem 4rem;
  }
  .header {
    border-bottom: 1px solid var(--border);
    padding-bottom: 1rem;
    margin-bottom: 1.5rem;
  }
  .header h1 {
    font-size: 16px;
    font-weight: 600;
    letter-spacing: .04em;
    color: var(--accent);
  }
  .header p {
    color: var(--muted);
    font-size: 12px;
    margin-top: 3px;
  }
  .model-tag {
    display: inline-block;
    font-size: 11px;
    background: #1e2028;
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 2px 8px;
    color: var(--muted);
    margin-top: 6px;
  }
  label {
    display: block;
    font-size: 11px;
    letter-spacing: .06em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 6px;
  }
  textarea {
    width: 100%;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    color: var(--text);
    font-family: var(--font);
    font-size: 12px;
    padding: 12px;
    resize: vertical;
    min-height: 200px;
    outline: none;
    transition: border-color .15s;
  }
  textarea:focus { border-color: var(--accent); }
  .btn-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 10px;
  }
  button {
    background: var(--accent);
    color: #0e0f11;
    border: none;
    border-radius: var(--radius);
    font-family: var(--font);
    font-size: 13px;
    font-weight: 600;
    padding: 9px 22px;
    cursor: pointer;
    letter-spacing: .02em;
    transition: opacity .15s, transform .1s;
  }
  button:hover { opacity: .88; }
  button:active { transform: scale(.97); }
  button:disabled { opacity: .4; cursor: not-allowed; }
  .status {
    font-size: 12px;
    color: var(--muted);
  }
  .status.err { color: var(--red); }

  /* ── Results ─────────────────────────────────────── */
  #results { margin-top: 2rem; display: none; }

  .overall-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
    flex-wrap: wrap;
  }
  .overall-score {
    font-size: 48px;
    font-weight: 700;
    line-height: 1;
    color: var(--text);
  }
  .overall-label {
    font-size: 11px;
    letter-spacing: .06em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 4px;
  }
  .decision-badge {
    font-size: 12px;
    font-weight: 600;
    padding: 5px 14px;
    border-radius: 4px;
    letter-spacing: .04em;
    text-transform: uppercase;
    align-self: flex-start;
  }
  .badge-ready    { background: #052e16; color: var(--green); border: 1px solid #166534; }
  .badge-clarify  { background: #1c1407; color: var(--amber); border: 1px solid #854d0e; }
  .badge-poor     { background: #1a0a0a; color: var(--red);   border: 1px solid #7f1d1d; }
  .rationale {
    font-size: 12px;
    color: var(--muted);
    margin-top: 8px;
    line-height: 1.7;
  }

  .metrics-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 1.25rem;
  }
  @media (max-width: 560px) { .metrics-grid { grid-template-columns: 1fr; } }

  .metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 12px 14px;
  }
  .metric-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 6px;
  }
  .metric-name {
    font-size: 11px;
    letter-spacing: .05em;
    text-transform: uppercase;
    color: var(--muted);
  }
  .metric-score {
    font-size: 18px;
    font-weight: 700;
  }
  .bar-track {
    height: 4px;
    background: #1e2028;
    border-radius: 2px;
    overflow: hidden;
    margin-bottom: 7px;
  }
  .bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width .6s ease;
  }
  .metric-note {
    font-size: 11px;
    color: var(--muted);
    line-height: 1.5;
  }

  .issues-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
  }
  .issues-title {
    font-size: 11px;
    letter-spacing: .06em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 10px;
  }
  .issue-item {
    display: flex;
    gap: 8px;
    margin-bottom: 8px;
    font-size: 12px;
    line-height: 1.6;
    color: var(--text);
  }
  .issue-num {
    color: var(--accent);
    font-weight: 700;
    flex-shrink: 0;
    width: 16px;
  }

  .separator {
    height: 1px;
    background: var(--border);
    margin: 1.5rem 0;
  }
</style>
</head>
<body>
<div class="shell">

  <div class="header">
    <h1>▸ Patent Readiness Analyzer</h1>
    <p>100% local · no data leaves your machine</p>
    <span class="model-tag" id="modelTag">loading model info…</span>
  </div>

  <div style="margin-bottom: 1rem;">
    <label for="draft">Patent draft</label>
    <textarea id="draft" placeholder="Paste your patent draft here…"></textarea>
  </div>

  <div class="btn-row">
    <button id="analyzeBtn" onclick="runAnalysis()">Analyze draft</button>
    <span class="status" id="statusMsg"></span>
  </div>

  <div id="results">
    <div class="separator"></div>

    <div class="overall-card">
      <div>
        <div class="overall-label">Patent readiness score</div>
        <div style="display:flex; align-items:baseline; gap:8px;">
          <div class="overall-score" id="overallScore">—</div>
          <div style="font-size:14px; color:var(--muted);">/ 100</div>
        </div>
        <div class="rationale" id="rationale"></div>
      </div>
      <div id="decisionBadge" class="decision-badge"></div>
    </div>

    <div class="metrics-grid" id="metricsGrid"></div>

    <div class="issues-card">
      <div class="issues-title">Top issues to address before review</div>
      <div id="issuesList"></div>
    </div>
  </div>

</div>

<script>
const METRIC_LABELS = {
  completeness:         'Completeness',
  technical_depth:      'Technical depth',
  claimability:         'Claimability',
  ambiguity:            'Ambiguity clarity',
  hallucination_risk:   'Hallucination risk',
  implementation_detail:'Implementation detail',
  terminology_quality:  'Terminology quality'
};

function scoreColor(s) {
  if (s >= 70) return 'var(--green)';
  if (s >= 45) return 'var(--amber)';
  return 'var(--red)';
}

async function runAnalysis() {
  const draft = document.getElementById('draft').value.trim();
  if (!draft) { setStatus('Paste a draft first.', true); return; }

  const btn = document.getElementById('analyzeBtn');
  btn.disabled = true;
  setStatus('Sending to local model… (may take 30–90 s)', false);
  document.getElementById('results').style.display = 'none';

  try {
    const resp = await fetch('/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ draft })
    });
    const data = await resp.json();
    if (!resp.ok || data.error) {
      setStatus(data.error || 'Unknown error', true);
      return;
    }
    renderResults(data);
    setStatus('', false);
  } catch (e) {
    setStatus('Request failed: ' + e.message, true);
  } finally {
    btn.disabled = false;
  }
}

function renderResults(data) {
  // Overall score
  const score = data.overall_score ?? 0;
  document.getElementById('overallScore').textContent = score;
  document.getElementById('overallScore').style.color = scoreColor(score);
  document.getElementById('rationale').textContent = data.decision_rationale ?? '';

  // Decision badge
  const badge = document.getElementById('decisionBadge');
  const d = data.decision ?? 'poor_quality';
  const labels = {
    ready_for_review: 'Ready for review',
    needs_clarification: 'Needs clarification',
    poor_quality: 'Poor quality'
  };
  const classes = {
    ready_for_review: 'badge-ready',
    needs_clarification: 'badge-clarify',
    poor_quality: 'badge-poor'
  };
  badge.textContent = labels[d] ?? d;
  badge.className = 'decision-badge ' + (classes[d] ?? 'badge-poor');

  // Metric cards
  const grid = document.getElementById('metricsGrid');
  grid.innerHTML = '';
  const scores = data.scores ?? {};
  for (const [key, label] of Object.entries(METRIC_LABELS)) {
    const m = scores[key] ?? { score: 0, note: 'No data' };
    const s = m.score ?? 0;
    const card = document.createElement('div');
    card.className = 'metric-card';
    card.innerHTML = `
      <div class="metric-header">
        <span class="metric-name">${label}</span>
        <span class="metric-score" style="color:${scoreColor(s)}">${s}</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill" style="width:${s}%; background:${scoreColor(s)};"></div>
      </div>
      <div class="metric-note">${m.note ?? ''}</div>`;
    grid.appendChild(card);
  }

  // Top issues
  const issuesList = document.getElementById('issuesList');
  issuesList.innerHTML = '';
  (data.top_issues ?? []).forEach((issue, i) => {
    const div = document.createElement('div');
    div.className = 'issue-item';
    div.innerHTML = `<span class="issue-num">${i + 1}.</span><span>${issue}</span>`;
    issuesList.appendChild(div);
  });

  document.getElementById('results').style.display = 'block';
}

function setStatus(msg, isErr) {
  const el = document.getElementById('statusMsg');
  el.textContent = msg;
  el.className = 'status' + (isErr ? ' err' : '');
}

// Load model info on startup
fetch('/models').then(r => r.json()).then(d => {
  const tag = document.getElementById('modelTag');
  if (d.error) {
    tag.textContent = '⚠ Ollama not reachable — run: ollama serve';
    tag.style.color = 'var(--red)';
  } else {
    tag.textContent = 'model: ' + (d.current || 'unknown') +
      (d.models.length ? '  ·  available: ' + d.models.join(', ') : '');
  }
});
</script>
</body>
</html>
"""

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"""
╔══════════════════════════════════════════════════════╗
║       Patent Readiness Analyzer — Local Edition      ║
╠══════════════════════════════════════════════════════╣
║  Model  : {OLLAMA_MODEL:<42}║
║  Ollama : {OLLAMA_BASE_URL:<42}║
║  UI     : http://localhost:{FLASK_PORT:<26}║
║                                                      ║
║  All data stays on your machine.                     ║
╚══════════════════════════════════════════════════════╝
""")
    app.run(host="127.0.0.1", port=FLASK_PORT, debug=False)