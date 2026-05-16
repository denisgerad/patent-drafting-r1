# AI Patent Drafting Tool – Production Refactor

## Architecture

```
patent_rag/
│
├── app.py                          ← Streamlit entry point
│
├── config/
│   └── settings.py                 ← ALL config (env vars + defaults)
│
├── core/
│   └── exceptions.py               ← Domain-specific exception hierarchy
│
├── services/                       ← Pure, UI-free, independently testable
│   ├── ollama_service.py           ← Ollama process lifecycle + model resolution
│   ├── document_loader.py          ← PDF / DOCX / TXT → chunks (with OCR fallback)
│   ├── vector_store.py             ← ChromaDB wrapper (stateless)
│   ├── project_manager.py          ← File-system project CRUD
│   └── export_service.py           ← DOCX / PDF export (BytesIO)
│
├── agents/
│   └── agent_factory.py            ← Build all CrewAI agents
│
├── tasks/
│   └── task_factory.py             ← Build all CrewAI tasks + post-process utilities
│
├── workflows/                      ← Orchestrate agents+tasks; thread-safe with timeout
│   ├── classification_workflow.py  ← Auto-classify domain → ClassificationResult
│   ├── scrutiny_workflow.py        ← Gap-analysis crew → ScrutinyResult
│   └── consolidation_workflow.py   ← Draft2 crew → ConsolidationResult
│
├── ui/
│   ├── session_state.py            ← All st.session_state keys + defaults
│   ├── sidebar.py                  ← Sidebar renderer (project mgmt, upload)
│   ├── tab_scrutiny.py             ← Step 1 tab (classify + scrutinise)
│   └── tab_consolidation.py        ← Step 2 tab (Q&A upload + consolidate)
│
├── patent_types.json               ← Domain configs (role, focus areas, units)
├── requirements.txt
└── .env.template                   ← Copy to .env and fill values
```

---

## Key Changes vs. Monolith

| Concern | Before | After |
|---|---|---|
| Configuration | `os.getenv()` everywhere | Single `config/settings.py` |
| Ollama lifecycle | Repeated `subprocess` + `requests` blocks | `OllamaService` singleton |
| Document loading | `fitz`/`docx` inline in UI | `document_loader.dispatch()` |
| Vector store | Inline `chromadb` calls | `vector_store.create_collection()` |
| Agent creation | Mixed with task logic | `agent_factory.build_*()` |
| Task creation | Inline in workflow function | `task_factory.build_*()` |
| Classification | **Not implemented** | `classification_workflow.run()` → `ClassificationResult` |
| Session state | Ad-hoc keys everywhere | `ui/session_state.py` with typed defaults |
| Exceptions | `print()` + bare `except` | Domain exception hierarchy |
| Thread safety | `threading.Thread` ad-hoc | All crews run via workflow layer with timeout |

---

## Domain Auto-Classification Flow (New in Step 1a)

```
Draft1 uploaded
      │
      ▼
RAG search (top-10 chunks: components, processes, materials…)
      │
      ▼
ClassifierAgent  ──(Ollama)──►  JSON { primary_domain, confidence, justification }
      │
      ▼
  confidence ≥ 0.55?
      │
   YES │                        NO │
      ▼                           ▼
Auto-apply domain          Show override panel
      │                    + orange confidence warning
      ▼                           │
  Show result banner ◄────────────┘
  with "Override" expander always visible
      │
      ▼
   User confirms or picks different domain
      │
      ▼
  classification_done = True  →  Step 1b (Scrutiny) unlocked
```

### Confidence Threshold
Set via `CLASSIFIER_CONFIDENCE` env var (default `0.55`).  
- `≥ 0.55` → green banner, override expander collapsed  
- `< 0.55` → orange warning, override expander auto-expanded  

---

## Setup

```bash
# 1. Clone / copy files
cd patent_rag

# 2. Python environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

pip install -r requirements.txt

# 3. Ollama
# Install from https://ollama.com
ollama pull mistral:7b-instruct-q4_K_M

# 4. Environment
cp .env.template .env
# Edit .env: set OLLAMA_MODEL, paths, etc.

# 5. Run
streamlit run app.py
```

---

## RAM Optimisation (16 GB)

The app deliberately stops Ollama while processing documents (Phase 1) and
restarts it only when the LLM is needed (Phase 2). After consolidation it
stops Ollama again before any Helsinki translation step.

Sequence:
```
Upload PDF → stop Ollama → embed → index → [Ollama stopped]
                                      │
                            User clicks "Auto-Classify"
                                      │
                            start Ollama → classify → stop Ollama*
                                      │
                            User clicks "Generate Questions"
                                      │
                            start Ollama → scrutiny → stop Ollama*
                                      │
                            User uploads Q&A → index → [Ollama stopped]
                                      │
                            User clicks "Generate Draft 2"
                                      │
                            start Ollama → consolidate → stop Ollama*

* Ollama.stop() is called automatically by the sidebar _process_draft1 and
  can also be triggered manually via "Clear Session".
```
