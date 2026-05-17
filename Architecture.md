# AI Patent Drafting Tool — Architecture

```text
╔══════════════════════════════════════════════════════════════════════════════════════╗
║               AI PATENT DRAFTING TOOL — COMPLETE ARCHITECTURE                        ║
║               Block Diagram — All components and data flows                          ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────────────┐
│  USER INTERFACE  (Streamlit — app.py)                                               │
│                                                                                     │
│   ┌──────────────┐   ┌────────────────────────────────────────────────────────┐     │
│   │   Sidebar    │   │  Main Content Area                                     │     │
│   │              │   │                                                        │     │
│   │ • Project    │   │  ┌───────────────────────┐  ┌──────────────────────┐   │     │
│   │   select /   │   │  │  STEP 1: SCRUTINY     │  │ STEP 2: CONSOLIDATE  │   │     │
│   │   create     │   │  │  tab_scrutiny.py      │  │ tab_consolidation.py │   │     │
│   │              │   │  │                       │  │                      │   │     │ 
│   │ • Draft1     │   │  │ 1a. Domain classify   │  │ • Upload Q&A doc     │   │     │
│   │   upload     │   │  │ 1b. Generate questions│  │ • Generate Draft 2   │   │     │
│   │   (PDF/DOCX) │   │  │ • Field banner        │  │ • Download           │   │     │
│   │              │   │  │ • Mechanism banner    │  │   MD / DOCX / PDF    │   │     │
│   │ • System     │   │  │ • Readiness gate panel│  │ • Audit log          │   │     │
│   │   status     │   │  │ • Augmentation badge  │  │                      │   │     │
│   │   (provider) │   │  │ • Download questions  │  └──────────────────────┘   │     │
│   │              │   │  └───────────────────────┘                             │     │
│   │ • Session    │   └────────────────────────────────────────────────────────┘     │
│   │   controls   │                                                                  │
│   └──────────────┘                                                                  │
└────────────────────────────────┬────────────────────────────────────────────────────┘
                                 │
                    ─────────────▼─────────────
                    WORKFLOWS (Orchestration)
                    ───────────────────────────

┌────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 1 WORKFLOW  (workflows/scrutiny_workflow.py)                                 │
│                                                                                    │
│   Draft1 uploaded                                                                  │
│         │                                                                          │
│         ▼                                                                          │
│   ┌─────────────────────────────┐                                                  │
│   │  3-PASS RAG RETRIEVAL       │  ← services/vector_store.py (ChromaDB)           │
│   │                             │                                                  │
│   │  Pass 1: broad query        │  "field of invention abstract summary..."        │
│   │     → extract_field_of_invention()                                             │
│   │     → extract_mechanism()                                                      │
│   │     → _match_checklist_type()   ─────────────────────────────────────────┐     │
│   │                             │                                            │     │
│   │  Pass 2: targeted query     │  field text  OR  checklist RAG query       │     │
│   │           (checklist overrides user domain selection)                    │     │
│   │                             │                                            │     │
│   │  Pass 3: checklist-specific │  always pulls product-type relevant chunks │     │
│   │                             │                                            │     │
│   │  → combined_context (deduplicated, 3 passes merged)                      │     │
│   └─────────────────────────────┘                                            │     │
│         │                                                                    │     │
│         ▼                                                                    │     │
│   ┌─────────────────────────────┐                                            │     │
│   │  TASK FACTORY               │  ← tasks/task_factory.py                   │     │
│   │                             │                                            │     │
│   │  build_scrutiny_task():     │                                            │     │
│   │  • field_anchor_block       │  pre-extracted FIELD injected              │     │
│   │  • mechanism_hint           │  pre-extracted MECHANISM injected          │     │
│   │  • checklist_block          │  LOCKED SECTION HEADINGS                   │     │
│   │  • section_reminder         │  repeated immediately before OUTPUT        │     │
│   │  • compact prompt (41 lines)│  ← shortened for small model attention     │     │
│   └─────────────────────────────┘                                            │     │
│         │                                                                          │
│         ▼                                                                          │
│   ┌─────────────────────────────┐                                                  │
│   │  CREWAI CREW (threaded)     │  timeout = CREW_TIMEOUT_SECONDS                  │
│   │                             │                                                  │
│   │  scrutinizer Agent          │  ← agents/agent_factory.py                       │
│   │  • role from patent_types   │  • build_scrutinizer()                           │
│   │  • expert backstory         │  • make_llm() → provider-agnostic                │
│   │  • llm = get_llm()          │                                                  │
│   │         │                   │                                                  │
│   │         ▼                   │                                                  │
│   │  LOCAL MODEL (Ollama)       │  Nemo 12B / Qwen 32B / Phi-4 14B                 │
│   │  Patent content stays LOCAL │  Nothing leaves the machine                      │
│   └─────────────────────────────┘                                                  │
│         │                                                                          │
│         ▼                                                                          │
│   ┌─────────────────────────────┐                                                  │
│   │  POST-PROCESSING            │                                                  │
│   │                             │                                                  │
│   │  _validate_novelty_line()   │  4-gram overlap check vs document                │
│   │  • if < 15% overlap         │  → replace with "NOT STATED"                     │
│   │  • hallucination guard      │  blocks biometric / LGP drift                    │
│   └─────────────────────────────┘                                                  │
│         │                                                                          │
│         ▼                                                                          │
│   ┌─────────────────────────────┐                                                  │
│   │  READINESS GATE             │  ← services/question_rater.py                    │
│   │                             │                                                  │
│   │  1. _match_checklist_type(field)   ─────────────────────────────┐              │
│   │  2. _match_checklist_type(mechanism)   ─────────────────────────┤              │
│   │  3. contradiction? → mechanism wins                             │              │
│   │  4. load reference JSON     │  domain_reference_questions/      │              │
│   │  5. score 3 dimensions:     │  • category completeness 25%      │ ◄────────────┘
│   │     • topic coverage   40%  │  • key terms present?             │
│   │     • depth markers    35%  │  • units, drawings, formulas?     │
│   │  6. verdict: READY /        │                                   │
│   │     BORDERLINE / NOT_READY  │                                   │
│   └─────────────────────────────┘                                   │
│         │                                                           │
│         ├── Score ≥ 65% ──────────────────────────► DONE            │
│         │                                           (Level 1)       │
│         │                                                           │
│         └── Score < 65% ─────────────────────────────────────────────────────────┐ │
│                                                                                  │ │
└──────────────────────────────────────────────────────────────────────────────────┘ │
                                                                                     │
┌────────────────────────────────────────────────────────────────────────────────────┘
│  HYBRID AUGMENTATION  (services/expert_augmentor.py)
│
│  Level 2 — Expert Bank (FREE, OFFLINE, no cloud call)
│  ┌──────────────────────────────────────────────────────────────────────┐
│  │  load_expert_bank(product_type)                                      │
│  │  domain_reference_questions/{type}_bank.json                         │
│  │  • review_status must be "APPROVED"                                  │
│  │  • get_bank_questions_for_categories(weak_categories)                │
│  │  • merge into model output                                           │
│  │  • re-score                                                          │
│  └──────────────────────────────────────────────────────────────────────┘
│         │
│         ├── Score ≥ 65% ──────────────────────────► DONE (Level 2, free)
│         │
│         └── Score < 65% AND ENABLE_CLOUD_AUGMENTATION=true
│
│  Level 3 — Claude Augmentation (ONLINE, per-run, ~$0.01)
│  ┌──────────────────────────────────────────────────────────────────────┐
│  │  augment_with_claude()                                               │
│  │                                                                      │
│  │  SENT TO CLAUDE (non-sensitive labels only):                         │
│  │    • product_type:    "flexible_heater_film"                         │
│  │    • mechanism:       "resistive heating element in polymer film"    │
│  │    • weak_categories: ["Material, Fabrication & Performance"]        │
│  │    • missing_terms:   ["injection moulding", "haze %"]               │
│  │                                                                      │
│  │  NEVER SENT TO CLAUDE:                                               │
│  │    ✗ Patent document content                                         │
│  │    ✗ Inventor details                                                │
│  │    ✗ Proprietary specifications                                      │
│  │    ✗ Any measurement or formula from the disclosure                  │
│  │                                                                      │
│  │  RETURNED FROM CLAUDE:                                               │
│  │    Expert questions for weak categories                              │
│  │    → merged into model output                                        │
│  │    → re-scored                                                       │
│  └──────────────────────────────────────────────────────────────────────┘
│         │
│         └── Final result → UI (augmentation badge shown)
│
└─────────────────────────────────────────────────────────────────────────

┌─────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 2 WORKFLOW  (workflows/consolidation_workflow.py)                             │
│                                                                                     │
│   Inventor answers Q&A offline → uploads Q&A document (PDF/DOCX/TXT)                │
│         │                                                                           │
│         ▼                                                                           │
│   ┌─────────────────────────────┐                                                   │
│   │  RAG over Q&A collection    │  separate ChromaDB collection per project         │
│   │  + RAG over Draft1          │  n=15 chunks from original patent sheet           │
│   └─────────────────────────────┘                                                   │
│         │                                                                           │
│         ▼                                                                           │
│   ┌─────────────────────────────┐                                                   │
│   │  CONSOLIDATOR AGENT         │  LOCAL MODEL (no cloud call)                      │
│   │                             │                                                   │
│   │  Strict non-lossy merge:    │                                                   │
│   │  • preserve Draft1 verbatim │                                                   │
│   │  • insert Q&A details only  │                                                   │
│   │  • append audit log         │  === AUDIT LOG ===                                │
│   └─────────────────────────────┘                                                   │
│         │                                                                           │
│         ▼                                                                           │
│   ┌─────────────────────────────┐                                                   │
│   │  verify_draft_inclusion()   │  checks Draft1 sentences present in Draft2        │
│   │  split_audit_log()          │  separates draft from audit trail                 │
│   └─────────────────────────────┘                                                   │
│         │                                                                           │
│         ▼                                                                           │
│   Draft 2  →  Download as MD / DOCX / PDF                                           │ 
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│  SERVICES LAYER  (stateless, independently testable, no UI imports)                 │
│                                                                                     │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐          │
│  │ document_loader.py  │  │  vector_store.py    │  │  ollama_service.py  │          │
│  │                     │  │                     │  │                     │          │
│  │ load_chunks()       │  │ create_collection() │  │ ensure_running()    │          │
│  │ • PDF (PyMuPDF)     │  │ search()            │  │ resolve_model()     │          │
│  │ • DOCX              │  │ delete_all()        │  │ test_generation()   │          │
│  │ • TXT               │  │                     │  │ diagnose_llm_stack()│          │
│  │ • OCR fallback      │  │ ChromaDB persistent │  │ stop()              │          │
│  └─────────────────────┘  │ per-project named   │  └─────────────────────┘          │
│                           │ collections         │                                   │
│  ┌─────────────────────┐  └─────────────────────┘    ┌─────────────────────┐        │
│  │ project_manager.py  │                             │ cloud_llm_service.py│        │
│  │                     │  ┌─────────────────────┐    │                     │        │
│  │ create_project()    │  │  question_rater.py  │    │ get_llm()           │        │
│  │ list_projects()     │  │                     │    │ • Strategy 1:       │        │
│  │ save_questions()    │  │ rate()              │    │   crewai.LLM        │        │
│  │ save_document()     │  │ • 3 dimensions      │    │   (crewai ≥ 0.80)   │        │
│  │ get_file_path()     │  │ • category scores   │    │ • Strategy 2:       │        │
│  │ update_metadata()   │  │ • verdict + reason  │    │   langchain_ollama  │        │
│  │ _restore_session()  │  │ • checklist fallback│    │ • Strategy 3:       │        │
│  └─────────────────────┘  └─────────────────────┘    │   plain string      │        │
│                                                      └─────────────────────┘        │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐          │
│  │  export_service.py  │  │ expert_augmentor.py │  │  LLM PROVIDERS      │          │
│  │                     │  │                     │  │                     │          │
│  │ to_docx()           │  │ get_augmentation()  │  │ ollama  (local)     │          │
│  │ to_pdf()            │  │ load_expert_bank()  │  │ azure   (cloud)     │          │
│  │ structure-preserve  │  │ augment_with_claude()│ │ claude  (cloud)     │          │
│  └─────────────────────┘  └─────────────────────┘  │ openai  (cloud)     │          │
│                                                    │                     │          │
│                                                    │ set via             │          │
│                                                    │ LLM_PROVIDER=       │          │
│                                                    └─────────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│  CONFIGURATION & KNOWLEDGE FILES                                                    │
│                                                                                     │
│  ┌─────────────────────────────┐  ┌───────────────────────────────────────────┐     │
│  │  patent_types.json          │  │  product_type_checklists.json             │     │
│  │                             │  │                                           │     │
│  │  Per domain:                │  │  Per product type (16 entries):           │     │
│  │  • role (agent persona)     │  │  • triggers (regex match on field text)   │     │
│  │  • focus_areas              │  │  • anti_patterns (scope prohibitions)     │     │
│  │  • technical_units          │  │  • expert_categories                      │     │
│  │  • anti_patterns            │  │    - name (locked section heading)        │     │
│  │                             │  │    - questions (depth examples)           │     │
│  │  7 domains:                 │  │                                           │     │
│  │  Electronics                │  │  16 product types:                        │     │
│  │  Optics/Display             │  │  flexible_heater_film                     │     │
│  │  Mechanical                 │  │  light_guide_plate                        │     │
│  │  Chemical                   │  │  oled_display                             │     │
│  │  Software                   │  │  power_electronics                        │     │
│  │  Medical Devices            │  │  digital_fpga_asic                        │     │
│  │  Materials                  │  │  sensor_signal_acquisition                │     │
│  └─────────────────────────────┘  │  wireless_communication                   │     │
│                                   │  motor_actuator_drive                     │     │
│  ┌─────────────────────────────┐  │  embedded_firmware                        │     │
│  │  domain_reference_questions/│  │  algorithm_method_patent                  │     │
│  │                             │  │  machine_learning_ai                      │     │
│  │  {type}.json  (scoring)     │  │  communication_protocol                   │     │
│  │  • reference_questions      │  │  database_data_structure                  │     │
│  │  • key_terms                │  │  identity_verification_system             │     │
│  │  • depth_markers            │  │  led_array_pcb                            │     │
│  │  • weights per category     │  │  optical_coating                          │     │
│  │  • readiness_threshold      │  └───────────────────────────────────────────┘     │
│  │                             │                                                    │
│  │  {type}_bank.json  (expert) │  ┌───────────────────────────────────────────┐     │
│  │  • Claude-generated Q bank  │  │  config/settings.py                       │     │
│  │  • review_status: APPROVED  │  │                                           │     │
│  │  • 6 enablement dimensions  │  │  LLM_PROVIDER         ollama/azure/claude │     │
│  │  • 30-40 questions each     │  │  OLLAMA_MODEL         model tag           │     │
│  │                             │  │  AZURE_OPENAI_*       endpoint + key      │     │
│  │  Currently available:       │  │  ANTHROPIC_API_KEY    claude key          │     │
│  │  light_guide_plate.json     │  │  ENABLE_CLOUD_AUGMENTATION  true/false    │     │
│  │  flexible_heater_film.json  │  │  CLASSIFIER_CONFIDENCE      0.55          │     │
│  │  power_electronics.json     │  │  CREW_TIMEOUT_SECONDS       300           │     │
│  │  algorithm_method.json      │  └───────────────────────────────────────────┘     │
│  │  oled_display.json          │                                                    │
│  └─────────────────────────────┘                                                    │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│  OFFLINE TOOLING  (scripts/)                                                        │
│                                                                                     │
│  scripts/generate_expert_bank.py                                                    │
│                                                                                     │
│  Run ONCE per product type to populate the expert bank.                             │
│  Sends to Claude: product type label + mechanism description only.                  │
│  Never sends patent content.                                                        │
│                                                                                     │
│  python scripts/generate_expert_bank.py --product-type flexible_heater_film         │
│  python scripts/generate_expert_bank.py --all                                       │
│  python scripts/generate_expert_bank.py --list                                      │
│                                                                                     │
│  Output: domain_reference_questions/{type}_bank.json (PENDING_EXPERT_REVIEW)        │
│  After domain expert review: change status to APPROVED → used automatically         │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│  DATA PROTECTION SUMMARY                                                            │
│                                                                                     │
│  LOCAL MACHINE (always):                                                            │
│    • Patent document text (PDF/DOCX chunks)                                         │
│    • Inventor details, specifications, measurements                                 │
│    • ChromaDB vector store                                                          │
│    • All project files and metadata                                                 │
│    • Draft 1 and Draft 2 documents                                                  │
│    • Q&A answers from inventor                                                      │
│                                                                                     │
│  TO CLOUD (only with explicit config, never patent content):                        │
│    • generate_expert_bank.py: product type label + mechanism string                 │
│    • augment_with_claude():   category names + missing term labels                  │
│    • LLM_PROVIDER=azure/claude: NO patent content, only prompt templates            │
│                                                                                     │
│  AUDIT:                                                                             │
│    • llm_audit.log: every cloud API call timestamped locally                        │
│    • Sidebar: shows API key status and provider name                                │
│    • UI: cloud augmentation badge shown when Phase 2 fires                          │
└─────────────────────────────────────────────────────────────────────────────────────┘


The three-layer quality system in the Step 1 workflow is the most important thing to understand operationally. Level 1 (local model alone) fires every time. Level 2 (expert bank) fires automatically when the score is below 65% and an approved bank file exists — this costs nothing and requires no internet. Level 3 (Claude augmentation) fires only when both Level 1 and Level 2 are insufficient AND you have set ENABLE_CLOUD_AUGMENTATION=true. Right now you are at Level 1 only because no bank files are approved yet.

The knowledge files are the most important things to maintain over time. The code is stable. What improves quality as you process more patents is: adding more product types to product_type_checklists.json, refining key terms in domain_reference_questions/ based on domain expert feedback, and running generate_expert_bank.py to populate approved banks. Each of those is a JSON edit or a one-time script run — no code changes.

The data protection boundary is precise: the patent document never crosses it under any configuration. Even with LLM_PROVIDER=azure or LLM_PROVIDER=claude, what the cloud receives is the prompt template (generic instructions) not the document chunks. The only cloud exposure is the labels — product type, mechanism description, category names — which are product-category descriptions, not proprietary technical content.
```
