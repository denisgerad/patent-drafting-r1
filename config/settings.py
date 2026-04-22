"""
config/settings.py  –  Single source of truth for all runtime configuration.
Import the module-level names directly:  from config.settings import OLLAMA_BASE_URL
"""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str       = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str          = os.getenv("OLLAMA_MODEL", "")
MISTRAL_MODEL: str         = os.getenv("MISTRAL_MODEL", "")          # legacy alias
OLLAMA_TIMEOUT: int        = int(os.getenv("OLLAMA_TIMEOUT", "5"))
OLLAMA_STARTUP_WAIT: int   = int(os.getenv("OLLAMA_STARTUP_WAIT", "15"))
OLLAMA_GEN_TIMEOUT: int    = int(os.getenv("OLLAMA_GEN_TIMEOUT", "120"))

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_PATH: str  = os.getenv("CHROMA_PATH", "./chromadb_store")

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int    = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

# ── Translation ───────────────────────────────────────────────────────────────
TRANSLATION_MODEL: str   = os.getenv("TRANSLATION_MODEL", "Helsinki-NLP/opus-mt-en-de")
TRANSLATION_MAX_LEN: int = int(os.getenv("TRANSLATION_MAX_LENGTH", "512"))

# ── Auto-classifier ───────────────────────────────────────────────────────────
CLASSIFIER_CONFIDENCE: float = float(os.getenv("CLASSIFIER_CONFIDENCE", "0.55"))
CLASSIFIER_RAG_RESULTS: int  = int(os.getenv("CLASSIFIER_RAG_RESULTS", "10"))

# ── Paths ─────────────────────────────────────────────────────────────────────
PATENT_TYPES_JSON: Path = Path(os.getenv("PATENT_TYPES_JSON", "patent_types.json"))
PRODUCT_CHECKLISTS_JSON: Path = Path(os.getenv("PRODUCT_CHECKLISTS_JSON", "product_type_checklists.json"))
PROJECTS_DIR: Path      = Path(os.getenv("PROJECTS_DIR", "./projects"))
TEMP_DIR: Path          = Path(os.getenv("TEMP_DIR", "./temp"))

# ── CrewAI ────────────────────────────────────────────────────────────────────
CREW_TIMEOUT_SECONDS: int = int(os.getenv("CREW_TIMEOUT_SECONDS", "300"))
