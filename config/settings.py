"""
config/settings.py  –  Single source of truth for all runtime configuration.

LLM_PROVIDER controls which backend is used:
  "ollama"  — local Ollama server (default, no API key needed)
  "azure"   — Azure OpenAI Service
  "claude"  — Anthropic Claude API
  "openai"  — OpenAI API direct

Set only the variables for the provider you are using.
"""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Provider selection ────────────────────────────────────────────────────────
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama").lower()
# Valid values: "ollama", "azure", "claude", "openai"

# ── Ollama (local, default) ───────────────────────────────────────────────────
OLLAMA_BASE_URL: str       = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str          = os.getenv("OLLAMA_MODEL", "")
MISTRAL_MODEL: str         = os.getenv("MISTRAL_MODEL", "")          # legacy alias
OLLAMA_TIMEOUT: int        = int(os.getenv("OLLAMA_TIMEOUT", "5"))
OLLAMA_STARTUP_WAIT: int   = int(os.getenv("OLLAMA_STARTUP_WAIT", "15"))
OLLAMA_GEN_TIMEOUT: int    = int(os.getenv("OLLAMA_GEN_TIMEOUT", "30"))

# ── Azure OpenAI ──────────────────────────────────────────────────────────────
AZURE_API_KEY: str          = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_API_BASE: str         = os.getenv("AZURE_OPENAI_ENDPOINT", "")
# e.g. https://YOUR-RESOURCE.openai.azure.com/
AZURE_API_VERSION: str      = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_DEPLOYMENT_NAME: str  = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
# The deployment name you created in Azure AI Studio (not the model name)

# ── Anthropic Claude ──────────────────────────────────────────────────────────
CLAUDE_API_KEY: str   = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL: str     = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
# Other options: claude-opus-4-20250514, claude-haiku-4-5-20251001

# ── OpenAI Direct ─────────────────────────────────────────────────────────────
OPENAI_API_KEY: str  = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str    = os.getenv("OPENAI_MODEL", "gpt-4o")

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
PROJECTS_DIR: Path      = Path(os.getenv("PROJECTS_DIR", "./projects"))
TEMP_DIR: Path          = Path(os.getenv("TEMP_DIR", "./temp"))

# ── CrewAI ────────────────────────────────────────────────────────────────────
CREW_TIMEOUT_SECONDS: int = int(os.getenv("CREW_TIMEOUT_SECONDS", "300"))
