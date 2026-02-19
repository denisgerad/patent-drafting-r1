"""core/exceptions.py  –  All domain-specific exceptions."""


class PatentRAGError(Exception):
    """Base exception for the Patent RAG system."""

class OllamaNotAvailableError(PatentRAGError):
    """Ollama cannot be reached or started."""

class NoModelsFoundError(PatentRAGError):
    """No LLM models installed in Ollama."""

class PDFLoadError(PatentRAGError):
    """PDF cannot be read or parsed."""

class VectorStoreError(PatentRAGError):
    """ChromaDB failure."""

class TranslationError(PatentRAGError):
    """Helsinki translation failure."""

class ClassificationError(PatentRAGError):
    """Domain classification failure."""

class ValidationError(PatentRAGError):
    """Agent output does not meet quality gate."""

class ProjectError(PatentRAGError):
    """Project management failure."""
