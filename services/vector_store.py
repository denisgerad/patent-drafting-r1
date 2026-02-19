"""
services/vector_store.py
Thin, stateless wrapper around ChromaDB.
Each caller names its collection explicitly – no global state here.
"""
from __future__ import annotations

import hashlib
import logging
from typing import List, Optional

import chromadb

import config.settings as cfg
from core.exceptions import VectorStoreError

logger = logging.getLogger(__name__)

_client: Optional[chromadb.PersistentClient] = None


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=cfg.CHROMA_PATH)
    return _client


def collection_name_for_project(project_path: str, doc_type: str) -> str:
    """
    Generate a stable, unique ChromaDB collection name tied to a project path.
    e.g.  project_name='MVB_v2', doc_type='draft1'  →  'proj_a1b2c3d4_draft1'
    """
    path_hash = hashlib.md5(project_path.encode()).hexdigest()[:8]
    return f"proj_{path_hash}_{doc_type}"


def create_collection(
    chunks: List[str],
    name: str,
    reuse_if_exists: bool = False,
) -> chromadb.Collection:
    """
    Upsert a ChromaDB collection with the given chunks.

    Parameters
    ----------
    chunks          : text segments to index
    name            : collection name (must be unique per project+doc_type)
    reuse_if_exists : if True and the collection already has data, reuse it
                      (useful when restoring a saved project session)
    """
    if not chunks:
        raise VectorStoreError("Cannot create vector store: no chunks provided.")
    if not name:
        raise VectorStoreError("Cannot create vector store: no collection name provided.")

    client = _get_client()

    if reuse_if_exists:
        try:
            col = client.get_collection(name=name)
            if col.count() > 0:
                logger.info("Reusing existing collection '%s' (%d items).", name, col.count())
                return col
            client.delete_collection(name=name)
        except Exception:
            pass  # collection does not exist yet – create fresh

    # Fresh creation
    try:
        client.delete_collection(name=name)
    except Exception:
        pass

    col = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})

    for i, chunk in enumerate(chunks):
        col.add(documents=[chunk], ids=[f"chunk_{i}"], metadatas=[{"source": "doc"}])

    logger.info("Created collection '%s' with %d chunks.", name, col.count())
    return col


def search(collection: chromadb.Collection, query: str, n_results: int = 5) -> str:
    """Return top-k chunks joined by double newline."""
    results = collection.query(query_texts=[query], n_results=n_results)
    docs: List[str] = results["documents"][0]
    return "\n\n".join(docs)


def delete_all_collections() -> None:
    """Hard reset – remove every collection (used by 'Reset All' button)."""
    client = _get_client()
    for col in client.list_collections():
        try:
            client.delete_collection(name=col.name)
        except Exception:
            pass
    logger.info("All ChromaDB collections deleted.")
