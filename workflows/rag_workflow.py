"""
workflows/rag_workflow.py
RAG workflow for patent drafting — retrieves domain-specific context
and builds two-pass context for model prompts.
"""

from typing import Tuple, Optional
import chromadb
import logging

logger = logging.getLogger(__name__)


def build_two_pass_context(
    collection: chromadb.Collection,
    domain_type: str,
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Build domain-specific context for patent drafting using two passes:
    1. First pass: general patent context
    2. Second pass: domain-specific constraints and patterns
    
    Parameters
    ----------
    collection : chromadb.Collection
        The vector store collection containing patent and domain documents
    domain_type : str
        Domain/product type key from product_type_checklists.json
        e.g., "flexible_heater_film", "optical_coating"
    
    Returns
    -------
    Tuple[str, Optional[str], Optional[str]]
        (rag_context, field, mechanism)
        - rag_context: concatenated domain-specific context
        - field: primary field of technology (optional)
        - mechanism: mechanism of operation (optional)
    """
    
    if not collection or collection.count() == 0:
        logger.warning("Collection is empty; returning empty RAG context")
        return "", None, None
    
    # First pass: general patent context
    general_query = "patent drafting best practices claims structure"
    general_context = _search_context(collection, general_query, n_results=3)
    
    # Second pass: domain-specific context
    domain_query = f"{domain_type} technical specifications constraints"
    domain_context = _search_context(collection, domain_query, n_results=5)
    
    # Combine contexts
    rag_context = f"{general_context}\n\n{domain_context}".strip()
    
    # Extract field and mechanism (optional metadata)
    field = domain_type  # Can be expanded to extract from collection metadata
    mechanism = None     # Can be expanded to extract from documents
    
    logger.info(
        f"Built RAG context for domain '{domain_type}': "
        f"{collection.count()} total docs, {len(rag_context)} chars context"
    )
    
    return rag_context, field, mechanism


def _search_context(collection: chromadb.Collection, query: str, n_results: int = 5) -> str:
    """
    Search the collection and format results as context.
    
    Parameters
    ----------
    collection : chromadb.Collection
        The collection to search
    query : str
        Search query
    n_results : int
        Number of results to retrieve
    
    Returns
    -------
    str
        Formatted context from search results
    """
    try:
        results = collection.query(query_texts=[query], n_results=n_results)
        docs = results.get("documents", [[]])[0]
        if not docs:
            return ""
        return "\n\n".join(docs)
    except Exception as e:
        logger.error(f"Error searching collection: {e}")
        return ""
