"""
vectorstore.py — ChromaDB collection setup and management.
Handles persistent storage, collection naming, and embedding function.
"""

import logging
from typing import Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

VALID_CHUNK_SIZES = (256, 512, 1024)

# ---------------------------------------------------------------------------
# Embedding function (shared across ingest + retrieval)
# ---------------------------------------------------------------------------

def get_embedding_function() -> embedding_functions.SentenceTransformerEmbeddingFunction:
    """Return a cached SentenceTransformer embedding function."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def get_chroma_client() -> chromadb.HttpClient:
    """
    Return an HttpClient connected to the running ChromaDB server.
    ChromaDB must be running at CHROMA_HOST:CHROMA_PORT
    (typically via Docker: chromadb/chroma image).
    """
    client = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        settings=Settings(anonymized_telemetry=False),
    )
    logger.info("Connected to ChromaDB at %s:%s", CHROMA_HOST, CHROMA_PORT)
    return client

# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------

def collection_name(chunk_size: int) -> str:
    """
    Canonical collection name for a given chunk size.
    E.g. chunk_size=512 → 'pubmed_512'
    """
    if chunk_size not in VALID_CHUNK_SIZES:
        raise ValueError(
            f"chunk_size must be one of {VALID_CHUNK_SIZES}, got {chunk_size}"
        )
    return f"pubmed_{chunk_size}"


def get_or_create_collection(
    chunk_size: int,
    client: Optional[chromadb.HttpClient] = None,
) -> chromadb.Collection:
    """
    Get an existing ChromaDB collection, or create it if absent.

    Args:
        chunk_size: Token chunk size — determines collection name.
        client:     Optional pre-built client (useful for testing).

    Returns:
        chromadb.Collection with cosine similarity metric and MiniLM embeddings.
    """
    client = client or get_chroma_client()
    name   = collection_name(chunk_size)
    ef     = get_embedding_function()

    collection = client.get_or_create_collection(
        name=name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},   # cosine similarity for dense retrieval
    )
    logger.info(
        "Collection '%s' ready — %d documents currently stored.",
        name,
        collection.count(),
    )
    return collection


def delete_collection(chunk_size: int, client: Optional[chromadb.HttpClient] = None) -> None:
    """Drop a collection entirely (useful for re-ingestion runs)."""
    client = client or get_chroma_client()
    name   = collection_name(chunk_size)
    client.delete_collection(name)
    logger.warning("Deleted collection '%s'.", name)


def collection_stats(chunk_size: int, client: Optional[chromadb.HttpClient] = None) -> dict:
    """Return basic stats for a collection."""
    client     = client or get_chroma_client()
    collection = get_or_create_collection(chunk_size, client)
    return {
        "collection": collection.name,
        "doc_count":  collection.count(),
        "embed_model": EMBED_MODEL,
        "chunk_size":  chunk_size,
    }
