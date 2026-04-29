# data/ingest.py
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.chroma_interface import get_embeddings as _get_embeddings
from src.chroma_interface import incremental_upsert as _incremental_upsert


def get_embeddings(chunk_size: int | None = None) -> NDArray[np.float32]:
    """
    Returns embedding matrix of shape (n_docs, 384).
    Called by Member 1 (MLOps) for drift detection baseline.

    Args:
        chunk_size: Which ``pubmed_{chunk_size}`` collection to query.
            Defaults to ``settings.baseline_chunk_size`` (currently 512).
    """
    if chunk_size is None:
        from src.config.settings import settings

        chunk_size = settings.baseline_chunk_size
    return _get_embeddings(chunk_size=chunk_size)


def incremental_upsert(new_docs: list[dict[str, Any]]) -> int:
    """
    Adds new documents to ChromaDB without re-embedding existing ones.
    new_docs: list of dicts with keys: pmid, text, title, pub_date, mesh_terms, word_count
    Returns count of newly added documents.
    """
    return _incremental_upsert(new_docs)


if __name__ == "__main__":
    print("Testing get_embeddings()...")
    matrix = get_embeddings()
    print(f"  Shape: {matrix.shape}")
    assert matrix.shape[1] == 384, "Wrong embedding dimension!"
    print("  get_embeddings() OK")
    print("\nAll interfaces working correctly.")