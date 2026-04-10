# data/ingest.py
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from chroma_interface import get_embeddings as _get_embeddings
from chroma_interface import incremental_upsert as _incremental_upsert


def get_embeddings() -> np.ndarray:
    """
    Returns embedding matrix of shape (n_docs, 384).
    Called by Member 1 (MLOps) for drift detection baseline.
    """
    return _get_embeddings()


def incremental_upsert(new_docs: list) -> int:
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