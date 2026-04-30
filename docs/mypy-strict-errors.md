# Mypy strict type-check errors (PR Checks)

**Run / job:** https://github.com/rishsoraganvi/ML_Ops-RAG-Based-Clinical-Decision-Support/actions/runs/25127375007/job/73643982536?pr=13  
**Ref:** `be36e49329ed0acff4d987167ea12891e0c37fab`

This repository runs **mypy in strict mode** in CI (`.github/workflows/pr_checks.yml`). The job currently fails with:

> Found **9 errors in 5 files** (checked 18 source files)

This document lists the errors reported in the failing job log.

---

## 1) `src/parser.py`

### Errors
- `src/parser.py:15: error: Missing type arguments for generic type "dict"  [type-arg]`
- `src/parser.py:20: error: Missing type arguments for generic type "dict"  [type-arg]`

### Fix
Parameterize dicts (minimal strict-mypy compliant typing):

```python
from typing import Any

def parse_xml_to_records(xml: str) -> list[dict[str, Any]]:
    ...

def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> int:
    ...
```

---

## 2) `serving/dashboard.py`

### Error
- `serving/dashboard.py:121: error: Unused "type: ignore[misc]" comment  [unused-ignore]`

### Why it happens
The decorator line contains:

```python
@st.cache_data(ttl=60)  # type: ignore[misc,untyped-decorator]
```

In this run, mypy did **not** emit `misc` for that line, so `misc` is considered unused and becomes an error under strict settings.

### Fix
Remove `misc` from the ignore list (keep only what is needed), e.g.:

```python
@st.cache_data(ttl=60)  # type: ignore[untyped-decorator]
```

Or remove the ignore entirely if mypy passes without it.

---

## 3) `rag_pipeline/vectorstore.py`

### Errors
- `rag_pipeline/vectorstore.py:45: error: Module has no attribute "SentenceTransformerEmbeddingFunction"  [attr-defined]`
- `rag_pipeline/vectorstore.py:45: note: Error code "attr-defined" not covered by "type: ignore" comment`
- `rag_pipeline/vectorstore.py:107: error: Argument "embedding_function" to "get_or_create_collection" of "ClientAPI" has incompatible type "Callable[[list[str]], list[list[float]]]"; expected "EmbeddingFunction[...]"  [arg-type]`
- `rag_pipeline/vectorstore.py:107: note: "function" is missing following "EmbeddingFunction" protocol member: embed_with_retries`

### Fix
These errors typically come from **Chroma version / stubs mismatch** and from passing a plain callable where Chroma expects its own `EmbeddingFunction` protocol.

Fix patterns:
- Avoid using `embedding_functions.SentenceTransformerEmbeddingFunction` if your installed `chromadb` version does not expose it.
- Avoid passing a bare `Callable` into `client.get_or_create_collection(..., embedding_function=...)` under strict mypy. Either:
  - remove `embedding_function=` at collection creation time and embed client-side, or
  - switch to a Chroma-supported embedding-function object that satisfies the protocol for your installed version.

---

## 4) `rag_pipeline/ingest.py`

### Error
- `rag_pipeline/ingest.py:178: error: Argument "metadatas" to "upsert" of "Collection" has incompatible type "list[dict[str, str | int | float | bool]]"; expected "Mapping[str, str | int | float | bool] | list[Mapping[str, str | int | float | bool]] | None"  [arg-type]`

### Why it happens
Even though `dict[...]` is a `Mapping[...]` at runtime, mypy can be strict about the exact container type for some third-party stubs.

### Fix
Cast the list to `list[Mapping[...]]` when calling `upsert`, e.g.:

```python
from typing import Mapping, cast

AllowedMetaValue = str | int | float | bool

metadatas = [_sanitize_metadata(c.metadata) for c in chunks]
metadatas_mapping = cast(list[Mapping[str, AllowedMetaValue]], metadatas)

collection.upsert(..., metadatas=metadatas_mapping[start:end])
```

---

## 5) `rag_pipeline/retriever.py`

### Errors
- `rag_pipeline/retriever.py:88: error: Argument "embedding_function" to "Chroma" has incompatible type "Callable[[list[str]], list[list[float]]]"; expected "Embeddings | None"  [arg-type]`
- `rag_pipeline/retriever.py:302: error: Argument "query_embeddings" to "query" of "Collection" has incompatible type "list[list[float]]"; expected "ndarray[...] | list[ndarray[...]]"  [arg-type]`
- `rag_pipeline/retriever.py:502: error: Argument "query_embeddings" to "query" of "Collection" has incompatible type "list[list[float]]"; expected "ndarray[...] | list[ndarray[...]]"  [arg-type]`

### Fix
- For LangChain `Chroma(...)`, pass a LangChain `Embeddings` implementation (not a plain callable).
- For Chroma `collection.query(query_embeddings=...)`, convert embeddings to numpy arrays (`np.ndarray`) (or list of ndarrays) per the stubs.

---

## Notes / workflow context

The strict mypy step is defined here:

```yaml
- name: Mypy type check
  run: |
    mypy src/ mlops/ serving/ \
      --ignore-missing-imports \
      --strict \
      --exclude src/infra/test
```
