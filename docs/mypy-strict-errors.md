# Mypy strict type-check errors (PR Checks)

**Run / job:** https://github.com/rishsoraganvi/ML_Ops-RAG-Based-Clinical-Decision-Support/actions/runs/25124957548/job/73635360991  
**Ref:** `1982b34d7eb60f54ad86b0a42cc098fc3a2f1e0e`

This repository runs **mypy in strict mode** in CI (`.github/workflows/pr_checks.yml`). The job currently fails with:

> Found **88 errors in 12 files**

This document lists the errors reported in the failing job log and provides concrete guidance on how to fix each category.

---

## 1) `serving/dashboard.py`

### Error
- `serving/dashboard.py:121: error: Untyped decorator makes function "_fetch_mlflow_runs" untyped  [untyped-decorator]`
- `serving/dashboard.py:121: note: Error code "untyped-decorator" not covered by "type: ignore" comment`

### Why it happens
`streamlit.cache_data` is not typed strongly enough for strict mypy, so decorating a typed function causes mypy to treat the function as untyped.

### Fix options
**Preferred:** ignore the correct error code at the decorator line:

```python
@st.cache_data(ttl=60)  # type: ignore[untyped-decorator]
def _fetch_mlflow_runs(...) -> pd.DataFrame:
    ...
```

Alternative: wrap caching behind a helper function or disable strictness for this module in `mypy.ini` (less preferred).

---

## 2) `mlops/mlflow_tracker.py`

### Errors
- `mlops/mlflow_tracker.py:215: error: Returning Any from function declared to return "str"  [no-any-return]`
- `mlops/mlflow_tracker.py:227: error: Returning Any from function declared to return "str"  [no-any-return]`
- `mlops/mlflow_tracker.py:236: error: Returning Any from function declared to return "str"  [no-any-return]`
- `mlops/mlflow_tracker.py:514: error: Argument "run_id" to "QualityGateResult" has incompatible type "str | None"; expected "str"  [arg-type]`

### Why it happens
- MLflow APIs are often typed as `Any` which leaks into your return values.
- `active_run_id` is declared as `str | None`, but `QualityGateResult.run_id` expects `str`.

### Fix
- Cast MLflow experiment IDs to `str` in `_get_or_create_experiment()` and anywhere else returning experiment IDs.
- Assert `run_id is not None` in `log_quality_gate_results()` before passing into `QualityGateResult`.

Example patterns:

```python
from typing import cast

return cast(str, experiment.experiment_id)
```

```python
run_id = self.active_run_id
if run_id is None:
    raise RuntimeError("No active MLflow run_id found")
```

---

## 3) `src/infra/baseline_store.py`

### Errors
Reported many times:
- `Missing type arguments for generic type "ndarray"  [type-arg]`
- `Returning Any from function declared to return "ndarray[Any, Any] | None"  [no-any-return]`

### Why it happens
With strict mypy, `np.ndarray` must be parameterized (or replaced with `numpy.typing.NDArray[...]`). Also, `np.load()` can be typed as returning `Any` unless properly annotated/cast.

### Fix
Use `numpy.typing.NDArray` and specify element dtypes, e.g.:

```python
from numpy.typing import NDArray
from typing import Any
import numpy as np

EmbeddingMatrix = NDArray[np.floating[Any]]

def load_psi_baseline(self) -> EmbeddingMatrix | None:
    ...
```

Also, cast `np.load(...)` return values if needed:

```python
return cast(EmbeddingMatrix, np.load(self._psi_path))
```

---

## 4) `rag_pipeline/vectorstore.py`

### Errors
- `rag_pipeline/vectorstore.py:42: error: Name "embedding_functions.SentenceTransformerEmbeddingFunction" is not defined  [name-defined]`
- `rag_pipeline/vectorstore.py:45: error: Module has no attribute "SentenceTransformerEmbeddingFunction"  [attr-defined]`
- `rag_pipeline/vectorstore.py:105: error: Redundant cast to "Collection"  [redundant-cast]`

### Why it happens
Chroma’s `embedding_functions` module can differ by version and may not export the type symbols in a way mypy understands.

### Fix
- Avoid annotating the return type as `embedding_functions.SentenceTransformerEmbeddingFunction`.
  Instead annotate as a `Callable[[list[str]], list[list[float]]]` (or a Protocol), which matches how it’s used.
- Remove the redundant cast if mypy already infers `Collection`.

---

## 5) `mlops/explanation_monitor.py`

### Errors
Multiple occurrences:
- `Missing type arguments for generic type "ndarray"  [type-arg]`

### Fix
Replace `np.ndarray` with `NDArray[...]` or parameterize arrays.

---

## 6) `mlops/drift_detector.py`

### Errors
Multiple occurrences:
- `Missing type arguments for generic type "ndarray"  [type-arg]`

### Fix
Use `numpy.typing.NDArray[...]` everywhere drift detector expects arrays.

---

## 7) `rag_pipeline/ingest.py`

### Error
- `rag_pipeline/ingest.py:159: error: Argument "metadatas" to "upsert" of "Collection" has incompatible type "list[dict[Any, Any]]"; expected "Mapping[str, str | int | float | bool] | list[Mapping[str, str | int | float | bool]] | None"  [arg-type]`

### Why it happens
Chroma metadata values are restricted to primitive JSON-like types: `str|int|float|bool`. LangChain `Document.metadata` can contain arbitrary values (`Any`).

### Fix
Sanitize metadata before upserting:
- Convert non-primitive values to strings
- Ensure keys are `str`

Example:

```python
AllowedMeta = str | int | float | bool

def sanitize(meta: dict[str, Any]) -> dict[str, AllowedMeta]:
    ...

metadatas = [sanitize(c.metadata) for c in chunks]
```

---

## 8) `rag_pipeline/retriever.py`

### Errors
- `rag_pipeline/retriever.py:304: error: List item 0 has incompatible type "str"; expected "IncludeEnum"  [list-item]` (also items 1,2)
- `rag_pipeline/retriever.py:309: error: Value of type "list[list[str]] | None" is not indexable  [index]`
- `rag_pipeline/retriever.py:310: error: Value of type "list[list[Mapping[str, str | int | float | bool]]] | None" is not indexable  [index]`
- `rag_pipeline/retriever.py:357: error: Missing type arguments for generic type "dict"  [type-arg]`
- `rag_pipeline/retriever.py:358: error: Missing type arguments for generic type "dict"  [type-arg]`
- `rag_pipeline/retriever.py:505: error: List item 0 has incompatible type "str"; expected "IncludeEnum"  [list-item]` (also items 1,2)
- `rag_pipeline/retriever.py:509: error: Value of type "list[list[str]] | None" is not indexable  [index]`
- `rag_pipeline/retriever.py:510: error: Value of type "list[list[Mapping[str, str | int | float | bool]]] | None" is not indexable  [index]`
- `rag_pipeline/retriever.py:511: error: Value of type "list[list[float]] | None" is not indexable  [index]`

### Why it happens
- Chroma’s `include=` parameter is typed as an `IncludeEnum` list in newer stubs.
- Chroma query returns optional lists (`None` possible), so indexing needs guards.
- Bare `dict` types violate strict mypy.

### Fix
- Cast include list: `include=cast(Any, ["documents", "metadatas", "distances"])` or import and use `IncludeEnum`.
- Guard `results.get("documents")` with `or [[]]` before indexing.
- Type `scores` and `doc_map` as `dict[str, float]` / `dict[str, Document]`.

---

## 9) `rag_pipeline/chain.py`

### Errors
- `rag_pipeline/chain.py:79: error: Missing type arguments for generic type "dict"  [type-arg]`
- `rag_pipeline/chain.py:119: error: Missing type arguments for generic type "dict"  [type-arg]`
- `rag_pipeline/chain.py:121: error: Missing type arguments for generic type "dict"  [type-arg]`
- `rag_pipeline/chain.py:170: error: Item "list[str | dict[Any, Any]]" of "str | list[str | dict[Any, Any]]" has no attribute "strip"  [union-attr]`
- `rag_pipeline/chain.py:211: error: Missing type arguments for generic type "dict"  [type-arg]`
- `rag_pipeline/chain.py:232: error: Function is missing a return type annotation  [no-untyped-def]`
- `rag_pipeline/chain.py:269: error: Missing type arguments for generic type "dict"  [type-arg]`
- `rag_pipeline/chain.py:270: error: Missing type arguments for generic type "dict"  [type-arg]`
- `rag_pipeline/chain.py:313: error: Missing type arguments for generic type "list"  [type-arg]`

### Fix
- Replace `dict` / `list` with typed versions, e.g. `dict[str, Any]`, `list[str]`, etc.
- Narrow unions before calling `.strip()`:
  - if value may be list, branch on `isinstance(value, str)`.
- Add explicit return type annotation for the missing function.

---

## 10) `evaluation/explainability.py`

### Errors
Many occurrences:
- `Missing type arguments for generic type "ndarray"  [type-arg]`
- `Missing type arguments for generic type "dict"  [type-arg]`

### Fix
- Use `NDArray[...]` for all numpy arrays.
- Use typed dicts: `dict[str, Any]` or more specific.

---

## 11) `serving/api.py`

### Errors
Many occurrences:
- `Missing type arguments for generic type "dict"  [type-arg]`
- `Missing type arguments for generic type "list"  [type-arg]`

### Fix
Replace untyped containers with parameterized ones:
- `dict[str, Any]`
- `list[SomeType]`

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

So `rag_pipeline/` is still being type-checked indirectly when imported by checked modules; and `evaluation/` also gets checked by ruff but not by mypy step unless imported (however the job log clearly shows mypy is finding errors in `rag_pipeline/` and `evaluation/`, meaning those modules are being checked in this run context).

---

## Next steps (recommended order)

1. Fix `serving/dashboard.py` decorator ignore (`untyped-decorator`).
2. Fix `rag_pipeline/vectorstore.py` embedding function type + redundant cast.
3. Fix Chroma `include` typing + optional indexing in `rag_pipeline/retriever.py`.
4. Sanitize metadata types in `rag_pipeline/ingest.py`.
5. Replace `np.ndarray` with `NDArray[...]` in baseline/drift/explainability modules.
6. Parameterize all bare `dict`/`list` in API/chain modules.

Once these are applied, rerun CI to see remaining errors (should drop dramatically).
