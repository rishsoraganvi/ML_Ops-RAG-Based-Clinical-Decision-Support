# Contributing to RAGOps

Welcome to the RAGOps project. This guide covers everything you need to get your environment running, understand the branch strategy, and get a PR merged.

Read this before writing your first line of code.

---

## Team & Ownership

| Member | Components | Branch prefix |
|--------|-----------|---------------|
| Member 1 (Infra) | 1 Docker · 2 MLflow · 3 CI/CD · 4 PSI drift · 5 KB refresh · 6 XAI monitoring | `infra/` |
| Member 2 (RAG) | RAG pipeline · LangChain chain | `rag/` |
| Member 3 (Eval) | RAGAS runner · Explainability | `eval/` |
| Member 4 (Data) | Ingestion · Embeddings | `data/` |

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/<org>/ragops.git
cd ragops

# 2. Create your .env
cp .env.example .env
# Edit .env — ask Member 1 if you need values

# 3. Start the stack
docker compose up -d

# 4. Wait for all services to be healthy
python scripts/healthcheck.py --timeout 300

# 5. Verify
open http://localhost:5000   # MLflow UI
open http://localhost:8080/docs  # FastAPI Swagger
```

**NVIDIA GPU required for Ollama.** Confirm the toolkit is installed:
```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

---

## Branch Strategy

### Permanent branches

| Branch | Purpose | Who merges here |
|--------|---------|----------------|
| `main` | Always deployable. Protected. | PRs only — never push direct |
| `develop` | Integration branch. All feature branches merge here first | PRs from feature branches |

### Feature branches

Branch off `develop`. Never branch off `main`.

```
<prefix>/<short-description>

# Examples
infra/psi-drift-detector
rag/retrieval-chain-v2
eval/ragas-benchmark-questions
data/pubmed-ingest-pipeline
```

Prefixes must match the table above. PRs from a wrong prefix will be asked to rename.

### Lifecycle

```
develop
  └── infra/psi-drift-detector    ← branch off develop
        ↓  (work, commits)
      Pull Request → develop       ← PR reviewed + CI green
        ↓
      develop (merged)
        ↓  (when milestone complete)
      Pull Request → main          ← release PR, all owners review
```

### Hotfixes

Branch off `main`, prefix `fix/`, PR back to both `main` and `develop`.

```bash
git checkout main
git checkout -b fix/chroma-healthcheck-timeout
# fix it
git push origin fix/chroma-healthcheck-timeout
# open PRs to main AND develop
```

---

## Interface Contracts

These are frozen. Do not change a function signature without opening an issue and getting sign-off from the affected member.

| Exposed by | Import path | Signature |
|-----------|-------------|-----------|
| Member 2 | `rag_pipeline.chain` | `query(question: str) -> dict` |
| Member 3 | `evaluation.ragas_runner` | `run_eval(qa_pairs: list) -> dict` |
| Member 3 | `evaluation.explainability` | `explain(q: str, docs: list) -> dict` |
| Member 4 | `data.ingest` | `get_embeddings() -> np.ndarray` |

Return shapes are documented in `PROGRESS.md`.

---

## Code Conventions

These are enforced by CI. A PR that violates them will not pass.

### Must-haves on every function and class
```python
def compute_psi(
    current: np.ndarray,
    baseline: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Population Stability Index between two embedding distributions.

    Parameters
    ----------
    current : np.ndarray
        Shape (n_docs, 384) — current embedding matrix.
    baseline : np.ndarray
        Shape (n_docs, 384) — reference embedding matrix.
    n_bins : int
        Number of histogram bins (default 10).

    Returns
    -------
    float
        PSI score. > 0.25 triggers KB refresh.  # PAPER CONTRIBUTION
    """
```

- **Type hints** on every parameter and return value — no bare `def f(x):`
- **Docstrings** on every public function, class, and module
- **`logging` only** — `print()` anywhere in `src/` will be caught by CI lint
- **`# PAPER CONTRIBUTION`** on any line implementing PSI drift detection
- **`# XAI CONTRIBUTION`** on any line implementing SHAP/explanation monitoring

### Imports
```python
# Standard library first
import logging
from dataclasses import dataclass

# Third-party second
import numpy as np
import mlflow

# Internal last
from src.config.settings import settings
from src.infra.mlflow_tracker import RAGOpsTracker
```

### Naming
- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Test files: `test_<module_name>.py` in a `tests/` subdirectory next to the module

### No secrets in code
Never hardcode URLs, credentials, thresholds, or model names. Everything goes in `.env` and is read through `src/config/settings.py`.

---

## Running Tests Locally

```bash
# Install deps (no Docker needed for unit tests)
pip install -r docker/fastapi/requirements.txt

# Run all tests
pytest src/ -v

# Run only your module's tests
pytest src/infra/tests/ -v

# With coverage
pytest src/ --cov=src --cov-report=term-missing
```

Tests must pass locally before you open a PR. CI runs the exact same command.

---

## Opening a Pull Request

1. Push your branch: `git push origin <your-branch>`
2. Open a PR against **`develop`** (not `main`)
3. Fill in the PR template completely — incomplete templates will be sent back
4. Assign at least one reviewer from a different component area
5. Wait for CI to go green before requesting review
6. Do not merge your own PR

### PR title format
```
[prefix] short description in sentence case

[infra] Add PSI embedding drift detector
[eval]  Fix RAGAS context recall calculation
[data]  Add PubMed PDF ingestion pipeline
```

---

## What CI Checks

Every PR to `develop` or `main` runs:

| Check | Tool | Fails on |
|-------|------|----------|
| Lint | `ruff` | Any error or warning |
| Type check | `mypy` | Any type error |
| No `print()` | `grep` | Any `print(` in `src/` |
| Unit tests | `pytest` | Any failure or < 80% coverage |
| Docker build | `docker build` | Build failure |

All checks must be green. No exceptions.

---

## Commit Messages

```
<type>: short description (≤72 chars)

Optional longer body explaining why, not what.
Reference issues: closes #12
```

Types: `feat` · `fix` · `test` · `docs` · `refactor` · `chore`

```bash
# Good
feat: add PSI histogram baseline persistence
fix: handle zero-count bins in PSI computation
test: add edge cases for XAI cosine similarity

# Bad
update stuff
WIP
fixed it
```
