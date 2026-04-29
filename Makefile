# =============================================================================
# RAGOps — Developer Makefile
#
# Windows note: the targets below use POSIX shell idioms (rm -rf, pipes, etc.).
# On Windows run them from WSL or Git Bash. CI uses Ubuntu runners — no issue.
#
# Common workflows:
#   make setup          install Python deps + pull llama3.2:1b
#   make up             docker compose up -d
#   make healthcheck    poll /health until green
#   make test-unit      pytest (mock-based, no docker required)
#   make test-e2e       pytest integration tier (needs docker stack)
#   make eval-ci        RAGAS 5-question CI smoke eval
#   make eval-full      full 9-config ablation sweep (hours)
#   make drift-check    POST /drift/check against the running API
#   make xai-check      POST /xai/check against the running API
# =============================================================================

.PHONY: help setup up down logs healthcheck lint format typecheck \
        test-unit test-e2e test eval-ci eval-full baseline baseline-host \
        drift-check xai-check clean

PYTHON ?= python
PIP    ?= pip
API    ?= http://localhost:8080

help:
	@echo "RAGOps Makefile targets:"
	@echo "  setup         install deps + pull llama3.2:1b"
	@echo "  up / down     docker compose up/down"
	@echo "  logs          tail FastAPI logs"
	@echo "  healthcheck   poll /health via scripts/healthcheck.py"
	@echo "  lint          ruff check"
	@echo "  format        ruff format"
	@echo "  typecheck     mypy (strict)"
	@echo "  test-unit     pytest (mock-based, no docker required)"
	@echo "  test-e2e      pytest -m integration (needs docker stack)"
	@echo "  test          test-unit + test-e2e"
	@echo "  eval-ci       RAGAS 5-question CI smoke eval"
	@echo "  eval-full     full 9-config ablation sweep (hours)"
	@echo "  baseline      scripts/run_baseline_eval.py — capture week-2 baseline"
	@echo "  drift-check   POST /drift/check"
	@echo "  xai-check     POST /xai/check"
	@echo "  clean         remove __pycache__, .pytest_cache, coverage artifacts"

# ── Environment setup ────────────────────────────────────────────────────────
setup:
	$(PIP) install --upgrade pip
	$(PIP) install -r docker/fastapi/requirements.txt
	$(PIP) install ruff mypy pytest pytest-asyncio pytest-cov
	@echo "Pulling llama3.2:1b via Ollama CLI (ignore if Ollama container handles it)…"
	-ollama pull llama3.2:1b

# ── Docker lifecycle ─────────────────────────────────────────────────────────
up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f fastapi

healthcheck:
	$(PYTHON) scripts/healthcheck.py --timeout 300 --poll-interval 5

# ── Static checks ────────────────────────────────────────────────────────────
lint:
	ruff check src/ mlops/ serving/ evaluation/ rag_pipeline/ tests/

format:
	ruff format src/ mlops/ serving/ evaluation/ rag_pipeline/ tests/

typecheck:
	mypy src/ mlops/ serving/ --ignore-missing-imports --strict --exclude src/infra/test

# ── Tests ────────────────────────────────────────────────────────────────────
test-unit:
	pytest tests/ src/ mlops/ -m "not integration" -v \
		--cov=src --cov=mlops --cov-report=term-missing --cov-fail-under=80

test-e2e:
	RAGOPS_E2E=1 pytest tests/test_e2e.py -m integration -v

test: test-unit test-e2e

# ── RAG evaluation ───────────────────────────────────────────────────────────
eval-ci:
	$(PYTHON) -c "from evaluation.ragas_runner import run_ci_eval; import json; print(json.dumps(run_ci_eval(), indent=2))"

eval-full:
	$(PYTHON) evaluation/ablations/run_ablations.py \
		--qa-file evaluation/benchmarks/qa_pairs.json \
		--workers 1

baseline:
	docker compose exec fastapi python scripts/run_baseline_eval.py

baseline-host:
	$(PYTHON) scripts/run_baseline_eval.py

# ── Live monitoring probes ───────────────────────────────────────────────────
drift-check:
	curl -sS -X POST $(API)/drift/check | $(PYTHON) -m json.tool

xai-check:
	curl -sS -X POST $(API)/xai/check \
		-H "Content-Type: application/json" \
		-d '{"current_vectors": null}' | $(PYTHON) -m json.tool

# ── Housekeeping ─────────────────────────────────────────────────────────────
clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
