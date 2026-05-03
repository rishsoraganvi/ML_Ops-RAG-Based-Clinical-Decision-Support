# =============================================================================
# RAGOps — Developer Makefile
#
# Windows note: the targets below use POSIX shell idioms (rm -rf, pipes, etc.).
# On Windows run them from WSL or Git Bash. CI uses Ubuntu runners — no issue.
#
# Common workflows:
#   make setup          install Python deps + pull phi3:mini (CI judge)
#   make up             docker compose up -d
#   make healthcheck    poll /health until green
#   make test-unit      pytest (mock-based, no docker required)
#   make test-e2e       pytest integration tier (needs docker stack)
#   make eval-ci        RAGAS 3-question CI smoke eval (phi3:mini judge)
#   make eval-full      full 9-config ablation sweep (hours, llama3:8b judge)
#   make eval-fast      9-config sweep on stratified-20 subset, phi3:mini judge
#   make eval-retrieval judge-free retrieval ablation (recall/precision/nDCG/MRR)
#   make eval-latency   per-config end-to-end latency profile (P50/P95/P99)
#   make drift-check    POST /drift/check against the running API
#   make xai-check      POST /xai/check against the running API
#
# Note on Ollama models:
#   The docker-compose stack runs Ollama in a container (`ragops_ollama`).
#   Models pulled with the host-side `ollama` CLI (e.g. `ollama pull llama3:8b`)
#   live in the host daemon, NOT the container — the eval scripts hit the
#   container at localhost:11434 and will 404 if the model is missing there.
#   Pull into the container: `docker exec ragops_ollama ollama pull <model>`.
#
# Judge model selection:
#   - CI smoke (eval-ci):      phi3:mini       (fast, JSON-friendly)
#   - Ablation (eval-full):    llama3:8b       (more JSON-stable on faithfulness
#                                               + context_recall, which decompose
#                                               answers into atomic statements)
#   - Production /query:       llama3.2:3b     (default OLLAMA_MODEL in .env)
#   Override per-target by exporting OLLAMA_MODEL before invocation.
# =============================================================================

.PHONY: help setup up down logs healthcheck lint format typecheck \
        test-unit test-e2e test eval-ci eval-full eval-fast eval-retrieval \
        eval-latency baseline baseline-host drift-check xai-check clean

PYTHON           ?= python
PIP              ?= pip
API              ?= http://localhost:8080
# Judge model used by `make eval-full` (ablation sweep). Override on the CLI
# (e.g. `make eval-full ABLATION_JUDGE=llama3.2:3b`) if you want a smaller
# judge — at the cost of more JSON-parse failures on faithfulness/context_recall.
ABLATION_JUDGE   ?= llama3:8b

help:
	@echo "RAGOps Makefile targets:"
	@echo "  setup         install deps + pull phi3:mini + llama3:8b + nomic-embed-text"
	@echo "  up / down     docker compose up/down"
	@echo "  logs          tail FastAPI logs"
	@echo "  healthcheck   poll /health via scripts/healthcheck.py"
	@echo "  lint          ruff check"
	@echo "  format        ruff format"
	@echo "  typecheck     mypy (strict)"
	@echo "  test-unit     pytest (mock-based, no docker required)"
	@echo "  test-e2e      pytest -m integration (needs docker stack)"
	@echo "  test          test-unit + test-e2e"
	@echo "  eval-ci       RAGAS 3-question CI smoke eval (phi3:mini judge)"
	@echo "  eval-full     full 9-config ablation sweep (judge=$(ABLATION_JUDGE))"
	@echo "  eval-fast     9-config sweep on stratified-20 subset (phi3:mini judge)"
	@echo "  eval-retrieval judge-free retrieval ablation (recall@k, precision@k, nDCG, MRR)"
	@echo "  eval-latency  per-config end-to-end latency profile (P50/P95/P99)"
	@echo "  baseline      scripts/run_baseline_eval.py — capture week-2 baseline"
	@echo "  drift-check   POST /drift/check"
	@echo "  xai-check     POST /xai/check"
	@echo "  clean         remove __pycache__, .pytest_cache, coverage artifacts"

# ── Environment setup ────────────────────────────────────────────────────────
setup:
	$(PIP) install --upgrade pip
	$(PIP) install -r docker/fastapi/requirements.txt
	$(PIP) install ruff mypy pytest pytest-asyncio pytest-cov
	@echo "Pulling judge + embedding models via Ollama CLI (ignore if Ollama container handles it)…"
	-ollama pull phi3:mini
	-ollama pull $(ABLATION_JUDGE)
	-ollama pull nomic-embed-text

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
	@echo "Running ablation sweep with judge=$(ABLATION_JUDGE) (override via ABLATION_JUDGE=…)"
	OLLAMA_MODEL=$(ABLATION_JUDGE) $(PYTHON) evaluation/ablations/run_ablations.py \
		--qa-file evaluation/benchmarks/qa_pairs.json \
		--workers 1

eval-fast:
	@echo "Fast 9-config sweep on stratified-20 subset (judge=phi3:mini)"
	OLLAMA_MODEL=phi3:mini $(PYTHON) evaluation/ablations/run_ablations.py \
		--qa-file evaluation/benchmarks/qa_pairs_stratified20.json \
		--workers 1 \
		--output-dir ablation_outputs/phi3_stratified20

eval-retrieval:
	$(PYTHON) evaluation/retrieval_only.py \
		--qa-file evaluation/benchmarks/qa_pairs.json

eval-latency:
	$(PYTHON) evaluation/latency_profile.py \
		--qa-file evaluation/benchmarks/qa_pairs_stratified20.json

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
