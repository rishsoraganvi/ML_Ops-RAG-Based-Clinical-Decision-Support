# RAGOps — Branch Strategy

## Branch Map

```
main ─────────────────────────────────────────────────────── (protected, releases only)
  │
  └─ develop ───────────────────────────────────────────────── (integration, default PR target)
       │
       ├─ infra/psi-drift-detector        (Member 1)
       ├─ infra/xai-consistency-monitor   (Member 1)
       ├─ rag/retrieval-chain             (Member 2)
       ├─ eval/ragas-benchmark            (Member 3)
       ├─ data/pubmed-ingest              (Member 4)
       └─ fix/chroma-healthcheck          (anyone — hotfix off main, PR to both)
```

## Rules at a Glance

| Rule | Detail |
|------|--------|
| Branch source | Always off `develop`. Never off `main` (except `fix/`) |
| PR target | Always `develop`. Only release PRs target `main` |
| Direct push to `main` | Blocked — GitHub branch protection enforced |
| Direct push to `develop` | Blocked — PRs only |
| Self-merge | Not allowed — minimum 1 approving review required |
| Stale branches | Delete within 48 h of merge |

## Branch Prefixes

| Prefix | Who | When |
|--------|-----|------|
| `infra/` | Member 1 | Docker, MLflow, CI/CD, monitoring |
| `rag/` | Member 2 | RAG pipeline, LangChain |
| `eval/` | Member 3 | RAGAS, explainability |
| `data/` | Member 4 | Ingestion, embeddings |
| `fix/` | Anyone | Hotfix off `main` |
| `docs/` | Anyone | Documentation-only changes |
| `chore/` | Anyone | Tooling, dependency bumps |

## Naming Convention

```
<prefix>/<short-description-in-kebab-case>

# Good
infra/psi-drift-detector
eval/ragas-faithfulness-gate
data/pubmed-batch-ingest
fix/ollama-healthcheck-timeout

# Bad
member1-work
new-feature
patch1
```

## Commit Convention

```
<type>: description in sentence case

feat:     new capability
fix:      bug fix
test:     test additions or changes
docs:     documentation only
refactor: restructure without behaviour change
chore:    tooling, deps, config
```

## Release Flow

When a milestone (e.g., all 6 components complete) is ready:

```bash
# 1. Ensure develop is green on CI
# 2. Open PR: develop → main
# 3. All four members review and approve
# 4. Squash merge with message: "release: v<N> — <milestone description>"
# 5. Tag the merge commit
git tag -a v1.0.0 -m "RAGOps v1.0.0 — all components complete"
git push origin v1.0.0
```

## GitHub Branch Protection Settings

Apply these to both `main` and `develop` via **Settings → Branches → Add rule**:

**For `main`:**
- [x] Require a pull request before merging
- [x] Require approvals: **2**
- [x] Require status checks to pass: `lint-and-type-check`, `unit-tests`, `docker-build`
- [x] Require branches to be up to date before merging
- [x] Do not allow bypassing the above settings

**For `develop`:**
- [x] Require a pull request before merging
- [x] Require approvals: **1**
- [x] Require status checks to pass: `lint-and-type-check`, `unit-tests`, `docker-build`
- [x] Require branches to be up to date before merging

## Day-to-Day Workflow

```bash
# Start new work
git checkout develop
git pull origin develop
git checkout -b infra/my-feature

# Work, commit often
git add -p
git commit -m "feat: add PSI histogram computation"

# Keep up to date with develop
git fetch origin
git rebase origin/develop

# Push and open PR
git push origin infra/my-feature
# → open PR on GitHub targeting develop
# → fill in PR template
# → wait for CI green, then request review
```
