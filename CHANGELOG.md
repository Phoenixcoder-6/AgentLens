# AgentLens Changelog

All notable changes to the AgentLens project are documented in this file.  
The project adheres to [Semantic Versioning](https://semver.org/).

---

## [2.0.0] - 2026-11-15 (Day 66 Release — Enterprise Release Complete)

Major milestone release delivering the complete 20-day AgentLens v2.0 development roadmap (Phases A through D, Days 47–66). Transforms AgentLens from a deterministic failure detector into an enterprise causal attribution, blast radius accounting, and automated prompt remediation platform.

### Added
- **Phase D: Causal Attribution & Prompt Auto-Diff (Days 62–66):**
  - **Causal Fault Tree Engine (`analyzers/causal/fault_tree.py`):** Models failure propagation across multi-agent execution graphs, categorizing every node into `ROOT_CAUSE_ORIGIN` (first invariant breach), `CASCADING_FAILURE` (downstream tainted execution), or `PASSTHROUGH_ESCAPED` (verifier/guardrail rubber-stamping).
  - **Blast Radius & Financial Waste Calculator (`analyzers/causal/blast_radius.py`):** Computes computational and financial waste caused by root failures, tracking Tainted Steps Ratio, Wasted Tokens, Wasted USD ($0.0051 on benchmark), and Wasted Latency.
  - **Automated Prompt Auto-Diff Engine (`analyzers/remediation/prompt_fixer.py`):** Generates targeted prompt patches (`SOURCE_PRESERVATION`, `ENTITY_FIDELITY`, `CONTRADICTION_GUARD`, `SCHEMA_CONFORMANCE`, `TOOL_DEFENSE`) with confidence ratings, regression risk scoring, and unified diff recommendations.
  - **Interactive Visual DAG Dashboard (`dashboard/dag_renderer.py`):** Renders interactive DAGs and Mermaid flowcharts in the NiceGUI dashboard, color-coding nodes by causal classification (🔴 Root Cause, 🟠 Cascade, 🟡 Passthrough, 🟢 Healthy) with integrated Blast Radius metrics and a one-click prompt fix modal.
  - **Extended 30-Trace Regression Suite (`scripts/run_v2_regression.py`):** 30-trace enterprise corpus combining 20 frozen baseline traces, 5 Phase B multi-agent traces, and 5 enterprise demo traces; achieved **86.7%** exact attribution accuracy (26/30, exceeding $\ge 85\%$ threshold) and **100.0%** binary failure detection (30/30).
  - **Release Gate Test Suite (`tests/test_v2_regression_gate.py`):** 8 comprehensive release-gate tests enforcing regression metric persistence, report schemas, and causal/remediation pipeline integration with **94%** test coverage on v2 core modules.

- **Phase C: Developer Experience & Framework Integrations (Days 57–61):**
  - **Drop-in Framework Auto-Tracers (`agentlens/tracers/`):** Zero-code instrumentation hooks for LangGraph (`AgentLensLangGraphTracer`), CrewAI (`AgentLensCrewAITracer`), AutoGen (`AgentLensAutoGenTracer`), and LangChain callbacks.
  - **Public Python SDK (`agentlens/`):** Fluent top-level imports (`agentlens.analyze()`, `agentlens.causal_graph()`, `agentlens.blast_radius()`, `agentlens.remediate()`).
  - **Unified CLI:** Enhanced `agentlens` CLI supporting `agentlens analyze`, `agentlens causal`, `agentlens blast-radius`, `agentlens patch`, and `agentlens diff`.
  - **Synthetic Fault Injection Engine:** Configurable synthetic generator injecting tool timeouts, context truncations, hallucinated assertions, and verifier bypasses.

- **Phase B: Universal DAGs & Semantic NLI (Days 52–56):**
  - **Universal DAG Engine (`analyzers/detection/topology.py`):** Dynamic multi-agent topology validator supporting arbitrary non-linear DAGs, fan-out/fan-in parallel branches, and iterative reflection loops.
  - **Cross-Agent Semantic Contradiction Engine:** NLI-based premise validation detecting subtle semantic drift between upstream agent findings and downstream agent synthesis.
  - **Granular Token & Cost Allocation:** Per-agent step-level token accounting with real-time budget threshold monitoring and Slack alerting.

- **Phase A: Stabilization & High-Throughput Telemetry (Days 47–51):**
  - **ContextVar Concurrency Isolation (`capture/context.py`):** Zero trace cross-talk across 50+ concurrent async tasks and multi-threaded agent executions.
  - **Non-Blocking Telemetry Ring Buffer (`capture/queue.py`):** Bounded ring buffer with background daemon flushing (`TraceIngestionWorker`), reducing `@trace_step` overhead to $\le 0.5\text{ms}$.
  - **SQLite WAL & Connection Pooling (`storage/db.py`):** Configured `PRAGMA journal_mode=WAL`, `busy_timeout=10000`, and thread-safe connection pool eliminating locked database errors.
  - **Pluggable Storage Adapters (`storage/base_adapter.py`):** Polymorphic storage layer supporting SQLite and PostgreSQL backend switching via `AGENTLENS_DB_URL`.

### Changed
- Refactored `analyzers/detection/workflow_validator.py` to resolve required agents and handoff edges from the configured pipeline topology, supporting custom non-linear topologies.
- Expanded public API exports in `agentlens/__init__.py` to include `CausalGraphEngine`, `CausalGraph`, `CausalNode`, `BlastRadiusCalculator`, `BlastRadiusReport`, `PromptFixer`, `RemediationPatch`, and `DAGRenderer`.
- Updated test suite to 210 passing tests across 11 test modules.

---

## [1.0.0] - 2026-10-10 (Day 44 / Day 46 Release)

### Added
- **Production Packaging & CLI:**
  - Editable package installation via `pyproject.toml` (`pip install -e .`).
  - Added CLI entry points: `agentlens` (CLI runner), `agentlens-dashboard` (UI), `agentlens-seed` (demo seeder).
  - PEP 561 `py.typed` markers across all 10 project modules.
- **Zero-Setup Sample Data:**
  - `sample_data/generate_demo_traces.py` generating 6 deterministic sample traces (normal PASS, grounded P1, heuristic P2/P3, and diff pairs).
  - Automatic SQLite seeding on first dashboard startup if the database is empty (`AGENTLENS_AUTO_SEED=1`).
- **Production Health Check & Docker:**
  - `GET /health` endpoint returning `{"status": "ok", "db": "connected", "llm": "reachable", "uptime_seconds": ...}`.
  - Production `Dockerfile` (`python:3.11-slim`, `curl` healthcheck, non-root ready).
  - Production `docker-compose.yml` with environment variable injection (`GROQ_API_KEY`, `DB_PATH`, `SLACK_WEBHOOK_URL`) and persistent `./data:/app/data` volume.
- **Documentation Suite (Day 46):**
  - Full `README.md`, `ARCHITECTURE.md`, `RULES.md`, `LIMITATIONS.md`, `CONTRIBUTING.md`, and `CHANGELOG.md`.

---

## [0.8.0] - 2026-10-09 (Week 8 Hardening — Days 39–43)

### Added
- **Day 39 (Fail-Safe Telemetry & PII):**
  - Fail-safe capture wrappers in `@trace_step` and `CaptureSession` preventing pipeline interruptions.
  - Configurable regex-based `PIIScrubber` redacting emails, API keys, bearer tokens, and IPv4 addresses.
  - Trace retention policy with automated TTL cleanup.
- **Day 40 (Configuration Sweep & Cost Controls):**
  - Dynamic budget tracking (`cost_usd` vs. `budget_alert_usd`) with red banner warnings in dashboard.
  - Global `config.yaml` centralization.
- **Day 40a (Multi-Agent Generalization):**
  - Topology engine (`config/topology.py`) resolving roles (`information_gatherer`, `synthesizer`, `quality_checker`) and `receives_from` handoffs dynamically.
- **Day 41 (Analyzer Interface Validation & Typing):**
  - Unified `Analyzer` abstract protocol across all 11 detection engines (`.analyze(trace)` & `.run(trace)`).
  - 100% clean Mypy type-checking across all 46 source files.
- **Day 42 (Unit Testing & 75% Coverage Gate):**
  - 6-pillar unit test suite (`tests/test_day42_unit_suite.py`) covering normalizer, diff alignment, Alembic migrations, 24-permutation Arbiter determinism, LLM caching, and alerts.
  - Enforced 75% coverage gate (`pytest --cov-fail-under=75`, achieving 93% actual coverage).
- **Day 43 (Automated CI Regression Gate):**
  - Automated 20-trace regression runner (`scripts/run_day43_regression.py`) wired into GitHub Actions CI (`.github/workflows/ci.yml`).
  - Guaranteed 0 regressions vs. Day 35 baseline (80.0% exact attribution, 100.0% binary match).

---

## [0.7.0] - 2026-10-08 (Week 7 — Validation & Error Injection — Days 34–38)

### Added
- 20 frozen labeled benchmark traces (`sample_data/labels.json`).
- Day 34 labeled validation pipeline (`scripts/run_day34_validation.py`).
- Day 35 human vs. AgentLens accuracy evaluator (`scripts/evaluate_day35_accuracy.py`).
- Day 36 error injection test suite (`tests/test_error_injection.py`) testing tool timeouts, writer hallucinations, skipped nodes, and ground-truth contradictions.
- Day 37 prompt ablation benchmarks.
- Day 38 rule refinement and Arbiter tie-break optimization.

---

## [0.6.0] - 2026-10-06 (Week 6 — Dashboard v2 & REST API — Days 29–33)

### Added
- FastAPI REST service (`api/router.py`) mounted at `/api` with endpoints for runs, verdicts, analysis, metrics, and health.
- NiceGUI interactive Dashboard v2 with 6 views: Run Explorer, Run Timeline, Evidence View, Diff View, Metrics View, and Rule Explorer.
- Real-time Slack alerting (`analyzers/alerter.py`) firing on P1/P2 failures.

---

## [0.5.0] - 2026-10-02 (Week 5 — Trace Diffing & Statistical Anomalies — Days 23–28)

### Added
- Graph-aligned trace comparison (`diff_engine/aligner.py`) matching steps across diverged executions.
- Local semantic similarity scoring (`diff_engine/similarity.py`) with `all-MiniLM-L6-v2` embeddings and Jaccard token overlap fallback.
- Statistical anomaly detection (`analyzers/detection/statistical_detector.py`) computing Z-scores for latency and token consumption.

---

## [0.4.0] - 2026-09-24 (Week 4 — Rule Engine & Arbiter — Days 17–22)

### Added
- 5-tier Arbiter priority resolver (`analyzers/arbiter.py`) with deterministic tie-breaking.
- Ground Truth Validator (`analyzers/detection/ground_truth.py`) for P1 assertions.
- Consistency Validator (`analyzers/detection/consistency_validator.py`) detecting verifier passthrough.
- Workflow Validator (`analyzers/detection/workflow_validator.py`) detecting skipped steps.

---

## [0.2.0] - 2026-09-08 (Weeks 2–3 — Capture & SQLite Storage — Days 5–16)

### Added
- `@trace_step` decorator and `HandoffCapture` recording 3-state input/filtered/output snapshots.
- SQLite schema with foreign key integrity and performance indexes.
- Structured LLM evidence extractor with JSON schema validation.

---

## [0.1.0] - 2026-08-01 (Week 1 — Reference Pipeline)

### Added
- 3-agent research report pipeline (`researcher` → `writer` → `verifier`) built on LangGraph.
- Initial schema models (`RunTrace`, `AgentStep`).
