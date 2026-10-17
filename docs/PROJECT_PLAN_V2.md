# AgentLens v2.0 — Day-Wise Project Plan (Days 47 to 66)

**Cadence:** Exactly **4 concrete, verifiable tasks per day** across 20 days (Days 47 to 66).  
**Baseline:** Built on top of verified **v1.0.0** (`16/20 = 80%` exact attribution, `20/20 = 100%` binary detection, `93%` test coverage).

---

## Roadmap Overview

```
WEEK 10: Phase A — Stabilization & Async Telemetry ─────── Days 47–51 (20 tasks)
WEEK 11: Phase B — Universal DAG & Semantic NLI ────────── Days 52–56 (20 tasks)
WEEK 12: Phase C — Developer Experience & Auto-Tracers ─── Days 57–61 (20 tasks)
WEEK 13: Phase D — Causal Attribution & Remediation ────── Days 62–66 (20 tasks)
Total: 80 Actionable Engineering Tasks
```

---

## WEEK 10: Phase A — Stabilization & High-Throughput Telemetry

### Day 47 — `ContextVar` Concurrency & Thread Isolation
- [x] **Task 47.1:** Create `capture/context.py` implementing `current_trace_var: ContextVar[RunTrace | None]` and `pending_tokens_var: ContextVar[tuple[int, int] | None]` with `copy_context()` propagation helpers.
- [x] **Task 47.2:** Refactor `capture/session.py` to replace class-level `_current_trace` and `_pending_tokens` with context variables, preserving the fail-safe try-except boundary.
- [x] **Task 47.3:** Write `tests/test_concurrency_isolation.py` launching 50 parallel asynchronous pipeline runs; assert **zero trace cross-talk or token leakage** across concurrent tasks.
- [x] **Task 47.4:** Create and verify `scripts/verify_day47.py` (8 checks: contextvar isolation, async task safety, thread safety, backward compatibility with v1.0.0 capture decorators).

---

### Day 48 — Non-Blocking Async Telemetry Ring Buffer
- [x] **Task 48.1:** Create `capture/queue.py` with an in-memory bounded ring buffer queue (`TraceIngestionQueue`) supporting configurable `max_size=5000` and dropped-span counters.
- [x] **Task 48.2:** Implement `TraceIngestionWorker` background daemon thread that batches trace flushes (every $100\text{ms}$ or $50$ spans) and handles clean shutdown on `atexit` / `signal`.
- [x] **Task 48.3:** Benchmark `@trace_step` execution overhead under heavy load; verify interception adds $\le 0.5\text{ms}$ to host pipeline execution.
- [x] **Task 48.4:** Create and verify `scripts/verify_day48.py` (8 checks: queue buffering, batch flush timing, dropped-span telemetry, graceful shutdown on SIGINT/exit).

---

### Day 49 — SQLite WAL Mode & Connection Pooling
- [x] **Task 49.1:** Update `storage/db.py` to execute `PRAGMA journal_mode=WAL;`, `PRAGMA synchronous=NORMAL;`, and set `busy_timeout=10000`.
- [x] **Task 49.2:** Add thread-safe connection pooling with retry backoff in `DatabaseManager` to eliminate `sqlite3.OperationalError: database is locked`.
- [x] **Task 49.3:** Write `tests/test_storage_concurrency.py` executing 100 simultaneous concurrent writes; assert zero locked database errors and 100% data integrity.
- [x] **Task 49.4:** Create and verify `scripts/verify_day49.py` (8 checks: WAL file generation, connection pooling, concurrent write throughput, clean transaction rollback).

---

### Day 50 — Pluggable Storage Adapter Abstraction
- [x] **Task 50.1:** Define `storage/base_adapter.py` specifying `BaseStorageAdapter` with abstract methods: `save_run`, `get_run`, `list_runs`, `save_analysis`, `query_metrics`.
- [x] **Task 50.2:** Refactor `storage/writer.py` and `DatabaseManager` into `SQLiteStorageAdapter` implementing `BaseStorageAdapter`.
- [x] **Task 50.3:** Create `storage/postgres_adapter.py` providing a production-ready PostgreSQL adapter stub driven by `AGENTLENS_DB_URL=postgresql://...`.
- [x] **Task 50.4:** Create and verify `scripts/verify_day50.py` (8 checks: adapter polymorphism, SQLite adapter parity, connection string switching, schema validation).

---

### Day 51 — Phase A Stabilization Gate & Regression Run
- [x] **Task 51.1:** Run full v1.0 regression suite (`scripts/run_day43_regression.py`); verify **zero regressions** on all 20 frozen benchmark traces ($16/20 = 80\%$, $20/20 = 100\%$).
- [x] **Task 51.2:** Update `tests/test_api.py` and `tests/test_day44_package_docker.py` under multi-threaded concurrency; verify all 26 existing API/Docker tests pass.
- [x] **Task 51.3:** Enforce test coverage gate (`pytest --cov-fail-under=75`) and verify 100% clean typing (`mypy`) and linting (`ruff`).
- [x] **Task 51.4:** Create and verify `scripts/verify_day51.py` certifying **Phase A Stabilization Complete**.

---

## WEEK 11: Phase B — Universal DAG Topology & Semantic NLI

### Day 52 — OpenInference Span DAG Schema
- [x] **Task 52.1:** Create `schema/span.py` implementing `TraceSpan` with `span_id`, `trace_id`, `parent_span_id`, `span_type` (`agent` | `tool` | `chain`), and `iteration_index`.
- [x] **Task 52.2:** Update `RunTrace` model to support both legacy linear `steps: list[AgentStep]` and hierarchical `spans: list[TraceSpan]` with bidirectional converters.
- [x] **Task 52.3:** Write `tests/test_span_dag.py` modeling parallel execution branches (3 agents in parallel) and reflection loops ($A \to B \to A$).
- [x] **Task 52.4:** Create and verify `scripts/verify_day52.py` (8 checks: span hierarchy validation, cycle iteration indexing, parent-child tree serialization, legacy step conversion).

---

### Day 53 — Graph Aligner Upgrade for DAGs & Loops
- [x] **Task 53.1:** Upgrade `diff_engine/aligner.py` (`GraphAligner`) to match steps across multi-agent loops by pairing identical `(agent, iteration_index)` coordinates.
- [x] **Task 53.2:** Upgrade `GraphAligner` to support parallel fork diffing, aligning branches independently by agent role and call stack depth.
- [x] **Task 53.3:** Add unit tests in `tests/test_diff_engine_dag.py` diffing a 3-iteration looping trace against a single-pass trace; verify correct alignment.
- [x] **Task 53.4:** Create and verify `scripts/verify_day53.py` (8 checks: loop alignment, parallel branch matching, missing node isolation in cycles, backward compatibility with v1.0 linear diffs).

---

### Day 54 — Semantic Entity Set Overlap Engine
- [ ] **Task 54.1:** Create `analyzers/semantic/entity_tracker.py` using regex and entity extraction to parse actual named entity strings from agent outputs.
- [ ] **Task 54.2:** Implement mathematical set difference: $E_{\text{invented}} = E_{\text{downstream}} \setminus E_{\text{upstream}}$ and $E_{\text{dropped}} = E_{\text{upstream}} \setminus E_{\text{downstream}}$.
- [ ] **Task 54.3:** Update `RuleEngine` to register `hallucination_v2`, firing $P_2$ when $|E_{\text{invented}}| > 0$ even if total entity counts remain identical.
- [ ] **Task 54.4:** Create and verify `scripts/verify_day54.py` (8 checks: exact set difference, entity substitution detection, preservation ratio calculation, zero false positives on identical sets).

---

### Day 55 — Local Cross-Encoder NLI Engine
- [ ] **Task 55.1:** Create `analyzers/semantic/nli_engine.py` integrating a local, fast cross-encoder (`cross-encoder/nli-deberta-v3-small` with ONNX/PyTorch runtime).
- [ ] **Task 55.2:** Implement claim-level verification: split downstream synthesis into claims and evaluate $[P(\text{Entailment}), P(\text{Neutral}), P(\text{Contradiction})]$ against upstream findings.
- [ ] **Task 55.3:** Add graceful fallback: if PyTorch/Transformers is not installed, fall back to Jaccard token overlap without raising exceptions.
- [ ] **Task 55.4:** Create and verify `scripts/verify_day55.py` (8 checks: contradiction thresholding, entailment verification, local model caching, graceful fallback mode).

---

### Day 56 — Phase B Universal Core Verification & Accuracy Gate
- [ ] **Task 56.1:** Generate 5 synthetic multi-agent test traces containing loops, parallel forks, and entity-swapping hallucinations into `sample_data/v2_traces/`.
- [ ] **Task 56.2:** Evaluate `RuleEngine` + `NLIEngine` on the new traces; assert $\ge 90\%$ detection accuracy on complex reasoning bugs.
- [ ] **Task 56.3:** Re-run the frozen 20-trace v1.0 benchmark (`scripts/run_day43_regression.py`); confirm **zero regressions** on v1.0 cases.
- [ ] **Task 56.4:** Create and verify `scripts/verify_day56.py` certifying **Phase B Universal Core Complete**.

---

## WEEK 12: Phase C — Developer Experience & Auto-Instrumentation

### Day 57 — Unified Top-Level Public SDK
- [ ] **Task 57.1:** Build `agentlens/__init__.py` exposing clean public APIs: `init()`, `@trace_agent`, `trace_run()`, `get_current_trace()`, and `evaluate_run()`.
- [ ] **Task 57.2:** Implement `@agentlens.trace_agent(role=..., name=...)` with automatic framework inspection (detects function name, docstring, and input kwargs).
- [ ] **Task 57.3:** Write `tests/test_public_sdk.py` verifying a complete multi-agent pipeline can be traced and analyzed in **under 5 lines of code**.
- [ ] **Task 57.4:** Create and verify `scripts/verify_day57.py` (8 checks: top-level imports, `@trace_agent` decorator options, `with agentlens.trace_run` context manager, SDK exception safety).

---

### Day 58 — LangGraph & LangChain Auto-Tracer
- [ ] **Task 58.1:** Create `agentlens/integrations/langchain.py` implementing `AgentLensCallbackHandler` (inheriting from `BaseCallbackHandler`).
- [ ] **Task 58.2:** Hook into `on_chain_start`, `on_tool_start`, `on_tool_end`, `on_llm_end`, and `on_chain_end` to construct `TraceSpan` DAGs automatically.
- [ ] **Task 58.3:** Write `tests/test_langgraph_integration.py` executing a standard LangGraph `StateGraph` with conditional routing; verify full trace capture with zero manual decorators.
- [ ] **Task 58.4:** Create and verify `scripts/verify_day58.py` (8 checks: callback handler lifecycle, conditional routing capture, tool invocation tracking, token aggregation).

---

### Day 59 — CrewAI & AutoGen Interoperability
- [ ] **Task 59.1:** Create `agentlens/integrations/crewai.py` providing task and agent lifecycle listeners for CrewAI swarms (`Crew.kickoff()`).
- [ ] **Task 59.2:** Create `agentlens/integrations/autogen.py` capturing group-chat conversational turns and speaker transitions into structured spans.
- [ ] **Task 59.3:** Write `tests/test_swarm_integrations.py` mocking CrewAI and AutoGen message exchanges and verifying correct DAG span conversion.
- [ ] **Task 59.4:** Create and verify `scripts/verify_day59.py` (8 checks: CrewAI task span creation, AutoGen speaker turns, state handoff capture, error extraction).

---

### Day 60 — OpenTelemetry (OTLP) Ingestion Endpoint
- [ ] **Task 60.1:** Add `POST /v1/traces` endpoint in `api/router.py` accepting standard OpenTelemetry JSON / Protobuf HTTP trace payloads.
- [ ] **Task 60.2:** Implement `normalizer/otlp_normalizer.py` mapping standard OpenTelemetry / OpenInference spans into AgentLens `RunTrace` and `TraceSpan` models.
- [ ] **Task 60.3:** Write `tests/test_otlp_ingestion.py` sending mock OpenInference spans via `TestClient(fastapi_app)`; verify spans normalize and store correctly.
- [ ] **Task 60.4:** Create and verify `scripts/verify_day60.py` (8 checks: OTLP endpoint schema validation, span attribute mapping, token parsing, HTTP status codes).

---

### Day 61 — Phase C Developer Experience Verification Gate
- [ ] **Task 61.1:** Update `pyproject.toml` dependencies and optional extras (`agentlens[langchain]`, `agentlens[crewai]`, `agentlens[otlp]`).
- [ ] **Task 61.2:** Re-run `pip install -e .` and verify all entry points (`agentlens`, `agentlens-dashboard`, `agentlens-seed`) work cleanly.
- [ ] **Task 61.3:** Run all unit and integration test suites (`pytest tests/ -v`); assert coverage $\ge 75\%$.
- [ ] **Task 61.4:** Create and verify `scripts/verify_day61.py` certifying **Phase C Developer Experience Complete**.

---

## WEEK 13: Phase D — Causal Attribution & Remediation

### Day 62 — Causal Fault Propagation Graph
- [ ] **Task 62.1:** Create `analyzers/causal/fault_tree.py` implementing `CausalGraphEngine` to model error propagation across downstream spans.
- [ ] **Task 62.2:** Classify every faulty node into one of three causal states:
  - `ROOT_CAUSE_ORIGIN`: First node where invariant broke.
  - `CASCADING_FAILURE`: Downstream node failing as a direct consequence of upstream corrupted state.
  - `PASSTHROUGH_ESCAPED`: Quality-control node that failed to intercept the error.
- [ ] **Task 62.3:** Write `tests/test_causal_fault_tree.py` simulating a 3-agent cascade (Researcher crash $\to$ Writer hallucination $\to$ Verifier approval); assert correct causal classification.
- [ ] **Task 62.4:** Create and verify `scripts/verify_day62.py` (8 checks: causal tree construction, root-cause isolation, cascade path resolution, passthrough flagging).

---

### Day 63 — Error Blast Radius & Financial Impact Calculator
- [ ] **Task 63.1:** Create `analyzers/causal/blast_radius.py` computing financial and computational waste caused by the primary root cause.
- [ ] **Task 63.2:** Calculate metrics:
  - `Tainted Steps Ratio`: $\frac{\text{Steps after error}}{\text{Total steps}}$
  - `Wasted Token Cost`: Estimated USD cost of all downstream steps executed after the root cause.
  - `Wasted Latency`: Milliseconds spent computing tainted steps.
- [ ] **Task 63.3:** Add blast radius metrics to `AnalysisBundle` and update `storage/db.py` to persist cost impact.
- [ ] **Task 63.4:** Create and verify `scripts/verify_day63.py` (8 checks: wasted token computation, latency impact, financial waste calculation, zero blast radius on clean runs).

---

### Day 64 — Automated Prompt Auto-Diff Engine
- [ ] **Task 64.1:** Create `analyzers/remediation/prompt_fixer.py` comparing failed agent inputs/outputs against historical successful runs.
- [ ] **Task 64.2:** Implement targeted constraint generator: generate GitHub-diff style prompt recommendations (e.g., adding explicit entity preservation constraints).
- [ ] **Task 64.3:** Add `remediation_patch` field to the Explain view in `dashboard/state.py` and `api/router.py`.
- [ ] **Task 64.4:** Create and verify `scripts/verify_day64.py` (8 checks: prompt diff generation, constraint formatting, regression avoidance check, API serialization).

---

### Day 65 — Interactive Visual DAG Dashboard (NiceGUI v2)
- [ ] **Task 65.1:** Update `dashboard/app.py` to render interactive DAG visualizers using Cytoscape.js / Mermaid in the Run Explorer and Timeline views.
- [ ] **Task 65.2:** Color-code DAG nodes by causal state (🔴 Root Cause, 🟠 Cascade, 🟡 Passthrough, 🟢 Healthy).
- [ ] **Task 65.3:** Add a "Blast Radius & Remediation" panel displaying wasted cost and prompt patches with a one-click "Copy Prompt Fix" button.
- [ ] **Task 65.4:** Create and verify `scripts/verify_day65.py` (8 checks: visual DAG rendering, color-coding status mapping, remediation panel layout, responsive UI).

---

### Day 66 — AgentLens v2.0 Final Release Gate & Benchmark Sign-Off
- [ ] **Task 66.1:** Execute comprehensive 30-trace extended regression suite (`scripts/run_v2_regression.py`); assert $\ge 85\%$ exact attribution and $100\%$ binary accuracy.
- [ ] **Task 66.2:** Run full test suite with coverage enforcement: `pytest --cov=. --cov-fail-under=80` (targeting $\ge 85\%$ coverage).
- [ ] **Task 66.3:** Verify 100% clean typing (`mypy . --strict`) and zero lint errors (`ruff check .`, `ruff format --check .`).
- [ ] **Task 66.4:** Create and verify `scripts/verify_day66.py` certifying **AgentLens v2.0 Enterprise Release Complete**. Tag `v2.0.0` on GitHub.
