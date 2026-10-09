# AgentLens v1.0.0

[![AgentLens CI](https://github.com/Phoenixcoder-6/AgentLens/actions/workflows/ci.yml/badge.svg)](https://github.com/Phoenixcoder-6/AgentLens/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Coverage >= 75%](https://img.shields.io/badge/coverage-%3E%3D%2075%25-brightgreen.svg)](https://github.com/Phoenixcoder-6/AgentLens)
[![Version: 1.0.0](https://img.shields.io/badge/version-1.0.0-purple.svg)](CHANGELOG.md)

**Multi-Agent Failure Attribution, Trace Diffing & Explainability Platform**

AgentLens answers a single question with mathematical precision:  
*Why did a multi-agent workflow fail, and which agent was responsible?*

> **Core Philosophy:**  
> The LLM never decides what happened — it only explains what deterministic analysis has already proven.

---

## 1. Why AgentLens?

When complex LLM agent swarms fail (hallucinating facts, dropping instructions, crashing tools, or rubber-stamping bad output), debugging with standard logs or APMs is overwhelming:
- Traditional tracing tools (Datadog, OpenTelemetry) record *latencies* and *spans*, but understand nothing about *agent semantics*, *handoff contracts*, or *reasoning integrity*.
- LLM-as-a-judge approaches are probabilistic, non-deterministic, and frequently hallucinate their own blame attributions.

**AgentLens introduces Deterministic Multi-Tier Failure Attribution:**
1. **Deterministic Rule Engine (P1–P3):** Isolates the exact faulty agent using verifiable rules (tool failures, information loss/gain, step omission, verifier passthrough).
2. **Deterministic 5-Tier Arbiter:** Guarantees that the exact same evidence always yields the exact same verdict ($P_1 \to P_5$).
3. **Graph-Aligned Trace Diffing:** Aligns multi-agent execution traces step-by-step to pinpoint the exact moment two runs diverged.
4. **Grounded vs. Heuristic Separation:** Clearly flags whether a failure was proven against external ground-truth ($P_1$) or detected via behavioral heuristics ($P_2–P_4$).

---

### 2. Architecture: Evolution from Legacy Prototype to v1.0.0 Production

AgentLens evolved from a simple linear capture script with manual error guessing into a production-grade, fail-safe observability and deterministic failure attribution platform.

### Phase 1: Legacy Early Architecture (v0.1 Prototype)
In the initial prototype, tracing was synchronous and unshielded. Detection relied on basic post-hoc parsing with unranked, conflicting heuristic outputs and no tie-break determinism:

```mermaid
flowchart TD
    subgraph S1["1. Pipeline Execution"]
        A1["Researcher Agent"] --> A2["Writer Agent"]
        A2 --> A3["Verifier Agent"]
    end

    subgraph S2["2. Raw Capture (Unshielded)"]
        A1 -.->|"Synchronous hooks\n(Crash risk)"| B1["Raw Logs & Console Print"]
        A2 -.-> B1
        A3 -.-> B1
        B1 --> B2[("Unindexed Flat JSON")]
    end

    subgraph S3["3. Ad-Hoc Heuristics"]
        B2 --> C1["Manual String Inspection"]
        B2 --> C2["Unranked Heuristics\n(Conflicting outputs)"]
        C1 --> D1["LLM-as-a-Judge Prompt\n(Probabilistic & Non-deterministic)"]
        C2 --> D1
    end

    subgraph S4["4. Output"]
        D1 --> E1["Terminal Output / Basic Script"]
    end

    classDef legacy fill:#ffebee,stroke:#c62828,stroke-width:1px,color:#b71c1c;
    class A1,A2,A3,B1,B2,C1,C2,D1,E1 legacy;
```

---

### Phase 2: Current Production Architecture (v1.0.0 Enterprise Core)
v1.0.0 introduces a fail-safe capture envelope, canonical normalization, 6 multi-tier detection analyzers, the deterministic 5-tier Arbiter ($P_1 \to P_5$), and dual presentation interfaces (NiceGUI + FastAPI):

```mermaid
flowchart TB
    subgraph G_PIPELINE["1. AGENT EXECUTION LAYER"]
        direction LR
        P_NODE["Multi-Agent Node<br/><i>LangGraph / Custom</i>"]
        P_WRAP["@trace_step Wrapper<br/><i>Fail-Safe Boundary</i>"]
        P_NODE --- P_WRAP
    end

    subgraph G_CAPTURE["2. FAIL-SAFE TELEMETRY & STORAGE"]
        direction TB
        HC["HandoffCapture Engine<br/><i>(input_state · filtered_state · output_state)</i>"]
        PII["PII Scrubber<br/><i>(Regex Redaction: emails, API keys, tokens)</i>"]
        CS["CaptureSession Coordinator"]
        
        P_WRAP -->|"Execute & Intercept"| HC
        HC --> PII
        PII --> CS
        
        DB_SQL[("SQLite DB<br/><i>Alembic Indexed</i>")]
        FS_BLOB[("JSON Trace Store<br/><i>data/traces/*.json</i>")]
        NORM["Trace Normalizer<br/><i>Schema v1.0 Standardizer</i>"]
        
        CS --> NORM
        NORM --> DB_SQL
        NORM --> FS_BLOB
    end

    subgraph G_ANALYZERS["3. PARALLEL DETECTION ENGINES"]
        direction TB
        A_P1["Ground Truth Validator [P1]<br/><i>SequenceMatcher vs expected_output</i>"]
        A_P2A["RuleEngine [P2]<br/><i>tool_failure_v1 · hallucination_v1</i>"]
        A_P2B["Consistency Validator [P2]<br/><i>verifier_passthrough_v1</i>"]
        A_P3A["Workflow Validator [P3]<br/><i>skipped_step_v1</i>"]
        A_P3B["Information Loss Rule [P3]<br/><i>Handoff source/entity drops</i>"]
        A_P4["Statistical Detector [P4]<br/><i>Per-agent Latency & Token Z-Scores</i>"]
    end

    DB_SQL --> A_P1
    DB_SQL --> A_P2A
    DB_SQL --> A_P2B
    DB_SQL --> A_P3A
    DB_SQL --> A_P3B
    DB_SQL --> A_P4

    subgraph G_ARBITER["4. THE ARBITER (Deterministic Resolver)"]
        direction TB
        ARB_SORT["Priority Filter<br/>P1 &gt; P2 &gt; P3 &gt; P4 &gt; P5"]
        ARB_TIE["Alphabetical Rule-ID Tie-Break<br/><i>Deterministic: Same Input &rarr; Same Verdict</i>"]
        ARB_BUNDLE["AnalysisBundle Payload<br/><i>(verdict · primary_cause · primary_agent · grounded)</i>"]
        
        ARB_SORT --> ARB_TIE
        ARB_TIE --> ARB_BUNDLE
    end

    A_P1 -->|"EvidenceRecord"| ARB_SORT
    A_P2A -->|"EvidenceRecord"| ARB_SORT
    A_P2B -->|"EvidenceRecord"| ARB_SORT
    A_P3A -->|"EvidenceRecord"| ARB_SORT
    A_P3B -->|"EvidenceRecord"| ARB_SORT
    A_P4 -->|"EvidenceRecord"| ARB_SORT

    subgraph G_DELIVERY["5. EXPLAINABILITY & PRESENTATION"]
        direction TB
        EXPLAIN["LLM Explainer<br/><i>(Explains deterministic finding)</i>"]
        ALERT["Alerter<br/><i>Slack Webhooks for P1/P2</i>"]
        DASH["NiceGUI Dashboard<br/><i>6 Views: Runs, Timeline, Evidence, Diff, Metrics, Rules</i>"]
        REST["FastAPI REST Service<br/><i>GET /health &middot; /api/runs &middot; /api/metrics</i>"]
        
        ARB_BUNDLE --> EXPLAIN
        ARB_BUNDLE --> ALERT
        ARB_BUNDLE --> DASH
        ARB_BUNDLE --> REST
    end

    classDef pipeClass fill:#e0f2fe,stroke:#0284c7,stroke-width:2px,color:#0369a1;
    classDef capClass fill:#f0fdf4,stroke:#16a34a,stroke-width:2px,color:#15803d;
    classDef anaClass fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#b45309;
    classDef arbClass fill:#ede9fe,stroke:#7c3aed,stroke-width:2px,color:#6d28d9;
    classDef delivClass fill:#fdf2f8,stroke:#db2777,stroke-width:2px,color:#be185d;

    class P_NODE,P_WRAP pipeClass;
    class HC,PII,CS,NORM,DB_SQL,FS_BLOB capClass;
    class A_P1,A_P2A,A_P2B,A_P3A,A_P3B,A_P4 anaClass;
    class ARB_SORT,ARB_TIE,ARB_BUNDLE arbClass;
    class EXPLAIN,ALERT,DASH,REST delivClass;
```

---

### Architectural Comparison: Prototype vs. v1.0.0 Production

| Dimension | Legacy Prototype (v0.1) | AgentLens v1.0.0 Production |
|---|---|---|
| **Pipeline Safety** | Unhandled logging crashes pipeline | **Fail-Safe Envelope**: telemetry errors are trapped; pipeline never fails |
| **Privacy & Security** | Plaintext state dump to disk | **PII Scrubber**: automated redaction of emails, API keys, bearer tokens |
| **Attribution Logic** | LLM-as-a-judge probabilistic guesswork | **Deterministic Arbiter**: strict 5-tier priority ladder with rule-id tie-breaking |
| **Co-occurrence Handling**| Conflicting rules produce chaotic alerts | **Upstream Root-Cause Prioritization**: upstream errors take precedence |
| **Ground Truth Support** | None (pure heuristic) | **Explicit Grounded Boundary**: $P_1$ factual contracts vs. $P_2–P_4$ structural heuristics |
| **Storage Engine** | Flat unstructured JSON files | **Indexed SQLite + Alembic Migrations**: 5 performance indexes & schema versioning |
| **Delivery Interfaces** | Terminal CLI script only | **NiceGUI Interactive Dashboard (6 Views) + FastAPI REST API + Docker** |

---

## 3. Quickstart (Zero-Setup in 3 Commands)

### Option A: Local Python Package Installation

```bash
# 1. Install editable package
pip install -e .

# 2. Seed pre-computed zero-setup demo traces (PASS, Grounded P1, Heuristic P2/P3, Diff Pairs)
agentlens-seed

# 3. Launch the AgentLens interactive dashboard
agentlens-dashboard
```
Open **`http://localhost:8080`** in your browser.

---

### Option B: One-Command Docker Setup

```bash
docker compose up --build
```
The NiceGUI dashboard and FastAPI REST endpoints are exposed at **`http://localhost:8080`**.  
Liveness/readiness is monitored via `GET http://localhost:8080/health`.

---

## 4. Benchmark Accuracy & Validation

AgentLens is evaluated on a frozen 20-trace benchmark (`sample_data/labels.json`) covering clean runs, tool crashes, reasoning hallucinations, skipped nodes, and verifier bypasses:

| Metric | Target | Verified Actual | Status |
|---|---|---|---|
| **Binary PASS / FAIL Detection** | $\ge 90\%$ | **100.0%** (20/20) | **PASS** |
| **Exact Category & Agent Attribution** | $\ge 75\%$ | **80.0%** (16/20) | **PASS** |
| **Test Suite Code Coverage Gate** | $\ge 75\%$ | **93.0%** | **PASS** |
| **Type Checking (`mypy`)** | Strict PEP 561 | **0 issues** (46 source files) | **PASS** |
| **Formatting & Linting (`ruff`)** | Clean | **0 issues** | **PASS** |

Continuous regression testing is enforced on every PR to `main` via [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

---

## 5. REST API

AgentLens includes a built-in FastAPI REST service mounted at `/api` and `/health`:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness & readiness check (`status`, `db`, `llm`, `uptime_seconds`) |
| `GET` | `/api/runs` | List recorded runs with pagination & verdict level filters |
| `GET` | `/api/runs/{run_id}` | Full trace details including per-agent steps & state diffs |
| `GET` | `/api/runs/{run_id}/verdict` | Final Arbiter verdict bundle & primary cause |
| `POST` | `/api/analyze/{run_id}` | Trigger deterministic analysis on an existing trace |
| `GET` | `/api/metrics` | Aggregated latency, token, and run counts per agent |

Interactive Swagger documentation is available at `http://localhost:8080/docs`.

---

## 6. Project Documentation Index

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Deep dive into the 5-tier Arbiter, component data flow, storage schema, and how to write custom rules.
- **[RULES.md](RULES.md)**: Exhaustive catalog of all built-in deterministic detection rules (P1–P4).
- **[LIMITATIONS.md](LIMITATIONS.md)**: Benchmark error analysis, scope boundaries, and dual-fault tie-break dynamics.
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: Developer setup, test guidelines, pre-commit hooks, and PR workflow.
- **[CHANGELOG.md](CHANGELOG.md)**: Complete release history from v0.1.0 to v1.0.0.

---

## 7. License

AgentLens is open-source software licensed under the [MIT License](LICENSE).
