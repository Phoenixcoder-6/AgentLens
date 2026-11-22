# AgentLens Architecture & Technical Specification (v2.0.0)

This document details the internal architecture, design guarantees, data flow, storage schema, and extensibility patterns of AgentLens v2.0 Enterprise.

---

## 1. System Design Principles

AgentLens is built around five core architectural tenets:

1. **Deterministic Attribution Over Probabilistic Guessing:**  
   LLM-based judges produce non-deterministic results that shift with temperature, model updates, and prompt phrasing. AgentLens resolves failure causes using deterministic, rule-based heuristics and a mathematically predictable priority order. The LLM is used strictly downstream to *explain* the deterministic finding.
2. **Fail-Safe Capture:**  
   Observability must never crash the workload it monitors. The `@trace_step` decorator and `CaptureSession` capture state, latency, and errors inside protective try-catch boundaries with non-blocking async ring buffers. If storage, serialization, or disk writes fail, the host pipeline executes unimpeded.
3. **Strict Priority Hierarchy ($P_1 \to P_5$):**  
   When multiple errors co-occur (e.g., an upstream hallucination followed by downstream verifier rubber-stamping), the Arbiter applies a strict priority ordering to isolate the primary root cause without human ambiguity.
4. **Causal Graph Propagation & Separation of Blame:**  
   Failures in multi-agent networks cascade. AgentLens isolates the exact `ROOT_CAUSE_ORIGIN` node, maps downstream `CASCADING_FAILURE` spans, and flags quality-control nodes that permitted corrupt state to escape (`PASSTHROUGH_ESCAPED`).
5. **Actionable Remediation & Waste Accounting:**  
   Detecting an error is insufficient. AgentLens computes the financial blast radius (wasted tokens, latency, USD spend) and automatically synthesizes prompt patches and auto-diffs to prevent regression.

---

## 2. End-to-End Architectural Pipeline & Data Flow

```mermaid
flowchart TD
    subgraph S_HOST["1. Host Execution & Fail-Safe Telemetry"]
        NODE["Multi-Agent Node<br/><i>LangGraph / CrewAI / Custom</i>"]
        DECOR["@trace_step Decorator<br/><i>Fail-Safe Boundary</i>"]
        RING["Ring Buffer Queue<br/><i>Non-blocking Async Ingestion</i>"]
        CTX["ContextVar Engine<br/><i>Concurrency Isolation</i>"]
        PII["PII Scrubber<br/><i>Regex Redaction Engine</i>"]

        NODE --> DECOR
        DECOR --> CTX
        CTX --> PII
        PII --> RING
    end

    subgraph S_STORE["2. Normalization & Pluggable Storage"]
        NORM["Normalizer<br/><i>SchemaValidator (v2.0.0)</i>"]
        SQL[("SQLite WAL / PostgreSQL<br/><i>Connection Pool & Retries</i>")]
        FS[("Raw Trace Store<br/><i>data/traces/{run_id}.json</i>")]

        RING --> NORM
        NORM --> SQL
        NORM --> FS
    end

    subgraph S_ENGINES["3. Multi-Tier Detection Analyzers"]
        direction TB
        GT["GroundTruthValidator [P1]<br/><i>Factual contradiction vs expected</i>"]
        RE["RuleEngine [P2]<br/><i>tool_failure · hallucination · reasoning</i>"]
        CV["ConsistencyValidator [P2]<br/><i>verifier_passthrough</i>"]
        WV["WorkflowValidator [P3]<br/><i>Universal DAG topology & skipped steps</i>"]
        IL["InformationLossRule [P3]<br/><i>Entity & citation delta tracking</i>"]
        SD["StatisticalDetector [P4]<br/><i>Latency & Token Z-Score Outliers</i>"]

        SQL --> GT
        SQL --> RE
        SQL --> CV
        SQL --> WV
        SQL --> IL
        SQL --> SD
    end

    subgraph S_ARBITER["4. Deterministic Arbiter"]
        RANK["5-Tier Priority Resolver<br/><i>P1 &gt; P2 &gt; P3 &gt; P4 &gt; P5</i>"]
        TIE["Ascending rule_id Tie-Breaker<br/><i>Guarantees 100% Determinism</i>"]
        BUNDLE["AnalysisBundle<br/><i>verdict · primary_cause · primary_agent</i>"]

        GT -->|"EvidenceRecord"| RANK
        RE -->|"EvidenceRecord"| RANK
        CV -->|"EvidenceRecord"| RANK
        WV -->|"EvidenceRecord"| RANK
        IL -->|"EvidenceRecord"| RANK
        SD -->|"EvidenceRecord"| RANK

        RANK --> TIE
        TIE --> BUNDLE
    end

    subgraph S_CAUSAL["5. Phase D Causal & Remediation Subsystem"]
        CFT["CausalGraphEngine<br/><i>Fault Tree & Blame Classification</i>"]
        BR["BlastRadiusCalculator<br/><i>Wasted Tokens, USD & Latency</i>"]
        FIX["PromptFixer<br/><i>Auto-Diff Prompt Patch Generator</i>"]

        BUNDLE --> CFT
        CFT --> BR
        CFT --> FIX
    end

    subgraph S_SURFACE["6. Storage, Alerting & UI Presentation"]
        EXPLAIN["LLM Explainer<br/><i>Natural language synthesis</i>"]
        ALERT["Alerter<br/><i>Slack webhook alerts for P1/P2</i>"]
        DASH["Interactive DAG Dashboard<br/><i>NiceGUI v2 + Mermaid Color-Coded Graphs</i>"]
        API["FastAPI REST Service<br/><i>/health · /api/runs · /api/causal</i>"]

        CFT --> DASH
        BR --> DASH
        FIX --> DASH
        BUNDLE --> EXPLAIN
        BUNDLE --> ALERT
        BUNDLE --> API
    end

    classDef hostStyle fill:#f0f9ff,stroke:#0369a1,stroke-width:2px,color:#0c4a6e;
    classDef storeStyle fill:#f0fdf4,stroke:#15803d,stroke-width:2px,color:#14532d;
    classDef engineStyle fill:#fffbeb,stroke:#b45309,stroke-width:2px,color:#78350f;
    classDef arbiterStyle fill:#faf5ff,stroke:#6b21a8,stroke-width:2px,color:#3b0764;
    classDef causalStyle fill:#fdf4ff,stroke:#a21caf,stroke-width:2px,color:#701a75;
    classDef surfaceStyle fill:#fff1f2,stroke:#be123c,stroke-width:2px,color:#881337;

    class NODE,DECOR,RING,CTX,PII hostStyle;
    class NORM,SQL,FS storeStyle;
    class GT,RE,CV,WV,IL,SD engineStyle;
    class RANK,TIE,BUNDLE arbiterStyle;
    class CFT,BR,FIX causalStyle;
    class EXPLAIN,ALERT,DASH,API surfaceStyle;
```

---

## 3. Causal Fault Propagation Graph Engine

The **Causal Fault Tree Engine** (`analyzers/causal/fault_tree.py`) constructs a directed execution graph of agent steps and classifies each node into one of four deterministic states:

- **🔴 `ROOT_CAUSE_ORIGIN`:** The first point of failure in the multi-agent graph where an invariant (contract, tool, hallucination, ground truth) was violated.
- **🟠 `CASCADING_FAILURE`:** A downstream agent node that failed as a direct consequence of corrupted inputs received from an upstream failure.
- **🟡 `PASSTHROUGH_ESCAPED`:** A quality-control or verifier node that received a corrupted payload from an upstream node and failed to intercept or flag it, rubber-stamping bad output.
- **🟢 `HEALTHY`:** An agent node that executed cleanly within contractual bounds.

```
[Researcher (ROOT_CAUSE)] ──(corrupt data)──> [Writer (CASCADE)] ──(corrupt report)──> [Verifier (PASSTHROUGH)]
```

---

## 4. Blast Radius & Financial Impact Calculator

The **Blast Radius Calculator** (`analyzers/causal/blast_radius.py`) quantifies the collateral damage caused by the primary root cause:

1. **Tainted Steps Ratio:**
   $$\text{Tainted Steps Ratio} = \frac{N_{\text{tainted}}}{N_{\text{total}}}$$
2. **Wasted Token Count:** Sum of prompt and completion tokens consumed by all steps executed *after* the root cause origin.
3. **Wasted Cost (USD):** Granular dollar calculation across token models (defaulting to standard blended pricing, e.g., \$0.0051 on benchmark runs).
4. **Wasted Latency (ms):** Sum of wall-clock milliseconds consumed by tainted downstream execution.

---

## 5. Automated Prompt Auto-Diff Remediation Engine

The **Prompt Auto-Diff Engine** (`analyzers/remediation/prompt_fixer.py`) inspects failing nodes and the arbiter's primary root cause to generate deterministic, targeted prompt modifications:

- **`SOURCE_PRESERVATION`:** Fired on information loss; instructs the agent to retain all primary citations and numeric data.
- **`ENTITY_FIDELITY`:** Fired on reasoning/hallucination matches; injects strict fact-checking constraints against input premises.
- **`CONTRADICTION_GUARD`:** Prevents contradictory assertions across collaborative agent handoffs.
- **`SCHEMA_CONFORMANCE`:** Fired on workflow state drops; enforces JSON state schema keys.
- **`TOOL_DEFENSE`:** Injects parameter validation guards for tools prone to parameter errors.

The engine produces unified diff patches:
```diff
--- SYSTEM PROMPT (agent=writer:original)
+++ SYSTEM PROMPT (agent=writer:remediated)
@@ -1,3 +1,5 @@
 You are a senior financial writer.
 Synthesize the research input into an executive briefing.
+CRITICAL CONSTRAINT: You must preserve all source citations, entity names, and numeric data exactly as provided in the research input without omission or substitution.
```

---

## 6. The 5-Tier Priority Resolution Hierarchy

The Arbiter resolves all collected `EvidenceRecord` items into a single final `AnalysisBundle` according to this strict ladder:

| Priority | Category | Evidence Source | Criteria | Default Verdict |
|---|---|---|---|---|
| **$P_1$** | **Grounded Failure** | `GROUND_TRUTH` | Direct factual contradiction against known ground truth | `FAIL` (grounded=True) |
| **$P_2$** | **Deterministic Rule** | `RULE_ENGINE`, `CONSISTENCY_VALIDATOR` | Execution crashes, missing tool output, severe hallucinations, verifier passthrough | `FAIL` (grounded=False) |
| **$P_3$** | **Workflow / Handoff** | `WORKFLOW_VALIDATOR`, `INFORMATION_LOSS` | Missing required agent, skipped nodes, moderate information loss/gain | `FAIL` (grounded=False) |
| **$P_4$** | **Statistical Anomaly** | `STATISTICAL_ANOMALY`, `METRICS_ANALYZER` | Latency or token consumption exceeds $>2.5\sigma$ of agent historical baseline | `FAIL` (grounded=True) |
| **$P_5$** | **Clean / Unknown** | None / Fallback | All checks passed, or no deterministic rule fired | `PASS` (or unknown) |

### Deterministic Tie-Breaking
If multiple pieces of evidence fire at the **same** priority tier (e.g., two $P_2$ rules fire: `hallucination_v1` on Writer and `verifier_passthrough_v1` on Verifier):
1. The Arbiter sorts candidate records alphabetically by `rule_id` ascending.
2. In the case above, `hallucination_v1` sorts before `verifier_passthrough_v1`.
3. The Arbiter crowns the upstream reasoning fault (`hallucination_v1`) as the primary cause, perfectly reflecting that downstream verifier passthrough was triggered by upstream corruption.
4. The exact same evidence list will yield the exact same verdict **100% of the time**.

---

## 4. Grounded vs. Heuristic Signals

AgentLens enforces a strict distinction in telemetry and reporting:

- **Grounded Attribution (`grounded = True`):**  
  The failure is verified against an objective anchor outside the LLM's opinion:
  - Factual mismatch against verified `expected_output` ($P_1$).
  - Mathematical statistical outlier exceeding $2.5\sigma$ of recorded population history ($P_4$).
- **Heuristic Attribution (`grounded = False`):**  
  The failure was inferred via behavioral rules and schema inspection ($P_2$, $P_3$). While highly accurate, it represents structural inference rather than mathematical ground-truth proof.

---

## 5. Storage Architecture & Alembic Schema

AgentLens stores telemetry in SQLite (`data/agentlens.db` by default, configurable via `DB_PATH` or `config.yaml`):

```sql
-- Core execution tables
runs (
    run_id TEXT PRIMARY KEY,
    workflow TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    status TEXT NOT NULL,
    total_latency_ms REAL,
    total_tokens INTEGER,
    schema_version TEXT,
    trace_path TEXT,
    trace_json TEXT,
    expected_output TEXT
);

steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    step INTEGER NOT NULL,
    agent TEXT NOT NULL,
    status TEXT NOT NULL,
    latency_ms REAL,
    tokens_prompt INTEGER,
    tokens_completion INTEGER,
    tokens_total INTEGER,
    diff_summary TEXT,
    error TEXT,
    timestamp TEXT NOT NULL,
    schema_version TEXT,
    UNIQUE (run_id, step)
);

-- Analysis & Observability tables
analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    step INTEGER,
    analyzer TEXT NOT NULL,
    category TEXT,
    verdict TEXT,
    confidence REAL,
    details_json TEXT,
    timestamp TEXT NOT NULL,
    schema_version TEXT
);

rule_matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    rule_id TEXT NOT NULL,
    rule_version TEXT NOT NULL,
    category TEXT NOT NULL,
    severity TEXT NOT NULL,
    agent TEXT,
    step_idx INTEGER,
    description TEXT,
    evidence_detail TEXT,
    timestamp TEXT NOT NULL
);

-- Performance Indexes (Alembic-managed)
CREATE INDEX ix_runs_timestamp ON runs (timestamp);
CREATE INDEX ix_steps_run_id ON steps (run_id);
CREATE INDEX ix_analysis_run_id ON analysis (run_id);
CREATE INDEX ix_rule_matches_run_id ON rule_matches (run_id);
CREATE INDEX ix_rule_matches_rule_id ON rule_matches (rule_id);
```

Database migrations are tracked via Alembic (`alembic/versions/`). Full test coverage guarantees that `upgrade("head")` and `downgrade("base")` execute cleanly.

---

## 6. How to Add a New Detection Rule

Adding a new deterministic failure rule takes 3 simple steps:

### Step 1: Register in `analyzers/rule_catalog.py`
```python
RuleDefinition(
    rule_id="custom_retry_loop_v1",
    version="1.0.0",
    category=FailureCategory.WORKFLOW,
    severity=RuleSeverity.MEDIUM,
    description="Agent entered an unconstrained retry loop exceeding max iterations.",
    priority=PriorityLevel.P3,
)
```

### Step 2: Implement Logic in the Target Analyzer
Implement the check inside `RuleEngine.analyze()`, `WorkflowValidator.analyze()`, or a dedicated analyzer:
```python
if agent_retry_count > max_retries:
    evidence.append(
        EvidenceRecord(
            source=EvidenceSource.WORKFLOW_VALIDATOR,
            description=f"Agent '{step.agent}' retried {agent_retry_count} times.",
            value="FAIL",
            rule_match=RuleMatch(
                rule_id="custom_retry_loop_v1",
                category=FailureCategory.WORKFLOW,
                severity=RuleSeverity.MEDIUM,
                agent=step.agent,
                step_idx=step.step,
                description="Retry loop exceeded limit.",
            ),
            agent=step.agent,
            confidence=1.0,
        )
    )
```

### Step 3: Run the Verification Suite
Ensure unit tests and the frozen benchmark pass:
```bash
pytest tests/ -v
python scripts/run_day43_regression.py
```
The regression runner will automatically verify that your new rule did not cause unintended verdict drift across the 20 benchmark traces.
