# AgentLens System Boundaries, Validation Results & Known Limitations

This document provides an honest, empirical accounting of **AgentLens**: its validation accuracy on the frozen 20-trace benchmark, known failure modes, per-rule false-positive/false-negative characteristics, and the Day 28 long-tail proof.

---

## 1. Observability Architecture Overview

AgentLens employs a **dual-layer detection architecture** to evaluate multi-agent pipeline executions before passing structured evidence to the deterministic **Arbiter (`P1`–`P5`)**:

```text
                             [ Agent Run Trace ]
                                      |
           +--------------------------+--------------------------+
           v                                                     v
+------------------------------+                      +------------------------------+
|  Deterministic Rule Engine   |                      | Statistical Anomaly Detector |
|  (P1 GT, P2 Rules, P3 Flow)  |                      |  (P4 Per-Agent Z-Score > N)  |
+--------------+---------------+                      +--------------+---------------+
               |                                                     |
               v                                                     v
     [ Known Failure Modes ]                                [ Long-Tail Outliers ]
     (Execution, Reasoning,                                 (Latency / Token Spikes
      Workflow, Verification)                                Matching Zero Rules)
               |                                                     |
               +--------------------------+--------------------------+
                                          v
                              +----------------------+
                              |    Arbiter Engine    |
                              |  (Priority Merge P1-5|
                              |   + Rule Tie-Break)  |
                              +----------------------+
```

---

## 2. Empirical Validation Results (Days 34–35)

AgentLens was evaluated against the **20 frozen labeled traces** (`sample_data/labels.json`, locked on Day 15) across 5 balanced categories (`4` traces each). Full outputs are stored in `validation/results_day34.json` and `validation/accuracy_day35.json`.

### Aggregate Metrics

| Metric | Score | Target | Status |
|---|---|---|---|
| **Exact Root-Cause Attribution** (`category + primary_agent`) | **16 / 20 (80.0%)** | `>= 15 / 20 (75.0%)` | **PASS** |
| **Binary Anomaly Detection** (`PASS` vs. `FAIL`) | **20 / 20 (100.0%)** | `>= 18 / 20 (90.0%)` | **PASS** (0 FP, 0 FN) |
| **Macro Rule-Level Recall** (Target detector fired on trace) | **20 / 20 (100.0%)** | `>= 15 / 20 (75.0%)` | **PASS** |
| **Error-Injection Pytest Suite** (`tests/test_error_injection.py`) | **10 / 10 (100.0%)** | `100%` | **PASS** |

### Per-Category Precision, Recall & F1 (Arbiter Single-Winner Level)

| Category | Support | Predicted | TP | FP | FN | Precision | Recall | F1 | Rule-Level Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pass` | 4 | 4 | 4 | 0 | 0 | `1.00` | `1.00` | `1.00` | **100%** (4/4) |
| `execution_failure` | 4 | 4 | 4 | 0 | 0 | `1.00` | `1.00` | `1.00` | **100%** (4/4) |
| `workflow_failure` | 4 | 4 | 4 | 0 | 0 | `1.00` | `1.00` | `1.00` | **100%** (4/4) |
| `reasoning_failure` | 4 | 8 | 4 | 4 | 0 | `0.50` | `1.00` | `0.67` | **100%** (4/4) |
| `verification_failure` | 4 | 0 | 0 | 0 | 4 | `0.00` | `0.00` | `0.00` | **100%** (4/4) |

---

## 3. Known Failure Modes & Disagreement Analysis

### Failure Mode 1: Dual-Fault `P2` Tie-Breaking (`run_lbl_verification_01`–`04`)
* **What happens**: All 4 disagreements on the 20-run benchmark occur on `run_lbl_verification_01` through `run_lbl_verification_04`.
* **Why it happens**: In the synthetic generator (`scripts/generate_labeled_set.py`), `verification_failure` traces are constructed by having the `writer` inflate entity counts by $+12$ to $+15$ entities over `researcher` **and** having the `verifier` approve those inflated counts unchanged. Consequently, two `P2` rules fire simultaneously with `confidence=1.0`:
  1. `hallucination_v1` (`category=reasoning`, `agent=writer`, `source=RULE_ENGINE` / `P2`)
  2. `verifier_passthrough_v1` (`category=verification`, `agent=verifier`, `source=RULE_ENGINE` / `P2`)
* **Arbiter behavior**: Because both rules share priority `P2`, `Arbiter._tiebreak_key()` breaks the tie deterministically by ascending alphabetical `rule_id` (`"hallucination_v1" < "verifier_passthrough_v1"`). The Arbiter blames the upstream `writer` fabrication as the primary cause while preserving `verifier_passthrough_v1` in `rules_fired` and `supporting_evidence`.
* **When `verifier` is blamed directly**: When the verifier rubber-stamps unverified claims without simultaneous writer entity-count inflation (verified in `tests/test_error_injection.py::TestAlwaysApproveVerifierInjection`), `verifier_passthrough_v1` wins `P2` directly and attributes blame to `verifier`.

### Failure Mode 2: Secondary `skipped_step_v1` Firing on Early Pipeline Halts
* **What happens**: On `run_lbl_execution_01`–`04`, when the `researcher` tool call fails (`tool_failure_v1` or `missing_tool_output_v1`), the pipeline aborts before invoking `verifier`. `WorkflowValidator` therefore also records `skipped_step_v1` (`P3`) in `rules_fired`.
* **Impact on Verdict**: **None** — `Arbiter` ranks `P2` (`EvidenceSource.RULE_ENGINE`) strictly above `P3` (`EvidenceSource.WORKFLOW_VALIDATOR`), so all 4 traces are accurately attributed to `execution` / `researcher`. However, operators inspecting the Evidence Panel will see `skipped_step_v1` listed as secondary supporting evidence.

### Failure Mode 3: Unstructured Prose Without `GROQ_API_KEY`
* **What happens**: When `GROQ_API_KEY` is unset and step outputs are free-form prose rather than structured JSON, `EvidenceExtractor` cannot call the LLM to extract `source_count`, `entity_count`, or `claims`.
* **Impact on Verdict**: Extraction-dependent rules (`hallucination_v1`, `researcher_quality_v1`, `verifier_passthrough_v1`, `claim_drift_v1`) are skipped gracefully (`extraction_failed=True` / `extractor=None`). Structural rules (`tool_failure_v1`, `missing_tool_output_v1`, `skipped_step_v1`, `wrong_order_v1`, `gt_mismatch_v1`, `STAT-LAT`, `STAT-TOK`) continue operating normally.

### Failure Mode 4: String-Similarity Sensitivity in `GroundTruthValidator` (`P1`)
* **What happens**: `gt_mismatch_v1` uses `difflib.SequenceMatcher` character-level ratio against `expected_output` (threshold `0.85`). Valid paraphrases with different wording can score below `0.85`.
* **Mitigation**: `expected_output` is optional; when omitted, `GroundTruthValidator` skips cleanly and sets `grounded=False` so the LLM Explainer uses hedged phrasing.

---

## 4. Known False Positives & Blind Spots Per Rule

| Rule ID | Known False-Positive Scenario | Known Blind Spot (False Negative) | Mitigation / Config Knob |
|---|---|---|---|
| `gt_mismatch_v1` | Semantically equivalent paraphrase with low lexical overlap (`ratio < 0.85`). | Subtle factual negation ("not approved" vs "approved") in a long paragraph where overall string similarity stays `>= 0.85`. | Tune `arbiter.ground_truth.p1_similarity_threshold` in `config/config.yaml`. |
| `tool_failure_v1` | Tool output legitimately discussing the literal substring `"Error:"` or `"Exception:"` (e.g., researching Python exceptions). | Tool returns HTTP 200 with a domain-level wrong answer that contains no error field or exception prefix. | Populate structured `tool_call["error"]` in `@trace_step` rather than relying on substring matching. |
| `missing_tool_output_v1` | Fire-and-forget void tool calls that intentionally return `""`. | Tool returns whitespace or a placeholder `"OK"` string with no real payload. | Ensure void tools return a status acknowledgement object. |
| `hallucination_v1` | Writer expands an acronym or splits a compound entity into 2 named entities (`entity_gain > 0` when threshold is `0`). | Writer fabricates a new entity while dropping one existing entity, keeping net `entity_count` unchanged. | Increase `arbiter.reasoning.hallucination_entity_gain_threshold` (e.g., `1` or `2`) and pair with `claim_drift_v1`. |
| `researcher_quality_v1` | Narrow or axiomatic topic that genuinely requires fewer than `researcher_min_sources` citations. | Researcher cites `>= min_sources` low-quality or irrelevant URLs. | Adjust `arbiter.reasoning.researcher_min_sources` in `config/config.yaml`. |
| `information_loss_v1` | Writer intentionally summarizes/condenses background entities while preserving core claims. | Writer replaces original entities with fabricated ones at a 1:1 ratio (`entity_diff == 0`). | Pair with `claim_drift_v1` and semantic `diff_engine` similarity scores. |
| `verifier_passthrough_v1` | Superseded at Arbiter winner level when `hallucination_v1` fires simultaneously at `P2` (`'h' < 'v'`). | Verifier approves a subtle factual contradiction where `entity_count` did not increase and `unverified_claims` is not flagged. | Inspect full `bundle.rule_matches` in the Evidence Panel alongside the primary winner. |
| `claim_drift_v1` | LLM extractor phrases the same underlying fact slightly differently between `researcher` and `writer` steps. | Skipped when `researcher` step yields zero extractable claims (`res_ev.claims == []`). | Temperature `0.0` in `EvidenceExtractor` + case-insensitive claim normalization. |
| `skipped_step_v1` | Fires as secondary `P3` evidence when an upstream `P2` execution crash halts the pipeline early. | Cannot detect a node that executed as a no-op stub returning unchanged state. | Arbiter `P2 > P3` priority ordering ensures upstream execution crash wins primary verdict. |
| `wrong_order_v1` | Dynamic cyclic graphs where an agent legitimately re-runs after a revision loop. | Does not flag intra-step sub-call reordering within a single agent node. | Configure `arbiter.workflow.required_agents` for the target pipeline topology. |
| `STAT-LAT` | Transient network or provider queueing spike on an otherwise healthy run. | Cold-start blind spot: inactive until at least `min_runs_for_baseline` (`5`) runs exist in SQLite. | Tune `metrics.latency_stddev_multiplier` (`2.5`) and `metrics.min_runs_for_baseline` (`5`). |
| `STAT-TOK` | Longer input topic naturally producing a longer, thorough report. | Cold-start blind spot (`< 5` baseline runs) or gradual baseline drift if many verbose runs accumulate. | Tune `metrics.token_stddev_multiplier` (`2.5`) in `config/config.yaml`. |

---

## 5. Empirical Long-Tail Proof (Day 28 Evaluation)

Deterministic rules can only catch failure patterns anticipated at authoring time. To prove that AgentLens closes the **long-tail observability gap** for novel, un-ruled failures, Day 28 (`scripts/run_long_tail_test.py`) evaluated a failure trace designed to bypass all 10 deterministic rules:

### Experimental Setup
* **Agent Behavior**: All three agents (`researcher -> writer -> verifier`) executed in the right order, cited all 5 sources, preserved all entities (`entity_gain = 0`), had zero tool errors, and passed verification (`APPROVED`).
* **Injected Un-Ruled Fault**: An internal reasoning loop inside `writer` caused extreme resource consumption:
  - **Latency**: `22,500 ms` (`5.5x` the historical `writer` baseline mean of `2,200 ms`)
  - **Tokens**: `9,500 tokens` (`6.0x` the historical `writer` baseline mean of `1,200 tokens`)

### Results

| Detection Layer | Outcome | Detail |
|---|---|---|
| **Deterministic Rules (`P1`–`P3`)** | `0` rules fired | Trace satisfied every schema, handoff, citation, and workflow invariant. |
| **Statistical Detector (`P4`)** | **FIRED** (`STAT-LAT`, `STAT-TOK`) | Flagged `writer` step for $Z > 4.2\sigma$ latency and token outliers (`confidence = 0.99`). |
| **Arbiter Final Verdict** | **`P4` (`PERFORMANCE` / `writer`)** | Correctly attributed the failure to `writer` via `EvidenceSource.STATISTICAL_ANOMALY`. |

This confirms that when a failure falls outside the deterministic rule catalog, the `P4` statistical anomaly layer catches the outlier and attributes it to the responsible agent.
