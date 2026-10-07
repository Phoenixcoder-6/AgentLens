# AgentLens Rule Catalog & Validation Results (`RULES.md`)

This document is the canonical reference for all **12 detection rules and statistical detectors** in AgentLens (`analyzers/rule_catalog.py`), together with their empirical **True Positive (TP)** and **False Positive (FP)** counts from the Week 7 validation suite (`validation/results_day34.json`, `validation/accuracy_day35.json`, `tests/test_error_injection.py`, and `scripts/run_long_tail_test.py`).

---

## 1. Complete Rule Catalog (`P1`–`P4`)

### P1 — Ground Truth (`EvidenceSource.GROUND_TRUTH`)

| Rule ID | Name | Version | Category | Owner Analyzer | Blames | Trigger Condition | Confidence |
|---|---|---|---|---|---|---|---|
| `gt_mismatch_v1` | Ground-Truth Mismatch | `1.0.0` | `reasoning` | `ground_truth` | Final step agent | `RunTrace.expected_output` is present AND `SequenceMatcher` similarity with final step output `< p1_similarity_threshold` (`0.85`). | `1.0 - similarity` |

### P2 — Deterministic Execution, Reasoning & Verification Rules (`EvidenceSource.RULE_ENGINE`)

| Rule ID | Name | Version | Category | Owner Analyzer | Blames | Trigger Condition | Confidence |
|---|---|---|---|---|---|---|---|
| `missing_tool_output_v1` | Missing Tool Output | `1.0.0` | `execution` | `rule_engine` | Step agent (`researcher`) | A tool call is recorded on the step with both `output`/`result` and `error` empty. | `1.0` |
| `tool_failure_v1` | Tool Failure | `1.0.0` | `execution` | `rule_engine` | Step agent (`researcher`) | A tool call has a non-empty `error` field OR its `output` contains `"Error:"` / `"Exception:"`. | `1.0` |
| `researcher_quality_v1` | Researcher Source Shortfall | `1.0.0` | `reasoning` | `rule_engine` | `researcher` | Researcher `source_count < researcher_min_sources` (`1`) and no researcher tool execution crash occurred. | `1.0` |
| `hallucination_v1` | Writer Hallucination | `1.0.0` | `reasoning` | `rule_engine` | `writer` | `writer.entity_count - researcher.entity_count > hallucination_entity_gain_threshold` (`0`). | `1.0` |
| `information_loss_v1` | Information Loss / Gain | `1.0.0` | `workflow` (loss) / `reasoning` (gain) | `information_loss` | `writer` | Sources or entities dropped (`FAIL`) or inflated (`WARNING`) across the `researcher -> writer` handoff. | `0.85`–`1.0` |
| `verifier_passthrough_v1` | Verifier Passthrough | `1.0.0` | `verification` | `consistency_validator` | `verifier` | Verifier entity count equals writer's inflated entity count (`entity_gain > threshold`), OR verifier rubber-stamps (`approved=True`) despite `always_approve=True` / `unverified_claims > 0`. | `1.0` |
| `claim_drift_v1` | Claim Drift | `1.0.0` | `verification` | `consistency_validator` | `writer` | Writer introduces extracted factual claims (`wr_ev.claims`) not present in `res_ev.claims` (case-insensitive). | `1.0` |

### P3 — Workflow Topology Rules (`EvidenceSource.WORKFLOW_VALIDATOR`)

| Rule ID | Name | Version | Category | Owner Analyzer | Blames | Trigger Condition | Confidence |
|---|---|---|---|---|---|---|---|
| `skipped_step_v1` | Skipped Step | `1.0.0` | `workflow` | `workflow_validator` | Missing agent (`researcher` / `writer` / `verifier`) | One or more `required_agents` (`["researcher", "writer", "verifier"]`) did not execute in the trace. | `1.0` |
| `wrong_order_v1` | Wrong Step Order | `1.0.0` | `workflow` | `workflow_validator` | Out-of-order agent | Present required agents executed out of the canonical sequence (`researcher -> writer -> verifier`). | `1.0` |

### P4 — Statistical Anomaly Detectors (`EvidenceSource.STATISTICAL_ANOMALY`)

| Rule ID | Name | Version | Category | Owner Analyzer | Blames | Trigger Condition | Confidence |
|---|---|---|---|---|---|---|---|
| `STAT-LAT` (`STAT-LAT-<AGENT>-<STEP>`) | Latency Outlier | `1.0.0` | `execution` / `performance` | `statistical_detector` | Outlier step agent | Step latency exceeds $\mu + 2.5\sigma$ of the agent's historical baseline ($\ge 5$ baseline runs). | `min(0.99, 0.50 + (z - 2.5) * 0.15)` |
| `STAT-TOK` (`STAT-TOK-<AGENT>-<STEP>`) | Token Outlier | `1.0.0` | `execution` / `performance` | `statistical_detector` | Outlier step agent | Step token usage exceeds $\mu + 2.5\sigma$ of the agent's historical baseline ($\ge 5$ baseline runs). | `min(0.99, 0.50 + (z - 2.5) * 0.15)` |

---

## 2. Validation Results Per Rule (20-Run Labeled Set + Error Injection Suite)

The table below reports validation performance at two levels:
1. **Condition-Level (Detector TP / FP on 20 Labeled Runs)**: Did the rule fire only when its underlying trace condition was genuinely present?
2. **Arbiter Primary-Winner Level (TP / FP vs. Single Human Label in `labels.json`)**: When the rule won the `Arbiter` priority/tie-break resolution, did it match the single human category label?

| Rule ID | Priority | Runs Fired (of 20) | Condition-Level TP | Condition-Level FP | Arbiter-Winner TP | Arbiter-Winner FP | Validation & Error-Injection Notes |
|---|---|---:|---:|---:|---:|---:|---|
| `gt_mismatch_v1` | `P1` | `0` | `0` *(1 in Day 36)* | `0` | `0` *(1 in Day 36)* | `0` | Labeled set has `expected_output=null`; verified in `TestWriterWrongFactsInjection` (`1/1 TP`). |
| `tool_failure_v1` | `P2` | `2` (`3` matches) | `2` | `0` | `2` | `0` | Fired on `run_lbl_execution_01` and `03` (`2/2 TP`, 0 FP). Also verified in `TestToolTimeoutInjection`. |
| `missing_tool_output_v1` | `P2` | `2` | `2` | `0` | `2` | `0` | Fired on `run_lbl_execution_02` and `04` (`2/2 TP`, 0 FP). Also verified in `TestToolTimeoutInjection`. |
| `researcher_quality_v1` | `P2` | `0` | `0` | `0` | `0` | `0` | Guarded by `not has_res_exec_failure` so researcher tool crashes are blamed on `execution` rather than `reasoning` (0 FP). |
| `hallucination_v1` | `P2` | `8` | `8` | `0` | `4` | `4` | Fired on all 4 `reasoning_failure` runs (`4 TP`) and all 4 dual-fault `verification_failure` runs where writer also inflated entities (`+12` to `+15`). Wins `P2` tie-break (`'h' < 'v'`). |
| `information_loss_v1` | `P2` | `8` | `8` | `0` | `0` | `0` | Fired as supporting `WARNING` evidence on the 8 entity-gain traces; ranks after `hallucination_v1` (`'h' < 'i'`) in `P2` tie-break. |
| `verifier_passthrough_v1` | `P2` | `8` | `8` | `0` | `0` *(2 in Day 36)* | `0` | **100% recall (`4/4`)** on `verification_failure` runs; superseded at Arbiter single-winner level only when `hallucination_v1` co-fires. Wins `P2` directly (`2/2 TP`) in `TestAlwaysApproveVerifierInjection`. |
| `claim_drift_v1` | `P2` | `0` | `0` *(2 in Day 22)* | `0` | `0` | `0` | Requires extracted `claims` list (`Day 20`); verified in `tests/test_consistency_validator.py::TestClaimDrift`. |
| `skipped_step_v1` | `P3` | `8` | `8` | `0` | `4` | `0` | Wins `P3` on all 4 `workflow_failure` runs (`4/4 TP`, 0 FP). Also fires as secondary `P3` evidence on the 4 `execution_failure` runs (correctly outranked by `P2` execution rules). |
| `wrong_order_v1` | `P3` | `0` | `0` *(3 in Day 21)* | `0` | `0` | `0` | Verified in `tests/test_workflow_validator.py` (0 FP on labeled set). |
| `STAT-LAT` | `P4` | `0` | `0` *(1 in Day 28)* | `0` | `0` *(1 in Day 28)* | `0` | 0 FP on the 20 normal-latency labeled runs; caught the `5.5x` latency spike (`Z > 4.2`) in `scripts/run_long_tail_test.py`. |
| `STAT-TOK` | `P4` | `0` | `0` *(1 in Day 28)* | `0` | `0` *(1 in Day 28)* | `0` | 0 FP on the 20 normal-token labeled runs; caught the `6.0x` token spike (`Z > 4.2`) in `scripts/run_long_tail_test.py`. |

---

## 3. Summary of Rule Engine Precision & Recall

* **Zero False Positives on Clean Runs (`pass`)**: Across `run_lbl_pass_01`–`04`, **0 rules fired** (`100%` specificity).
* **Condition-Level Precision**: **100% (`37/37` rule match instances)** — every rule that fired on the 20-run benchmark corresponded to an actual injected structural, tool, or entity-count fault in the trace.
* **Arbiter-Winner Attribution Accuracy**: **16 / 20 (`80.0%`)** — exceeds the $\ge 75\%$ validation target.
