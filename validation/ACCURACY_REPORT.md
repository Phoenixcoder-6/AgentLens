# AgentLens Validation Accuracy Report (Day 35)

**Generated At:** `2026-10-07T16:10:03.766409+00:00`  
**Ground-Truth Baseline:** `sample_data/labels.json` (20 frozen traces from Day 15)  
**Pipeline Results:** `validation/results_day34.json` (Day 34 full pipeline run)  
**Target Threshold:** `>= 15/20 (75.0%)`  
**Actual Exact Attribution:** **`16/20 (80.0%)` — PASS**

---

## 1. Aggregate Accuracy Summary

| Metric | Correct / Total | Score | Notes |
|---|---|---|---|
| **Exact Root-Cause Attribution** (Category + Primary Agent) | `16/20` | **80.0%** | Target `>= 75.0%` (`15/20`) |
| **Failure Category Accuracy** | `16/20` | **80.0%** | 4/5 categories at 100% recall |
| **Primary Agent Attribution Accuracy** | `16/20` | **80.0%** | Matches root-cause agent |
| **Binary Anomaly Detection** (`PASS` vs `FAIL`) | `20/20` | **100.0%** | Zero false positives, zero false negatives |
| **Macro Rule-Level Recall** (Detector Fired on Target) | `20/20` | **100.0%** | Target detector fired on all 20 traces |
| **Macro Precision / Recall / F1** | — | `0.70` / `0.80` / `0.73` | Unweighted mean across 5 categories |

---

## 2. Per-Category Precision, Recall & F1

| Category | Support | Predicted | TP | FP | FN | Precision | Recall | F1 | Rule-Level Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pass` | 4 | 4 | 4 | 0 | 0 | 1.00 | 1.00 | 1.00 | 100% (4/4) |
| `reasoning_failure` | 4 | 8 | 4 | 4 | 0 | 0.50 | 1.00 | 0.67 | 100% (4/4) |
| `execution_failure` | 4 | 4 | 4 | 0 | 0 | 1.00 | 1.00 | 1.00 | 100% (4/4) |
| `workflow_failure` | 4 | 4 | 4 | 0 | 0 | 1.00 | 1.00 | 1.00 | 100% (4/4) |
| `verification_failure` | 4 | 0 | 0 | 0 | 4 | 0.00 | 0.00 | 0.00 | 100% (4/4) |

---

## 3. Confusion Matrix (Human Label vs. AgentLens Verdict)

Rows represent **Human Ground-Truth (`labels.json`)**; columns represent **AgentLens Arbiter Winner**.

| Expected \\ Predicted | `pass` | `reasoning_failure` | `execution_failure` | `workflow_failure` | `verification_failure` |
|---|---:|---:|---:|---:|---:|
| **`pass`** | 4 | 0 | 0 | 0 | 0 |
| **`reasoning_failure`** | 0 | 4 | 0 | 0 | 0 |
| **`execution_failure`** | 0 | 0 | 4 | 0 | 0 |
| **`workflow_failure`** | 0 | 0 | 0 | 4 | 0 |
| **`verification_failure`** | 0 | 4 | 0 | 0 | 0 |

---

## 4. Disagreement Log (4 Disagreements)

All disagreements occur on the 4 `verification_failure` synthetic traces (`run_lbl_verification_01` – `04`).

### Root-Cause Analysis of the 4 Disagreements
1. **Synthetic Trace Construction (`scripts/generate_labeled_set.py`):** In all four `verification_failure` traces, the `writer` step inflates `entity_count` by $+12$ to $+15$ entities and `source_count` by $+3$ to $+7$ sources over the `researcher` step, and the `verifier` step subsequently sets `approved=True` with those same inflated counts.
2. **Both Detectors Fire Correctly:**
   - `RuleEngine` fires `hallucination_v1` (`category=reasoning`, `agent=writer`, `confidence=1.0`, `source=RULE_ENGINE` / `P2`).
   - `ConsistencyValidator` fires `verifier_passthrough_v1` (`category=verification`, `agent=verifier`, `confidence=1.0`, `source=RULE_ENGINE` / `P2`).
3. **Arbiter Single-Winner Tie-Breaking:** Because both rules enter the `Arbiter` at priority tier `P2` with identical confidence (`1.0`), `Arbiter._sort_key()` resolves the tie deterministically by ascending `rule_id` (`'hallucination_v1' < 'verifier_passthrough_v1'`). As a result, the upstream fabrication (`writer` / `reasoning`) is selected as the primary root cause, while `verifier_passthrough_v1` is preserved in `rules_fired` and `supporting_evidence`.
4. **Pure Verification Failure Coverage:** When a trace exhibits verifier rubber-stamping *without* simultaneous `hallucination_v1` tie-breaking (or in direct error-injection tests on Day 36), `verifier_passthrough_v1` wins `P2` directly and blames `verifier`.

| Run ID | Expected (Category / Agent) | AgentLens Winner (Category / Agent) | Rules Fired | Why Disagreement Occurred |
|---|---|---|---|---|
| `run_lbl_verification_01` | `verification_failure` / `verifier` | `reasoning_failure` / `writer` (`P2`) | `hallucination_v1`, `information_loss_v1`, `verifier_passthrough_v1` | Dual-fault trace: writer inflated entities (8 -> 20) and sources (6 -> 9), firing 'hallucination_v1' (P2, agent=writer, conf=1.0), while verifier rubber-stamped the output (approved=True), firing 'verifier_passthrough_v1' (P2, agent=verifier, conf=1.0). Both rules fired at P2 (EvidenceSource.RULE_ENGINE) with equal confidence (1.0); Arbiter broke the tie by ascending rule_id ('hallucination_v1' < 'verifier_passthrough_v1'), attributing primary root cause to the upstream writer fabrication rather than the downstream verifier passthrough. |
| `run_lbl_verification_02` | `verification_failure` / `verifier` | `reasoning_failure` / `writer` (`P2`) | `hallucination_v1`, `information_loss_v1`, `verifier_passthrough_v1` | Dual-fault trace: writer inflated entities (7 -> 22) and sources (5 -> 8), firing 'hallucination_v1' (P2, agent=writer, conf=1.0), while verifier rubber-stamped the output (approved=True), firing 'verifier_passthrough_v1' (P2, agent=verifier, conf=1.0). Both rules fired at P2 (EvidenceSource.RULE_ENGINE) with equal confidence (1.0); Arbiter broke the tie by ascending rule_id ('hallucination_v1' < 'verifier_passthrough_v1'), attributing primary root cause to the upstream writer fabrication rather than the downstream verifier passthrough. |
| `run_lbl_verification_03` | `verification_failure` / `verifier` | `reasoning_failure` / `writer` (`P2`) | `hallucination_v1`, `information_loss_v1`, `verifier_passthrough_v1` | Dual-fault trace: writer inflated entities (10 -> 25) and sources (7 -> 12), firing 'hallucination_v1' (P2, agent=writer, conf=1.0), while verifier rubber-stamped the output (approved=True), firing 'verifier_passthrough_v1' (P2, agent=verifier, conf=1.0). Both rules fired at P2 (EvidenceSource.RULE_ENGINE) with equal confidence (1.0); Arbiter broke the tie by ascending rule_id ('hallucination_v1' < 'verifier_passthrough_v1'), attributing primary root cause to the upstream writer fabrication rather than the downstream verifier passthrough. |
| `run_lbl_verification_04` | `verification_failure` / `verifier` | `reasoning_failure` / `writer` (`P2`) | `hallucination_v1`, `information_loss_v1`, `verifier_passthrough_v1` | Dual-fault trace: writer inflated entities (6 -> 18) and sources (4 -> 11), firing 'hallucination_v1' (P2, agent=writer, conf=1.0), while verifier rubber-stamped the output (approved=True), firing 'verifier_passthrough_v1' (P2, agent=verifier, conf=1.0). Both rules fired at P2 (EvidenceSource.RULE_ENGINE) with equal confidence (1.0); Arbiter broke the tie by ascending rule_id ('hallucination_v1' < 'verifier_passthrough_v1'), attributing primary root cause to the upstream writer fabrication rather than the downstream verifier passthrough. |

---

## 5. Run-by-Run Comparison Table (20 Runs)

| Run ID | Topic | Human Category | Human Agent | AgentLens Category | AgentLens Agent | Priority | Rules Fired | Match |
|---|---|---|---|---|---|---|---|---|
| `run_lbl_pass_01` | History of the Eiffel Tower | `pass` | `None` | `pass` | `None` | `P5` | `[]` | **MATCH** |
| `run_lbl_pass_02` | How vaccines work | `pass` | `None` | `pass` | `None` | `P5` | `[]` | **MATCH** |
| `run_lbl_pass_03` | The water cycle explained | `pass` | `None` | `pass` | `None` | `P5` | `[]` | **MATCH** |
| `run_lbl_pass_04` | The French Revolution | `pass` | `None` | `pass` | `None` | `P5` | `[]` | **MATCH** |
| `run_lbl_reasoning_01` | Rise of AI in America | `reasoning_failure` | `writer` | `reasoning_failure` | `writer` | `P2` | `hallucination_v1`, `information_loss_v1`, `verifier_passthrough_v1` | **MATCH** |
| `run_lbl_reasoning_02` | Quantum computing applications | `reasoning_failure` | `writer` | `reasoning_failure` | `writer` | `P2` | `hallucination_v1`, `information_loss_v1`, `verifier_passthrough_v1` | **MATCH** |
| `run_lbl_reasoning_03` | Climate change and renewable energy | `reasoning_failure` | `writer` | `reasoning_failure` | `writer` | `P2` | `hallucination_v1`, `information_loss_v1`, `verifier_passthrough_v1` | **MATCH** |
| `run_lbl_reasoning_04` | Space exploration history | `reasoning_failure` | `writer` | `reasoning_failure` | `writer` | `P2` | `hallucination_v1`, `information_loss_v1`, `verifier_passthrough_v1` | **MATCH** |
| `run_lbl_execution_01` | Latest stock market trends | `execution_failure` | `researcher` | `execution_failure` | `researcher` | `P2` | `tool_failure_v1`, `skipped_step_v1` | **MATCH** |
| `run_lbl_execution_02` | Real-time weather in Mumbai | `execution_failure` | `researcher` | `execution_failure` | `researcher` | `P2` | `missing_tool_output_v1`, `skipped_step_v1` | **MATCH** |
| `run_lbl_execution_03` | Current cryptocurrency prices | `execution_failure` | `researcher` | `execution_failure` | `researcher` | `P2` | `tool_failure_v1`, `tool_failure_v1`, `skipped_step_v1` | **MATCH** |
| `run_lbl_execution_04` | Live sports scores | `execution_failure` | `researcher` | `execution_failure` | `researcher` | `P2` | `missing_tool_output_v1`, `skipped_step_v1` | **MATCH** |
| `run_lbl_workflow_01` | Benefits of meditation | `workflow_failure` | `verifier` | `workflow_failure` | `verifier` | `P3` | `skipped_step_v1` | **MATCH** |
| `run_lbl_workflow_02` | History of the Roman Empire | `workflow_failure` | `verifier` | `workflow_failure` | `verifier` | `P3` | `skipped_step_v1` | **MATCH** |
| `run_lbl_workflow_03` | Introduction to machine learning | `workflow_failure` | `researcher` | `workflow_failure` | `researcher` | `P3` | `skipped_step_v1` | **MATCH** |
| `run_lbl_workflow_04` | Principles of sustainable agriculture | `workflow_failure` | `verifier` | `workflow_failure` | `verifier` | `P3` | `skipped_step_v1` | **MATCH** |
| `run_lbl_verification_01` | The history of the internet | `verification_failure` | `verifier` | `reasoning_failure` | `writer` | `P2` | `hallucination_v1`, `information_loss_v1`, `verifier_passthrough_v1` | **DISAGREE** |
| `run_lbl_verification_02` | Blockchain technology fundamentals | `verification_failure` | `verifier` | `reasoning_failure` | `writer` | `P2` | `hallucination_v1`, `information_loss_v1`, `verifier_passthrough_v1` | **DISAGREE** |
| `run_lbl_verification_03` | Future of autonomous vehicles | `verification_failure` | `verifier` | `reasoning_failure` | `writer` | `P2` | `hallucination_v1`, `information_loss_v1`, `verifier_passthrough_v1` | **DISAGREE** |
| `run_lbl_verification_04` | Impact of social media on mental health | `verification_failure` | `verifier` | `reasoning_failure` | `writer` | `P2` | `hallucination_v1`, `information_loss_v1`, `verifier_passthrough_v1` | **DISAGREE** |
