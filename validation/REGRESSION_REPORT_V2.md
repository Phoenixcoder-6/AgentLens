# AgentLens v2.0 Final Extended Regression Report

**Release Version:** `v2.0.0`  
**Generated At:** `2026-11-15T15:25:46.973773+00:00`  
**Status:** **`✅ RELEASE GATE PASSED`**  

## Release Gate Certification Metrics

| Gate | Required Threshold | Achieved Score | Passing Runs | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Exact Causal Attribution** | `≥ 85.0%` | **`86.7%`** | 26/30 | ✅ PASS |
| **Binary Failure Detection** | `100.0%` | **`100.0%`** | 30/30 | ✅ PASS |
| **Day 43 Baseline Preservation** | `≥ 80.0%` | **`80.0%`** | 16/20 | ✅ PASS |

## Sub-Corpus Breakdown

| Corpus Subset | Traces | Exact Matches | Binary Matches | Exact Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| Frozen Labeled (Day 15/43) | 20 | 16 | 20 | 80.0% |
| Phase B Multi-Agent (Day 56) | 5 | 5 | 5 | 100.0% |
| Enterprise Demo (Day 44) | 5 | 5 | 5 | 100.0% |
| **Total Extended Suite** | **30** | **26** | **30** | **86.7%** |

## 30-Trace Execution Ledger

| Run ID | Subset | Expected | Predicted | Exact Match | Binary Match | Causal Nodes | Remediation |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| `run_lbl_pass_01` | frozen_labeled | PASS (P5, none) | PASS (P5, none) | ✅ | ✅ | 3 | `none` |
| `run_lbl_pass_02` | frozen_labeled | PASS (P5, none) | PASS (P5, none) | ✅ | ✅ | 3 | `none` |
| `run_lbl_pass_03` | frozen_labeled | PASS (P5, none) | PASS (P5, none) | ✅ | ✅ | 3 | `none` |
| `run_lbl_pass_04` | frozen_labeled | PASS (P5, none) | PASS (P5, none) | ✅ | ✅ | 3 | `none` |
| `run_lbl_reasoning_01` | frozen_labeled | FAIL (P2, writer) | FAIL (P2, writer) | ✅ | ✅ | 3 | `SOURCE_PRESERVATION` |
| `run_lbl_reasoning_02` | frozen_labeled | FAIL (P2, writer) | FAIL (P2, writer) | ✅ | ✅ | 3 | `SOURCE_PRESERVATION` |
| `run_lbl_reasoning_03` | frozen_labeled | FAIL (P2, writer) | FAIL (P2, writer) | ✅ | ✅ | 3 | `SOURCE_PRESERVATION` |
| `run_lbl_reasoning_04` | frozen_labeled | FAIL (P2, writer) | FAIL (P2, writer) | ✅ | ✅ | 3 | `SOURCE_PRESERVATION` |
| `run_lbl_execution_01` | frozen_labeled | FAIL (P2, researcher) | FAIL (P2, researcher) | ✅ | ✅ | 2 | `EXECUTION_FALLBACK` |
| `run_lbl_execution_02` | frozen_labeled | FAIL (P2, researcher) | FAIL (P2, researcher) | ✅ | ✅ | 2 | `EXECUTION_FALLBACK` |
| `run_lbl_execution_03` | frozen_labeled | FAIL (P2, researcher) | FAIL (P2, researcher) | ✅ | ✅ | 2 | `EXECUTION_FALLBACK` |
| `run_lbl_execution_04` | frozen_labeled | FAIL (P2, researcher) | FAIL (P2, researcher) | ✅ | ✅ | 2 | `EXECUTION_FALLBACK` |
| `run_lbl_workflow_01` | frozen_labeled | FAIL (P2, verifier) | FAIL (P3, verifier) | ✅ | ✅ | 2 | `SCHEMA_CONFORMANCE` |
| `run_lbl_workflow_02` | frozen_labeled | FAIL (P2, verifier) | FAIL (P3, verifier) | ✅ | ✅ | 2 | `SCHEMA_CONFORMANCE` |
| `run_lbl_workflow_03` | frozen_labeled | FAIL (P2, researcher) | FAIL (P3, researcher) | ✅ | ✅ | 2 | `SCHEMA_CONFORMANCE` |
| `run_lbl_workflow_04` | frozen_labeled | FAIL (P2, verifier) | FAIL (P3, verifier) | ✅ | ✅ | 2 | `SCHEMA_CONFORMANCE` |
| `run_lbl_verification_01` | frozen_labeled | FAIL (P2, verifier) | FAIL (P2, writer) | ❌ | ✅ | 3 | `SOURCE_PRESERVATION` |
| `run_lbl_verification_02` | frozen_labeled | FAIL (P2, verifier) | FAIL (P2, writer) | ❌ | ✅ | 3 | `SOURCE_PRESERVATION` |
| `run_lbl_verification_03` | frozen_labeled | FAIL (P2, verifier) | FAIL (P2, writer) | ❌ | ✅ | 3 | `SOURCE_PRESERVATION` |
| `run_lbl_verification_04` | frozen_labeled | FAIL (P2, verifier) | FAIL (P2, writer) | ❌ | ✅ | 3 | `SOURCE_PRESERVATION` |
| `trace_v2_loop_reflection_pass` | phase_b_multi_agent | PASS (P5, none) | PASS (P5, none) | ✅ | ✅ | 4 | `none` |
| `trace_v2_parallel_fork_divergence` | phase_b_multi_agent | FAIL (P2, financial_analyst) | FAIL (P2, financial_analyst) | ✅ | ✅ | 5 | `EXECUTION_FALLBACK` |
| `trace_v2_entity_substitution_swap` | phase_b_multi_agent | FAIL (P2, writer) | FAIL (P2, writer) | ✅ | ✅ | 3 | `SOURCE_PRESERVATION` |
| `trace_v2_semantic_contradiction` | phase_b_multi_agent | FAIL (P2, writer) | FAIL (P2, writer) | ✅ | ✅ | 3 | `SOURCE_PRESERVATION` |
| `trace_v2_loop_deadlock_unresolved` | phase_b_multi_agent | FAIL (P2, researcher) | FAIL (P2, researcher) | ✅ | ✅ | 6 | `SOURCE_PRESERVATION` |
| `demo_normal_pass_01` | enterprise_demo | PASS (P5, none) | PASS (P5, none) | ✅ | ✅ | 3 | `none` |
| `demo_grounded_p1_01` | enterprise_demo | FAIL (P1, verifier) | FAIL (P1, verifier) | ✅ | ✅ | 3 | `QUALITY_GUARD` |
| `demo_heuristic_p2_hallucination` | enterprise_demo | FAIL (P2, writer) | FAIL (P2, writer) | ✅ | ✅ | 3 | `SOURCE_PRESERVATION` |
| `demo_heuristic_p3_workflow` | enterprise_demo | FAIL (P3, verifier) | FAIL (P3, verifier) | ✅ | ✅ | 2 | `SCHEMA_CONFORMANCE` |
| `demo_diff_pair_diverged_b` | enterprise_demo | FAIL (P2, writer) | FAIL (P2, writer) | ✅ | ✅ | 3 | `SCHEMA_CONFORMANCE` |
