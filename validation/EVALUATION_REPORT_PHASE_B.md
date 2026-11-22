# AgentLens v2.0 — Phase B Accuracy Evaluation Report

**Generated:** 2026-10-18T20:27:22.289980+00:00  
**Status:** PASSED (Certified)  
**Exact Attribution Accuracy:** 5/5 (100.0%)  
**Binary Classification Accuracy:** 5/5 (100.0%)  
**Target Threshold:** >= 90%  

## Evaluation Summary Table

| Run ID | Scenario | Expected | Predicted | Rules Fired | Cause | Primary Agent | Result |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: |
| `trace_v2_loop_reflection_pass` | loop_reflection | PASS (P5) | PASS (P5) | `None` | unknown | None | **PASS** |
| `trace_v2_parallel_fork_divergence` | parallel_fork | FAIL (P2) | FAIL (P2) | `tool_failure_v1` | execution | financial_analyst | **PASS** |
| `trace_v2_entity_substitution_swap` | entity_substitution | FAIL (P2) | FAIL (P2) | `hallucination_v2` | reasoning | writer | **PASS** |
| `trace_v2_semantic_contradiction` | semantic_contradiction | FAIL (P2) | FAIL (P2) | `contradiction_v1` | reasoning | writer | **PASS** |
| `trace_v2_loop_deadlock_unresolved` | loop_deadlock | FAIL (P2) | FAIL (P2) | `researcher_quality_v1` | reasoning | researcher | **PASS** |

## Deep Component Evaluation Details

1. **Reflection Loop Tracking (`trace_v2_loop_reflection_pass`):**
   - Validated cyclic span alignment by `(agent, iteration_index)`.
   - Verified loop convergence to PASS (P5) with 0 false positive rule firings.

2. **Parallel Fork Concurrency (`trace_v2_parallel_fork_divergence`):**
   - Verified DAG parallel branching from coordinator into market & financial analysts.
   - Root cause attributed correctly to `financial_analyst` via `tool_failure_v1` (P2).

3. **Semantic Entity Substitution (`trace_v2_entity_substitution_swap`):**
   - Verified `EntityTracker` flagged 3-for-3 entity replacement (Tech -> Pharma).
   - Fired `hallucination_v2` with exact attribution to `writer` (P2).

4. **Cross-Encoder NLI Contradiction (`trace_v2_semantic_contradiction`):**
   - Verified `NLIEngine` detected direct polar relation inversion (failed -> met).
   - Fired `contradiction_v1` with exact attribution to `writer` (P2).

5. **Cyclic Loop Non-Convergence (`trace_v2_loop_deadlock_unresolved`):**
   - Verified detection of 3 failed reflection iterations with 0 sources.
   - Fired `researcher_quality_v1` with exact attribution to `researcher` (P2).

---
**Phase B Universal Core Verification Status: COMPLETE & PASSING**