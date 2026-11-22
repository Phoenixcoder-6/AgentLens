"""
scripts/verify_day53.py — Day 53 Verification Gate
===================================================
Phase B: Graph Aligner Upgrade for DAGs & Loops (AgentLens v2.0)

Checks:
  1. Multi-Agent Reflection Loop Alignment by (agent, iteration_index)
  2. Asymmetric Loop Iteration Isolation (MISSING_IN_B / MISSING_IN_A)
  3. Parallel Fork Branch Matching across Out-of-Order Execution
  4. Missing Parallel Fork Branch Detection
  5. Direct OpenInference TraceSpan Alignment (align_spans)
  6. Dual-Mode RunTrace Alignment & In-Place Immutability
  7. Analyzer Protocol Integration (EvidenceRecord & Workflow category)
  8. Static Analysis & Day 43 Frozen Benchmark Gate (16/20 = 80.0%)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from diff_engine import (
    AlignmentStatus,
    DiffEngine,
    GraphAligner,
    align_spans,
    align_traces,
)
from schema.models import AgentStep, EvidenceSource, FailureCategory, RunTrace, StepStatus
from schema.span import SpanType, TraceSpan

REPO_ROOT = Path(__file__).resolve().parent.parent


def make_step(
    agent: str,
    step: int,
    output: str = "",
    parent_step: str | None = None,
    iteration_index: int = 0,
) -> AgentStep:
    return AgentStep(
        run_id="run_verify_53",
        step=step,
        agent=agent,
        output=output or f"{agent} output (iter {iteration_index})",
        status=StepStatus.SUCCESS,
        parent_step=parent_step,
        iteration_index=iteration_index,
    )


def run_checks() -> bool:
    print("=" * 70)
    print("  AGENTLENS v2.0 -- DAY 53 VERIFICATION GATE")
    print("  Graph Aligner Upgrade for Multi-Agent DAGs & Reflection Loops")
    print("=" * 70)
    all_passed = True

    # ─────────────────────────────────────────────────────────────────────────
    # Check 1: Multi-Agent Reflection Loop Alignment by (agent, iteration_index)
    # ─────────────────────────────────────────────────────────────────────────
    try:
        trace_a = RunTrace(
            run_id="run_a_loop",
            workflow="reflection_loop",
            steps=[
                make_step("writer", 1, iteration_index=0),
                make_step("critic", 2, parent_step="1", iteration_index=0),
                make_step("writer", 3, parent_step="2", iteration_index=1),
                make_step("critic", 4, parent_step="3", iteration_index=1),
            ],
        )
        trace_b = RunTrace(
            run_id="run_b_loop",
            workflow="reflection_loop",
            steps=[
                make_step("writer", 1, iteration_index=0),
                make_step("critic", 2, parent_step="1", iteration_index=0),
                make_step("writer", 3, parent_step="2", iteration_index=1),
                make_step("critic", 4, parent_step="3", iteration_index=1),
            ],
        )
        res1 = align_traces(trace_a, trace_b)
        assert res1.matched_count == 4, f"Expected 4 matched steps, got {res1.matched_count}"
        assert res1.is_fully_aligned is True
        for pair in res1.pairs:
            assert pair.status == AlignmentStatus.MATCHED
            assert pair.iteration_index_a == pair.iteration_index_b
        assert res1.get_loop_iterations() == {0, 1}
        print("[PASS] Check 1: Multi-Agent Reflection Loop Alignment by (agent, iteration_index)")
    except Exception as exc:
        print(f"[FAIL] Check 1: Reflection loop alignment failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 2: Asymmetric Loop Iteration Isolation
    # ─────────────────────────────────────────────────────────────────────────
    try:
        # Trace A had 3 iterations (6 steps), Trace B had 1 iteration (2 steps)
        trace_a = RunTrace(
            run_id="run_a_3iters",
            workflow="reflection_loop",
            steps=[
                make_step("writer", 1, iteration_index=0),
                make_step("critic", 2, parent_step="1", iteration_index=0),
                make_step("writer", 3, parent_step="2", iteration_index=1),
                make_step("critic", 4, parent_step="3", iteration_index=1),
                make_step("writer", 5, parent_step="4", iteration_index=2),
                make_step("critic", 6, parent_step="5", iteration_index=2),
            ],
        )
        trace_b = RunTrace(
            run_id="run_b_1iter",
            workflow="reflection_loop",
            steps=[
                make_step("writer", 1, iteration_index=0),
                make_step("critic", 2, parent_step="1", iteration_index=0),
            ],
        )
        res2 = align_traces(trace_a, trace_b)
        assert res2.matched_count == 2, f"Expected 2 matched, got {res2.matched_count}"
        assert res2.missing_in_b_count == 4, (
            f"Expected 4 missing in B, got {res2.missing_in_b_count}"
        )
        assert res2.missing_in_a_count == 0
        assert res2.is_fully_aligned is False

        # Verify missing pairs belong to iterations 1 and 2
        missing = res2.get_missing_steps()
        assert len(missing) == 4
        iter1_missing = [p for p in missing if p.iteration_index_a == 1]
        iter2_missing = [p for p in missing if p.iteration_index_a == 2]
        assert len(iter1_missing) == 2
        assert len(iter2_missing) == 2
        print("[PASS] Check 2: Asymmetric Loop Iteration Isolation (iterations 1 and 2 isolated)")
    except Exception as exc:
        print(f"[FAIL] Check 2: Asymmetric loop isolation failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 3: Parallel Fork Branch Matching across Out-of-Order Execution
    # ─────────────────────────────────────────────────────────────────────────
    try:
        # Coordinator spawns 3 parallel workers: search, db, calc
        trace_a = RunTrace(
            run_id="run_a_fork",
            workflow="parallel_fork",
            steps=[
                make_step("coordinator", 1),
                make_step("search_worker", 2, parent_step="1"),
                make_step("db_worker", 3, parent_step="1"),
                make_step("calc_worker", 4, parent_step="1"),
                make_step("synthesizer", 5, parent_step="1"),
            ],
        )
        # Trace B finished in different thread completion order: db, calc, search
        trace_b = RunTrace(
            run_id="run_b_fork",
            workflow="parallel_fork",
            steps=[
                make_step("coordinator", 1),
                make_step("db_worker", 2, parent_step="1"),
                make_step("calc_worker", 3, parent_step="1"),
                make_step("search_worker", 4, parent_step="1"),
                make_step("synthesizer", 5, parent_step="1"),
            ],
        )
        res3 = align_traces(trace_a, trace_b)
        assert res3.matched_count == 5, f"Expected 5 matched, got {res3.matched_count}"
        assert res3.missing_in_a_count == 0
        assert res3.missing_in_b_count == 0
        assert res3.is_fully_aligned is True

        # Assert individual steps preserved original step indices
        search_pair = res3.get_pairs_by_agent("search_worker")[0]
        assert search_pair.step_index_a == 2
        assert search_pair.step_index_b == 4
        print("[PASS] Check 3: Parallel Fork Branch Matching across Out-of-Order Execution")
    except Exception as exc:
        print(f"[FAIL] Check 3: Parallel fork matching failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 4: Missing Parallel Fork Branch Detection
    # ─────────────────────────────────────────────────────────────────────────
    try:
        # Trace A had 3 workers, Trace B omitted db_worker
        trace_a = RunTrace(
            run_id="run_a_fork",
            workflow="parallel_fork",
            steps=[
                make_step("coordinator", 1),
                make_step("search_worker", 2, parent_step="1"),
                make_step("db_worker", 3, parent_step="1"),
                make_step("calc_worker", 4, parent_step="1"),
            ],
        )
        trace_b = RunTrace(
            run_id="run_b_fork_missing",
            workflow="parallel_fork",
            steps=[
                make_step("coordinator", 1),
                make_step("search_worker", 2, parent_step="1"),
                make_step("calc_worker", 3, parent_step="1"),
            ],
        )
        res4 = align_traces(trace_a, trace_b)
        assert res4.matched_count == 3
        assert res4.missing_in_b_count == 1
        assert res4.missing_in_a_count == 0
        missing = res4.get_missing_steps()
        assert len(missing) == 1
        assert missing[0].agent == "db_worker"
        assert missing[0].status == AlignmentStatus.MISSING_IN_B
        print("[PASS] Check 4: Missing Parallel Fork Branch Detection (db_worker isolated)")
    except Exception as exc:
        print(f"[FAIL] Check 4: Missing parallel branch detection failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 5: Direct OpenInference TraceSpan Alignment (align_spans)
    # ─────────────────────────────────────────────────────────────────────────
    try:
        spans_a = [
            TraceSpan(span_id="s1", trace_id="run_a", name="planner", span_type=SpanType.AGENT),
            TraceSpan(
                span_id="s2",
                trace_id="run_a",
                parent_span_id="s1",
                name="analyst_a",
                span_type=SpanType.AGENT,
            ),
            TraceSpan(
                span_id="s3",
                trace_id="run_a",
                parent_span_id="s1",
                name="analyst_b",
                span_type=SpanType.AGENT,
            ),
        ]
        spans_b = [
            TraceSpan(span_id="sb1", trace_id="run_b", name="planner", span_type=SpanType.AGENT),
            TraceSpan(
                span_id="sb2",
                trace_id="run_b",
                parent_span_id="sb1",
                name="analyst_b",
                span_type=SpanType.AGENT,
            ),
            TraceSpan(
                span_id="sb3",
                trace_id="run_b",
                parent_span_id="sb1",
                name="analyst_a",
                span_type=SpanType.AGENT,
            ),
        ]
        res5 = align_spans(spans_a, spans_b, run_id_a="run_a", run_id_b="run_b")
        assert res5.matched_count == 3
        assert res5.is_fully_aligned is True
        analyst_a_pair = res5.get_pairs_by_agent("analyst_a")[0]
        assert analyst_a_pair.span_id_a == "s2"
        assert analyst_a_pair.span_id_b == "sb3"
        print("[PASS] Check 5: Direct OpenInference TraceSpan Alignment (align_spans)")
    except Exception as exc:
        print(f"[FAIL] Check 5: Direct span alignment failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 6: Dual-Mode RunTrace Alignment & In-Place Immutability
    # ─────────────────────────────────────────────────────────────────────────
    try:
        # Trace A has only spans, Trace B has only steps
        spans_a = [
            TraceSpan(
                span_id="s10", trace_id="run_spans", name="fetcher", span_type=SpanType.AGENT
            ),
            TraceSpan(
                span_id="s11",
                trace_id="run_spans",
                parent_span_id="s10",
                name="processor",
                span_type=SpanType.AGENT,
            ),
        ]
        trace_a = RunTrace(run_id="run_spans", workflow="wf", spans=spans_a)
        snap_a_before = trace_a.model_dump()

        steps_b = [
            make_step("fetcher", 1),
            make_step("processor", 2, parent_step="1"),
        ]
        trace_b = RunTrace(run_id="run_steps", workflow="wf", steps=steps_b)
        snap_b_before = trace_b.model_dump()

        res6 = align_traces(trace_a, trace_b)
        assert res6.matched_count == 2
        assert res6.is_fully_aligned is True

        # Assert immutability: align_traces must NOT mutate input traces in-place
        assert trace_a.model_dump() == snap_a_before, "trace_a was mutated in-place!"
        assert trace_b.model_dump() == snap_b_before, "trace_b was mutated in-place!"
        print("[PASS] Check 6: Dual-Mode RunTrace Alignment & In-Place Immutability")
    except Exception as exc:
        print(f"[FAIL] Check 6: Dual-mode alignment / immutability failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 7: Analyzer Protocol Integration
    # ─────────────────────────────────────────────────────────────────────────
    try:
        trace_ref = RunTrace(
            run_id="baseline",
            workflow="wf",
            steps=[
                make_step("writer", 1, iteration_index=0),
                make_step("critic", 2, iteration_index=0),
                make_step("writer", 3, iteration_index=1),
                make_step("critic", 4, iteration_index=1),
            ],
        )
        trace_curr = RunTrace(
            run_id="current",
            workflow="wf",
            steps=[
                make_step("writer", 1, iteration_index=0),
                make_step("critic", 2, iteration_index=0),
            ],
        )
        aligner = GraphAligner(baseline_trace=trace_ref)
        an_res = aligner.analyze(trace_curr)
        assert an_res.skipped is False
        assert len(an_res.evidence) == 2, f"Expected 2 evidence records, got {len(an_res.evidence)}"
        assert all(e.source == EvidenceSource.DIFF_ENGINE for e in an_res.evidence)
        assert all(e.rule_match.category == FailureCategory.WORKFLOW for e in an_res.evidence)
        assert any("iteration 1" in e.description for e in an_res.evidence)

        # Unified DiffEngine integration
        diff_engine = DiffEngine(baseline_trace=trace_ref)
        diff_res = diff_engine.analyze(trace_curr)
        assert diff_res.skipped is False
        assert len(diff_res.evidence) >= 2
        print("[PASS] Check 7: Analyzer Protocol Integration (EvidenceRecord & WORKFLOW category)")
    except Exception as exc:
        print(f"[FAIL] Check 7: Analyzer protocol integration failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 8: Static Analysis & Day 43 Frozen Benchmark Gate
    # ─────────────────────────────────────────────────────────────────────────
    try:
        # Static analysis check
        py_exe = sys.executable
        ruff_res = subprocess.run(
            [
                py_exe,
                "-m",
                "ruff",
                "check",
                "diff_engine/aligner.py",
                "tests/test_diff_engine_dag.py",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert ruff_res.returncode == 0, f"Ruff failed: {ruff_res.stdout}\n{ruff_res.stderr}"

        # Day 43 frozen benchmark regression check
        reg_cmd = [py_exe, "scripts/run_day43_regression.py"]
        reg_res = subprocess.run(reg_cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
        assert "16/20 (80.0%)" in reg_res.stdout and "Overall Regression Gate" in reg_res.stdout, (
            f"Day 43 benchmark gate check failed:\n{reg_res.stdout}"
        )
        print(
            "[PASS] Check 8: Static Analysis (Ruff clean) & Day 43 Frozen Benchmark Preserved (16/20 = 80.0%)"
        )
    except Exception as exc:
        print(f"[FAIL] Check 8: Regression gate / static analysis failed: {exc}")
        all_passed = False

    print("=" * 70)
    if all_passed:
        print(">>> ALL 8 DAY 53 VERIFICATION CHECKS PASSED SUCCESSFULLY! <<<")
        print(">>> DAY 53 GRAPH ALIGNER UPGRADE CERTIFIED <<<")
    else:
        print(">>> SOME DAY 53 VERIFICATION CHECKS FAILED! <<<")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
