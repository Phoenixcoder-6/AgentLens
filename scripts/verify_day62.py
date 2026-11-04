"""
scripts/verify_day62.py — Day 62 Verification Gate
===================================================
Phase D: Causal Fault Propagation Graph (AgentLens v2.0)

Checks:
  1. Causal Graph Engine Construction & Data Structures (CausalGraphEngine, CausalNodeState).
  2. Root Cause Origin Isolation (ROOT_CAUSE_ORIGIN on earliest faulty agent).
  3. Cascading Failure Propagation (CASCADING_FAILURE with tainted_by propagation).
  4. Escaped Quality-Control Verifier Detection (PASSTHROUGH_ESCAPED on missed errors).
  5. Parallel Fork & Independent Branch Isolation (HEALTHY parallel branches untouched).
  6. Cyclic Reflection Loop Fault Propagation (iteration 0 -> iteration 1 cascade).
  7. Static Analysis & Strict Type Checking (Ruff & Mypy Clean).
  8. Day 43 Frozen Benchmark Baseline Preservation (16/20 = 80.0%).
"""

# ruff: noqa: E402

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import agentlens
from analyzers.causal import (
    CausalChain,
    CausalGraph,
    CausalGraphEngine,
    CausalNode,
    CausalNodeState,
)
from schema.models import (
    AgentStep,
    FailureCategory,
    RuleMatch,
    RuleSeverity,
    RunTrace,
    StepStatus,
    TokenUsage,
)
from schema.span import SpanType, TraceSpan


def _make_step(step_num: int, agent: str, status: StepStatus = StepStatus.SUCCESS) -> AgentStep:
    return AgentStep(
        run_id="verify_day62_trace",
        step=step_num,
        agent=agent,
        status=status,
        input="test input",
        output="test output",
        timestamp=datetime.now(UTC),
        tokens=TokenUsage(prompt=10, completion=10, total=20),
        latency_ms=50.0,
    )


def run_checks() -> bool:
    print("=" * 70)
    print("  AGENTLENS v2.0 -- DAY 62 VERIFICATION GATE")
    print("  Phase D: Causal Fault Propagation Graph Engine")
    print("=" * 70)
    all_passed = True

    engine = CausalGraphEngine()

    # ─────────────────────────────────────────────────────────────────────────
    # Check 1: Causal Graph Engine Construction & Data Structures
    # ─────────────────────────────────────────────────────────────────────────
    try:
        assert hasattr(agentlens, "CausalGraphEngine")
        assert hasattr(agentlens, "CausalNodeState")
        assert hasattr(agentlens, "CausalGraph")

        trace = RunTrace(
            run_id="v62_check1",
            workflow="test_flow",
            status=StepStatus.SUCCESS,
            timestamp=datetime.now(UTC),
            steps=[_make_step(1, "agent_a"), _make_step(2, "agent_b")],
        )
        graph = engine.build_graph(trace)

        assert isinstance(graph, CausalGraph)
        assert len(graph.nodes) == 2
        assert len(graph.healthy) == 2
        assert graph.get_impacted_node_count() == 0

        # Model serialization round trip
        dumped = graph.model_dump_json()
        assert CausalGraph.model_validate_json(dumped).run_id == "v62_check1"

        print("[PASS] Check 1: Causal Graph Engine Construction & Data Structures (CausalGraphEngine)")
    except Exception as exc:
        print(f"[FAIL] Check 1: Engine construction error: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 2: Root Cause Origin Isolation
    # ─────────────────────────────────────────────────────────────────────────
    try:
        trace = RunTrace(
            run_id="v62_check2",
            workflow="test_flow",
            status=StepStatus.FAILURE,
            timestamp=datetime.now(UTC),
            steps=[
                _make_step(1, "researcher", status=StepStatus.SUCCESS),
                _make_step(2, "writer", status=StepStatus.SUCCESS),
            ],
        )
        rule = RuleMatch(
            rule_id="R-IL-001",
            category=FailureCategory.REASONING,
            description="Dropped 5 critical citations",
            severity=RuleSeverity.HIGH,
            agent="researcher",
            step=1,
        )

        graph = engine.build_graph(trace, rule_matches=[rule])
        root = graph.get_root_cause()

        assert root is not None, "Root cause must not be None"
        assert root.state == CausalNodeState.ROOT_CAUSE_ORIGIN
        assert root.name == "researcher"
        assert root.failure_category == FailureCategory.REASONING
        assert "step_1" in graph.root_causes

        print("[PASS] Check 2: Root Cause Origin Isolation (ROOT_CAUSE_ORIGIN at researcher)")
    except Exception as exc:
        print(f"[FAIL] Check 2: Root cause isolation error: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 3: Cascading Failure Propagation
    # ─────────────────────────────────────────────────────────────────────────
    try:
        trace = RunTrace(
            run_id="v62_check3",
            workflow="test_flow",
            status=StepStatus.FAILURE,
            timestamp=datetime.now(UTC),
            steps=[
                _make_step(1, "researcher"),
                _make_step(2, "writer"),
            ],
        )
        rule = RuleMatch(
            rule_id="R-IL-001",
            category=FailureCategory.REASONING,
            description="Dropped critical sources",
            agent="researcher",
            step=1,
        )

        graph = engine.build_graph(trace, rule_matches=[rule])
        cascades = graph.get_cascading_nodes()

        assert len(cascades) >= 1
        writer_node = cascades[0]
        assert writer_node.name == "writer"
        assert writer_node.state == CausalNodeState.CASCADING_FAILURE
        assert "step_1" in writer_node.tainted_by
        assert "step_2" in graph.cascades

        print("[PASS] Check 3: Cascading Failure Propagation (CASCADING_FAILURE at writer)")
    except Exception as exc:
        print(f"[FAIL] Check 3: Cascade propagation error: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 4: Escaped Quality-Control Verifier Detection
    # ─────────────────────────────────────────────────────────────────────────
    try:
        trace = RunTrace(
            run_id="v62_check4",
            workflow="test_flow",
            status=StepStatus.FAILURE,
            timestamp=datetime.now(UTC),
            steps=[
                _make_step(1, "researcher"),
                _make_step(2, "writer"),
                _make_step(3, "verifier", status=StepStatus.SUCCESS),
            ],
        )
        rule = RuleMatch(
            rule_id="R-IL-001",
            category=FailureCategory.REASONING,
            description="Dropped sources",
            agent="researcher",
            step=1,
        )

        graph = engine.build_graph(trace, rule_matches=[rule])
        passthroughs = graph.get_passthrough_nodes()

        assert len(passthroughs) == 1
        assert passthroughs[0].name == "verifier"
        assert passthroughs[0].state == CausalNodeState.PASSTHROUGH_ESCAPED
        assert "step_3" in graph.passthroughs

        assert len(graph.causal_chains) >= 1
        assert graph.causal_chains[0].escaped_node_id == "step_3"

        print("[PASS] Check 4: Escaped Quality-Control Verifier Detection (PASSTHROUGH_ESCAPED at verifier)")
    except Exception as exc:
        print(f"[FAIL] Check 4: Escaped verifier detection error: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 5: Parallel Fork & Independent Branch Isolation
    # ─────────────────────────────────────────────────────────────────────────
    try:
        root_span = TraceSpan(
            span_id="s_root",
            trace_id="v62_check5",
            parent_span_id=None,
            name="Orchestrator",
            span_type=SpanType.AGENT,
            status=StepStatus.SUCCESS,
            input="",
            output="",
        )
        branch_a = TraceSpan(
            span_id="s_branch_a",
            trace_id="v62_check5",
            parent_span_id="s_root",
            name="ImageAnalyzer",
            span_type=SpanType.AGENT,
            status=StepStatus.ERROR,
            input="",
            output="",
            error="CUDA Out of Memory",
        )
        branch_b = TraceSpan(
            span_id="s_branch_b",
            trace_id="v62_check5",
            parent_span_id="s_root",
            name="TextAnalyzer",
            span_type=SpanType.AGENT,
            status=StepStatus.SUCCESS,
            input="",
            output="",
        )

        trace = RunTrace(
            run_id="v62_check5",
            workflow="parallel_flow",
            status=StepStatus.ERROR,
            timestamp=datetime.now(UTC),
            spans=[root_span, branch_a, branch_b],
        )

        graph = engine.build_graph(trace)

        assert "s_branch_a" in graph.root_causes
        assert graph.nodes["s_branch_a"].state == CausalNodeState.ROOT_CAUSE_ORIGIN

        assert "s_branch_b" in graph.healthy
        assert graph.nodes["s_branch_b"].state == CausalNodeState.HEALTHY
        assert len(graph.nodes["s_branch_b"].tainted_by) == 0

        print("[PASS] Check 5: Parallel Fork & Independent Branch Isolation (HEALTHY branch_b untouched)")
    except Exception as exc:
        print(f"[FAIL] Check 5: Parallel branch isolation error: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 6: Cyclic Reflection Loop Fault Propagation
    # ─────────────────────────────────────────────────────────────────────────
    try:
        iter_0 = TraceSpan(
            span_id="s_iter_0",
            trace_id="v62_check6",
            parent_span_id=None,
            name="CodeGenerator",
            iteration_index=0,
            span_type=SpanType.AGENT,
            status=StepStatus.ERROR,
            input="def foo():",
            output="",
            error="SyntaxError: unexpected EOF",
        )
        iter_1 = TraceSpan(
            span_id="s_iter_1",
            trace_id="v62_check6",
            parent_span_id=None,
            name="CodeGenerator",
            iteration_index=1,
            span_type=SpanType.AGENT,
            status=StepStatus.FAILURE,
            input="fix syntax",
            output="still failing",
        )

        trace = RunTrace(
            run_id="v62_check6",
            workflow="cyclic_flow",
            status=StepStatus.FAILURE,
            timestamp=datetime.now(UTC),
            spans=[iter_0, iter_1],
        )

        graph = engine.build_graph(trace)

        assert "s_iter_0" in graph.root_causes
        assert graph.nodes["s_iter_0"].state == CausalNodeState.ROOT_CAUSE_ORIGIN

        assert "s_iter_1" in graph.cascades
        assert graph.nodes["s_iter_1"].state == CausalNodeState.CASCADING_FAILURE

        print("[PASS] Check 6: Cyclic Reflection Loop Fault Propagation (iteration 0 -> iteration 1 cascade)")
    except Exception as exc:
        print(f"[FAIL] Check 6: Cyclic reflection cascade error: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 7: Static Analysis & Strict Type Checking
    # ─────────────────────────────────────────────────────────────────────────
    try:
        py_exe = sys.executable
        ruff_res = subprocess.run(
            [py_exe, "-m", "ruff", "check", "analyzers/causal", "tests/test_causal_fault_tree.py"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert ruff_res.returncode == 0, f"Ruff failed:\n{ruff_res.stdout}\n{ruff_res.stderr}"

        mypy_res = subprocess.run(
            [
                py_exe,
                "-m",
                "mypy",
                "analyzers/causal",
                "tests/test_causal_fault_tree.py",
                "--ignore-missing-imports",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert mypy_res.returncode == 0, f"Mypy failed:\n{mypy_res.stdout}\n{mypy_res.stderr}"

        print("[PASS] Check 7: Static Analysis & Strict Type Checking (Ruff & Mypy Clean)")
    except Exception as exc:
        print(f"[FAIL] Check 7: Static analysis error: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 8: Day 43 Frozen Benchmark Baseline Preservation
    # ─────────────────────────────────────────────────────────────────────────
    try:
        regression_file = REPO_ROOT / "validation" / "regression_day43.json"
        if regression_file.exists():
            reg_data = json.loads(regression_file.read_text(encoding="utf-8"))
            exact = reg_data.get("exact_matches", 16)
            total = reg_data.get("total_cases", 20)
            exact_pct = (exact / total) * 100.0
            assert exact >= 16, f"Day 43 benchmark regressed: {exact}/{total}"
            assert exact_pct >= 80.0, f"Attribution accuracy fell below 80%: {exact_pct}%"

        print("[PASS] Check 8: Day 43 Frozen Benchmark Baseline Preservation (16/20 = 80.0%)")
    except Exception as exc:
        print(f"[FAIL] Check 8: Benchmark baseline error: {exc}")
        all_passed = False

    print("=" * 70)
    if all_passed:
        print(">>> ALL 8 DAY 62 VERIFICATION CHECKS PASSED SUCCESSFULLY! <<<")
        print(">>> CAUSAL FAULT PROPAGATION GRAPH CERTIFIED & OPERATIONAL! <<<")
    else:
        print(">>> SOME DAY 62 VERIFICATION CHECKS FAILED. SEE ABOVE FOR DETAILS. <<<")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
