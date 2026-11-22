"""
scripts/verify_day52.py — Day 52 OpenInference Span DAG Schema Verification Suite
================================================================================
Automated verification of the universal DAG span schema and reflection loop models:
1. TraceSpan schema validation & OpenInference serialization
2. Span hierarchy, parent-child DAG pointers, and tree depth calculation
3. Cyclic reflection loop iteration indexing (A -> B -> A -> B)
4. Parallel fork branch DAG modeling
5. Bidirectional conversion integrity (steps_to_spans and spans_to_steps)
6. RunTrace dual-mode compatibility (legacy linear steps + DAG spans)
7. Static analysis & strict typing certification (ruff & mypy clean)
8. Day 43 frozen regression benchmark preservation (16/20 exact attribution)
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from schema.models import (
    AgentStep,
    HandoffState,
    NodeType,
    RunTrace,
    StepStatus,
    TokenUsage,
)
from schema.span import (
    SpanType,
    TraceSpan,
    build_span_tree,
    calculate_span_depth,
    find_reflection_cycles,
    get_child_spans,
    get_root_spans,
    spans_to_steps,
    steps_to_spans,
)


def check_1_span_schema_and_serialization() -> bool:
    print("\n--- Check 1: TraceSpan Schema & OpenInference Serialization ---")
    span = TraceSpan(
        span_id="span_verify_01",
        trace_id="run_v52_001",
        parent_span_id=None,
        name="RouterAgent",
        span_type=SpanType.ROUTER,
        iteration_index=0,
        start_time=datetime(2026, 10, 18, 12, 0, 0, tzinfo=UTC),
        latency_ms=150.0,
        status=StepStatus.SUCCESS,
        input="Query router",
        output="Route to researcher",
        tokens=TokenUsage(prompt=50, completion=20, total=70),
        attributes={"openinference.span.kind": "chain", "dag_version": "2.0"},
    )

    if span.span_type != SpanType.ROUTER or span.tokens.total != 70:
        print("[FAIL] TraceSpan attributes mismatch")
        return False

    raw_json = span.model_dump_json()
    reloaded = TraceSpan.model_validate_json(raw_json)
    if reloaded.span_id != "span_verify_01" or reloaded.attributes.get("dag_version") != "2.0":
        print("[FAIL] TraceSpan JSON roundtrip failed")
        return False

    print("[PASS] TraceSpan schema, serialization, and OpenInference attributes verified.")
    return True


def check_2_span_hierarchy_and_depth() -> bool:
    print("\n--- Check 2: Span Hierarchy & DAG Depth Calculation ---")
    root = TraceSpan(
        span_id="sp_root", trace_id="run_dag_h", name="Workflow", span_type=SpanType.CHAIN
    )
    agent = TraceSpan(
        span_id="sp_agent",
        trace_id="run_dag_h",
        parent_span_id="sp_root",
        name="Agent",
        span_type=SpanType.AGENT,
    )
    tool = TraceSpan(
        span_id="sp_tool",
        trace_id="run_dag_h",
        parent_span_id="sp_agent",
        name="Tool",
        span_type=SpanType.TOOL,
    )

    spans = [root, agent, tool]

    roots = get_root_spans(spans)
    if len(roots) != 1 or roots[0].span_id != "sp_root":
        print(f"[FAIL] get_root_spans failed: {roots}")
        return False

    children = get_child_spans(spans, "sp_root")
    if len(children) != 1 or children[0].span_id != "sp_agent":
        print(f"[FAIL] get_child_spans failed: {children}")
        return False

    if calculate_span_depth(spans, "sp_root") != 0:
        print(f"[FAIL] Root depth should be 0, got {calculate_span_depth(spans, 'sp_root')}")
        return False

    if calculate_span_depth(spans, "sp_agent") != 1:
        print(f"[FAIL] Agent depth should be 1, got {calculate_span_depth(spans, 'sp_agent')}")
        return False

    if calculate_span_depth(spans, "sp_tool") != 2:
        print(f"[FAIL] Tool depth should be 2, got {calculate_span_depth(spans, 'sp_tool')}")
        return False

    tree = build_span_tree(spans)
    if (
        len(tree) != 1
        or len(tree[0]["children"]) != 1
        or len(tree[0]["children"][0]["children"]) != 1
    ):
        print(f"[FAIL] Tree nesting mismatch: {tree}")
        return False

    print("[PASS] Span hierarchy, parent-child pointers, and depth calculations verified.")
    return True


def check_3_reflection_loop_cycles() -> bool:
    print("\n--- Check 3: Reflection Loop Cycle Iteration Indexing ---")
    trace_id = "run_cycle_v52"
    spans = [
        TraceSpan(
            span_id="s1",
            trace_id=trace_id,
            name="Researcher",
            span_type=SpanType.AGENT,
            iteration_index=0,
        ),
        TraceSpan(
            span_id="s2",
            trace_id=trace_id,
            parent_span_id="s1",
            name="Reviewer",
            span_type=SpanType.AGENT,
            iteration_index=0,
        ),
        TraceSpan(
            span_id="s3",
            trace_id=trace_id,
            parent_span_id="s2",
            name="Researcher",
            span_type=SpanType.AGENT,
            iteration_index=1,
        ),
        TraceSpan(
            span_id="s4",
            trace_id=trace_id,
            parent_span_id="s3",
            name="Reviewer",
            span_type=SpanType.AGENT,
            iteration_index=1,
        ),
    ]

    cycles = find_reflection_cycles(spans)
    if "Researcher" not in cycles or "Reviewer" not in cycles:
        print(f"[FAIL] Expected Researcher and Reviewer in reflection cycles: {cycles}")
        return False

    if [s.iteration_index for s in cycles["Researcher"]] != [0, 1]:
        print(f"[FAIL] Researcher iteration indices incorrect: {cycles['Researcher']}")
        return False

    print("[PASS] Multi-agent cyclic reflection loops and iteration indices verified.")
    return True


def check_4_parallel_fork_branches() -> bool:
    print("\n--- Check 4: Parallel Fork Branch DAG Modeling ---")
    trace_id = "run_fork_v52"
    coord = TraceSpan(
        span_id="sp_coord", trace_id=trace_id, name="Coordinator", span_type=SpanType.AGENT
    )
    branch_a = TraceSpan(
        span_id="sp_a",
        trace_id=trace_id,
        parent_span_id="sp_coord",
        name="BranchA",
        span_type=SpanType.AGENT,
    )
    branch_b = TraceSpan(
        span_id="sp_b",
        trace_id=trace_id,
        parent_span_id="sp_coord",
        name="BranchB",
        span_type=SpanType.AGENT,
    )
    branch_c = TraceSpan(
        span_id="sp_c",
        trace_id=trace_id,
        parent_span_id="sp_coord",
        name="BranchC",
        span_type=SpanType.AGENT,
    )

    spans = [coord, branch_a, branch_b, branch_c]
    children = get_child_spans(spans, "sp_coord")
    if len(children) != 3:
        print(f"[FAIL] Expected 3 parallel children, got {len(children)}")
        return False

    for b in [branch_a, branch_b, branch_c]:
        if calculate_span_depth(spans, b.span_id) != 1:
            print(f"[FAIL] Expected depth 1 for parallel branch {b.name}")
            return False

    print("[PASS] Parallel execution forks and shared parent routing verified.")
    return True


def check_5_bidirectional_conversion() -> bool:
    print("\n--- Check 5: Lossless Bidirectional Conversion Integrity ---")
    run_id = "run_conv_v52"
    original_steps = [
        AgentStep(
            run_id=run_id,
            step=1,
            agent="Extractor",
            node_type=NodeType.LLM,
            input="Raw input text",
            output="Extracted entities",
            latency_ms=120.0,
            tokens=TokenUsage(prompt=80, completion=40, total=120),
            tool_calls=[{"name": "entity_matcher", "result": "match_ok"}],
            handoff=HandoffState(output_state={"entities": ["A", "B"]}),
        ),
        AgentStep(
            run_id=run_id,
            step=2,
            agent="Classifier",
            node_type=NodeType.LLM,
            input="Extracted entities",
            output="Classified labels",
            latency_ms=250.0,
            tokens=TokenUsage(prompt=150, completion=60, total=210),
            parent_step="1",
            handoff=HandoffState(
                input_state={"entities": ["A", "B"]},
                output_state={"labels": [1, 2]},
            ),
        ),
    ]

    spans = steps_to_spans(original_steps, run_id=run_id)
    if len(spans) != 3:  # 2 agents + 1 child tool
        print(f"[FAIL] Expected 3 spans (2 agents + 1 tool), got {len(spans)}")
        return False

    reconstructed_steps = spans_to_steps(spans, run_id=run_id)
    if len(reconstructed_steps) != 2:
        print(f"[FAIL] Expected 2 reconstructed steps, got {len(reconstructed_steps)}")
        return False

    s1 = reconstructed_steps[0]
    if s1.agent != "Extractor" or s1.tokens.total != 120 or len(s1.tool_calls) != 1:
        print(f"[FAIL] Step 1 attributes did not roundtrip cleanly: {s1}")
        return False

    s2 = reconstructed_steps[1]
    if s2.agent != "Classifier" or s2.tokens.total != 210 or s2.parent_step != "1":
        print(f"[FAIL] Step 2 attributes did not roundtrip cleanly: {s2}")
        return False

    print("[PASS] Bidirectional conversion (steps_to_spans <-> spans_to_steps) verified.")
    return True


def check_6_runtrace_dual_mode() -> bool:
    print("\n--- Check 6: RunTrace Dual-Mode Compatibility (steps + spans) ---")
    trace = RunTrace(
        run_id="run_dual_v52",
        workflow="DualWorkflow",
        steps=[
            AgentStep(run_id="run_dual_v52", step=1, agent="AgentAlpha", output="AlphaOut"),
            AgentStep(
                run_id="run_dual_v52", step=2, agent="AgentBeta", output="BetaOut", parent_step="1"
            ),
        ],
    )
    if len(trace.spans) != 0:
        print(f"[FAIL] Initial trace should have 0 spans, got {len(trace.spans)}")
        return False

    spans = trace.ensure_spans()
    if len(spans) != 2 or trace.spans[0].name != "AgentAlpha":
        print(f"[FAIL] ensure_spans failed to populate spans: {spans}")
        return False

    # Spans to steps
    trace_from_spans = RunTrace(
        run_id="run_dual_v52_b",
        workflow="SpansWorkflow",
        spans=[
            TraceSpan(
                span_id="sp_x",
                trace_id="run_dual_v52_b",
                name="AgentGamma",
                span_type=SpanType.AGENT,
            ),
        ],
    )
    steps = trace_from_spans.ensure_steps()
    if len(steps) != 1 or trace_from_spans.steps[0].agent != "AgentGamma":
        print(f"[FAIL] ensure_steps failed to populate steps: {steps}")
        return False

    print("[PASS] RunTrace dual-mode support verified with dynamic ensure_spans and ensure_steps.")
    return True


def check_7_static_analysis_and_typing() -> bool:
    print("\n--- Check 7: Static Analysis & Strict Typing Certification (ruff & mypy) ---")
    ruff_cmd = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        "schema/",
        "tests/test_span_dag.py",
    ]
    res_ruff = subprocess.run(ruff_cmd, capture_output=True, text=True)
    if res_ruff.returncode != 0:
        print(f"[FAIL] Ruff linting failed:\n{res_ruff.stdout}")
        return False

    mypy_cmd = [
        sys.executable,
        "-m",
        "mypy",
        "schema/span.py",
        "tests/test_span_dag.py",
    ]
    res_mypy = subprocess.run(mypy_cmd, capture_output=True, text=True)
    if res_mypy.returncode != 0:
        print(f"[FAIL] Mypy typing failed:\n{res_mypy.stdout}")
        return False

    print("[PASS] Static analysis and typing clean across schema/span.py and test suites.")
    return True


def check_8_regression_benchmark() -> bool:
    print("\n--- Check 8: Day 43 Frozen Regression Benchmark (Zero Regressions) ---")
    script_path = Path(__file__).resolve().parent / "run_day43_regression.py"
    cmd = [sys.executable, str(script_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(
            f"[FAIL] Day 43 regression failed with returncode {res.returncode}:\n{res.stdout}\n{res.stderr}"
        )
        return False

    if "16/20 (80.0%)" not in res.stdout or "Overall Regression Gate" not in res.stdout:
        print(f"[FAIL] Benchmark output missing required gate success:\n{res.stdout}")
        return False

    print(
        "[PASS] Day 43 regression benchmark preserved: 16/20 (80.0%) exact attribution, zero regressions."
    )
    return True


def main() -> int:
    print("=====================================================================")
    print("AgentLens v2.0 Day 52 Verification: OpenInference Span DAG Schema")
    print("=====================================================================")

    checks = [
        check_1_span_schema_and_serialization(),
        check_2_span_hierarchy_and_depth(),
        check_3_reflection_loop_cycles(),
        check_4_parallel_fork_branches(),
        check_5_bidirectional_conversion(),
        check_6_runtrace_dual_mode(),
        check_7_static_analysis_and_typing(),
        check_8_regression_benchmark(),
    ]

    passed = sum(1 for c in checks if c)
    total = len(checks)
    print("\n=====================================================================")
    print(f"Day 52 Verification Summary: {passed}/{total} checks PASSED")
    print("=====================================================================")

    if passed == total:
        print("\n>>> DAY 52 OPENINFERENCE SPAN DAG SCHEMA FULLY CERTIFIED <<<")
        return 0
    else:
        print(f"\n>>> DAY 52 FAILED: {total - passed} checks failed <<<")
        return 1


if __name__ == "__main__":
    sys.exit(main())
