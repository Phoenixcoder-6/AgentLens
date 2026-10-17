"""
tests/test_span_dag.py — Day 52 OpenInference Span DAG Schema & Utilities Test Suite
=====================================================================================
Tests for:
    - TraceSpan Pydantic v2 schema validation, serialization, and deserialization
    - Span DAG hierarchy (parent-child links, depth calculation, tree construction)
    - Parallel execution forks (Coordinator -> 3 parallel worker agents)
    - Cyclic reflection loops (Writer -> Critic -> Writer -> Critic) with iteration_index
    - Bidirectional conversion between linear AgentStep lists and hierarchical TraceSpan DAGs
    - RunTrace dual-mode support (ensure_spans and ensure_steps)
"""

from __future__ import annotations

from datetime import UTC, datetime

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


def test_trace_span_schema_and_serialization():
    """Verify TraceSpan initialization, OpenInference attribute mapping, and JSON roundtrip."""
    span = TraceSpan(
        span_id="span_root_01",
        trace_id="run_dag_001",
        parent_span_id=None,
        name="orchestrator",
        span_type=SpanType.CHAIN,
        iteration_index=0,
        start_time=datetime(2026, 10, 18, 10, 0, 0, tzinfo=UTC),
        end_time=datetime(2026, 10, 18, 10, 0, 2, tzinfo=UTC),
        latency_ms=2000.0,
        status=StepStatus.SUCCESS,
        input={"goal": "Synthesize distributed AI report"},
        output={"summary": "Distributed AI report completed"},
        tokens=TokenUsage(prompt=300, completion=150, total=450),
        attributes={"openinference.span.kind": "chain", "framework": "agentlens_v2"},
    )

    assert span.span_id == "span_root_01"
    assert span.span_type == SpanType.CHAIN
    assert span.iteration_index == 0
    assert span.attributes["framework"] == "agentlens_v2"

    # JSON Roundtrip
    json_str = span.model_dump_json()
    reconstructed = TraceSpan.model_validate_json(json_str)
    assert reconstructed.span_id == span.span_id
    assert reconstructed.trace_id == span.trace_id
    assert reconstructed.tokens.total == 450
    assert reconstructed.attributes["openinference.span.kind"] == "chain"


def test_span_hierarchy_and_tree_construction():
    """Verify parent-child DAG traversal, depth calculation, and nested tree building."""
    root = TraceSpan(
        span_id="span_wf",
        trace_id="run_dag_002",
        name="root_pipeline",
        span_type=SpanType.CHAIN,
    )
    child_agent1 = TraceSpan(
        span_id="span_a1",
        trace_id="run_dag_002",
        parent_span_id="span_wf",
        name="researcher",
        span_type=SpanType.AGENT,
    )
    child_tool1 = TraceSpan(
        span_id="span_t1",
        trace_id="run_dag_002",
        parent_span_id="span_a1",
        name="web_search",
        span_type=SpanType.TOOL,
    )
    child_agent2 = TraceSpan(
        span_id="span_a2",
        trace_id="run_dag_002",
        parent_span_id="span_wf",
        name="writer",
        span_type=SpanType.AGENT,
    )

    spans = [root, child_agent1, child_tool1, child_agent2]

    # Roots & Children
    roots = get_root_spans(spans)
    assert len(roots) == 1
    assert roots[0].span_id == "span_wf"

    wf_children = get_child_spans(spans, "span_wf")
    assert len(wf_children) == 2
    assert {c.span_id for c in wf_children} == {"span_a1", "span_a2"}

    a1_children = get_child_spans(spans, "span_a1")
    assert len(a1_children) == 1
    assert a1_children[0].span_id == "span_t1"

    # Depths
    assert calculate_span_depth(spans, "span_wf") == 0
    assert calculate_span_depth(spans, "span_a1") == 1
    assert calculate_span_depth(spans, "span_t1") == 2
    assert calculate_span_depth(spans, "span_a2") == 1

    # Nested tree structure
    tree = build_span_tree(spans)
    assert len(tree) == 1
    root_node = tree[0]
    assert root_node["span_id"] == "span_wf"
    assert len(root_node["children"]) == 2

    # Verify tool is nested inside researcher
    a1_node = next(n for n in root_node["children"] if n["span_id"] == "span_a1")
    assert len(a1_node["children"]) == 1
    assert a1_node["children"][0]["span_id"] == "span_t1"


def test_parallel_fork_branches():
    """Verify modeling of parallel execution forks (3 agents executing under one coordinator)."""
    trace_id = "run_fork_001"
    coordinator = TraceSpan(
        span_id="span_coord",
        trace_id=trace_id,
        name="Coordinator",
        span_type=SpanType.AGENT,
    )

    # 3 parallel workers
    worker_web = TraceSpan(
        span_id="span_w_web",
        trace_id=trace_id,
        parent_span_id="span_coord",
        name="WebResearcher",
        span_type=SpanType.AGENT,
        iteration_index=0,
    )
    worker_code = TraceSpan(
        span_id="span_w_code",
        trace_id=trace_id,
        parent_span_id="span_coord",
        name="CodeResearcher",
        span_type=SpanType.AGENT,
        iteration_index=0,
    )
    worker_doc = TraceSpan(
        span_id="span_w_doc",
        trace_id=trace_id,
        parent_span_id="span_coord",
        name="DocResearcher",
        span_type=SpanType.AGENT,
        iteration_index=0,
    )

    aggregator = TraceSpan(
        span_id="span_agg",
        trace_id=trace_id,
        parent_span_id="span_coord",
        name="Aggregator",
        span_type=SpanType.AGENT,
        iteration_index=0,
    )

    spans = [coordinator, worker_web, worker_code, worker_doc, aggregator]

    children = get_child_spans(spans, "span_coord")
    assert len(children) == 4
    worker_names = {c.name for c in children}
    assert worker_names == {"WebResearcher", "CodeResearcher", "DocResearcher", "Aggregator"}

    for w in [worker_web, worker_code, worker_doc, aggregator]:
        assert calculate_span_depth(spans, w.span_id) == 1


def test_reflection_loop_cycles():
    """Verify modeling of cyclic reflection loops (Writer -> Critic -> Writer -> Critic -> Writer)."""
    trace_id = "run_cycle_001"

    spans = [
        TraceSpan(
            span_id="span_w_0",
            trace_id=trace_id,
            name="Writer",
            span_type=SpanType.AGENT,
            iteration_index=0,
            output="Draft 1",
        ),
        TraceSpan(
            span_id="span_c_0",
            trace_id=trace_id,
            parent_span_id="span_w_0",
            name="Critic",
            span_type=SpanType.AGENT,
            iteration_index=0,
            output="Critique 1: Needs more data",
        ),
        TraceSpan(
            span_id="span_w_1",
            trace_id=trace_id,
            parent_span_id="span_c_0",
            name="Writer",
            span_type=SpanType.AGENT,
            iteration_index=1,
            output="Draft 2 with data",
        ),
        TraceSpan(
            span_id="span_c_1",
            trace_id=trace_id,
            parent_span_id="span_w_1",
            name="Critic",
            span_type=SpanType.AGENT,
            iteration_index=1,
            output="Critique 2: Minor grammar fixes",
        ),
        TraceSpan(
            span_id="span_w_2",
            trace_id=trace_id,
            parent_span_id="span_c_1",
            name="Writer",
            span_type=SpanType.AGENT,
            iteration_index=2,
            output="Draft 3 final",
        ),
    ]

    cycles = find_reflection_cycles(spans)
    assert "Writer" in cycles
    assert "Critic" in cycles

    assert len(cycles["Writer"]) == 3
    assert [s.iteration_index for s in cycles["Writer"]] == [0, 1, 2]

    assert len(cycles["Critic"]) == 2
    assert [s.iteration_index for s in cycles["Critic"]] == [0, 1]


def test_bidirectional_conversion_steps_and_spans():
    """Verify lossless bidirectional transformation: AgentStep list <-> TraceSpan DAG."""
    run_id = "run_conv_001"
    steps = [
        AgentStep(
            run_id=run_id,
            step=1,
            agent="Researcher",
            node_type=NodeType.LLM,
            input="Find latest developments",
            output="Found 3 developments",
            latency_ms=300.0,
            tokens=TokenUsage(prompt=100, completion=50, total=150),
            tool_calls=[{"name": "arxiv_search", "args": {"query": "AI"}, "result": "paper 1"}],
            handoff=HandoffState(output_state={"findings": ["paper 1"]}),
        ),
        AgentStep(
            run_id=run_id,
            step=2,
            agent="Writer",
            node_type=NodeType.LLM,
            input="Summarize findings",
            output="Summary report",
            latency_ms=500.0,
            tokens=TokenUsage(prompt=200, completion=100, total=300),
            parent_step="1",
            handoff=HandoffState(
                input_state={"findings": ["paper 1"]},
                output_state={"summary": "Summary report"},
            ),
        ),
    ]

    # Convert Steps -> Spans
    spans = steps_to_spans(steps, run_id=run_id)
    assert len(spans) == 3  # Researcher, arxiv_search tool, Writer

    agent_spans = [s for s in spans if s.span_type == SpanType.AGENT]
    tool_spans = [s for s in spans if s.span_type == SpanType.TOOL]

    assert len(agent_spans) == 2
    assert len(tool_spans) == 1
    assert tool_spans[0].name == "arxiv_search"
    assert tool_spans[0].parent_span_id == agent_spans[0].span_id

    # Convert Spans -> Steps
    reconstructed_steps = spans_to_steps(spans, run_id=run_id)
    assert len(reconstructed_steps) == 2

    s1, s2 = reconstructed_steps[0], reconstructed_steps[1]
    assert s1.agent == "Researcher"
    assert s1.tokens.total == 150
    assert len(s1.tool_calls) == 1
    assert s1.tool_calls[0]["name"] == "arxiv_search"
    assert s1.handoff.output_state == {"findings": ["paper 1"]}

    assert s2.agent == "Writer"
    assert s2.tokens.total == 300
    assert s2.parent_step == "1"


def test_runtrace_dual_mode_support():
    """Verify RunTrace seamlessly supports both linear steps and hierarchical spans."""
    # 1. Start with steps only
    trace = RunTrace(
        run_id="run_dual_001",
        workflow="DualModePipeline",
        steps=[
            AgentStep(
                run_id="run_dual_001",
                step=1,
                agent="Ingester",
                output="Data ready",
            ),
            AgentStep(
                run_id="run_dual_001",
                step=2,
                agent="Processor",
                output="Processed",
                parent_step="1",
            ),
        ],
    )
    assert len(trace.spans) == 0

    # Ensure spans creates the DAG
    spans = trace.ensure_spans()
    assert len(spans) == 2
    assert trace.spans[0].name == "Ingester"
    assert trace.spans[1].name == "Processor"

    # 2. Start with spans only
    trace2 = RunTrace(
        run_id="run_dual_002",
        workflow="SpansOnlyPipeline",
        spans=[
            TraceSpan(
                span_id="sp_1",
                trace_id="run_dual_002",
                name="Planner",
                span_type=SpanType.AGENT,
                output="Plan created",
            ),
            TraceSpan(
                span_id="sp_2",
                trace_id="run_dual_002",
                parent_span_id="sp_1",
                name="Executor",
                span_type=SpanType.AGENT,
                output="Task executed",
            ),
        ],
    )
    assert len(trace2.steps) == 0

    # Ensure steps reconstructs linear steps
    steps2 = trace2.ensure_steps()
    assert len(steps2) == 2
    assert trace2.steps[0].agent == "Planner"
    assert trace2.steps[1].agent == "Executor"
