"""
tests/test_causal_fault_tree.py — Day 62: Causal Fault Propagation Graph Tests
================================================================================
Unit and integration tests certifying the CausalGraphEngine (Day 62).

Tests:
  1. Clean pipeline run: all nodes marked HEALTHY.
  2. Classic 3-agent cascade: Researcher (Root) -> Writer (Cascade) -> Verifier (Escaped).
  3. Tool failure cascade: Tool crash leading to downstream agent cascade.
  4. Parallel fork isolation: Error in one branch does not taint healthy parallel branches.
  5. Cyclic reflection cascade: Loop iteration 1 fault propagates to iteration 2.
  6. Multiple independent root causes in disjoint execution branches.
  7. Integration with Arbiter verdict and evidence records.
  8. Graph serialization, round-trip model validation, and ASCII tree formatting.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyzers.arbiter import Arbiter  # noqa: E402
from analyzers.causal import (  # noqa: E402
    CausalChain,
    CausalGraph,
    CausalGraphEngine,
    CausalNode,
    CausalNodeState,
)
from schema.models import (  # noqa: E402
    AgentStep,
    EvidenceRecord,
    EvidenceSource,
    FailureCategory,
    RuleMatch,
    RuleSeverity,
    RunTrace,
    StepStatus,
    TokenUsage,
)
from schema.span import SpanType, TraceSpan  # noqa: E402


def _make_step(
    step_num: int,
    agent: str,
    status: StepStatus = StepStatus.SUCCESS,
    input: str = "input data",
    output: str = "output data",
    error: str | None = None,
) -> AgentStep:
    return AgentStep(
        run_id="run_causal_test",
        step=step_num,
        agent=agent,
        status=status,
        input=input,
        output=output,
        error=error,
        timestamp=datetime.now(UTC),
        tokens=TokenUsage(prompt=100, completion=50, total=150),
        latency_ms=120.0,
    )


class TestCausalFaultTree:
    """Comprehensive test suite for CausalGraphEngine."""

    def test_clean_pipeline_run_all_healthy(self) -> None:
        """Verify that a successful run with zero rule violations results in 100% HEALTHY nodes."""
        trace = RunTrace(
            run_id="clean_run_01",
            workflow="research_pipeline",
            status=StepStatus.SUCCESS,
            timestamp=datetime.now(UTC),
            steps=[
                _make_step(1, "researcher"),
                _make_step(2, "writer"),
                _make_step(3, "verifier"),
            ],
        )

        engine = CausalGraphEngine()
        graph = engine.build_graph(trace)

        assert len(graph.nodes) == 3
        assert len(graph.root_causes) == 0
        assert len(graph.cascades) == 0
        assert len(graph.passthroughs) == 0
        assert len(graph.healthy) == 3
        assert graph.get_root_cause() is None
        assert graph.get_impacted_node_count() == 0

        for nid in ("step_1", "step_2", "step_3"):
            assert graph.nodes[nid].state == CausalNodeState.HEALTHY

    def test_classic_3_agent_cascade(self) -> None:
        """
        Verify classic 3-agent cascade:
          - Researcher (step 1) drops sources (ROOT_CAUSE_ORIGIN)
          - Writer (step 2) produces hallucination (CASCADING_FAILURE)
          - Verifier (step 3) approves faulty draft (PASSTHROUGH_ESCAPED)
        """
        trace = RunTrace(
            run_id="cascade_run_01",
            workflow="research_pipeline",
            status=StepStatus.FAILURE,
            timestamp=datetime.now(UTC),
            steps=[
                _make_step(1, "researcher", output='{"sources": 1, "entities": ["AI"]}'),
                _make_step(2, "writer", output='{"report": "Hallucinated quantum claims"}'),
                _make_step(3, "verifier", output='{"verified": true}'),
            ],
        )

        rules = [
            RuleMatch(
                rule_id="R-IL-001",
                category=FailureCategory.REASONING,
                description="Information loss: researcher dropped 7 sources",
                severity=RuleSeverity.HIGH,
                agent="researcher",
                step=1,
            ),
            RuleMatch(
                rule_id="R-RE-001",
                category=FailureCategory.REASONING,
                description="Hallucination: invented entities",
                severity=RuleSeverity.HIGH,
                agent="writer",
                step=2,
            ),
        ]

        engine = CausalGraphEngine()
        graph = engine.build_graph(trace, rule_matches=rules)

        assert "step_1" in graph.root_causes
        assert graph.nodes["step_1"].state == CausalNodeState.ROOT_CAUSE_ORIGIN
        assert graph.nodes["step_1"].name == "researcher"

        assert "step_2" in graph.cascades
        assert graph.nodes["step_2"].state == CausalNodeState.CASCADING_FAILURE
        assert graph.nodes["step_2"].name == "writer"
        assert "step_1" in graph.nodes["step_2"].tainted_by

        assert "step_3" in graph.passthroughs
        assert graph.nodes["step_3"].state == CausalNodeState.PASSTHROUGH_ESCAPED
        assert graph.nodes["step_3"].name == "verifier"

        root = graph.get_root_cause()
        assert root is not None
        assert root.name == "researcher"
        assert len(graph.causal_chains) >= 1
        assert graph.causal_chains[0].path == ["step_1", "step_2", "step_3"]

    def test_tool_failure_cascade(self) -> None:
        """Verify that a tool span crash propagates as root cause to calling agent."""
        trace_id = f"trace_tool_{uuid4().hex[:6]}"
        agent_span = TraceSpan(
            span_id="span_agent_1",
            trace_id=trace_id,
            parent_span_id=None,
            name="DatabaseSyncAgent",
            span_type=SpanType.AGENT,
            status=StepStatus.FAILURE,
            input="sync request",
            output="partial sync",
            error="Child tool crashed",
        )
        tool_span = TraceSpan(
            span_id="span_tool_1",
            trace_id=trace_id,
            parent_span_id="span_agent_1",
            name="sql_query_executor",
            span_type=SpanType.TOOL,
            status=StepStatus.ERROR,
            input="SELECT * FROM users",
            output="",
            error="OperationalError: connection lost",
        )

        trace = RunTrace(
            run_id=trace_id,
            workflow="db_sync_flow",
            status=StepStatus.ERROR,
            timestamp=datetime.now(UTC),
            spans=[agent_span, tool_span],
        )

        rule = RuleMatch(
            rule_id="R-EX-002",
            category=FailureCategory.EXECUTION,
            description="Tool execution crashed with connection error",
            severity=RuleSeverity.CRITICAL,
            agent="sql_query_executor",
        )

        engine = CausalGraphEngine()
        graph = engine.build_graph(trace, rule_matches=[rule])

        assert "span_tool_1" in graph.root_causes or "span_agent_1" in graph.root_causes
        assert graph.get_impacted_node_count() >= 1

    def test_parallel_fork_isolation(self) -> None:
        """Verify that an error in one parallel branch leaves the parallel sibling HEALTHY."""
        trace_id = "parallel_trace_01"
        root_agent = TraceSpan(
            span_id="span_root",
            trace_id=trace_id,
            parent_span_id=None,
            name="Dispatcher",
            span_type=SpanType.AGENT,
            status=StepStatus.SUCCESS,
            input="dispatch tasks",
            output="tasks dispatched",
        )
        # Fork A: fails
        branch_a = TraceSpan(
            span_id="span_branch_a",
            trace_id=trace_id,
            parent_span_id="span_root",
            name="FinancialAnalyst",
            span_type=SpanType.AGENT,
            status=StepStatus.ERROR,
            input="analyze portfolio",
            output="",
            error="ZeroDivisionError",
        )
        # Fork B: succeeds
        branch_b = TraceSpan(
            span_id="span_branch_b",
            trace_id=trace_id,
            parent_span_id="span_root",
            name="NewsSummarizer",
            span_type=SpanType.AGENT,
            status=StepStatus.SUCCESS,
            input="summarize news",
            output="news summary",
        )

        trace = RunTrace(
            run_id=trace_id,
            workflow="parallel_flow",
            status=StepStatus.ERROR,
            timestamp=datetime.now(UTC),
            spans=[root_agent, branch_a, branch_b],
        )

        engine = CausalGraphEngine()
        graph = engine.build_graph(trace)

        # Branch A is faulty root cause
        assert "span_branch_a" in graph.root_causes
        assert graph.nodes["span_branch_a"].state == CausalNodeState.ROOT_CAUSE_ORIGIN

        # Branch B is untouched and HEALTHY
        assert "span_branch_b" in graph.healthy
        assert graph.nodes["span_branch_b"].state == CausalNodeState.HEALTHY

    def test_cyclic_reflection_cascade(self) -> None:
        """Verify cyclic reflection tracking where iteration 1 fault cascades into iteration 2."""
        trace_id = "cyclic_trace_01"
        iter1 = TraceSpan(
            span_id="span_iter_1",
            trace_id=trace_id,
            parent_span_id=None,
            name="CodeGenerator",
            iteration_index=0,
            span_type=SpanType.AGENT,
            status=StepStatus.FAILURE,
            input="generate code",
            output="syntax error line 4",
            error="SyntaxError: invalid syntax",
        )
        iter2 = TraceSpan(
            span_id="span_iter_2",
            trace_id=trace_id,
            parent_span_id=None,
            name="CodeGenerator",
            iteration_index=1,
            span_type=SpanType.AGENT,
            status=StepStatus.FAILURE,
            input="fix syntax error",
            output="still broken",
        )

        trace = RunTrace(
            run_id=trace_id,
            workflow="cyclic_flow",
            status=StepStatus.FAILURE,
            timestamp=datetime.now(UTC),
            spans=[iter1, iter2],
        )

        engine = CausalGraphEngine()
        graph = engine.build_graph(trace)

        assert "span_iter_1" in graph.root_causes
        assert graph.nodes["span_iter_1"].state == CausalNodeState.ROOT_CAUSE_ORIGIN
        assert "span_iter_2" in graph.cascades
        assert graph.nodes["span_iter_2"].state == CausalNodeState.CASCADING_FAILURE

    def test_multiple_independent_root_causes(self) -> None:
        """Verify handling of disjoint parallel tasks having independent root causes."""
        trace_id = "disjoint_trace_01"
        task1 = TraceSpan(
            span_id="task_1",
            trace_id=trace_id,
            parent_span_id=None,
            name="ServiceA",
            span_type=SpanType.AGENT,
            status=StepStatus.ERROR,
            input="",
            output="",
            error="ConnectionRefused",
        )
        task2 = TraceSpan(
            span_id="task_2",
            trace_id=trace_id,
            parent_span_id=None,
            name="ServiceB",
            span_type=SpanType.AGENT,
            status=StepStatus.ERROR,
            input="",
            output="",
            error="AuthenticationFailed",
        )

        trace = RunTrace(
            run_id=trace_id,
            workflow="disjoint_flow",
            status=StepStatus.ERROR,
            timestamp=datetime.now(UTC),
            spans=[task1, task2],
        )

        engine = CausalGraphEngine()
        graph = engine.build_graph(trace)

        assert len(graph.root_causes) == 2
        assert "task_1" in graph.root_causes
        assert "task_2" in graph.root_causes

    def test_arbiter_primary_agent_integration(self) -> None:
        """Verify that Arbiter verdict seamlessly aligns with CausalGraph root cause."""
        trace = RunTrace(
            run_id="arbiter_causal_run",
            workflow="pipeline_flow",
            status=StepStatus.FAILURE,
            timestamp=datetime.now(UTC),
            steps=[
                _make_step(1, "researcher"),
                _make_step(2, "writer"),
            ],
        )

        rule = RuleMatch(
            rule_id="R-IL-002",
            category=FailureCategory.REASONING,
            description="Dropped entities",
            agent="researcher",
            step=1,
        )
        ev = EvidenceRecord(
            source=EvidenceSource.RULE_ENGINE,
            description="Dropped entities rule match",
            rule_match=rule,
            agent="researcher",
            step=1,
            confidence=1.0,
        )

        arbiter = Arbiter()
        bundle = arbiter.run(run_id="arbiter_causal_run", evidence=[ev])
        assert bundle.primary_agent == "researcher"

        engine = CausalGraphEngine()
        graph = engine.build_graph(trace, analysis_bundle=bundle)

        root = graph.get_root_cause()
        assert root is not None
        assert root.name == bundle.primary_agent
        assert root.failure_category == bundle.primary_cause

    def test_causal_graph_serialization_and_ascii_tree(self) -> None:
        """Verify model validation, dictionary serialization, and ASCII tree output."""
        node1 = CausalNode(
            node_id="n1",
            name="researcher",
            state=CausalNodeState.ROOT_CAUSE_ORIGIN,
            failure_category=FailureCategory.REASONING,
        )
        node2 = CausalNode(
            node_id="n2",
            name="writer",
            state=CausalNodeState.CASCADING_FAILURE,
            failure_category=FailureCategory.REASONING,
            tainted_by=["n1"],
        )
        node3 = CausalNode(
            node_id="n3",
            name="verifier",
            state=CausalNodeState.PASSTHROUGH_ESCAPED,
            failure_category=FailureCategory.VERIFICATION,
            tainted_by=["n1"],
        )

        chain = CausalChain(
            root_node_id="n1",
            path=["n1", "n2", "n3"],
            escaped_node_id="n3",
            impacted_agents=["researcher", "writer", "verifier"],
            primary_category=FailureCategory.REASONING,
        )

        graph = CausalGraph(
            run_id="test_ser_run",
            nodes={"n1": node1, "n2": node2, "n3": node3},
            edges=[["n1", "n2"], ["n2", "n3"]],
            root_causes=["n1"],
            cascades=["n2"],
            passthroughs=["n3"],
            healthy=[],
            causal_chains=[chain],
            primary_root_cause_node="n1",
            summary="Test summary",
        )

        # JSON Round trip
        json_data = graph.model_dump_json()
        restored = CausalGraph.model_validate_json(json_data)
        assert restored.run_id == "test_ser_run"
        assert len(restored.nodes) == 3
        assert restored.get_impacted_node_count() == 3

        # ASCII tree formatting
        ascii_tree = graph.format_ascii_tree()
        assert "[ROOT: researcher]" in ascii_tree
        assert "[CASCADE: writer]" in ascii_tree
        assert "[ESCAPED: verifier]" in ascii_tree
