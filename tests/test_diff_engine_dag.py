"""
tests/test_diff_engine_dag.py — Day 53: DAG & Loop Alignment Tests
===================================================================
Tests:
  - Reflection loop alignment: 3-iteration trace vs 1-iteration trace
  - Reflection loop alignment: asymmetric extra iterations in Trace B
  - Reflection loop alignment: missing step inside a loop iteration
  - Parallel fork alignment: out-of-order execution completion matching
  - Parallel fork alignment: missing / dropped branch detection
  - OpenInference TraceSpan direct alignment via align_spans
  - Dual-mode RunTrace alignment (steps vs spans auto-conversion)
  - DiffEngine Analyzer integration & workflow evidence generation
  - Helper methods on GraphAlignmentResult (iteration querying)
"""

from __future__ import annotations

from diff_engine import (
    AlignmentStatus,
    DiffEngine,
    GraphAligner,
    GraphAlignmentResult,
    align_spans,
    align_traces,
)
from schema.models import AgentStep, EvidenceSource, FailureCategory, RunTrace, StepStatus
from schema.span import SpanType, TraceSpan


def _make_step(
    agent: str,
    step: int,
    output: str = "",
    parent_step: str | None = None,
    iteration_index: int = 0,
) -> AgentStep:
    return AgentStep(
        run_id="test_run",
        step=step,
        agent=agent,
        output=output or f"{agent} output (iter {iteration_index})",
        status=StepStatus.SUCCESS,
        parent_step=parent_step,
        iteration_index=iteration_index,
    )


def _make_trace(run_id: str, *steps: AgentStep) -> RunTrace:
    return RunTrace(
        run_id=run_id,
        workflow="test_dag_pipeline",
        steps=list(steps),
    )


class TestReflectionLoopAlignment:
    """Test alignment of cyclic multi-agent reflection loops (Writer -> Critic -> Writer -> Critic)."""

    def test_three_iterations_vs_one_iteration(self):
        """
        Run A executed 3 iterations of (writer, critic): 6 steps total.
        Run B converged in 1 iteration of (writer, critic): 2 steps total.
        Iterations 1 and 2 should be marked MISSING_IN_B.
        """
        trace_a = _make_trace(
            "run_a_3_iters",
            _make_step("writer", 1, iteration_index=0),
            _make_step("critic", 2, parent_step="1", iteration_index=0),
            _make_step("writer", 3, parent_step="2", iteration_index=1),
            _make_step("critic", 4, parent_step="3", iteration_index=1),
            _make_step("writer", 5, parent_step="4", iteration_index=2),
            _make_step("critic", 6, parent_step="5", iteration_index=2),
        )

        trace_b = _make_trace(
            "run_b_1_iter",
            _make_step("writer", 1, iteration_index=0),
            _make_step("critic", 2, parent_step="1", iteration_index=0),
        )

        res = align_traces(trace_a, trace_b)

        assert isinstance(res, GraphAlignmentResult)
        assert res.matched_count == 2
        assert res.missing_in_a_count == 0
        assert res.missing_in_b_count == 4
        assert res.is_fully_aligned is False

        # Matched steps belong to iteration 0
        matched = res.get_matched_pairs()
        assert len(matched) == 2
        assert matched[0].agent == "writer"
        assert matched[0].iteration_index_a == 0
        assert matched[0].iteration_index_b == 0
        assert matched[1].agent == "critic"
        assert matched[1].iteration_index_a == 0
        assert matched[1].iteration_index_b == 0

        # Missing steps belong to iterations 1 and 2
        missing_b = [p for p in res.pairs if p.status == AlignmentStatus.MISSING_IN_B]
        assert len(missing_b) == 4
        assert [(p.agent, p.iteration_index_a) for p in missing_b] == [
            ("writer", 1),
            ("critic", 1),
            ("writer", 2),
            ("critic", 2),
        ]

        # Test iteration querying helpers
        iter0_pairs = res.get_pairs_by_iteration(0)
        assert len(iter0_pairs) == 2
        assert all(p.is_matched for p in iter0_pairs)

        iter1_pairs = res.get_pairs_by_iteration(1)
        assert len(iter1_pairs) == 2
        assert all(p.status == AlignmentStatus.MISSING_IN_B for p in iter1_pairs)

        assert res.get_loop_iterations() == {0, 1, 2}

    def test_asymmetric_extra_iterations_in_trace_b(self):
        """
        Run A had 1 iteration: (writer, critic).
        Run B required 2 iterations: (writer, critic, writer, critic).
        Iteration 1 in Trace B should be marked MISSING_IN_A.
        """
        trace_a = _make_trace(
            "run_a",
            _make_step("writer", 1),
            _make_step("critic", 2, parent_step="1"),
        )
        trace_b = _make_trace(
            "run_b",
            _make_step("writer", 1),
            _make_step("critic", 2, parent_step="1"),
            _make_step("writer", 3, parent_step="2"),
            _make_step("critic", 4, parent_step="3"),
        )

        res = align_traces(trace_a, trace_b)

        assert res.matched_count == 2
        assert res.missing_in_a_count == 2
        assert res.missing_in_b_count == 0
        assert res.is_fully_aligned is False

        missing_a = [p for p in res.pairs if p.status == AlignmentStatus.MISSING_IN_A]
        assert len(missing_a) == 2
        assert [(p.agent, p.iteration_index_b) for p in missing_a] == [
            ("writer", 1),
            ("critic", 1),
        ]

    def test_missing_step_inside_second_iteration(self):
        """
        Run A: (writer_0, critic_0, writer_1, critic_1)
        Run B: (writer_0, critic_0, writer_1) [critic_1 failed/omitted]
        Only critic_1 should be MISSING_IN_B.
        """
        trace_a = _make_trace(
            "run_a",
            _make_step("writer", 1),
            _make_step("critic", 2, parent_step="1"),
            _make_step("writer", 3, parent_step="2"),
            _make_step("critic", 4, parent_step="3"),
        )
        trace_b = _make_trace(
            "run_b",
            _make_step("writer", 1),
            _make_step("critic", 2, parent_step="1"),
            _make_step("writer", 3, parent_step="2"),
        )

        res = align_traces(trace_a, trace_b)

        assert res.matched_count == 3
        assert res.missing_in_b_count == 1
        assert res.missing_in_a_count == 0

        missing = res.get_missing_steps()
        assert len(missing) == 1
        assert missing[0].agent == "critic"
        assert missing[0].iteration_index_a == 1
        assert missing[0].status == AlignmentStatus.MISSING_IN_B


class TestParallelForkAlignment:
    """Test alignment of parallel execution forks with sibling branches."""

    def test_parallel_fork_out_of_order_completion(self):
        """
        Coordinator spawns 3 workers in parallel: search, db, and calc.
        Run A completed: [search, db, calc].
        Run B completed: [db, calc, search] (different thread completion order).
        Followed by synthesizer.
        All 5 steps should match 1-to-1 without false gaps.
        """
        trace_a = _make_trace(
            "run_a",
            _make_step("coordinator", 1),
            _make_step("search_worker", 2, parent_step="1"),
            _make_step("db_worker", 3, parent_step="1"),
            _make_step("calc_worker", 4, parent_step="1"),
            _make_step("synthesizer", 5, parent_step="1"),
        )

        trace_b = _make_trace(
            "run_b",
            _make_step("coordinator", 1),
            _make_step("db_worker", 2, parent_step="1"),
            _make_step("calc_worker", 3, parent_step="1"),
            _make_step("search_worker", 4, parent_step="1"),
            _make_step("synthesizer", 5, parent_step="1"),
        )

        res = align_traces(trace_a, trace_b)

        assert res.matched_count == 5
        assert res.missing_in_a_count == 0
        assert res.missing_in_b_count == 0
        assert res.is_fully_aligned is True

        # Verify step index preservation
        search_pair = res.get_pairs_by_agent("search_worker")[0]
        assert search_pair.step_index_a == 2
        assert search_pair.step_index_b == 4

        db_pair = res.get_pairs_by_agent("db_worker")[0]
        assert db_pair.step_index_a == 3
        assert db_pair.step_index_b == 2

        calc_pair = res.get_pairs_by_agent("calc_worker")[0]
        assert calc_pair.step_index_a == 4
        assert calc_pair.step_index_b == 3

    def test_parallel_fork_missing_branch(self):
        """
        Run A: Coordinator spawns [search, db, calc].
        Run B: Coordinator spawns [search, calc] (db failed/omitted).
        db_worker should be isolated as MISSING_IN_B.
        """
        trace_a = _make_trace(
            "run_a",
            _make_step("coordinator", 1),
            _make_step("search_worker", 2, parent_step="1"),
            _make_step("db_worker", 3, parent_step="1"),
            _make_step("calc_worker", 4, parent_step="1"),
        )

        trace_b = _make_trace(
            "run_b",
            _make_step("coordinator", 1),
            _make_step("search_worker", 2, parent_step="1"),
            _make_step("calc_worker", 3, parent_step="1"),
        )

        res = align_traces(trace_a, trace_b)

        assert res.matched_count == 3
        assert res.missing_in_b_count == 1
        assert res.missing_in_a_count == 0
        assert res.is_fully_aligned is False

        missing = res.get_missing_steps()
        assert len(missing) == 1
        assert missing[0].agent == "db_worker"
        assert missing[0].status == AlignmentStatus.MISSING_IN_B


class TestSpanDAGAlignment:
    """Test alignment directly using OpenInference TraceSpan objects."""

    def test_direct_span_alignment(self):
        """Aligning TraceSpans directly using align_spans."""
        spans_a = [
            TraceSpan(span_id="s1", trace_id="run_a", name="planner", span_type=SpanType.AGENT),
            TraceSpan(
                span_id="s2",
                trace_id="run_a",
                parent_span_id="s1",
                name="worker_a",
                span_type=SpanType.AGENT,
            ),
            TraceSpan(
                span_id="s3",
                trace_id="run_a",
                parent_span_id="s1",
                name="worker_b",
                span_type=SpanType.AGENT,
            ),
        ]
        spans_b = [
            TraceSpan(span_id="sb1", trace_id="run_b", name="planner", span_type=SpanType.AGENT),
            # Sibling order reversed in Run B
            TraceSpan(
                span_id="sb2",
                trace_id="run_b",
                parent_span_id="sb1",
                name="worker_b",
                span_type=SpanType.AGENT,
            ),
            TraceSpan(
                span_id="sb3",
                trace_id="run_b",
                parent_span_id="sb1",
                name="worker_a",
                span_type=SpanType.AGENT,
            ),
        ]

        res = align_spans(spans_a, spans_b, run_id_a="run_a", run_id_b="run_b")

        assert res.matched_count == 3
        assert res.is_fully_aligned is True

        worker_a = res.get_pairs_by_agent("worker_a")[0]
        assert worker_a.span_id_a == "s2"
        assert worker_a.span_id_b == "sb3"

    def test_dual_mode_run_trace_alignment(self):
        """Trace A has only spans, Trace B has only steps; align_traces auto-converts cleanly."""
        spans_a = [
            TraceSpan(span_id="sa1", trace_id="run_a", name="agent_x", span_type=SpanType.AGENT),
            TraceSpan(
                span_id="sa2",
                trace_id="run_a",
                parent_span_id="sa1",
                name="agent_y",
                span_type=SpanType.AGENT,
            ),
        ]
        trace_a = RunTrace(run_id="run_a", workflow="wf", spans=spans_a)

        steps_b = [
            _make_step("agent_x", 1),
            _make_step("agent_y", 2, parent_step="1"),
        ]
        trace_b = RunTrace(run_id="run_b", workflow="wf", steps=steps_b)

        res = align_traces(trace_a, trace_b)

        assert res.matched_count == 2
        assert res.is_fully_aligned is True


class TestDiffEngineIntegration:
    """Test Analyzer protocol and EvidenceRecord generation for DAG anomalies."""

    def test_graph_aligner_analyzer_evidence(self):
        """GraphAligner.analyze() emits WORKFLOW EvidenceRecord for missing loop iterations."""
        trace_a = _make_trace(
            "baseline",
            _make_step("writer", 1),
            _make_step("critic", 2),
            _make_step("writer", 3),
            _make_step("critic", 4),
        )
        trace_b = _make_trace(
            "current",
            _make_step("writer", 1),
            _make_step("critic", 2),
        )

        aligner = GraphAligner(baseline_trace=trace_a)
        result = aligner.analyze(trace_b)

        assert result.skipped is False
        assert len(result.evidence) == 2
        assert all(e.source == EvidenceSource.DIFF_ENGINE for e in result.evidence)
        assert all(e.rule_match.category == FailureCategory.WORKFLOW for e in result.evidence)

    def test_unified_diff_engine_analyzer(self):
        """DiffEngine.analyze() combines alignment evidence with semantic scoring."""
        trace_a = _make_trace(
            "baseline",
            _make_step("researcher", 1, output="Detailed factual analysis of Q3 earnings."),
            _make_step("writer", 2, output="Final executive summary of revenue growth."),
        )
        trace_b = _make_trace(
            "current",
            _make_step("researcher", 1, output="Detailed factual analysis of Q3 earnings."),
            # writer omitted in trace B
        )

        diff_engine = DiffEngine(baseline_trace=trace_a)
        res = diff_engine.analyze(trace_b)

        assert res.skipped is False
        assert len(res.evidence) >= 1
        missing_evidence = [e for e in res.evidence if "writer" in e.agent.lower()]
        assert len(missing_evidence) >= 1
