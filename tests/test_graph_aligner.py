"""
tests/test_graph_aligner.py — Day 24: Graph Alignment Engine Tests
===================================================================
Tests:
  - Identical trace alignment (100% matched)
  - Missing step in Trace B (e.g. verifier skipped in B)
  - Missing step in Trace A (e.g. extra step in B)
  - Middle step insertion / sequence shifts
  - Empty trace handling (no crashes)
  - Helper methods on GraphAlignmentResult
  - Export completeness from diff_engine package
"""

from __future__ import annotations

from diff_engine import (
    AlignmentStatus,
    GraphAligner,
    GraphAlignmentResult,
    align_traces,
)
from schema.models import AgentStep, RunTrace, StepStatus


def _make_step(agent: str, step: int, output: str = "", parent_step: str | None = None) -> AgentStep:
    return AgentStep(
        run_id="test_run",
        step=step,
        agent=agent,
        output=output or f"{agent} output",
        status=StepStatus.SUCCESS,
        parent_step=parent_step,
    )


def _make_trace(run_id: str, *steps: AgentStep) -> RunTrace:
    return RunTrace(
        run_id=run_id,
        workflow="test_pipeline",
        steps=list(steps),
    )


class TestGraphAlignerUnit:
    def test_identical_traces_alignment(self):
        """Two identical 3-step runs should align 1-to-1 with MATCHED status."""
        step1_a = _make_step("researcher", 1)
        step2_a = _make_step("writer", 2)
        step3_a = _make_step("verifier", 3)
        trace_a = _make_trace("run_a", step1_a, step2_a, step3_a)

        step1_b = _make_step("researcher", 1)
        step2_b = _make_step("writer", 2)
        step3_b = _make_step("verifier", 3)
        trace_b = _make_trace("run_b", step1_b, step2_b, step3_b)

        res = align_traces(trace_a, trace_b)

        assert isinstance(res, GraphAlignmentResult)
        assert res.matched_count == 3
        assert res.missing_in_a_count == 0
        assert res.missing_in_b_count == 0
        assert res.is_fully_aligned is True
        assert len(res.pairs) == 3

        for pair in res.pairs:
            assert pair.status == AlignmentStatus.MATCHED
            assert pair.step_a is not None
            assert pair.step_b is not None
            assert pair.step_a.agent == pair.step_b.agent

    def test_missing_step_in_trace_b(self):
        """Run A has [researcher, writer, verifier], Run B has [researcher, writer]."""
        trace_a = _make_trace(
            "run_a",
            _make_step("researcher", 1),
            _make_step("writer", 2),
            _make_step("verifier", 3),
        )
        trace_b = _make_trace(
            "run_b",
            _make_step("researcher", 1),
            _make_step("writer", 2),
        )

        res = GraphAligner.align_traces(trace_a, trace_b)

        assert res.matched_count == 2
        assert res.missing_in_b_count == 1
        assert res.is_fully_aligned is False

        missing = res.get_missing_steps()
        assert len(missing) == 1
        assert missing[0].agent == "verifier"
        assert missing[0].status == AlignmentStatus.MISSING_IN_B
        assert missing[0].step_a is not None
        assert missing[0].step_b is None

    def test_missing_step_in_trace_a(self):
        """Run A has [researcher, writer], Run B has [researcher, writer, verifier]."""
        trace_a = _make_trace(
            "run_a",
            _make_step("researcher", 1),
            _make_step("writer", 2),
        )
        trace_b = _make_trace(
            "run_b",
            _make_step("researcher", 1),
            _make_step("writer", 2),
            _make_step("verifier", 3),
        )

        res = align_traces(trace_a, trace_b)

        assert res.matched_count == 2
        assert res.missing_in_a_count == 1
        assert res.is_fully_aligned is False

        missing = res.get_missing_steps()
        assert len(missing) == 1
        assert missing[0].agent == "verifier"
        assert missing[0].status == AlignmentStatus.MISSING_IN_A
        assert missing[0].step_a is None
        assert missing[0].step_b is not None

    def test_inserted_middle_step(self):
        """Run A has [researcher, writer], Run B has [researcher, editor, writer]."""
        trace_a = _make_trace(
            "run_a",
            _make_step("researcher", 1),
            _make_step("writer", 2),
        )
        trace_b = _make_trace(
            "run_b",
            _make_step("researcher", 1),
            _make_step("editor", 2),
            _make_step("writer", 3),
        )

        res = align_traces(trace_a, trace_b)

        assert res.matched_count == 2
        assert res.missing_in_a_count == 1
        missing_a = [p for p in res.pairs if p.status == AlignmentStatus.MISSING_IN_A]
        assert len(missing_a) == 1
        assert missing_a[0].agent == "editor"

    def test_empty_traces_no_crash(self):
        """Aligning empty traces should return 0 pairs without throwing exceptions."""
        trace_a = _make_trace("run_a")
        trace_b = _make_trace("run_b")

        res = align_traces(trace_a, trace_b)
        assert res.matched_count == 0
        assert res.pairs == []
        assert res.is_fully_aligned is True

    def test_helper_methods(self):
        """Test get_matched_pairs, get_missing_steps, and get_pairs_by_agent."""
        trace_a = _make_trace(
            "run_a",
            _make_step("researcher", 1),
            _make_step("writer", 2),
        )
        trace_b = _make_trace(
            "run_b",
            _make_step("researcher", 1),
        )

        res = align_traces(trace_a, trace_b)

        matched = res.get_matched_pairs()
        assert len(matched) == 1
        assert matched[0].agent == "researcher"

        missing = res.get_missing_steps()
        assert len(missing) == 1
        assert missing[0].agent == "writer"

        res_pairs = res.get_pairs_by_agent("researcher")
        assert len(res_pairs) == 1
        assert res_pairs[0].is_matched is True


class TestPackageExports:
    def test_diff_engine_exports(self):
        from diff_engine import AlignmentStatus, GraphAligner, align_traces

        assert callable(align_traces)
        assert GraphAligner is not None
        assert AlignmentStatus.MATCHED == "MATCHED"
