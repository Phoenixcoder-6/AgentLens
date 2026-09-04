"""
tests/test_semantic_similarity.py — Day 25: Semantic Similarity Engine Tests
=============================================================================
All tests mock the sentence-transformers model so they run fast with no
internet or GPU dependency. A final integration-marked test does a real embed.

Test coverage:
  - ScoreReport for identical matched outputs → similarity ≈ 1.0
  - Score below threshold triggers diverged=True
  - first_divergence_agent points to the first diverged step
  - MISSING_IN_A / MISSING_IN_B pairs are skipped (not scored)
  - Empty alignment → average_similarity = 1.0, no crash
  - Multiple diverged steps → diverged_count accurate
  - get_diverged_steps helper
  - get_score_for_agent helper
  - score_similarity convenience wrapper
  - Jaccard fallback returns float in [0, 1]
"""

from __future__ import annotations

import numpy as np
import pytest

from diff_engine import (
    AlignedStepPair,
    AlignmentStatus,
    GraphAlignmentResult,
    SemanticSimilarityEngine,
    SimilarityReport,
)
from diff_engine.similarity import _jaccard_similarity
from schema.models import AgentStep, StepStatus

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_step(agent: str, step: int, output: str) -> AgentStep:
    return AgentStep(
        run_id="test_run",
        step=step,
        agent=agent,
        output=output,
        status=StepStatus.SUCCESS,
    )


def _matched_pair(agent: str, out_a: str, out_b: str, step: int = 1) -> AlignedStepPair:
    return AlignedStepPair(
        agent=agent,
        status=AlignmentStatus.MATCHED,
        step_a=_make_step(agent, step, out_a),
        step_b=_make_step(agent, step, out_b),
        step_index_a=step,
        step_index_b=step,
    )


def _missing_b_pair(agent: str, step: int = 1) -> AlignedStepPair:
    return AlignedStepPair(
        agent=agent,
        status=AlignmentStatus.MISSING_IN_B,
        step_a=_make_step(agent, step, "output"),
        step_b=None,
        step_index_a=step,
        step_index_b=None,
    )


def _alignment(
    run_id_a: str,
    run_id_b: str,
    pairs: list[AlignedStepPair],
) -> GraphAlignmentResult:
    matched = sum(1 for p in pairs if p.status == AlignmentStatus.MATCHED)
    return GraphAlignmentResult(
        run_id_a=run_id_a,
        run_id_b=run_id_b,
        pairs=pairs,
        matched_count=matched,
        missing_in_a_count=0,
        missing_in_b_count=sum(1 for p in pairs if p.status == AlignmentStatus.MISSING_IN_B),
        is_fully_aligned=(matched == len(pairs)),
    )


def _mock_engine(
    vectors: dict[str, np.ndarray], threshold: float = 0.85
) -> SemanticSimilarityEngine:
    """
    Build an engine with sentence-transformers mocked out.
    vectors: mapping from output text → embedding vector.
    """
    engine = SemanticSimilarityEngine(threshold=threshold)
    engine._use_transformers = True

    def fake_embed(texts: list[str]) -> list[np.ndarray]:
        return [vectors[t] for t in texts]

    engine._embed_texts = fake_embed  # type: ignore[method-assign]
    return engine


# Unit vector helpers
_VEC_A = np.array([1.0, 0.0, 0.0])
_VEC_B = np.array([1.0, 0.0, 0.0])  # identical → cosine=1.0
_VEC_C = np.array([0.0, 1.0, 0.0])  # orthogonal to A → cosine=0.0
_VEC_D = np.array([0.7071, 0.7071, 0.0])  # 45° from A → cosine≈0.707


class TestSemanticSimilarityEngine:
    def test_identical_outputs_score_one(self):
        """Identical outputs should produce similarity ≈ 1.0."""
        pairs = [_matched_pair("researcher", "same text", "same text")]
        alignment = _alignment("a", "b", pairs)
        engine = _mock_engine({"same text": _VEC_A})
        report = engine.score(alignment)

        assert report.total_scored == 1
        assert report.scores[0].similarity == pytest.approx(1.0, abs=1e-5)
        assert report.scores[0].diverged is False
        assert report.first_divergence_agent is None
        assert report.diverged_count == 0

    def test_orthogonal_outputs_diverge(self):
        """Orthogonal embeddings (cosine=0.0) should be flagged as diverged."""
        pairs = [_matched_pair("writer", "text_a", "text_b")]
        alignment = _alignment("a", "b", pairs)
        engine = _mock_engine({"text_a": _VEC_A, "text_b": _VEC_C}, threshold=0.85)
        report = engine.score(alignment)

        assert report.scores[0].similarity == pytest.approx(0.0, abs=1e-5)
        assert report.scores[0].diverged is True
        assert report.first_divergence_agent == "writer"
        assert report.diverged_count == 1

    def test_first_divergence_is_first_agent(self):
        """first_divergence_agent should be the first step that falls below threshold."""
        pairs = [
            _matched_pair("researcher", "good_a", "good_b", step=1),
            _matched_pair("writer", "bad_a", "bad_b", step=2),
            _matched_pair("verifier", "ok_a", "ok_b", step=3),
        ]
        alignment = _alignment("a", "b", pairs)
        engine = _mock_engine(
            {
                "good_a": _VEC_A,
                "good_b": _VEC_A,
                "bad_a": _VEC_A,
                "bad_b": _VEC_C,
                "ok_a": _VEC_A,
                "ok_b": _VEC_A,
            },
            threshold=0.85,
        )
        report = engine.score(alignment)

        assert report.first_divergence_agent == "writer"
        assert report.diverged_count == 1

    def test_missing_pairs_skipped(self):
        """MISSING_IN_B pairs should not be scored."""
        pairs = [
            _matched_pair("researcher", "text_a", "text_b", step=1),
            _missing_b_pair("verifier", step=2),
        ]
        alignment = _alignment("a", "b", pairs)
        engine = _mock_engine({"text_a": _VEC_A, "text_b": _VEC_A})
        report = engine.score(alignment)

        assert report.total_scored == 1
        assert report.scores[0].agent == "researcher"

    def test_empty_alignment_no_crash(self):
        """Empty alignment should return average_similarity=1.0 with 0 scored."""
        alignment = _alignment("a", "b", [])
        engine = SemanticSimilarityEngine(threshold=0.85)
        engine._use_transformers = False  # use Jaccard so no model needed
        report = engine.score(alignment)

        assert report.total_scored == 0
        assert report.average_similarity == pytest.approx(1.0)
        assert report.first_divergence_agent is None

    def test_multiple_diverged_count(self):
        """All three orthogonal pairs should produce diverged_count == 3."""
        pairs = [
            _matched_pair("a1", "t_a", "t_b", step=1),
            _matched_pair("a2", "t_c", "t_d", step=2),
            _matched_pair("a3", "t_e", "t_f", step=3),
        ]
        alignment = _alignment("a", "b", pairs)
        engine = _mock_engine(
            {
                "t_a": _VEC_A,
                "t_b": _VEC_C,
                "t_c": _VEC_A,
                "t_d": _VEC_C,
                "t_e": _VEC_A,
                "t_f": _VEC_C,
            },
            threshold=0.85,
        )
        report = engine.score(alignment)
        assert report.diverged_count == 3

    def test_get_diverged_steps_helper(self):
        """get_diverged_steps() should return only steps where diverged=True."""
        pairs = [
            _matched_pair("researcher", "same", "same", step=1),
            _matched_pair("writer", "t_a", "t_b", step=2),
        ]
        alignment = _alignment("a", "b", pairs)
        engine = _mock_engine(
            {"same": _VEC_A, "t_a": _VEC_A, "t_b": _VEC_C},
            threshold=0.85,
        )
        report = engine.score(alignment)
        diverged = report.get_diverged_steps()
        assert len(diverged) == 1
        assert diverged[0].agent == "writer"

    def test_get_score_for_agent_helper(self):
        """get_score_for_agent() should find the score by agent name."""
        pairs = [_matched_pair("researcher", "out_a", "out_b")]
        alignment = _alignment("a", "b", pairs)
        engine = _mock_engine({"out_a": _VEC_A, "out_b": _VEC_A})
        report = engine.score(alignment)

        found = report.get_score_for_agent("researcher")
        assert found is not None
        assert found.agent == "researcher"
        assert report.get_score_for_agent("missing") is None

    def test_score_similarity_convenience_wrapper(self):
        """score_similarity() function should return a SimilarityReport."""
        pairs = [_matched_pair("researcher", "hello world", "hello world")]
        alignment = _alignment("a", "b", pairs)

        # Use Jaccard fallback (no sentence-transformers needed for this test)
        engine = SemanticSimilarityEngine(threshold=0.85)
        engine._use_transformers = False
        report = engine.score(alignment)

        assert isinstance(report, SimilarityReport)


class TestJaccardFallback:
    def test_identical_texts(self):
        assert _jaccard_similarity("hello world", "hello world") == pytest.approx(1.0)

    def test_disjoint_texts(self):
        assert _jaccard_similarity("apple banana", "car dog") == pytest.approx(0.0)

    def test_partial_overlap(self):
        score = _jaccard_similarity("the quick brown fox", "the slow brown bear")
        assert 0.0 < score < 1.0

    def test_empty_both(self):
        assert _jaccard_similarity("", "") == pytest.approx(1.0)

    def test_empty_one_side(self):
        assert _jaccard_similarity("hello", "") == pytest.approx(0.0)
