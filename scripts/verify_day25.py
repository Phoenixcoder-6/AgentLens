"""
scripts/verify_day25.py — Day 25 Verification Script
=====================================================
Verifies 10 critical Day 25 Semantic Similarity requirements:
  1. Package: diff_engine exports SemanticSimilarityEngine, SimilarityReport, StepSimilarityScore
  2. Jaccard fallback works without sentence-transformers
  3. Identical outputs score ≈ 1.0
  4. Orthogonal outputs score ≈ 0.0 and diverged=True
  5. MISSING_IN_B pairs not scored
  6. Empty alignment returns 1.0 average and no crash
  7. first_divergence_agent identifies first diverged step
  8. get_diverged_steps returns only diverged steps
  9. get_score_for_agent returns correct step
 10. Full pipeline: align_traces → score_similarity (real sentence-transformers)
"""

from __future__ import annotations

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from diff_engine import (
    AlignedStepPair,
    AlignmentStatus,
    GraphAlignmentResult,
    SemanticSimilarityEngine,
    SimilarityReport,
    StepSimilarityScore,
    align_traces,
    score_similarity,
)
from diff_engine.similarity import _jaccard_similarity
from schema.models import AgentStep, RunTrace, StepStatus

PASS = "[PASS]"
FAIL = "[FAIL]"


def _make_step(agent: str, step: int, output: str) -> AgentStep:
    return AgentStep(
        run_id="verify_run",
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


def _alignment(pairs: list[AlignedStepPair]) -> GraphAlignmentResult:
    matched = sum(1 for p in pairs if p.status == AlignmentStatus.MATCHED)
    return GraphAlignmentResult(
        run_id_a="run_a",
        run_id_b="run_b",
        pairs=pairs,
        matched_count=matched,
        missing_in_a_count=0,
        missing_in_b_count=sum(1 for p in pairs if p.status == AlignmentStatus.MISSING_IN_B),
        is_fully_aligned=(matched == len(pairs)),
    )


def _mock_engine(
    vectors: dict[str, np.ndarray], threshold: float = 0.85
) -> SemanticSimilarityEngine:
    engine = SemanticSimilarityEngine(threshold=threshold)
    engine._use_transformers = True

    def fake_embed(texts: list[str]) -> list[np.ndarray]:
        return [vectors[t] for t in texts]

    engine._embed_texts = fake_embed  # type: ignore[method-assign]
    return engine


_VA = np.array([1.0, 0.0, 0.0])
_VB = np.array([0.0, 1.0, 0.0])  # orthogonal → cosine=0.0


def main() -> int:
    print("=== Day 25 Verification: Semantic Similarity Engine ===\n")
    passed = 0
    total = 0

    # ── 1. Package exports ───────────────────────────────────────────────────
    total += 1
    try:
        assert SemanticSimilarityEngine is not None
        assert SimilarityReport is not None
        assert StepSimilarityScore is not None
        assert callable(score_similarity)
        print(
            f"  {PASS} [1] diff_engine exports SemanticSimilarityEngine, SimilarityReport, StepSimilarityScore"
        )
        passed += 1
    except Exception as e:
        print(f"  {FAIL} [1] Export check failed: {e}")

    # ── 2. Jaccard fallback ──────────────────────────────────────────────────
    total += 1
    try:
        score = _jaccard_similarity("machine learning", "machine learning")
        assert score == 1.0
        zero = _jaccard_similarity("apple banana", "car dog")
        assert zero == 0.0
        partial = _jaccard_similarity("the quick brown fox", "the slow brown bear")
        assert 0.0 < partial < 1.0
        print(f"  {PASS} [2] Jaccard fallback: identical=1.0, disjoint=0.0, partial={partial:.3f}")
        passed += 1
    except Exception as e:
        print(f"  {FAIL} [2] Jaccard fallback failed: {e}")

    # ── 3. Identical outputs → similarity ≈ 1.0 ──────────────────────────────
    total += 1
    try:
        pairs = [_matched_pair("researcher", "same text", "same text")]
        engine = _mock_engine({"same text": _VA})
        report = engine.score(_alignment(pairs))
        assert abs(report.scores[0].similarity - 1.0) < 1e-5
        assert report.scores[0].diverged is False
        print(f"  {PASS} [3] Identical outputs → similarity=1.0, diverged=False")
        passed += 1
    except Exception as e:
        print(f"  {FAIL} [3] Identical outputs failed: {e}")

    # ── 4. Orthogonal outputs → similarity=0.0, diverged=True ───────────────
    total += 1
    try:
        pairs = [_matched_pair("writer", "text_a", "text_b")]
        engine = _mock_engine({"text_a": _VA, "text_b": _VB}, threshold=0.85)
        report = engine.score(_alignment(pairs))
        assert abs(report.scores[0].similarity - 0.0) < 1e-5
        assert report.scores[0].diverged is True
        assert report.first_divergence_agent == "writer"
        print(f"  {PASS} [4] Orthogonal outputs → similarity=0.0, diverged=True")
        passed += 1
    except Exception as e:
        print(f"  {FAIL} [4] Orthogonal outputs failed: {e}")

    # ── 5. MISSING_IN_B pairs skipped ───────────────────────────────────────
    total += 1
    try:
        missing = AlignedStepPair(
            agent="verifier",
            status=AlignmentStatus.MISSING_IN_B,
            step_a=_make_step("verifier", 2, "some output"),
            step_b=None,
            step_index_a=2,
            step_index_b=None,
        )
        pairs = [_matched_pair("researcher", "ok", "ok", step=1), missing]
        engine = _mock_engine({"ok": _VA})
        report = engine.score(_alignment(pairs))
        assert report.total_scored == 1
        assert report.scores[0].agent == "researcher"
        print(f"  {PASS} [5] MISSING_IN_B pairs skipped (scored={report.total_scored})")
        passed += 1
    except Exception as e:
        print(f"  {FAIL} [5] Missing pair skip failed: {e}")

    # ── 6. Empty alignment ───────────────────────────────────────────────────
    total += 1
    try:
        engine = SemanticSimilarityEngine(threshold=0.85)
        engine._use_transformers = False
        report = engine.score(_alignment([]))
        assert report.total_scored == 0
        assert abs(report.average_similarity - 1.0) < 1e-5
        assert report.first_divergence_agent is None
        print(f"  {PASS} [6] Empty alignment → avg_similarity=1.0, no crash")
        passed += 1
    except Exception as e:
        print(f"  {FAIL} [6] Empty alignment failed: {e}")

    # ── 7. first_divergence_agent ─────────────────────────────────────────────
    total += 1
    try:
        pairs = [
            _matched_pair("researcher", "g1", "g2", step=1),
            _matched_pair("writer", "b1", "b2", step=2),
        ]
        engine = _mock_engine(
            {
                "g1": _VA,
                "g2": _VA,
                "b1": _VA,
                "b2": _VB,
            },
            threshold=0.85,
        )
        report = engine.score(_alignment(pairs))
        assert report.first_divergence_agent == "writer"
        print(f"  {PASS} [7] first_divergence_agent correctly identifies 'writer'")
        passed += 1
    except Exception as e:
        print(f"  {FAIL} [7] first_divergence_agent failed: {e}")

    # ── 8. get_diverged_steps ────────────────────────────────────────────────
    total += 1
    try:
        pairs = [
            _matched_pair("researcher", "same", "same", step=1),
            _matched_pair("writer", "t_a", "t_b", step=2),
        ]
        engine = _mock_engine({"same": _VA, "t_a": _VA, "t_b": _VB}, threshold=0.85)
        report = engine.score(_alignment(pairs))
        diverged = report.get_diverged_steps()
        assert len(diverged) == 1
        assert diverged[0].agent == "writer"
        print(f"  {PASS} [8] get_diverged_steps returns 1 step: '{diverged[0].agent}'")
        passed += 1
    except Exception as e:
        print(f"  {FAIL} [8] get_diverged_steps failed: {e}")

    # ── 9. get_score_for_agent ────────────────────────────────────────────────
    total += 1
    try:
        pairs = [_matched_pair("researcher", "out_a", "out_b")]
        engine = _mock_engine({"out_a": _VA, "out_b": _VA})
        report = engine.score(_alignment(pairs))
        found = report.get_score_for_agent("researcher")
        assert found is not None and found.agent == "researcher"
        assert report.get_score_for_agent("missing") is None
        print(f"  {PASS} [9] get_score_for_agent returns correct step")
        passed += 1
    except Exception as e:
        print(f"  {FAIL} [9] get_score_for_agent failed: {e}")

    # ── 10. Full pipeline: align → score (real sentence-transformers) ────────
    total += 1
    try:
        trace_a = RunTrace(
            run_id="run_a",
            workflow="pipeline",
            steps=[
                _make_step("researcher", 1, "The capital of France is Paris."),
                _make_step("writer", 2, "France is in Western Europe."),
            ],
        )
        trace_b = RunTrace(
            run_id="run_b",
            workflow="pipeline",
            steps=[
                _make_step("researcher", 1, "The capital of France is Paris."),
                _make_step("writer", 2, "Paris is the capital and largest city of France."),
            ],
        )
        alignment = align_traces(trace_a, trace_b)
        report = score_similarity(alignment)

        assert isinstance(report, SimilarityReport)
        assert report.total_scored == 2
        assert 0.0 <= report.average_similarity <= 1.0
        researcher_score = report.get_score_for_agent("researcher")
        assert researcher_score is not None
        assert researcher_score.similarity > 0.90, (
            f"Expected >0.90, got {researcher_score.similarity}"
        )
        print(
            f"  {PASS} [10] Full pipeline (real embeddings): "
            f"researcher={researcher_score.similarity:.3f}, "
            f"writer={report.get_score_for_agent('writer').similarity:.3f}, "  # type: ignore[union-attr]
            f"avg={report.average_similarity:.3f}"
        )
        passed += 1
    except Exception as e:
        print(f"  {FAIL} [10] Full pipeline failed: {e}")

    print(f"\n{passed}/{total} verifications passed.")
    if passed == total:
        print("✅ Day 25 Verification Complete — All 10 checks passed!")
        return 0
    else:
        print(f"❌ {total - passed} verifications failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
