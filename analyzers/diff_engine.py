"""
analyzers/diff_engine.py — Analyzer-compatible DiffEngine re-export (Day 41)
=============================================================================
Re-exports DiffEngine, GraphAligner, and SemanticSimilarityEngine from
`diff_engine` so all analyzers can be imported consistently from `analyzers.*`.
"""

from __future__ import annotations

from diff_engine import (
    AlignedStepPair,
    AlignmentStatus,
    DiffEngine,
    GraphAligner,
    GraphAlignmentResult,
    SemanticSimilarityEngine,
    SimilarityReport,
    StepSimilarityScore,
    align_traces,
    score_similarity,
)

__all__ = [
    "DiffEngine",
    "GraphAligner",
    "AlignedStepPair",
    "AlignmentStatus",
    "GraphAlignmentResult",
    "align_traces",
    "SemanticSimilarityEngine",
    "SimilarityReport",
    "StepSimilarityScore",
    "score_similarity",
]
