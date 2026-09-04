"""
diff_engine — Trace Alignment & Semantic Difference Layer.

Day 24: Graph Alignment Engine (align_traces).
Day 25: Semantic Similarity Engine (score_similarity).

Import key classes directly from diff_engine:
    from diff_engine import GraphAligner, AlignedStepPair, AlignmentStatus, GraphAlignmentResult
    from diff_engine import SemanticSimilarityEngine, SimilarityReport, StepSimilarityScore
"""

from diff_engine.aligner import (
    AlignedStepPair,
    AlignmentStatus,
    GraphAligner,
    GraphAlignmentResult,
)
from diff_engine.similarity import (
    SemanticSimilarityEngine,
    SimilarityReport,
    StepSimilarityScore,
)

align_traces = GraphAligner.align_traces


def score_similarity(
    alignment: GraphAlignmentResult,
    model_name: str | None = None,
    threshold: float | None = None,
) -> SimilarityReport:
    """Convenience wrapper: score a GraphAlignmentResult and return a SimilarityReport."""
    engine = SemanticSimilarityEngine(model_name=model_name, threshold=threshold)
    return engine.score(alignment)


__all__ = [
    # Day 24 — Alignment
    "GraphAligner",
    "AlignedStepPair",
    "AlignmentStatus",
    "GraphAlignmentResult",
    "align_traces",
    # Day 25 — Similarity
    "SemanticSimilarityEngine",
    "SimilarityReport",
    "StepSimilarityScore",
    "score_similarity",
]
