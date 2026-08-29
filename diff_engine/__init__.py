"""
diff_engine — Trace Alignment & Semantic Difference Layer.

Day 24: Graph Alignment Engine (align_traces).

Import key classes directly from diff_engine:
    from diff_engine import GraphAligner, AlignedStepPair, AlignmentStatus, GraphAlignmentResult
"""

from diff_engine.aligner import (
    AlignedStepPair,
    AlignmentStatus,
    GraphAligner,
    GraphAlignmentResult,
)

align_traces = GraphAligner.align_traces

__all__ = [
    "GraphAligner",
    "AlignedStepPair",
    "AlignmentStatus",
    "GraphAlignmentResult",
    "align_traces",
]
