"""
diff_engine — Trace Alignment & Semantic Difference Layer.

Day 24: Graph Alignment Engine (align_traces).
Day 25: Semantic Similarity Engine (score_similarity).

Import key classes directly from diff_engine:
    from diff_engine import GraphAligner, AlignedStepPair, AlignmentStatus, GraphAlignmentResult
    from diff_engine import SemanticSimilarityEngine, SimilarityReport, StepSimilarityScore
"""

from app.interfaces import AnalysisResult, Analyzer
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
from schema.models import RunTrace

align_traces = GraphAligner.align_traces


def score_similarity(
    alignment: GraphAlignmentResult,
    model_name: str | None = None,
    threshold: float | None = None,
) -> SimilarityReport:
    """Convenience wrapper: score a GraphAlignmentResult and return a SimilarityReport."""
    engine = SemanticSimilarityEngine(model_name=model_name, threshold=threshold)
    return engine.score(alignment)


class DiffEngine(Analyzer):
    """
    Unified DiffEngine implementing the Analyzer protocol (Day 41).
    Combines GraphAligner structural alignment with SemanticSimilarityEngine scoring.
    """

    def __init__(
        self,
        baseline_trace: RunTrace | None = None,
        model_name: str | None = None,
        threshold: float | None = None,
    ) -> None:
        self.baseline_trace = baseline_trace
        self.aligner = GraphAligner(baseline_trace=baseline_trace)
        self.similarity_engine = SemanticSimilarityEngine(
            model_name=model_name, threshold=threshold
        )

    @property
    def analyzer_id(self) -> str:
        return "diff_engine"

    def analyze(self, trace: RunTrace, baseline_trace: RunTrace | None = None) -> AnalysisResult:
        """Run structural alignment + semantic similarity diffing on a RunTrace."""
        try:
            if not trace or not getattr(trace, "steps", None):
                return AnalysisResult(
                    skipped=True,
                    skip_reason="No steps in trace",
                    analyzer_id=self.analyzer_id,
                )
            ref = baseline_trace or self.baseline_trace
            align_res = self.aligner.analyze(trace, baseline_trace=ref)
            sim_res = self.similarity_engine.analyze(trace, baseline_trace=ref)
            combined = list(align_res.evidence) + list(sim_res.evidence)
            return AnalysisResult(
                evidence=combined,
                analyzer_id=self.analyzer_id,
                skipped=False,
            )
        except Exception as exc:
            return AnalysisResult(
                skipped=True,
                skip_reason=f"DiffEngine error: {exc}",
                analyzer_id=self.analyzer_id,
            )

    def run(self, trace: RunTrace, baseline_trace: RunTrace | None = None) -> AnalysisResult:
        """Standard Analyzer execution alias."""
        return self.analyze(trace, baseline_trace=baseline_trace)

    def compare(self, trace_a: RunTrace, trace_b: RunTrace) -> SimilarityReport:
        """Align two traces and return their semantic SimilarityReport."""
        alignment = GraphAligner.align_traces(trace_a, trace_b)
        return self.similarity_engine.score(alignment)


__all__ = [
    "DiffEngine",
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
