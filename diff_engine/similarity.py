"""
diff_engine/similarity.py — Day 25: Semantic Similarity Engine
===============================================================
Computes per-step cosine similarity between matched agent outputs
in two aligned traces using sentence-transformers embeddings.

Features:
  - Local inference with all-MiniLM-L6-v2 (no API key required).
  - Configurable similarity threshold from config.yaml (diff.similarity_threshold).
  - Identifies first point of divergence across matched step pairs.
  - Graceful degradation: if sentence-transformers is unavailable, falls back
    to Jaccard token overlap similarity so the pipeline never crashes.
  - Singleton model cache: model loaded once per process.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from config.config_loader import get
from diff_engine.aligner import AlignedStepPair, AlignmentStatus, GraphAlignmentResult

log = logging.getLogger("agentlens.similarity")


# ─────────────────────────────────────────────────────────────────────────────
# Similarity Output Models
# ─────────────────────────────────────────────────────────────────────────────


class StepSimilarityScore(BaseModel):
    """
    Semantic similarity result for a single aligned step pair.
    """

    agent: str = Field(description="Agent name for this step pair")
    similarity: float = Field(
        description="Cosine similarity score in [0.0, 1.0] between step_a.output and step_b.output"
    )
    diverged: bool = Field(description="True if similarity is below the configured threshold")
    threshold: float = Field(description="Threshold used for the diverged flag")
    output_a: str = Field(default="", description="Output from Trace A (truncated for storage)")
    output_b: str = Field(default="", description="Output from Trace B (truncated for storage)")
    method: str = Field(
        default="cosine",
        description="Similarity method used: 'cosine' (sentence-transformers) or 'jaccard' (fallback)",
    )


class SimilarityReport(BaseModel):
    """
    Full semantic similarity report across all matched step pairs from a GraphAlignmentResult.
    """

    run_id_a: str
    run_id_b: str
    scores: list[StepSimilarityScore] = Field(default_factory=list)
    average_similarity: float = Field(
        default=1.0, description="Mean cosine similarity across all matched pairs"
    )
    first_divergence_agent: str | None = Field(
        default=None,
        description="Agent name of the first matched step where similarity fell below threshold",
    )
    diverged_count: int = Field(
        default=0, description="Number of step pairs where similarity < threshold"
    )
    total_scored: int = Field(default=0, description="Number of matched pairs scored")
    embedding_model: str = Field(default="", description="Model used for embedding")
    threshold_used: float = Field(default=0.85)

    def get_diverged_steps(self) -> list[StepSimilarityScore]:
        """Return only the step pairs that diverged."""
        return [s for s in self.scores if s.diverged]

    def get_score_for_agent(self, agent: str) -> StepSimilarityScore | None:
        """Look up similarity score by agent name."""
        for s in self.scores:
            if s.agent == agent:
                return s
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading Helpers
# ─────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _load_sentence_transformer(model_name: str) -> Any:
    """Load and cache the sentence-transformers model (loaded only once per process)."""
    from sentence_transformers import SentenceTransformer  # type: ignore[import]

    log.info(f"Loading sentence-transformers model: {model_name}")
    return SentenceTransformer(model_name)


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D float vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0 if norm_a == norm_b else 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """
    Fallback token-overlap similarity when sentence-transformers is not available.
    Returns |intersection| / |union| of word token sets.
    """
    tokens_a = set(re.findall(r"\w+", text_a.lower()))
    tokens_b = set(re.findall(r"\w+", text_b.lower()))
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


# ─────────────────────────────────────────────────────────────────────────────
# SemanticSimilarityEngine
# ─────────────────────────────────────────────────────────────────────────────

_OUTPUT_PREVIEW_CHARS = 200


class SemanticSimilarityEngine:
    """
    Computes semantic similarity scores for all MATCHED pairs in a GraphAlignmentResult.

    Usage:
        engine = SemanticSimilarityEngine()
        report = engine.score(alignment_result)
    """

    def __init__(
        self,
        model_name: str | None = None,
        threshold: float | None = None,
    ) -> None:
        self._model_name: str = model_name or str(
            get("diff", "embedding_model", "all-MiniLM-L6-v2")
        )
        self._threshold: float = (
            threshold if threshold is not None else float(get("diff", "similarity_threshold", 0.85))
        )
        self._use_transformers: bool = self._check_transformers_available()

    def _check_transformers_available(self) -> bool:
        try:
            import sentence_transformers  # noqa: F401

            return True
        except ImportError:
            log.warning(
                "sentence-transformers not installed — falling back to Jaccard token similarity"
            )
            return False

    def _embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a list of texts using sentence-transformers (or raise on failure)."""
        model = _load_sentence_transformer(self._model_name)
        embeddings = model.encode(texts, convert_to_numpy=True)
        return [embeddings[i] for i in range(len(texts))]

    def _score_pair(self, text_a: str, text_b: str) -> tuple[float, str]:
        """
        Score a single (output_a, output_b) pair.
        Returns (similarity, method).
        """
        if self._use_transformers:
            try:
                vec_a, vec_b = self._embed_texts([text_a, text_b])
                return _cosine_similarity(vec_a, vec_b), "cosine"
            except Exception as e:
                log.warning(f"Embedding failed, falling back to Jaccard: {e}")
        return _jaccard_similarity(text_a, text_b), "jaccard"

    def score(self, alignment: GraphAlignmentResult) -> SimilarityReport:
        """
        Compute semantic similarity for every MATCHED step pair in the alignment.

        MISSING_IN_A / MISSING_IN_B pairs are skipped (no output to compare).

        Args:
            alignment: GraphAlignmentResult from GraphAligner.align_traces()

        Returns:
            SimilarityReport with per-step scores, average similarity, and
            the first agent where divergence was detected.
        """
        matched_pairs: list[AlignedStepPair] = [
            p for p in alignment.pairs if p.status == AlignmentStatus.MATCHED
        ]

        step_scores: list[StepSimilarityScore] = []
        first_divergence: str | None = None

        for pair in matched_pairs:
            output_a = (pair.step_a.output if pair.step_a else "") or ""
            output_b = (pair.step_b.output if pair.step_b else "") or ""

            similarity, method = self._score_pair(output_a, output_b)
            diverged = similarity < self._threshold

            if diverged and first_divergence is None:
                first_divergence = pair.agent

            step_scores.append(
                StepSimilarityScore(
                    agent=pair.agent,
                    similarity=round(similarity, 6),
                    diverged=diverged,
                    threshold=self._threshold,
                    output_a=output_a[:_OUTPUT_PREVIEW_CHARS],
                    output_b=output_b[:_OUTPUT_PREVIEW_CHARS],
                    method=method,
                )
            )

        avg_sim = float(np.mean([s.similarity for s in step_scores])) if step_scores else 1.0
        diverged_count = sum(1 for s in step_scores if s.diverged)

        return SimilarityReport(
            run_id_a=alignment.run_id_a,
            run_id_b=alignment.run_id_b,
            scores=step_scores,
            average_similarity=round(avg_sim, 6),
            first_divergence_agent=first_divergence,
            diverged_count=diverged_count,
            total_scored=len(step_scores),
            embedding_model=self._model_name if self._use_transformers else "jaccard-fallback",
            threshold_used=self._threshold,
        )
