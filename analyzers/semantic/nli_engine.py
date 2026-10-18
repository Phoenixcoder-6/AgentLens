"""
analyzers/semantic/nli_engine.py — Day 55: Local Cross-Encoder NLI Engine
========================================================================
Performs granular claim-level Natural Language Inference (NLI) between
upstream premise and downstream agent synthesis.

Mathematical Formulation:
    For premise P and claim proposition c_i ∈ Downstream:
        NLI(P, c_i) -> [P(Entailment), P(Neutral), P(Contradiction)]
        where P(Entailment) + P(Neutral) + P(Contradiction) = 1.0

    Classification:
        Contradiction: P(Contradiction) >= tau_contradiction (default 0.70)
        Entailment:    P(Entailment)    >= tau_entailment    (default 0.75)
        Neutral:       Otherwise

Features:
    - Claim-level decomposition: splits synthesis text into discrete propositions.
    - Local cross-encoder support (e.g. cross-encoder/nli-deberta-v3-small).
    - Deterministic lexical/polar negation fallback mode (zero network / offline).
    - Step-level and trace-level evaluation helpers.
"""

from __future__ import annotations

import logging
import re
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from schema.models import AgentStep

logger = logging.getLogger("agentlens.semantic.nli")


# ─────────────────────────────────────────────────────────────────────────────
# Enums & Data Models
# ─────────────────────────────────────────────────────────────────────────────


class NLILabel(StrEnum):
    """NLI categorical labels."""

    ENTAILMENT = "entailment"
    NEUTRAL = "neutral"
    CONTRADICTION = "contradiction"


class NLIClaimScore(BaseModel):
    """Evaluation score for a single proposition claim against an upstream premise."""

    claim: str = Field(description="The extracted proposition statement evaluated")
    premise: str = Field(description="The upstream reference context / premise")
    entailment_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    neutral_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    contradiction_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    predicted_label: NLILabel = Field(default=NLILabel.NEUTRAL)
    is_contradiction: bool = Field(default=False)
    is_entailed: bool = Field(default=False)


class NLIEvaluationReport(BaseModel):
    """Aggregate NLI evaluation report across all claims in an agent transition."""

    upstream_agent: str = Field(description="Name of the upstream source agent")
    downstream_agent: str = Field(description="Name of the downstream synthesizing agent")
    claims: list[NLIClaimScore] = Field(
        default_factory=list, description="All evaluated proposition claims"
    )
    has_contradiction: bool = Field(
        default=False, description="True if at least one claim is a contradiction"
    )
    contradicted_claims: list[NLIClaimScore] = Field(
        default_factory=list, description="Claims flagged as direct contradictions"
    )
    entailed_claims: list[NLIClaimScore] = Field(
        default_factory=list, description="Claims flagged as faithful entailments"
    )
    overall_entailment_ratio: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Ratio of entailed claims over total claims"
    )
    fallback_mode_used: bool = Field(
        default=False, description="True if lexical fallback was used instead of model"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Proposition / Claim Decomposition
# ─────────────────────────────────────────────────────────────────────────────


def split_claims(text: str) -> list[str]:
    """
    Decompose a multi-sentence agent output into discrete proposition claims.
    Strips markdown formatting, list numerals, and empty lines.
    """
    if not text:
        return []

    # Split by sentence terminals, semicolons, and double newlines
    raw_segments = re.split(r"(?<=[.!?])\s+|\n+|;\s*", text)
    claims: list[str] = []

    for seg in raw_segments:
        # Strip bullets, numbering, blockquotes
        cleaned = seg.strip().strip("•-* \t")
        cleaned = re.sub(r"^(?:#+\s*|\d+[\.)]\s*|>\s*)", "", cleaned).strip()
        # Require minimal claim length (at least 15 chars and 3 words)
        if len(cleaned) >= 15 and len(cleaned.split()) >= 3:
            claims.append(cleaned)

    return claims


# ─────────────────────────────────────────────────────────────────────────────
# Lexical & Polar Negation Fallback Engine
# ─────────────────────────────────────────────────────────────────────────────

NEGATION_PATTERNS = [
    re.compile(r"\bfailed\s+to\b", re.I),
    re.compile(r"\bunable\s+to\b", re.I),
    re.compile(r"\bdid\s+not\b", re.I),
    re.compile(r"\bdoes\s+not\b", re.I),
    re.compile(r"\bwill\s+not\b", re.I),
    re.compile(r"\bnot\b", re.I),
    re.compile(r"\bnever\b", re.I),
    re.compile(r"\bno\b", re.I),
    re.compile(r"\bneither\b", re.I),
    re.compile(r"\bnor\b", re.I),
    re.compile(r"\bcannot\b", re.I),
    re.compile(r"\bwithout\b", re.I),
]

OPPOSING_VERBS = [
    (
        {"fail", "failed", "failing", "failure"},
        {"succeed", "succeeded", "succeeding", "success", "successful", "successfully"},
    ),
    (
        {"fail", "failed", "failing", "miss", "missed"},
        {"meet", "met", "meeting", "achieve", "achieved"},
    ),
    (
        {"reject", "rejected", "rejection", "refuse", "refused"},
        {"approve", "approved", "approval", "accept", "accepted"},
    ),
    (
        {"decrease", "decreased", "drop", "dropped", "fall", "fell", "loss", "losses"},
        {"increase", "increased", "rise", "rose", "gain", "gains"},
    ),
    (
        {"lost", "destroyed", "crash", "crashed", "exploded"},
        {"safe", "safely", "intact", "survived", "recovered", "landed"},
    ),
    ({"deny", "denied"}, {"allow", "allowed", "permit", "permitted", "grant", "granted"}),
    ({"false", "incorrect", "invalid"}, {"true", "correct", "valid"}),
    ({"negative"}, {"positive"}),
    ({"down"}, {"up"}),
    ({"worse"}, {"better"}),
    ({"decline", "declined"}, {"improve", "improved", "improvement"}),
]

STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "was",
        "are",
        "were",
        "and",
        "or",
        "to",
        "in",
        "of",
        "for",
        "with",
        "at",
        "by",
        "from",
        "on",
        "it",
        "its",
        "that",
        "this",
        "be",
        "been",
        "have",
        "has",
        "had",
        "however",
    }
)


def _check_single_polar_inversion(premise: str, hypothesis: str) -> bool:
    p_lower = premise.lower()
    h_lower = hypothesis.lower()

    p_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", p_lower))
    h_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", h_lower))

    p_content = p_words - STOPWORDS
    h_content = h_words - STOPWORDS
    shared = p_content & h_content

    if not h_content or not p_content:
        return False

    overlap_ratio = len(shared) / len(h_content)

    # 1. Opposing verbs with at least 1 shared subject / concept
    if len(shared) >= 1:
        for set_a, set_b in OPPOSING_VERBS:
            if (bool(p_words & set_a) and bool(h_words & set_b)) or (
                bool(p_words & set_b) and bool(h_words & set_a)
            ):
                return True

    # 2. Polar negation inversion with at least 2 shared concepts or >= 0.25 overlap
    if len(shared) >= 2 or overlap_ratio >= 0.25:
        p_neg = any(pat.search(p_lower) for pat in NEGATION_PATTERNS)
        h_neg = any(pat.search(h_lower) for pat in NEGATION_PATTERNS)
        if p_neg != h_neg:
            return True

    return False


def evaluate_pair_lexical_fallback(premise: str, hypothesis: str) -> tuple[float, float, float]:
    """
    Deterministic lexical fallback computing entailment, neutral, and contradiction
    probabilities using token containment, negation alignment, and polar opposites.

    Returns:
        (p_entailment, p_neutral, p_contradiction) summing to 1.0.
    """
    premise_sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", premise) if len(s.strip()) >= 5
    ]
    if not premise_sentences:
        premise_sentences = [premise]

    for p_sent in premise_sentences:
        if _check_single_polar_inversion(p_sent, hypothesis):
            return (0.05, 0.10, 0.85)

    best_overlap = 0.0
    h_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", hypothesis.lower())) - STOPWORDS
    if not h_words:
        return (0.20, 0.70, 0.10)

    for p_sent in [premise] + premise_sentences:
        p_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", p_sent.lower())) - STOPWORDS
        overlap = len(p_words & h_words) / len(h_words)
        if overlap > best_overlap:
            best_overlap = overlap

    if best_overlap >= 0.55:
        return (0.85, 0.10, 0.05)

    return (0.20, 0.70, 0.10)


# ─────────────────────────────────────────────────────────────────────────────
# Local Cross-Encoder NLIEngine
# ─────────────────────────────────────────────────────────────────────────────


class NLIEngine:
    """
    Granular Claim-Level Natural Language Inference Engine.
    Evaluates factual consistency between upstream evidence and downstream synthesis.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        mode: str = "auto",
        contradiction_threshold: float = 0.70,
        entailment_threshold: float = 0.75,
        cache_dir: Path | str | None = None,
    ) -> None:
        self.model_name = model_name
        self.mode = mode
        self.contradiction_threshold = contradiction_threshold
        self.entailment_threshold = entailment_threshold
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self._tokenizer: Any = None
        self._model: Any = None
        self._model_initialized = False
        self.fallback_mode = False

    def _ensure_model_initialized(self) -> None:
        """Lazy model initialization with fail-safe fallback."""
        if self._model_initialized:
            return

        self._model_initialized = True

        if self.mode == "fallback":
            self.fallback_mode = True
            logger.info("NLIEngine configured in explicit fallback mode.")
            return

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            # Attempt local files only first if cache exists
            local_first = False
            if self.cache_dir and (self.cache_dir / self.model_name.replace("/", "_")).exists():
                local_first = True

            target_path = (
                str(self.cache_dir / self.model_name.replace("/", "_"))
                if local_first and self.cache_dir
                else self.model_name
            )

            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    target_path, local_files_only=local_first
                )
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    target_path, local_files_only=local_first
                )
                self._model.eval()
                self.fallback_mode = False
                logger.info("NLIEngine successfully loaded model: %s", target_path)
            except Exception as load_err:
                if self.mode == "transformers":
                    raise load_err
                logger.warning(
                    "Model %s unavailable locally; defaulting to deterministic lexical NLI: %s",
                    self.model_name,
                    load_err,
                )
                self.fallback_mode = True

        except Exception as e:
            if self.mode == "transformers":
                raise e
            logger.warning(
                "Transformers or PyTorch not available (%s); using deterministic lexical NLI fallback.",
                e,
            )
            self.fallback_mode = True

    def score_claim(self, premise: str, claim: str) -> NLIClaimScore:
        """
        Evaluate a single proposition claim against an upstream reference premise.

        Returns:
            NLIClaimScore containing entailment, neutral, and contradiction probabilities.
        """
        self._ensure_model_initialized()

        if self.fallback_mode or self._model is None or self._tokenizer is None:
            p_entail, p_neut, p_contra = evaluate_pair_lexical_fallback(premise, claim)
        else:
            try:
                import torch

                inputs = self._tokenizer(
                    premise,
                    claim,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    logits = self._model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1)[0].tolist()

                # Label mapping for standard cross-encoder/nli-deberta-v3 models:
                # 0: contradiction, 1: entailment, 2: neutral (or 0: entailment, 1: neutral, 2: contradiction)
                # We normalize via config/id2label if available
                id2label = getattr(self._model.config, "id2label", None)
                if id2label and isinstance(id2label, dict):
                    prob_map = {id2label[i].lower(): probs[i] for i in range(len(probs))}
                    p_contra = prob_map.get("contradiction", probs[0])
                    p_entail = prob_map.get("entailment", probs[1] if len(probs) > 1 else 0.0)
                    p_neut = prob_map.get("neutral", probs[2] if len(probs) > 2 else 0.0)
                else:
                    # Default DeBERTa cross-encoder mapping: [contradiction, entailment, neutral]
                    p_contra, p_entail, p_neut = probs[0], probs[1], probs[2]

            except Exception as err:
                logger.warning(
                    "Error during cross-encoder forward pass: %s. Using lexical fallback.", err
                )
                p_entail, p_neut, p_contra = evaluate_pair_lexical_fallback(premise, claim)

        # Classify label
        if p_contra >= self.contradiction_threshold:
            pred_label = NLILabel.CONTRADICTION
            is_contra = True
            is_entail = False
        elif p_entail >= self.entailment_threshold:
            pred_label = NLILabel.ENTAILMENT
            is_contra = False
            is_entail = True
        else:
            pred_label = NLILabel.NEUTRAL
            is_contra = False
            is_entail = False

        return NLIClaimScore(
            claim=claim,
            premise=premise,
            entailment_prob=round(p_entail, 4),
            neutral_prob=round(p_neut, 4),
            contradiction_prob=round(p_contra, 4),
            predicted_label=pred_label,
            is_contradiction=is_contra,
            is_entailed=is_entail,
        )

    def evaluate_claims(
        self,
        premise: str,
        downstream_text: str,
        upstream_agent: str = "upstream",
        downstream_agent: str = "downstream",
    ) -> NLIEvaluationReport:
        """
        Decompose downstream text into claims and score each against the premise.
        """
        claims_text = split_claims(downstream_text)
        claim_scores: list[NLIClaimScore] = []

        for c_text in claims_text:
            score = self.score_claim(premise, c_text)
            claim_scores.append(score)

        contradicted = [c for c in claim_scores if c.is_contradiction]
        entailed = [c for c in claim_scores if c.is_entailed]
        total_claims = len(claim_scores)

        ratio = len(entailed) / total_claims if total_claims > 0 else 1.0

        return NLIEvaluationReport(
            upstream_agent=upstream_agent,
            downstream_agent=downstream_agent,
            claims=claim_scores,
            has_contradiction=len(contradicted) > 0,
            contradicted_claims=contradicted,
            entailed_claims=entailed,
            overall_entailment_ratio=round(ratio, 4),
            fallback_mode_used=self.fallback_mode,
        )

    def evaluate_steps(
        self,
        upstream_step: AgentStep,
        downstream_step: AgentStep,
    ) -> NLIEvaluationReport:
        """
        Evaluate semantic consistency between two sequential or linked AgentStep objects.
        """
        up_parts = [upstream_step.output or ""]
        if upstream_step.input:
            up_parts.append(str(upstream_step.input))
        if upstream_step.handoff:
            if upstream_step.handoff.output_state:
                up_parts.append(str(upstream_step.handoff.output_state))
            if upstream_step.handoff.input_state:
                up_parts.append(str(upstream_step.handoff.input_state))

        down_parts = [downstream_step.output or ""]
        if downstream_step.handoff and downstream_step.handoff.output_state:
            down_parts.append(str(downstream_step.handoff.output_state))

        return self.evaluate_claims(
            premise="\n".join(up_parts),
            downstream_text="\n".join(down_parts),
            upstream_agent=upstream_step.agent,
            downstream_agent=downstream_step.agent,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Singleton Getter
# ─────────────────────────────────────────────────────────────────────────────

_GLOBAL_NLI_ENGINE: NLIEngine | None = None


def get_nli_engine(
    model_name: str = "cross-encoder/nli-deberta-v3-small",
    mode: str = "auto",
    contradiction_threshold: float = 0.70,
    entailment_threshold: float = 0.75,
) -> NLIEngine:
    """Retrieve or initialize the global NLIEngine singleton."""
    global _GLOBAL_NLI_ENGINE
    if _GLOBAL_NLI_ENGINE is None:
        _GLOBAL_NLI_ENGINE = NLIEngine(
            model_name=model_name,
            mode=mode,
            contradiction_threshold=contradiction_threshold,
            entailment_threshold=entailment_threshold,
        )
    return _GLOBAL_NLI_ENGINE
