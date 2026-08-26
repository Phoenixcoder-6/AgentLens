"""
tests/test_consistency_validator.py — Day 22: ConsistencyValidator tests
=========================================================================
Tests:
  - verifier_passthrough_v1: verifier lets hallucinated entities through
  - claim_drift_v1: writer introduces claims not found in researcher output
  - Correct Analyzer interface compliance (analyzer_id, analyze())
  - Graceful handling when extraction fails or is absent
  - compute_grounded: ConsistencyValidator can contribute to grounded evidence

All tests are pure unit tests — no API key required (LLM calls mocked).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from analyzers.detection import ConsistencyValidator, RuleEngine, WorkflowValidator
from analyzers.evidence_extraction.extractor import ExtractedEvidence
from app.interfaces import AnalysisResult, Analyzer
from schema.models import (
    AgentStep,
    FailureCategory,
    RunTrace,
    StepStatus,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_step(agent: str, step: int, output: str = "") -> AgentStep:
    return AgentStep(
        run_id="test_run",
        step=step,
        agent=agent,
        output=output or f"{agent} output",
        status=StepStatus.SUCCESS,
    )


def _make_trace(*steps: AgentStep, workflow: str = "test_pipeline") -> RunTrace:
    return RunTrace(
        run_id="test_run",
        workflow=workflow,
        steps=list(steps),
    )


def _make_evidence(
    source_count: int = 0,
    entity_count: int = 0,
    claims: list[str] | None = None,
    failed: bool = False,
) -> ExtractedEvidence:
    return ExtractedEvidence(
        source_count=source_count,
        entity_count=entity_count,
        claims=claims or [],
        extraction_failed=failed,
    )


def _patch_extractor(*evidence_seq: ExtractedEvidence):
    """
    Context manager: patches EvidenceExtractor so that successive calls to
    .extract() return the given ExtractedEvidence objects in order.
    """
    mock_instance = MagicMock()
    mock_instance.extract.side_effect = list(evidence_seq)
    return patch(
        "analyzers.detection.consistency_validator.EvidenceExtractor",
        return_value=mock_instance,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Analyzer interface compliance
# ─────────────────────────────────────────────────────────────────────────────


class TestAnalyzerInterface:
    def test_is_analyzer_protocol(self):
        """ConsistencyValidator must satisfy the Analyzer Protocol."""
        assert isinstance(ConsistencyValidator(), Analyzer)

    def test_analyzer_id(self):
        assert ConsistencyValidator().analyzer_id == "consistency_validator"

    def test_analyze_returns_analysis_result(self):
        validator = ConsistencyValidator()
        trace = _make_trace(_make_step("researcher", 1))
        result = validator.analyze(trace)
        assert isinstance(result, AnalysisResult)

    def test_empty_trace_returns_skipped(self):
        validator = ConsistencyValidator()
        trace = RunTrace(run_id="x", workflow="test", steps=[])
        result = validator.analyze(trace)
        assert result.skipped is True
        assert result.evidence == []

    def test_analyze_never_raises(self):
        """analyze() must not raise — graceful degradation required."""
        validator = ConsistencyValidator()
        trace = _make_trace(_make_step("researcher", 1))
        try:
            validator.analyze(trace)
        except Exception as e:
            pytest.fail(f"analyze() raised unexpectedly: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# verifier_passthrough_v1
# ─────────────────────────────────────────────────────────────────────────────


class TestVerifierPassthrough:
    def test_fires_when_verifier_passes_hallucinated_entities(self):
        """
        Researcher: 5 entities
        Writer:     10 entities (gain of 5 — hallucinated)
        Verifier:   10 entities (same as writer — passed through unchanged)
        → verifier_passthrough_v1 should fire.
        """
        res = _make_step("researcher", 1)
        wr = _make_step("writer", 2)
        ver = _make_step("verifier", 3)
        trace = _make_trace(res, wr, ver)

        res_ev = _make_evidence(entity_count=5)
        wr_ev = _make_evidence(entity_count=10)
        ver_ev = _make_evidence(entity_count=10)

        with patch("os.getenv", return_value="fake_key"), _patch_extractor(res_ev, wr_ev, ver_ev):
            validator = ConsistencyValidator()
            result = validator.analyze(trace)

        rule_ids = [e.rule_match.rule_id for e in result.evidence if e.rule_match]
        assert "verifier_passthrough_v1" in rule_ids

    def test_does_not_fire_when_no_entity_gain(self):
        """
        Researcher: 5 entities, Writer: 5 entities → no hallucination → no fire.
        """
        res = _make_step("researcher", 1)
        wr = _make_step("writer", 2)
        ver = _make_step("verifier", 3)
        trace = _make_trace(res, wr, ver)

        res_ev = _make_evidence(entity_count=5)
        wr_ev = _make_evidence(entity_count=5)
        ver_ev = _make_evidence(entity_count=5)

        with patch("os.getenv", return_value="fake_key"), _patch_extractor(res_ev, wr_ev, ver_ev):
            validator = ConsistencyValidator()
            result = validator.analyze(trace)

        rule_ids = [e.rule_match.rule_id for e in result.evidence if e.rule_match]
        assert "verifier_passthrough_v1" not in rule_ids

    def test_does_not_fire_when_verifier_reduces_entities(self):
        """
        Writer had hallucinated entities, but verifier caught and removed them.
        → No fire (verifier did its job).
        """
        res = _make_step("researcher", 1)
        wr = _make_step("writer", 2)
        ver = _make_step("verifier", 3)
        trace = _make_trace(res, wr, ver)

        res_ev = _make_evidence(entity_count=5)
        wr_ev = _make_evidence(entity_count=10)
        ver_ev = _make_evidence(entity_count=7)  # verifier removed some

        with patch("os.getenv", return_value="fake_key"), _patch_extractor(res_ev, wr_ev, ver_ev):
            validator = ConsistencyValidator()
            result = validator.analyze(trace)

        rule_ids = [e.rule_match.rule_id for e in result.evidence if e.rule_match]
        assert "verifier_passthrough_v1" not in rule_ids

    def test_skips_when_extraction_failed(self):
        """If extraction fails for any step, verifier rule must be skipped."""
        res = _make_step("researcher", 1)
        wr = _make_step("writer", 2)
        ver = _make_step("verifier", 3)
        trace = _make_trace(res, wr, ver)

        res_ev = _make_evidence(entity_count=5, failed=True)  # extraction failed
        wr_ev = _make_evidence(entity_count=10)
        ver_ev = _make_evidence(entity_count=10)

        with patch("os.getenv", return_value="fake_key"), _patch_extractor(res_ev, wr_ev, ver_ev):
            validator = ConsistencyValidator()
            result = validator.analyze(trace)

        rule_ids = [e.rule_match.rule_id for e in result.evidence if e.rule_match]
        assert "verifier_passthrough_v1" not in rule_ids

    def test_verifier_passthrough_evidence_has_correct_category(self):
        res = _make_step("researcher", 1)
        wr = _make_step("writer", 2)
        ver = _make_step("verifier", 3)
        trace = _make_trace(res, wr, ver)

        res_ev = _make_evidence(entity_count=3)
        wr_ev = _make_evidence(entity_count=8)
        ver_ev = _make_evidence(entity_count=8)

        with patch("os.getenv", return_value="fake_key"), _patch_extractor(res_ev, wr_ev, ver_ev):
            validator = ConsistencyValidator()
            result = validator.analyze(trace)

        passthrough_records = [
            e for e in result.evidence
            if e.rule_match and e.rule_match.rule_id == "verifier_passthrough_v1"
        ]
        assert len(passthrough_records) == 1
        assert passthrough_records[0].rule_match.category == FailureCategory.VERIFICATION


# ─────────────────────────────────────────────────────────────────────────────
# claim_drift_v1
# ─────────────────────────────────────────────────────────────────────────────


class TestClaimDrift:
    def test_fires_when_writer_adds_new_claims(self):
        """
        Researcher claims: ["Apollo 11 landed July 20, 1969"]
        Writer claims:     ["Apollo 11 landed July 20, 1969", "Buzz Aldrin walked on Mars"]
        → claim_drift_v1 should fire for the new claim.
        """
        res = _make_step("researcher", 1)
        wr = _make_step("writer", 2)
        trace = _make_trace(res, wr)

        res_ev = _make_evidence(
            entity_count=3,
            claims=["Apollo 11 landed July 20, 1969"],
        )
        wr_ev = _make_evidence(
            entity_count=5,
            claims=["Apollo 11 landed July 20, 1969", "Buzz Aldrin walked on Mars"],
        )

        with patch("os.getenv", return_value="fake_key"), _patch_extractor(res_ev, wr_ev):
            validator = ConsistencyValidator()
            result = validator.analyze(trace)

        rule_ids = [e.rule_match.rule_id for e in result.evidence if e.rule_match]
        assert "claim_drift_v1" in rule_ids

    def test_does_not_fire_when_claims_are_subset(self):
        """
        Writer claims are a strict subset of researcher claims → no new claims → no fire.
        """
        res = _make_step("researcher", 1)
        wr = _make_step("writer", 2)
        trace = _make_trace(res, wr)

        res_ev = _make_evidence(claims=["Claim A", "Claim B", "Claim C"])
        wr_ev = _make_evidence(claims=["Claim A", "Claim B"])

        with patch("os.getenv", return_value="fake_key"), _patch_extractor(res_ev, wr_ev):
            validator = ConsistencyValidator()
            result = validator.analyze(trace)

        rule_ids = [e.rule_match.rule_id for e in result.evidence if e.rule_match]
        assert "claim_drift_v1" not in rule_ids

    def test_does_not_fire_when_researcher_has_no_claims(self):
        """
        If researcher had no extractable claims (empty list), claim_drift_v1
        cannot fire — we need a baseline to compare against.
        """
        res = _make_step("researcher", 1)
        wr = _make_step("writer", 2)
        trace = _make_trace(res, wr)

        res_ev = _make_evidence(claims=[])  # no claims
        wr_ev = _make_evidence(claims=["New claim from nowhere"])

        with patch("os.getenv", return_value="fake_key"), _patch_extractor(res_ev, wr_ev):
            validator = ConsistencyValidator()
            result = validator.analyze(trace)

        rule_ids = [e.rule_match.rule_id for e in result.evidence if e.rule_match]
        assert "claim_drift_v1" not in rule_ids

    def test_claim_comparison_is_case_insensitive(self):
        """
        'Apollo 11 landed...' and 'apollo 11 landed...' should be treated as the same claim.
        """
        res = _make_step("researcher", 1)
        wr = _make_step("writer", 2)
        trace = _make_trace(res, wr)

        res_ev = _make_evidence(claims=["Apollo 11 landed July 20, 1969"])
        wr_ev = _make_evidence(claims=["apollo 11 landed july 20, 1969"])  # same, lowercase

        with patch("os.getenv", return_value="fake_key"), _patch_extractor(res_ev, wr_ev):
            validator = ConsistencyValidator()
            result = validator.analyze(trace)

        rule_ids = [e.rule_match.rule_id for e in result.evidence if e.rule_match]
        assert "claim_drift_v1" not in rule_ids

    def test_claim_drift_evidence_has_correct_category(self):
        res = _make_step("researcher", 1)
        wr = _make_step("writer", 2)
        trace = _make_trace(res, wr)

        res_ev = _make_evidence(claims=["Claim A"])
        wr_ev = _make_evidence(claims=["Claim A", "Hallucinated claim"])

        with patch("os.getenv", return_value="fake_key"), _patch_extractor(res_ev, wr_ev):
            validator = ConsistencyValidator()
            result = validator.analyze(trace)

        drift_records = [
            e for e in result.evidence
            if e.rule_match and e.rule_match.rule_id == "claim_drift_v1"
        ]
        assert len(drift_records) == 1
        assert drift_records[0].rule_match.category == FailureCategory.VERIFICATION
        assert drift_records[0].agent == "writer"

    def test_skips_when_extraction_failed_for_writer(self):
        res = _make_step("researcher", 1)
        wr = _make_step("writer", 2)
        trace = _make_trace(res, wr)

        res_ev = _make_evidence(claims=["Claim A"])
        wr_ev = _make_evidence(claims=["Claim A", "Extra"], failed=True)  # failed

        with patch("os.getenv", return_value="fake_key"), _patch_extractor(res_ev, wr_ev):
            validator = ConsistencyValidator()
            result = validator.analyze(trace)

        rule_ids = [e.rule_match.rule_id for e in result.evidence if e.rule_match]
        assert "claim_drift_v1" not in rule_ids


# ─────────────────────────────────────────────────────────────────────────────
# No API key — graceful skip
# ─────────────────────────────────────────────────────────────────────────────


class TestNoApiKey:
    def test_no_evidence_without_api_key(self):
        """Without GROQ_API_KEY, extraction-dependent rules produce no evidence."""
        res = _make_step("researcher", 1, "research output")
        wr = _make_step("writer", 2, "writer output")
        ver = _make_step("verifier", 3, "verifier output")
        trace = _make_trace(res, wr, ver)

        with patch("os.getenv", return_value=None):
            validator = ConsistencyValidator()
            result = validator.analyze(trace)

        assert result.evidence == []
        assert result.skipped is False  # not skipped, just no evidence


# ─────────────────────────────────────────────────────────────────────────────
# Grounded evaluation
# ─────────────────────────────────────────────────────────────────────────────


class TestGroundedEvaluation:
    """
    Day 22: compute_grounded — ConsistencyValidator can independently
    contribute P1-quality evidence when extraction data is reliable.
    """

    def test_evidence_has_confidence_1_when_rule_fires(self):
        """All ConsistencyValidator evidence records carry confidence=1.0."""
        res = _make_step("researcher", 1)
        wr = _make_step("writer", 2)
        ver = _make_step("verifier", 3)
        trace = _make_trace(res, wr, ver)

        res_ev = _make_evidence(entity_count=3, claims=["Claim A"])
        wr_ev = _make_evidence(entity_count=8, claims=["Claim A", "Hallucinated"])
        ver_ev = _make_evidence(entity_count=8)

        with patch("os.getenv", return_value="fake_key"), _patch_extractor(res_ev, wr_ev, ver_ev):
            validator = ConsistencyValidator()
            result = validator.analyze(trace)

        for record in result.evidence:
            assert record.confidence == 1.0

    def test_multiple_rules_can_fire_independently(self):
        """Both verifier_passthrough_v1 and claim_drift_v1 can fire in same trace."""
        res = _make_step("researcher", 1)
        wr = _make_step("writer", 2)
        ver = _make_step("verifier", 3)
        trace = _make_trace(res, wr, ver)

        res_ev = _make_evidence(entity_count=3, claims=["Claim A"])
        wr_ev = _make_evidence(entity_count=8, claims=["Claim A", "Hallucinated claim"])
        ver_ev = _make_evidence(entity_count=8)

        with patch("os.getenv", return_value="fake_key"), _patch_extractor(res_ev, wr_ev, ver_ev):
            validator = ConsistencyValidator()
            result = validator.analyze(trace)

        rule_ids = {e.rule_match.rule_id for e in result.evidence if e.rule_match}
        assert "verifier_passthrough_v1" in rule_ids
        assert "claim_drift_v1" in rule_ids
        assert len(result.evidence) == 2

    def test_analyzer_id_is_correctly_set_in_result(self):
        validator = ConsistencyValidator()
        trace = _make_trace(_make_step("researcher", 1))
        result = validator.analyze(trace)
        assert result.analyzer_id == "consistency_validator"


# ─────────────────────────────────────────────────────────────────────────────
# Package-level import test (Day 22 __init__.py)
# ─────────────────────────────────────────────────────────────────────────────


class TestDetectionPackageExports:
    def test_all_three_analyzers_importable_from_package(self):
        from analyzers.detection import ConsistencyValidator

        assert RuleEngine().analyzer_id == "rule_engine"
        assert WorkflowValidator().analyzer_id == "workflow_validator"
        assert ConsistencyValidator().analyzer_id == "consistency_validator"

    def test_all_three_satisfy_analyzer_protocol(self):
        from analyzers.detection import ConsistencyValidator

        for cls in (RuleEngine, WorkflowValidator, ConsistencyValidator):
            assert isinstance(cls(), Analyzer), f"{cls.__name__} must implement Analyzer"
