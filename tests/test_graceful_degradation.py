"""
tests/test_graceful_degradation.py — Day 23: Graceful Degradation & Fallback Tests
===================================================================================
Tests:
  - Extraction failure fallback (both primary & retry fail -> returns extraction_failed=True, no crash)
  - Rule engines skip extraction rules gracefully when extraction_failed=True
  - Explainer fallback to rule-based summary when Groq API is offline / raises error
  - Primary model error triggers fallback model attempt before failing
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from analyzers.detection import ConsistencyValidator, RuleEngine
from analyzers.evidence_extraction.extractor import EvidenceExtractor, ExtractedEvidence
from analyzers.explainer import LLMExplainer
from schema.models import (
    AgentStep,
    AnalysisBundle,
    FailureCategory,
    PriorityLevel,
    RunTrace,
    StepStatus,
)


def _make_step(agent: str, step: int, output: str = "") -> AgentStep:
    return AgentStep(
        run_id="test_run",
        step=step,
        agent=agent,
        output=output or f"{agent} output",
        status=StepStatus.SUCCESS,
    )


def _make_trace(*steps: AgentStep) -> RunTrace:
    return RunTrace(
        run_id="test_run",
        workflow="test_workflow",
        steps=list(steps),
    )


class TestExtractionGracefulDegradation:
    def test_extraction_double_failure_returns_failed_object(self):
        """When LLM calls fail on both attempts, extract() returns extraction_failed=True without crashing."""
        extractor = EvidenceExtractor()

        # Mock _call_llm to raise exception on both attempts
        with patch.object(extractor, "_call_llm", side_effect=RuntimeError("API offline")):
            res = extractor.extract("Some raw output", agent="researcher")

        assert isinstance(res, ExtractedEvidence)
        assert res.extraction_failed is True
        assert "RuntimeError" in res.error_message

    def test_rule_engine_skips_rules_on_extraction_failure(self):
        """RuleEngine skips researcher_quality_v1 and hallucination_v1 when extraction_failed=True."""
        trace = _make_trace(_make_step("researcher", 1), _make_step("writer", 2))

        failed_ev = ExtractedEvidence(extraction_failed=True, error_message="Mocked failure")

        with patch("analyzers.detection.rule_engine.EvidenceExtractor") as MockExt:
            mock_inst = MagicMock()
            mock_inst.extract.return_value = failed_ev
            MockExt.return_value = mock_inst

            with patch("os.getenv", return_value="fake_key"):
                engine = RuleEngine()
                res = engine.analyze(trace)

        rule_ids = [e.rule_match.rule_id for e in res.evidence if e.rule_match]
        assert "researcher_quality_v1" not in rule_ids
        assert "hallucination_v1" not in rule_ids

    def test_consistency_validator_skips_rules_on_extraction_failure(self):
        """ConsistencyValidator skips verifier_passthrough_v1 and claim_drift_v1 when extraction_failed=True."""
        trace = _make_trace(_make_step("researcher", 1), _make_step("writer", 2), _make_step("verifier", 3))

        failed_ev = ExtractedEvidence(extraction_failed=True, error_message="Mocked failure")

        with patch("analyzers.detection.consistency_validator.EvidenceExtractor") as MockExt:
            mock_inst = MagicMock()
            mock_inst.extract.return_value = failed_ev
            MockExt.return_value = mock_inst

            with patch("os.getenv", return_value="fake_key"):
                validator = ConsistencyValidator()
                res = validator.analyze(trace)

        rule_ids = [e.rule_match.rule_id for e in res.evidence if e.rule_match]
        assert "verifier_passthrough_v1" not in rule_ids
        assert "claim_drift_v1" not in rule_ids


class TestExplainerGracefulDegradation:
    def test_explainer_falls_back_on_api_error(self):
        """When LLM API fails, LLMExplainer populates rule-based fallback summary/fix without raising."""
        bundle = AnalysisBundle(
            run_id="run_123",
            workflow="test_wf",
            primary_cause=FailureCategory.REASONING,
            priority_level=PriorityLevel.P2,
            primary_agent="writer",
            grounded=False,
        )

        with patch("os.getenv", return_value="fake_key"):
            explainer = LLMExplainer()
            explainer._cache = MagicMock()
            explainer._cache.get.return_value = None

            # Force LLM call to fail
            mock_llm = MagicMock()
            mock_llm.with_structured_output.side_effect = RuntimeError("Groq API 503")
            explainer._llm = mock_llm

            res_bundle = explainer.explain(bundle)

        assert res_bundle.summary is not None
        assert len(res_bundle.summary) > 0
        assert res_bundle.suggested_fix is not None
        assert "REASONING" in res_bundle.summary or "reasoning" in res_bundle.summary.lower()


class TestFallbackModelExecution:
    def test_extractor_uses_fallback_model_when_primary_fails(self):
        """When primary model fails, EvidenceExtractor uses fallback model if configured."""
        extractor = EvidenceExtractor()
        extractor._cache = MagicMock()
        extractor._cache.get.return_value = None

        mock_primary = MagicMock()
        mock_primary.invoke.side_effect = RuntimeError("Primary model 429 rate limit")

        mock_fallback = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = '{"source_count": 2, "entity_count": 4}'
        mock_resp.response_metadata = {"token_usage": {"total_tokens": 120}}
        mock_fallback.invoke.return_value = mock_resp

        extractor._llm = mock_primary
        extractor._fallback_llm = mock_fallback
        extractor._fallback_model_name = "llama-3.3-70b-versatile"

        raw, tokens = extractor._call_llm("sys", "user")
        assert raw == '{"source_count": 2, "entity_count": 4}'
        assert tokens == 120
        mock_primary.invoke.assert_called_once()
        mock_fallback.invoke.assert_called_once()
