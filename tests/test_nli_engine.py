"""
tests/test_nli_engine.py — Day 55: Local Cross-Encoder NLI Engine Tests
========================================================================
Tests:
  - Claim / proposition splitting from unstructured text and lists
  - Direct contradiction detection (failed to meet vs successfully met)
  - Opposing verb contradiction detection (rejected vs approved)
  - Faithful entailment scoring
  - Neutral / unrelated statement scoring
  - Multi-claim report evaluation and contradiction filtering
  - Step-level evaluation with AgentStep objects
  - Graceful fallback mode (zero external dependency / offline mode)
  - RuleEngine contradiction_v1 firing on semantic contradiction
  - RuleEngine contradiction_v1 suppression on upstream execution error
"""

from __future__ import annotations

from analyzers.detection.rule_engine import RuleEngine
from analyzers.semantic.nli_engine import (
    NLIEngine,
    NLIEvaluationReport,
    NLILabel,
    evaluate_pair_lexical_fallback,
    split_claims,
)
from schema.models import AgentStep, FailureCategory, RunTrace, StepStatus


class TestClaimDecomposition:
    """Test claim proposition extraction and splitting."""

    def test_split_claims_sentences_and_bullets(self) -> None:
        text = (
            "1. Apollo 11 landed on the lunar surface on July 20, 1969.\n"
            "- Astronauts Neil Armstrong and Buzz Aldrin explored the Moon.\n"
            "Michael Collins orbited in the Command Module. Ok."
        )
        claims = split_claims(text)

        assert len(claims) == 3
        assert "Apollo 11 landed on the lunar surface on July 20, 1969" in claims[0]
        assert "Neil Armstrong and Buzz Aldrin explored the Moon" in claims[1]
        assert "Michael Collins orbited in the Command Module" in claims[2]
        # "Ok." is too short (< 15 chars) so filtered out

    def test_split_claims_empty_or_whitespace(self) -> None:
        assert split_claims("") == []
        assert split_claims("   \n\t  ") == []


class TestNLIEvaluation:
    """Test NLI scoring, contradiction detection, and entailment."""

    def test_direct_polar_contradiction_detection(self) -> None:
        engine = NLIEngine(mode="fallback")
        premise = "Clinical trial NCT-0492 failed to achieve primary efficacy endpoints."
        hypothesis = "Clinical trial NCT-0492 successfully met primary efficacy endpoints."

        score = engine.score_claim(premise, hypothesis)

        assert score.is_contradiction is True
        assert score.predicted_label == NLILabel.CONTRADICTION
        assert score.contradiction_prob >= 0.70
        assert score.entailment_prob < 0.20

    def test_opposing_verbs_contradiction(self) -> None:
        engine = NLIEngine(mode="fallback")
        premise = "The FDA regulatory committee rejected the new drug application."
        hypothesis = "The FDA regulatory committee approved the new drug application."

        score = engine.score_claim(premise, hypothesis)

        assert score.is_contradiction is True
        assert score.contradiction_prob >= 0.70

    def test_faithful_synthesis_entailment(self) -> None:
        engine = NLIEngine(mode="fallback")
        premise = (
            "DeepMind released AlphaFold predicting protein 3D structures with atomic accuracy."
        )
        hypothesis = (
            "AlphaFold was released by DeepMind predicting protein structures with high accuracy."
        )

        score = engine.score_claim(premise, hypothesis)

        assert score.is_entailed is True
        assert score.is_contradiction is False
        assert score.predicted_label == NLILabel.ENTAILMENT
        assert score.entailment_prob >= 0.75

    def test_neutral_claim_scoring(self) -> None:
        engine = NLIEngine(mode="fallback")
        premise = "Alphabet reported Q3 revenue of $88.2 billion in mountain view."
        hypothesis = "Quantum computing research requires cryogenic cooling hardware."

        score = engine.score_claim(premise, hypothesis)

        assert score.predicted_label == NLILabel.NEUTRAL
        assert score.is_contradiction is False
        assert score.neutral_prob >= 0.50

    def test_evaluate_claims_multi_claim_report(self) -> None:
        engine = NLIEngine(mode="fallback")
        premise = (
            "SpaceX Starship completed its third flight test reaching orbital velocity. "
            "However, Starship was lost during atmospheric reentry over the Indian Ocean."
        )
        downstream = (
            "SpaceX Starship completed its third test flight reaching orbital speed. "
            "Starship safely landed intact on the launch pad without any heat shield damage."
        )

        report = engine.evaluate_claims(premise, downstream, "researcher", "writer")

        assert isinstance(report, NLIEvaluationReport)
        assert len(report.claims) == 2
        assert report.has_contradiction is True
        assert len(report.contradicted_claims) == 1
        assert "safely landed intact" in report.contradicted_claims[0].claim

    def test_evaluate_steps_integration(self) -> None:
        engine = NLIEngine(mode="fallback")
        step1 = AgentStep(
            run_id="run_nli_01",
            step=1,
            agent="researcher",
            output="Quarterly operating expenses increased by 14 percent year-over-year.",
            status=StepStatus.SUCCESS,
        )
        step2 = AgentStep(
            run_id="run_nli_01",
            step=2,
            agent="writer",
            output="Quarterly operating expenses decreased significantly compared to last year.",
            status=StepStatus.SUCCESS,
            parent_step="1",
        )

        report = engine.evaluate_steps(step1, step2)

        assert report.has_contradiction is True
        assert report.upstream_agent == "researcher"
        assert report.downstream_agent == "writer"

    def test_lexical_fallback_deterministic_output(self) -> None:
        p_ent, p_neu, p_con = evaluate_pair_lexical_fallback(
            "The model refused the user query.",
            "The model accepted the user query.",
        )
        assert p_con >= 0.70
        assert abs((p_ent + p_neu + p_con) - 1.0) < 1e-4


class TestRuleEngineNLIIntegration:
    """Test contradiction_v1 integration into RuleEngine."""

    def test_rule_engine_fires_contradiction_v1(self) -> None:
        step1 = AgentStep(
            run_id="contra_run",
            step=1,
            agent="researcher",
            output="The clinical trial failed to meet primary endpoints in adult patients.",
            status=StepStatus.SUCCESS,
        )
        step2 = AgentStep(
            run_id="contra_run",
            step=2,
            agent="writer",
            output="The clinical trial successfully met primary endpoints in adult patients.",
            status=StepStatus.SUCCESS,
            parent_step="1",
        )
        trace = RunTrace(run_id="contra_run", workflow="pharma_eval", steps=[step1, step2])

        rule_engine = RuleEngine()
        result = rule_engine.analyze(trace)

        rule_ids = [e.rule_match.rule_id for e in result.evidence if e.rule_match]
        assert "contradiction_v1" in rule_ids

        contra_ev = next(
            e
            for e in result.evidence
            if e.rule_match and e.rule_match.rule_id == "contradiction_v1"
        )

        rule_match = contra_ev.rule_match
        assert rule_match is not None
        assert rule_match.category == FailureCategory.REASONING
        assert contra_ev.agent == "writer"
        assert "contradicting upstream" in contra_ev.description

    def test_rule_engine_contradiction_suppressed_on_upstream_execution_failure(self) -> None:
        step1 = AgentStep(
            run_id="contra_exec_err",
            step=1,
            agent="researcher",
            output="",
            tool_calls=[{"name": "fetch_data", "error": "DatabaseConnectionTimeout"}],
            status=StepStatus.FAILURE,
        )
        step2 = AgentStep(
            run_id="contra_exec_err",
            step=2,
            agent="writer",
            output="The system reported complete success and zero errors.",
            status=StepStatus.SUCCESS,
            parent_step="1",
        )
        trace = RunTrace(run_id="contra_exec_err", workflow="pharma_eval", steps=[step1, step2])

        rule_engine = RuleEngine()
        result = rule_engine.analyze(trace)

        rule_ids = [e.rule_match.rule_id for e in result.evidence if e.rule_match]
        assert "tool_failure_v1" in rule_ids
        assert "contradiction_v1" not in rule_ids
