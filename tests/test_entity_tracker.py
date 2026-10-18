"""
tests/test_entity_tracker.py — Day 54: Entity Tracker & Set Overlap Tests
===========================================================================
Tests:
  - Extraction of named entities, acronyms, CamelCase, and financial units
  - Entity normalization (casing, punctuation, possessives)
  - Mathematical set difference logic (E_invented, E_dropped, E_shared)
  - Preservation ratio and hallucination rate calculations
  - Entity substitution detection (where scalar counts match 3 vs 3)
  - Zero false positives on faithful paraphrasing
  - RuleEngine hallucination_v2 firing on substitution
  - RuleEngine hallucination_v2 suppression on upstream execution failure
"""

from __future__ import annotations

from analyzers.detection.rule_engine import RuleEngine
from analyzers.semantic.entity_tracker import (
    EntityDeltaReport,
    EntityTracker,
    compare_entity_sets,
    extract_entities,
    normalize_entity,
)
from schema.models import AgentStep, FailureCategory, RunTrace, StepStatus


class TestEntityExtraction:
    """Test entity parsing, normalization, and pattern matching."""

    def test_proper_nouns_and_acronyms_extraction(self):
        text = (
            "Satya Nadella discussed Microsoft Azure and OpenAI partnerships with the FDA and SEC."
        )
        entities = extract_entities(text)

        assert "satya nadella" in entities or ("satya" in entities and "nadella" in entities)
        assert "microsoft azure" in entities or "microsoft" in entities
        assert "openai" in entities
        assert "fda" in entities
        assert "sec" in entities

    def test_model_identifiers_and_camel_case(self):
        text = "AgentLens benchmarks show that GPT-4o outperforms Claude-3-Opus and Llama-3-70b on PyTorch."
        entities = extract_entities(text)

        assert "gpt-4o" in entities
        assert "claude-3-opus" in entities
        assert "llama-3-70b" in entities
        assert "pytorch" in entities
        assert "agentlens" in entities

    def test_financial_metrics_and_percentages(self):
        text = "Alphabet reported revenue of $88.2 billion, up 15% year-over-year."
        entities = extract_entities(text)

        assert "alphabet" in entities
        assert "$88.2 billion" in entities
        assert "15%" in entities

    def test_normalization_casing_and_possessives(self):
        assert normalize_entity("Apple's") == "apple"
        assert normalize_entity("MICROSOFT,") == "microsoft"
        assert normalize_entity("  Google  ") == "google"
        assert normalize_entity('"Pfizer"') == "pfizer"


class TestMathematicalSetDifference:
    """Test exact mathematical set difference and preservation ratio metrics."""

    def test_set_difference_math(self):
        upstream = {"apple", "google", "microsoft"}
        downstream = {"google", "microsoft", "amazon"}

        report = compare_entity_sets(upstream, downstream, "researcher", "writer")

        assert isinstance(report, EntityDeltaReport)
        assert report.shared_entities == {"google", "microsoft"}
        assert report.invented_entities == {"amazon"}
        assert report.dropped_entities == {"apple"}
        assert abs(report.preservation_ratio - (2 / 3)) < 1e-4
        assert abs(report.hallucination_rate - (1 / 3)) < 1e-4
        assert report.has_hallucination is True
        assert report.has_information_loss is True

    def test_entity_substitution_detection(self):
        """
        Classic substitution failure mode:
        Upstream: 3 tech companies.
        Downstream: 3 pharma companies.
        Scalar count difference is 0 (3 - 3 = 0), but all entities were swapped!
        """
        up_text = "Analysis focused on Apple, Google, and Microsoft performance in Q3."
        down_text = "Analysis focused on Pfizer, Moderna, and BioNTech performance in Q3."

        report = EntityTracker.compare_texts(up_text, down_text, "researcher", "writer")

        assert report.has_hallucination is True
        assert report.substitution_detected is True
        assert "pfizer" in report.invented_entities
        assert "moderna" in report.invented_entities
        assert "biontech" in report.invented_entities
        assert "apple" in report.dropped_entities
        assert report.preservation_ratio < 0.5

    def test_zero_false_positives_on_faithful_synthesis(self):
        """When downstream faithfully preserves upstream entities, zero hallucinations flagged."""
        up_text = "Earnings report: Microsoft and Google achieved record revenue in Seattle."
        down_text = (
            "Executive Summary: Microsoft and Google delivered strong financial results in Seattle."
        )

        report = EntityTracker.compare_texts(up_text, down_text, "researcher", "writer")

        assert report.invented_entities == set()
        assert report.has_hallucination is False
        assert report.preservation_ratio >= 0.9
        assert report.substitution_detected is False


class TestRuleEngineIntegration:
    """Test hallucination_v2 detection in RuleEngine."""

    def test_hallucination_v2_fires_on_substitution_where_v1_fails(self):
        """
        Demonstrate that hallucination_v2 catches entity substitution where hallucination_v1
        is completely blind due to identical scalar entity counts.
        """
        step1 = AgentStep(
            run_id="test_run",
            step=1,
            agent="researcher",
            output="Research completed on Apple, Google, and Microsoft.",
            status=StepStatus.SUCCESS,
        )
        step2 = AgentStep(
            run_id="test_run",
            step=2,
            agent="writer",
            output="Executive report detailing findings for Pfizer, Moderna, and BioNTech.",
            status=StepStatus.SUCCESS,
            parent_step="1",
        )
        trace = RunTrace(run_id="test_run", workflow="test_pipeline", steps=[step1, step2])

        rule_engine = RuleEngine()
        result = rule_engine.analyze(trace)

        assert result.skipped is False
        rule_ids = [e.rule_match.rule_id for e in result.evidence if e.rule_match]

        # hallucination_v1 should NOT fire because scalar entity counts match (3 in, 3 out)
        assert "hallucination_v1" not in rule_ids

        # hallucination_v2 MUST fire because Pfizer, Moderna, BioNTech were invented
        assert "hallucination_v2" in rule_ids

        v2_record = next(
            e
            for e in result.evidence
            if e.rule_match and e.rule_match.rule_id == "hallucination_v2"
        )
        assert v2_record.rule_match.category == FailureCategory.REASONING
        assert v2_record.agent == "writer"
        assert "Entity substitution detected" in v2_record.description

    def test_hallucination_v2_suppressed_on_upstream_execution_failure(self):
        """
        When upstream has an execution error (tool failure), downstream is not falsely
        penalized for hallucination, preserving the upstream root cause.
        """
        step1 = AgentStep(
            run_id="test_run",
            step=1,
            agent="researcher",
            output="",
            tool_calls=[{"name": "search", "error": "TimeoutError after 30s"}],
            status=StepStatus.FAILURE,
        )
        step2 = AgentStep(
            run_id="test_run",
            step=2,
            agent="writer",
            output="Report on general market trends.",
            status=StepStatus.SUCCESS,
            parent_step="1",
        )
        trace = RunTrace(run_id="test_run", workflow="test_pipeline", steps=[step1, step2])

        rule_engine = RuleEngine()
        result = rule_engine.analyze(trace)

        rule_ids = [e.rule_match.rule_id for e in result.evidence if e.rule_match]
        assert "tool_failure_v1" in rule_ids
        assert "hallucination_v2" not in rule_ids
