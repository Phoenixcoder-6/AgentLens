from __future__ import annotations

from analyzers.detection.ground_truth import GroundTruthValidator
from schema.models import AgentStep, EvidenceSource, FailureCategory, RunTrace


def test_ground_truth_skipped_when_no_expected_output():
    trace = RunTrace(
        workflow="test",
        steps=[AgentStep(run_id="run1", step=1, agent="writer", output="hello")],
        expected_output=None,
    )
    validator = GroundTruthValidator()
    result = validator.analyze(trace)

    assert result.skipped is True
    assert "No expected_output" in result.skip_reason
    assert not result.evidence


def test_ground_truth_skipped_when_no_steps():
    trace = RunTrace(
        workflow="test",
        steps=[],
        expected_output="expected",
    )
    validator = GroundTruthValidator()
    result = validator.analyze(trace)

    assert result.skipped is True
    assert "No steps" in result.skip_reason
    assert not result.evidence


def test_ground_truth_fires_on_mismatch():
    trace = RunTrace(
        workflow="test",
        steps=[
            AgentStep(run_id="run1", step=1, agent="writer", output="This is completely different.")
        ],
        expected_output="The expected output is this exact sentence.",
    )
    validator = GroundTruthValidator()
    result = validator.analyze(trace)

    assert result.skipped is False
    assert len(result.evidence) == 1

    ev = result.evidence[0]
    assert ev.source == EvidenceSource.GROUND_TRUTH
    assert ev.value == "FAIL"
    assert ev.agent == "writer"
    assert ev.rule_match is not None
    assert ev.rule_match.rule_id == "gt_mismatch_v1"
    assert ev.rule_match.category == FailureCategory.REASONING
    assert ev.confidence > 0.0


def test_ground_truth_passes_on_high_similarity():
    # Only minor differences like a missing period
    trace = RunTrace(
        workflow="test",
        steps=[
            AgentStep(
                run_id="run1",
                step=1,
                agent="writer",
                output="The expected output is this exact sentence",
            )
        ],
        expected_output="The expected output is this exact sentence.",
    )
    validator = GroundTruthValidator()
    result = validator.analyze(trace)

    assert result.skipped is False
    assert len(result.evidence) == 0
