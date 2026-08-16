from __future__ import annotations

from analyzers.detection.workflow_validator import WorkflowValidator
from schema.models import AgentStep, FailureCategory, RunTrace


def test_workflow_validator_skipped_when_no_steps():
    trace = RunTrace(workflow="test", steps=[])
    validator = WorkflowValidator()
    result = validator.analyze(trace)
    assert result.skipped is True


def test_skipped_step_v1():
    trace = RunTrace(
        workflow="test",
        steps=[
            AgentStep(run_id="run1", step=1, agent="researcher"),
            AgentStep(run_id="run1", step=2, agent="writer"),
            # missing verifier
        ],
    )
    validator = WorkflowValidator()
    result = validator.analyze(trace)

    assert result.skipped is False
    assert len(result.evidence) == 1
    ev = result.evidence[0]
    assert ev.rule_match.rule_id == "skipped_step_v1"
    assert ev.rule_match.category == FailureCategory.WORKFLOW
    assert ev.agent == "verifier"


def test_wrong_order_v1():
    trace = RunTrace(
        workflow="test",
        steps=[
            AgentStep(run_id="run1", step=1, agent="writer"),
            AgentStep(run_id="run1", step=2, agent="researcher"),
            AgentStep(run_id="run1", step=3, agent="verifier"),
        ],
    )
    validator = WorkflowValidator()
    result = validator.analyze(trace)

    assert result.skipped is False
    # writer before researcher -> wrong order
    # The rule checks required_agents = ["researcher", "writer", "verifier"]
    # So researcher should be before writer. In this trace, researcher is at idx 1, writer at idx 0
    # idx of writer < idx of researcher -> false. Wait, the code checks:
    # idx_b < idx_a where agent_a="researcher", agent_b="writer".
    # idx_b (writer) = 0, idx_a (researcher) = 1. So 0 < 1 is True.
    assert len(result.evidence) == 1
    ev = result.evidence[0]
    assert ev.rule_match.rule_id == "wrong_order_v1"
    assert ev.rule_match.category == FailureCategory.WORKFLOW
    assert ev.agent == "writer"
