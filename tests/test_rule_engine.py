from __future__ import annotations

from analyzers.detection.rule_engine import RuleEngine
from schema.models import AgentStep, FailureCategory, RunTrace


def test_rule_engine_skipped_when_no_steps():
    trace = RunTrace(workflow="test", steps=[])
    engine = RuleEngine()
    result = engine.analyze(trace)
    assert result.skipped is True
    assert "No steps" in result.skip_reason


def test_tool_failure_v1():
    trace = RunTrace(
        workflow="test",
        steps=[
            AgentStep(
                run_id="run1",
                step=1,
                agent="researcher",
                tool_calls=[
                    {"name": "search", "args": {}, "output": "Error: timeout", "error": ""}
                ],
            )
        ],
    )
    engine = RuleEngine()
    result = engine.analyze(trace)

    assert result.skipped is False
    assert len(result.evidence) == 1
    ev = result.evidence[0]
    assert ev.rule_match.rule_id == "tool_failure_v1"
    assert ev.rule_match.category == FailureCategory.EXECUTION
    assert ev.agent == "researcher"


def test_missing_tool_output_v1():
    trace = RunTrace(
        workflow="test",
        steps=[
            AgentStep(
                run_id="run1",
                step=1,
                agent="researcher",
                tool_calls=[{"name": "search", "args": {}, "output": "", "error": ""}],
            )
        ],
    )
    engine = RuleEngine()
    result = engine.analyze(trace)

    assert result.skipped is False
    assert len(result.evidence) == 1
    ev = result.evidence[0]
    assert ev.rule_match.rule_id == "missing_tool_output_v1"
    assert ev.rule_match.category == FailureCategory.EXECUTION
    assert ev.agent == "researcher"
