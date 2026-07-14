# tests/conftest.py
"""
Shared pytest fixtures for the AgentLens test suite.
All tests import fixtures from here via pytest's automatic discovery.
"""
import sys
import os

import pytest

# Ensure project root is on sys.path for all tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def sample_handoff_state():
    from schema import HandoffState
    return HandoffState(
        input_state={"sources": 10, "topic": "AI"},
        filtered_state={"sources": 3},
        output_state={"summary_sources_cited": 3},
    )


@pytest.fixture
def sample_agent_step(sample_handoff_state):
    from schema import AgentStep, TokenUsage, NodeType, StepStatus
    return AgentStep(
        run_id="run_test01",
        step=1,
        agent="researcher",
        node_type=NodeType.LLM,
        input="Summarize findings from 10 sources",
        output="Tesla was founded in 2003...",
        latency_ms=1400.0,
        tokens=TokenUsage(prompt=512, completion=180),
        handoff=sample_handoff_state,
        status=StepStatus.SUCCESS,
    )


@pytest.fixture
def sample_run_trace(sample_agent_step):
    from schema import RunTrace
    return RunTrace(
        workflow="research_report_pipeline",
        steps=[sample_agent_step],
    )


@pytest.fixture
def sample_rule_match():
    from schema import RuleMatch, FailureCategory, RuleSeverity
    return RuleMatch(
        rule_id="R-WF-001",
        category=FailureCategory.WORKFLOW,
        description="Information loss: sources dropped from 10 to 3",
        severity=RuleSeverity.HIGH,
        agent="writer",
        step=2,
        evidence_detail="sources: 10 -> 3",
    )


@pytest.fixture
def sample_evidence_record(sample_rule_match):
    from schema import EvidenceRecord, EvidenceSource
    return EvidenceRecord(
        source=EvidenceSource.RULE_ENGINE,
        description="Information loss in writer handoff",
        rule_match=sample_rule_match,
        agent="writer",
        step=2,
        confidence=1.0,
    )
