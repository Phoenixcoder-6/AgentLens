# tests/test_schema.py
"""
Tests for schema/models.py — all 7 Pydantic models.

Covers:
  - Instantiation of every model
  - schema_version stamp on every record
  - TokenUsage auto-computation
  - HandoffState field capture
  - RunTrace auto-generated run_id
  - JSON round-trip (serialize → deserialize → equality)
"""

import pytest
from schema import (
    SCHEMA_VERSION,
    RunTrace, AgentStep, WorkflowState, HandoffState,
    AnalysisBundle, EvidenceRecord, RuleMatch,
    FailureCategory, PriorityLevel, EvidenceSource,
    StepStatus, NodeType, RuleSeverity,
    TokenUsage, GenerationParams,
)


# ── SCHEMA_VERSION ────────────────────────────────────────────────────────────

def test_schema_version_is_1_0():
    assert SCHEMA_VERSION == "1.0"


# ── TokenUsage ────────────────────────────────────────────────────────────────

def test_token_usage_auto_total():
    t = TokenUsage(prompt=512, completion=180)
    assert t.total == 692


def test_token_usage_explicit_total():
    t = TokenUsage(prompt=512, completion=180, total=999)
    assert t.total == 999  # explicit value is preserved


def test_token_usage_zeros():
    t = TokenUsage()
    assert t.prompt == 0 and t.completion == 0 and t.total == 0


# ── HandoffState ──────────────────────────────────────────────────────────────

def test_handoff_state_instantiates(sample_handoff_state):
    assert sample_handoff_state is not None


def test_handoff_state_input_fields(sample_handoff_state):
    assert sample_handoff_state.input_state["sources"] == 10
    assert sample_handoff_state.filtered_state["sources"] == 3
    assert sample_handoff_state.output_state["summary_sources_cited"] == 3


def test_handoff_state_detects_loss(sample_handoff_state):
    """Core capability: input_state vs output_state reveals dropped keys."""
    dropped = set(sample_handoff_state.input_state.keys()) - set(sample_handoff_state.output_state.keys())
    assert "topic" in dropped  # topic was in input, gone in output


# ── WorkflowState ─────────────────────────────────────────────────────────────

def test_workflow_state_instantiates():
    ws = WorkflowState(run_id="run_test01", step_index=1, state_data={"topic": "AI"})
    assert ws is not None


def test_workflow_state_schema_version():
    ws = WorkflowState(run_id="run_test01", step_index=1)
    assert ws.schema_version == SCHEMA_VERSION


# ── AgentStep ─────────────────────────────────────────────────────────────────

def test_agent_step_instantiates(sample_agent_step):
    assert sample_agent_step is not None


def test_agent_step_schema_version(sample_agent_step):
    assert sample_agent_step.schema_version == SCHEMA_VERSION


def test_agent_step_tokens_auto_computed(sample_agent_step):
    assert sample_agent_step.tokens.total == 692


def test_agent_step_handoff_captured(sample_agent_step):
    assert sample_agent_step.handoff.input_state["sources"] == 10


def test_agent_step_status_default(sample_agent_step):
    assert sample_agent_step.status == StepStatus.SUCCESS


# ── RunTrace ──────────────────────────────────────────────────────────────────

def test_run_trace_instantiates(sample_run_trace):
    assert sample_run_trace is not None


def test_run_trace_schema_version(sample_run_trace):
    assert sample_run_trace.schema_version == SCHEMA_VERSION


def test_run_trace_run_id_auto_generated(sample_run_trace):
    assert sample_run_trace.run_id.startswith("run_")
    assert len(sample_run_trace.run_id) == 12  # "run_" + 8 hex chars


def test_run_trace_contains_steps(sample_run_trace):
    assert len(sample_run_trace.steps) == 1
    assert sample_run_trace.steps[0].agent == "researcher"


# ── RuleMatch ─────────────────────────────────────────────────────────────────

def test_rule_match_instantiates(sample_rule_match):
    assert sample_rule_match is not None


def test_rule_match_category(sample_rule_match):
    assert sample_rule_match.category == FailureCategory.WORKFLOW


def test_rule_match_severity(sample_rule_match):
    assert sample_rule_match.severity == RuleSeverity.HIGH


def test_rule_match_rule_id_format(sample_rule_match):
    assert sample_rule_match.rule_id.startswith("R-")


# ── EvidenceRecord ────────────────────────────────────────────────────────────

def test_evidence_record_instantiates(sample_evidence_record):
    assert sample_evidence_record is not None


def test_evidence_record_confidence(sample_evidence_record):
    assert sample_evidence_record.confidence == 1.0


def test_evidence_record_rule_match_attached(sample_evidence_record):
    assert sample_evidence_record.rule_match is not None
    assert sample_evidence_record.rule_match.rule_id == "R-WF-001"


def test_evidence_record_confidence_bounds():
    with pytest.raises(Exception):
        EvidenceRecord(
            source=EvidenceSource.RULE_ENGINE,
            description="test",
            confidence=1.5,  # invalid — must be 0.0–1.0
        )


# ── AnalysisBundle ────────────────────────────────────────────────────────────

def test_analysis_bundle_instantiates(sample_run_trace, sample_evidence_record, sample_rule_match):
    bundle = AnalysisBundle(
        run_id=sample_run_trace.run_id,
        primary_cause=FailureCategory.WORKFLOW,
        priority_level=PriorityLevel.P2,
        grounded=False,
        evidence=[sample_evidence_record],
        rule_matches=[sample_rule_match],
        primary_agent="writer",
    )
    assert bundle is not None


def test_analysis_bundle_schema_version(sample_run_trace, sample_evidence_record, sample_rule_match):
    bundle = AnalysisBundle(
        run_id=sample_run_trace.run_id,
        primary_cause=FailureCategory.WORKFLOW,
        priority_level=PriorityLevel.P2,
        grounded=False,
    )
    assert bundle.schema_version == SCHEMA_VERSION


def test_analysis_bundle_grounded_flag(sample_run_trace):
    heuristic = AnalysisBundle(
        run_id=sample_run_trace.run_id,
        primary_cause=FailureCategory.UNKNOWN,
        priority_level=PriorityLevel.P5,
        grounded=False,
    )
    grounded = AnalysisBundle(
        run_id=sample_run_trace.run_id,
        primary_cause=FailureCategory.REASONING,
        priority_level=PriorityLevel.P1,
        grounded=True,
    )
    assert heuristic.grounded is False
    assert grounded.grounded is True


# ── JSON Round-trip ───────────────────────────────────────────────────────────

def test_run_trace_json_round_trip(sample_run_trace):
    json_str = sample_run_trace.model_dump_json()
    restored = RunTrace.model_validate_json(json_str)
    assert restored.run_id == sample_run_trace.run_id
    assert restored.schema_version == SCHEMA_VERSION
    assert len(restored.steps) == 1


def test_analysis_bundle_json_round_trip(sample_run_trace, sample_evidence_record):
    bundle = AnalysisBundle(
        run_id=sample_run_trace.run_id,
        primary_cause=FailureCategory.WORKFLOW,
        priority_level=PriorityLevel.P2,
        grounded=False,
        evidence=[sample_evidence_record],
    )
    json_str = bundle.model_dump_json()
    restored = AnalysisBundle.model_validate_json(json_str)
    assert restored.run_id == bundle.run_id
    assert restored.primary_cause == FailureCategory.WORKFLOW
    assert len(restored.evidence) == 1
