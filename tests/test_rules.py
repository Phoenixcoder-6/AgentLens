"""
tests/test_rules.py

Day 19 — Rule Testing against the 20-run labeled set.

Goals:
  1. Every implemented rule fires at least once against the labeled traces.
  2. No rule fires on any PASS run (zero false positives).
  3. All rules correctly identify the expected agent on FAIL runs.

Labeled set breakdown (sample_data/labels.json):
  - 4 PASS runs            → no rules should fire
  - 4 execution failures   → tool_failure_v1 or missing_tool_output_v1
  - 4 reasoning failures   → hallucination_v1
  - 4 workflow failures    → skipped_step_v1
  - 4 verification failures → hallucination_v1 + verifier_passthrough_v1
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analyzers.arbiter import determine_primary_cause
from analyzers.detection.rule_engine import RuleEngine
from analyzers.detection.workflow_validator import WorkflowValidator
from schema.models import AgentStep, FailureCategory, PriorityLevel, RunTrace

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

LABELED_DIR = Path(__file__).parent.parent / "sample_data" / "labeled_traces"
LABELS_FILE = Path(__file__).parent.parent / "sample_data" / "labels.json"


def _load_labels() -> dict:
    return json.loads(LABELS_FILE.read_text())


def _load_trace(run_id: str) -> RunTrace:
    """Load a labeled trace JSON and parse it into a RunTrace object."""
    path = LABELED_DIR / f"{run_id}.json"
    raw = json.loads(path.read_text())

    steps = []
    for s in raw["steps"]:
        # Map JSON keys → AgentStep fields
        tool_calls_raw = s.get("tool_calls", [])
        # Normalise tool_call dict: traces use 'result'/'error' keys
        tool_calls = []
        for tc in tool_calls_raw:
            tool_calls.append(
                {
                    "name": tc.get("name", ""),
                    "args": tc.get("args", {}),
                    "output": tc.get("result") or "",
                    "error": tc.get("error") or "",
                }
            )

        steps.append(
            AgentStep(
                run_id=s["run_id"],
                step=s["step"],
                agent=s["agent"],
                output=s.get("output", ""),
                tool_calls=tool_calls,
                error=s.get("error"),
                status=s.get("status", "SUCCESS"),
            )
        )

    return RunTrace(
        workflow=raw.get("workflow", "research_report_pipeline"),
        steps=steps,
        expected_output=raw.get("expected_output"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def labels():
    return _load_labels()


@pytest.fixture(scope="module")
def rule_engine():
    return RuleEngine()


@pytest.fixture(scope="module")
def workflow_validator():
    return WorkflowValidator()


# ─────────────────────────────────────────────────────────────────────────────
# 1. False Positive Check — PASS Runs: No Rule Should Fire
# ─────────────────────────────────────────────────────────────────────────────


class TestPassRunsNoFalsePositives:
    """
    Strict requirement: no deterministic rule should fire on any PASS run.
    Even a single false positive on a PASS run is a regression.
    """

    @pytest.mark.parametrize(
        "run_id",
        [r["run_id"] for r in _load_labels()["runs"] if r["category"] == "pass"],
    )
    def test_no_rule_fires_on_pass(self, run_id, rule_engine, workflow_validator):
        trace = _load_trace(run_id)

        re_result = rule_engine.analyze(trace)
        wv_result = workflow_validator.analyze(trace)

        all_evidence = re_result.evidence + wv_result.evidence

        fired_rules = [e.rule_match.rule_id for e in all_evidence if e.rule_match]
        assert fired_rules == [], (
            f"False positive on PASS run '{run_id}': rules fired = {fired_rules}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Execution Rules Fire on Execution Failure Runs
# ─────────────────────────────────────────────────────────────────────────────


class TestExecutionRules:
    """
    Execution failure traces have tool calls with errors or null output.
    Expect tool_failure_v1 or missing_tool_output_v1 to fire on every run.
    """

    EXECUTION_RULES = {"tool_failure_v1", "missing_tool_output_v1"}

    @pytest.mark.parametrize(
        "run_id",
        [r["run_id"] for r in _load_labels()["runs"] if r["category"] == "execution_failure"],
    )
    def test_execution_rule_fires(self, run_id, rule_engine):
        trace = _load_trace(run_id)
        result = rule_engine.analyze(trace)

        fired = {e.rule_match.rule_id for e in result.evidence if e.rule_match}
        assert fired & self.EXECUTION_RULES, (
            f"Expected an execution rule to fire on '{run_id}', got: {fired}"
        )

    @pytest.mark.parametrize(
        "run_id",
        [r["run_id"] for r in _load_labels()["runs"] if r["category"] == "execution_failure"],
    )
    def test_execution_evidence_category(self, run_id, rule_engine):
        trace = _load_trace(run_id)
        result = rule_engine.analyze(trace)

        exec_evidence = [
            e
            for e in result.evidence
            if e.rule_match and e.rule_match.category == FailureCategory.EXECUTION
        ]
        assert len(exec_evidence) > 0, f"No EXECUTION-category evidence on '{run_id}'"

    def test_at_least_one_tool_failure_fires(self, rule_engine):
        """Regression: tool_failure_v1 must fire at least once across all execution runs."""
        execution_ids = [
            r["run_id"] for r in _load_labels()["runs"] if r["category"] == "execution_failure"
        ]
        fired_any = False
        for run_id in execution_ids:
            trace = _load_trace(run_id)
            result = rule_engine.analyze(trace)
            for e in result.evidence:
                if e.rule_match and e.rule_match.rule_id == "tool_failure_v1":
                    fired_any = True
                    break
        assert fired_any, "tool_failure_v1 never fired across any execution failure run"

    def test_at_least_one_missing_output_fires(self, rule_engine):
        """Regression: missing_tool_output_v1 must fire at least once."""
        execution_ids = [
            r["run_id"] for r in _load_labels()["runs"] if r["category"] == "execution_failure"
        ]
        fired_any = False
        for run_id in execution_ids:
            trace = _load_trace(run_id)
            result = rule_engine.analyze(trace)
            for e in result.evidence:
                if e.rule_match and e.rule_match.rule_id == "missing_tool_output_v1":
                    fired_any = True
                    break
        assert fired_any, "missing_tool_output_v1 never fired across any execution failure run"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Workflow Rules Fire on Workflow Failure Runs
# ─────────────────────────────────────────────────────────────────────────────


class TestWorkflowRules:
    """
    Workflow failure traces have a missing required agent step.
    Expect skipped_step_v1 to fire. wrong_order_v1 may also fire depending
    on the trace, but skipped_step_v1 is the primary signal here.
    """

    @pytest.mark.parametrize(
        "run_id,skip_step",
        [
            (r["run_id"], r["skip_step"])
            for r in _load_labels()["runs"]
            if r["category"] == "workflow_failure"
        ],
    )
    def test_skipped_step_fires(self, run_id, skip_step, workflow_validator):
        trace = _load_trace(run_id)
        result = workflow_validator.analyze(trace)

        fired = {e.rule_match.rule_id for e in result.evidence if e.rule_match}
        assert "skipped_step_v1" in fired, (
            f"skipped_step_v1 not fired on workflow failure '{run_id}', got: {fired}"
        )

    @pytest.mark.parametrize(
        "run_id,skip_step",
        [
            (r["run_id"], r["skip_step"])
            for r in _load_labels()["runs"]
            if r["category"] == "workflow_failure"
        ],
    )
    def test_skipped_agent_is_correct(self, run_id, skip_step, workflow_validator):
        """The skipped agent in evidence must match the label."""
        trace = _load_trace(run_id)
        result = workflow_validator.analyze(trace)

        skipped_agents = [
            e.agent
            for e in result.evidence
            if e.rule_match and e.rule_match.rule_id == "skipped_step_v1"
        ]
        assert skip_step in skipped_agents, (
            f"On '{run_id}', expected skipped agent '{skip_step}', got: {skipped_agents}"
        )

    @pytest.mark.parametrize(
        "run_id",
        [r["run_id"] for r in _load_labels()["runs"] if r["category"] == "workflow_failure"],
    )
    def test_workflow_routes_to_p3(self, run_id, workflow_validator):
        """Workflow evidence should route to P3 in the Arbiter."""
        trace = _load_trace(run_id)
        result = workflow_validator.analyze(trace)

        bundle = determine_primary_cause(result.evidence, run_id)
        assert bundle.priority_level == PriorityLevel.P3, (
            f"Expected P3 for workflow failure '{run_id}', got {bundle.priority_level}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Execution Failures Route to P2
# ─────────────────────────────────────────────────────────────────────────────


class TestExecutionArbiterRouting:
    @pytest.mark.parametrize(
        "run_id",
        [r["run_id"] for r in _load_labels()["runs"] if r["category"] == "execution_failure"],
    )
    def test_execution_routes_to_p2(self, run_id, rule_engine):
        trace = _load_trace(run_id)
        result = rule_engine.analyze(trace)

        if not result.evidence:
            pytest.skip(f"No evidence generated for {run_id}")

        bundle = determine_primary_cause(result.evidence, run_id)
        assert bundle.priority_level == PriorityLevel.P2, (
            f"Expected P2 for execution failure '{run_id}', got {bundle.priority_level}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. P2 Outranks P3 When Both Fire
# ─────────────────────────────────────────────────────────────────────────────


class TestPriorityRanking:
    """Regression: P2 execution rule must override P3 workflow violation."""

    def test_p2_beats_p3_when_both_fire(self, rule_engine, workflow_validator):
        # Build a trace that has BOTH a tool error AND a missing verifier
        trace = RunTrace(
            workflow="test",
            steps=[
                AgentStep(
                    run_id="priority_test",
                    step=1,
                    agent="researcher",
                    tool_calls=[
                        {
                            "name": "search",
                            "args": {},
                            "output": "",
                            "error": "TimeoutError: timed out",
                        }
                    ],
                ),
                AgentStep(run_id="priority_test", step=2, agent="writer"),
                # verifier missing → P3 will also fire
            ],
        )

        re_result = rule_engine.analyze(trace)
        wv_result = workflow_validator.analyze(trace)
        all_evidence = re_result.evidence + wv_result.evidence

        bundle = determine_primary_cause(all_evidence, "priority_test")
        assert bundle.priority_level == PriorityLevel.P2, (
            f"Expected P2 to outrank P3, got {bundle.priority_level}"
        )
        assert bundle.primary_cause == FailureCategory.EXECUTION
