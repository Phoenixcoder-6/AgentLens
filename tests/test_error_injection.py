"""
tests/test_error_injection.py — Day 36: Error Injection Sweep
=============================================================
Deliberately constructs fault-injected multi-agent traces and runs them through
the complete AgentLens deterministic pipeline (Normalizer -> GroundTruthValidator
-> RuleEngine -> InformationLossRule -> WorkflowValidator -> ConsistencyValidator
-> Arbiter) to verify root-cause category and primary agent attribution:

  1. Tool timeout / tool error       -> FailureCategory.EXECUTION, researcher blamed
  2. Wrong facts in writer output    -> FailureCategory.REASONING, writer blamed
  3. Skipped verifier node           -> FailureCategory.WORKFLOW, verifier blamed
  4. Always-approve verifier         -> FailureCategory.VERIFICATION, verifier blamed
  5. Clean pipeline control          -> PriorityLevel.P5 (PASS), no agent blamed
"""

from __future__ import annotations

import json
from typing import Any

from analyzers.arbiter import Arbiter, evidence_from_information_loss
from analyzers.detection.consistency_validator import ConsistencyValidator
from analyzers.detection.ground_truth import GroundTruthValidator
from analyzers.detection.information_loss import InformationLossRule
from analyzers.detection.rule_engine import RuleEngine, _prestructured_evidence
from analyzers.detection.workflow_validator import WorkflowValidator
from normalizer.normalizer import Normalizer
from schema.models import (
    AgentStep,
    AnalysisBundle,
    EvidenceRecord,
    FailureCategory,
    PriorityLevel,
    RunTrace,
    StepStatus,
    TokenUsage,
)

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Runner & Trace Builders
# ─────────────────────────────────────────────────────────────────────────────


def _step_json(
    source_count: int,
    entity_count: int,
    *,
    verified: bool | None = None,
    verification_result: str | None = None,
    always_approve: bool = False,
    unverified_claims: int = 0,
    claims: list[str] | None = None,
    extra_text: str = "",
) -> str:
    payload: dict[str, Any] = {
        "source_count": source_count,
        "entity_count": entity_count,
    }
    if verified is not None:
        payload["verified"] = verified
    if verification_result is not None:
        payload["verification_result"] = verification_result
    if always_approve:
        payload["always_approve"] = True
    if unverified_claims:
        payload["unverified_claims"] = unverified_claims
    if claims is not None:
        payload["claims"] = claims
    if extra_text:
        payload["summary"] = extra_text
    return json.dumps(payload)


def _make_step(
    run_id: str,
    step_num: int,
    agent: str,
    output: str,
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    expected_output: str | None = None,
    latency_ms: float = 1000.0,
) -> AgentStep:
    return AgentStep(
        run_id=run_id,
        step=step_num,
        agent=agent,
        output=output,
        expected_output=expected_output,
        latency_ms=latency_ms,
        tokens=TokenUsage(prompt=150, completion=250, total=400),
        tool_calls=tool_calls or [],
        status=StepStatus.SUCCESS,
    )


def run_injected_trace(trace: RunTrace) -> AnalysisBundle:
    """Execute a RunTrace through the full deterministic AgentLens pipeline."""
    normalizer = Normalizer()
    gt_validator = GroundTruthValidator()
    rule_engine = RuleEngine()
    info_loss_rule = InformationLossRule()
    workflow_validator = WorkflowValidator()
    consistency_validator = ConsistencyValidator()
    arbiter = Arbiter()

    # 1. Normalize trace
    normalizer.normalize_run(trace)

    evidence: list[EvidenceRecord] = []

    # 2. P1 Ground Truth Validator
    gt_res = gt_validator.analyze(trace)
    if not gt_res.skipped:
        evidence.extend(gt_res.evidence)

    # 3. P2 Rule Engine (Execution + Reasoning)
    re_res = rule_engine.analyze(trace)
    if not re_res.skipped:
        evidence.extend(re_res.evidence)

    # 4. P2 Information Loss Rule (Researcher -> Writer handoff)
    res_steps = [s for s in trace.steps if s.agent == "researcher"]
    wr_steps = [s for s in trace.steps if s.agent == "writer"]
    if res_steps and wr_steps:
        res_ev = _prestructured_evidence(res_steps[-1])
        wr_ev = _prestructured_evidence(wr_steps[-1])
        if res_ev is not None and wr_ev is not None:
            loss_res = info_loss_rule.evaluate(
                run_id=trace.run_id,
                researcher_evidence=res_ev,
                writer_evidence=wr_ev,
            )
            loss_ev = evidence_from_information_loss(loss_res)
            if loss_ev is not None:
                evidence.append(loss_ev)

    # 5. P3 Workflow Validator
    wf_res = workflow_validator.analyze(trace)
    if not wf_res.skipped:
        evidence.extend(wf_res.evidence)

    # 6. P2/P3 Consistency Validator (Verification + Claim Drift)
    cv_res = consistency_validator.analyze(trace)
    if not cv_res.skipped:
        evidence.extend(cv_res.evidence)

    # 7. Arbitrate
    return arbiter.run(run_id=trace.run_id, evidence=evidence)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Tool Timeout -> Execution Failure Category, Researcher Blamed
# ─────────────────────────────────────────────────────────────────────────────


class TestToolTimeoutInjection:
    """Inject tool timeouts and missing tool outputs into the researcher step."""

    def test_tool_timeout_error_field_triggers_execution_failure(self) -> None:
        """Tool call with a timeout error in the 'error' field -> execution failure on researcher."""
        run_id = "inj_tool_timeout_01"
        trace = RunTrace(
            run_id=run_id,
            workflow="research_report_pipeline",
            steps=[
                _make_step(
                    run_id,
                    1,
                    "researcher",
                    _step_json(source_count=0, entity_count=0),
                    tool_calls=[
                        {
                            "name": "web_search",
                            "args": {"query": " semiconductor supply chain "},
                            "output": "",
                            "error": "TimeoutError: web_search timed out after 30000ms",
                        }
                    ],
                ),
                _make_step(run_id, 2, "writer", _step_json(source_count=0, entity_count=0)),
                _make_step(
                    run_id,
                    3,
                    "verifier",
                    _step_json(
                        source_count=0,
                        entity_count=0,
                        verified=False,
                        verification_result="REJECTED",
                    ),
                ),
            ],
        )

        bundle = run_injected_trace(trace)
        fired_rules = [rm.rule_id for rm in bundle.rule_matches]

        assert bundle.priority_level == PriorityLevel.P2
        assert bundle.primary_cause == FailureCategory.EXECUTION
        assert bundle.primary_agent == "researcher"
        assert "tool_failure_v1" in fired_rules

    def test_tool_timeout_exception_in_output_triggers_execution_failure(self) -> None:
        """Tool call returning 'Error: Request timed out' in output -> execution failure on researcher."""
        run_id = "inj_tool_timeout_02"
        trace = RunTrace(
            run_id=run_id,
            workflow="research_report_pipeline",
            steps=[
                _make_step(
                    run_id,
                    1,
                    "researcher",
                    _step_json(source_count=0, entity_count=0),
                    tool_calls=[
                        {
                            "name": "arxiv_fetch",
                            "args": {"paper_id": "2401.00001"},
                            "output": "Error: ReadTimeoutException while fetching arxiv API",
                        }
                    ],
                ),
                _make_step(run_id, 2, "writer", _step_json(source_count=0, entity_count=0)),
                _make_step(run_id, 3, "verifier", _step_json(source_count=0, entity_count=0)),
            ],
        )

        bundle = run_injected_trace(trace)
        fired_rules = [rm.rule_id for rm in bundle.rule_matches]

        assert bundle.priority_level == PriorityLevel.P2
        assert bundle.primary_cause == FailureCategory.EXECUTION
        assert bundle.primary_agent == "researcher"
        assert "tool_failure_v1" in fired_rules

    def test_missing_tool_output_triggers_execution_failure(self) -> None:
        """Tool call that returns neither output nor error -> missing_tool_output_v1 on researcher."""
        run_id = "inj_tool_missing_output_03"
        trace = RunTrace(
            run_id=run_id,
            workflow="research_report_pipeline",
            steps=[
                _make_step(
                    run_id,
                    1,
                    "researcher",
                    _step_json(source_count=2, entity_count=3),
                    tool_calls=[
                        {
                            "name": "sec_filings_lookup",
                            "args": {"ticker": "NVDA"},
                            "output": "",
                            "error": None,
                        }
                    ],
                ),
                _make_step(run_id, 2, "writer", _step_json(source_count=2, entity_count=3)),
                _make_step(run_id, 3, "verifier", _step_json(source_count=2, entity_count=3)),
            ],
        )

        bundle = run_injected_trace(trace)
        fired_rules = [rm.rule_id for rm in bundle.rule_matches]

        assert bundle.priority_level == PriorityLevel.P2
        assert bundle.primary_cause == FailureCategory.EXECUTION
        assert bundle.primary_agent == "researcher"
        assert "missing_tool_output_v1" in fired_rules


# ─────────────────────────────────────────────────────────────────────────────
# 2. Wrong Facts in Writer Output -> Reasoning Failure, Writer Blamed
# ─────────────────────────────────────────────────────────────────────────────


class TestWriterWrongFactsInjection:
    """Inject fabricated entities/facts into the writer step."""

    def test_writer_entity_fabrication_triggers_reasoning_failure(self) -> None:
        """Writer inflates entities from 8 to 22 -> reasoning failure, writer blamed."""
        run_id = "inj_writer_wrong_facts_01"
        trace = RunTrace(
            run_id=run_id,
            workflow="research_report_pipeline",
            steps=[
                _make_step(run_id, 1, "researcher", _step_json(source_count=6, entity_count=8)),
                _make_step(run_id, 2, "writer", _step_json(source_count=10, entity_count=22)),
                # Verifier catches and trims entities to 8 so only writer is at fault
                _make_step(
                    run_id,
                    3,
                    "verifier",
                    _step_json(
                        source_count=6,
                        entity_count=8,
                        verified=False,
                        verification_result="REJECTED",
                    ),
                ),
            ],
        )

        bundle = run_injected_trace(trace)
        fired_rules = [rm.rule_id for rm in bundle.rule_matches]

        assert bundle.priority_level == PriorityLevel.P2
        assert bundle.primary_cause == FailureCategory.REASONING
        assert bundle.primary_agent == "writer"
        assert "hallucination_v1" in fired_rules

    def test_writer_ground_truth_contradiction_triggers_p1_grounded_failure(self) -> None:
        """Pipeline final output contradicts RunTrace.expected_output -> P1 grounded failure."""
        run_id = "inj_writer_gt_contradiction_02"
        trace = RunTrace(
            run_id=run_id,
            workflow="research_report_pipeline",
            expected_output=_step_json(
                source_count=5,
                entity_count=6,
                extra_text="The James Webb Space Telescope was launched on December 25, 2021 to the Sun-Earth L2 point where its infrared instruments observe early galaxies.",
            ),
            steps=[
                _make_step(run_id, 1, "researcher", _step_json(source_count=5, entity_count=6)),
                _make_step(
                    run_id,
                    2,
                    "writer",
                    _step_json(
                        source_count=5,
                        entity_count=6,
                        extra_text="Wrong facts: Hubble replacement launched in 1985 to Mars orbit.",
                    ),
                ),
            ],
        )

        bundle = run_injected_trace(trace)
        fired_rules = [rm.rule_id for rm in bundle.rule_matches]

        assert bundle.priority_level == PriorityLevel.P1
        assert bundle.primary_cause == FailureCategory.REASONING
        assert bundle.grounded is True
        assert bundle.primary_agent == "writer"
        assert "gt_mismatch_v1" in fired_rules


# ─────────────────────────────────────────────────────────────────────────────
# 3. Skipped Verifier Node -> Workflow Failure, Verifier Blamed
# ─────────────────────────────────────────────────────────────────────────────


class TestSkippedVerifierNodeInjection:
    """Inject workflow topology faults (skipped verifier node, skipped researcher, wrong order)."""

    def test_skipped_verifier_node_triggers_workflow_failure(self) -> None:
        """Pipeline terminates after writer without running verifier -> workflow failure, verifier blamed."""
        run_id = "inj_skipped_verifier_01"
        trace = RunTrace(
            run_id=run_id,
            workflow="research_report_pipeline",
            steps=[
                _make_step(run_id, 1, "researcher", _step_json(source_count=8, entity_count=10)),
                _make_step(run_id, 2, "writer", _step_json(source_count=8, entity_count=10)),
                # Step 3 (verifier) deliberately omitted
            ],
        )

        bundle = run_injected_trace(trace)
        fired_rules = [rm.rule_id for rm in bundle.rule_matches]

        assert bundle.priority_level == PriorityLevel.P3
        assert bundle.primary_cause == FailureCategory.WORKFLOW
        assert bundle.primary_agent == "verifier"
        assert "skipped_step_v1" in fired_rules

    def test_skipped_researcher_node_triggers_workflow_failure(self) -> None:
        """Pipeline starts at writer without running researcher -> workflow failure, researcher blamed."""
        run_id = "inj_skipped_researcher_02"
        trace = RunTrace(
            run_id=run_id,
            workflow="research_report_pipeline",
            steps=[
                # Step 1 (researcher) deliberately omitted
                _make_step(run_id, 2, "writer", _step_json(source_count=5, entity_count=7)),
                _make_step(run_id, 3, "verifier", _step_json(source_count=5, entity_count=7)),
            ],
        )

        bundle = run_injected_trace(trace)
        fired_rules = [rm.rule_id for rm in bundle.rule_matches]

        assert bundle.priority_level == PriorityLevel.P3
        assert bundle.primary_cause == FailureCategory.WORKFLOW
        assert bundle.primary_agent == "researcher"
        assert "skipped_step_v1" in fired_rules


# ─────────────────────────────────────────────────────────────────────────────
# 4. Always-Approve Verifier -> Verification Failure, Verifier Blamed
# ─────────────────────────────────────────────────────────────────────────────


class TestAlwaysApproveVerifierInjection:
    """Inject an always-approve (rubber-stamp) verifier and confirm verifier is blamed."""

    def test_always_approve_rubber_stamp_verifier_blamed_in_full_pipeline(self) -> None:
        """Always-approve verifier rubber-stamps unverified claims -> verification failure, verifier blamed."""
        run_id = "inj_always_approve_verifier_01"
        trace = RunTrace(
            run_id=run_id,
            workflow="research_report_pipeline",
            steps=[
                _make_step(run_id, 1, "researcher", _step_json(source_count=7, entity_count=9)),
                _make_step(run_id, 2, "writer", _step_json(source_count=7, entity_count=9)),
                _make_step(
                    run_id,
                    3,
                    "verifier",
                    _step_json(
                        source_count=7,
                        entity_count=9,
                        verified=True,
                        verification_result="APPROVED",
                        always_approve=True,
                        unverified_claims=4,
                    ),
                ),
            ],
        )

        bundle = run_injected_trace(trace)
        fired_rules = [rm.rule_id for rm in bundle.rule_matches]

        assert bundle.priority_level == PriorityLevel.P2
        assert bundle.primary_cause == FailureCategory.VERIFICATION
        assert bundle.primary_agent == "verifier"
        assert "verifier_passthrough_v1" in fired_rules

    def test_always_approve_verifier_passthrough_on_hallucinated_entities(self) -> None:
        """ConsistencyValidator + Arbiter attributes verification failure to verifier on entity passthrough."""
        run_id = "inj_always_approve_verifier_02"
        trace = RunTrace(
            run_id=run_id,
            workflow="research_report_pipeline",
            steps=[
                _make_step(run_id, 1, "researcher", _step_json(source_count=5, entity_count=6)),
                _make_step(run_id, 2, "writer", _step_json(source_count=5, entity_count=14)),
                _make_step(
                    run_id,
                    3,
                    "verifier",
                    _step_json(
                        source_count=5,
                        entity_count=14,
                        verified=True,
                        verification_result="APPROVED",
                    ),
                ),
            ],
        )

        cv_result = ConsistencyValidator().analyze(trace)
        bundle = Arbiter().run(run_id=run_id, evidence=cv_result.evidence)
        fired_rules = [rm.rule_id for rm in bundle.rule_matches]

        assert bundle.priority_level == PriorityLevel.P2
        assert bundle.primary_cause == FailureCategory.VERIFICATION
        assert bundle.primary_agent == "verifier"
        assert "verifier_passthrough_v1" in fired_rules


# ─────────────────────────────────────────────────────────────────────────────
# 5. Clean Baseline Control -> PASS (P5), Zero False Positives
# ─────────────────────────────────────────────────────────────────────────────


class TestCleanPipelineControl:
    """Verify that a faithful 3-agent trace produces zero false positives."""

    def test_clean_three_agent_pipeline_passes_with_zero_false_positives(self) -> None:
        run_id = "inj_clean_control_01"
        trace = RunTrace(
            run_id=run_id,
            workflow="research_report_pipeline",
            steps=[
                _make_step(run_id, 1, "researcher", _step_json(source_count=8, entity_count=10)),
                _make_step(run_id, 2, "writer", _step_json(source_count=8, entity_count=10)),
                _make_step(
                    run_id,
                    3,
                    "verifier",
                    _step_json(
                        source_count=8,
                        entity_count=10,
                        verified=True,
                        verification_result="APPROVED",
                    ),
                ),
            ],
        )

        bundle = run_injected_trace(trace)

        assert bundle.priority_level == PriorityLevel.P5
        assert bundle.primary_cause == FailureCategory.UNKNOWN
        assert bundle.primary_agent is None
        assert bundle.rule_matches == []
