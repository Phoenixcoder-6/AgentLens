"""
tests/test_prompt_fixer.py — Day 64: Automated Prompt Auto-Diff Engine Tests
=============================================================================
Unit and integration tests certifying PromptFixer and RemediationPatch (Day 64).

Tests:
  1. Clean pipeline run: zero remediation needed (generate_patch returns None).
  2. Information loss patch (R-IL): generates SOURCE_PRESERVATION constraint.
  3. Hallucination guard patch (R-HL): generates ENTITY_FIDELITY constraint.
  4. Contradiction guard patch (R-CT): generates CONTRADICTION_GUARD constraint.
  5. Workflow schema patch (R-WF): generates SCHEMA_CONFORMANCE constraint.
  6. Unified diff syntax validation and apply_patch functionality.
  7. Historical baseline comparison and regression avoidance score calculation.
  8. SQLite persistence, AnalysisBundle, and REST API /verdict integration.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyzers.remediation import (  # noqa: E402
    PromptFixer,
    RemediationPatch,
)
from api.main import fastapi_app  # noqa: E402
from dashboard import state as dash_state  # noqa: E402
from schema.models import (  # noqa: E402
    AgentStep,
    AnalysisBundle,
    FailureCategory,
    HandoffState,
    PriorityLevel,
    RuleMatch,
    RuleSeverity,
    RunTrace,
    StepStatus,
    TokenUsage,
)
from storage.db import DatabaseManager  # noqa: E402


def _make_step(
    run_id: str,
    step_num: int,
    agent: str,
    *,
    status: StepStatus = StepStatus.SUCCESS,
    prompt: str = "",
    output: str = "agent output",
) -> AgentStep:
    return AgentStep(
        run_id=run_id,
        step=step_num,
        agent=agent,
        status=status,
        input="input payload",
        output=output,
        prompt=prompt,
        timestamp=datetime(2026, 11, 14, 12, 0, step_num, tzinfo=UTC),
        tokens=TokenUsage(prompt=100, completion=50, total=150),
        latency_ms=100.0,
        handoff=HandoffState(),
    )


class TestPromptFixer:
    """Comprehensive test suite for PromptFixer and automated prompt diff generation."""

    def test_clean_pipeline_zero_remediation_needed(self) -> None:
        """Verify successful runs with zero rule violations return None (no patch needed)."""
        run_id = "fix_clean_01"
        trace = RunTrace(
            run_id=run_id,
            workflow="research_pipeline",
            status=StepStatus.SUCCESS,
            timestamp=datetime.now(UTC),
            steps=[
                _make_step(run_id, 1, "researcher"),
                _make_step(run_id, 2, "writer"),
            ],
        )
        bundle = AnalysisBundle(
            run_id=run_id,
            primary_cause=FailureCategory.UNKNOWN,
            priority_level=PriorityLevel.P5,
            grounded=False,
            primary_agent=None,
        )

        fixer = PromptFixer()
        patch = fixer.generate_patch(trace, analysis_bundle=bundle)
        assert patch is None

    def test_information_loss_source_preservation_patch(self) -> None:
        """Verify R-IL rule match triggers SOURCE_PRESERVATION constraint."""
        run_id = "fix_il_01"
        base_prompt = (
            "You are a report writer. Summarize the research findings into clear paragraphs."
        )
        trace = RunTrace(
            run_id=run_id,
            workflow="research_pipeline",
            status=StepStatus.FAILURE,
            timestamp=datetime.now(UTC),
            steps=[
                _make_step(run_id, 1, "researcher"),
                _make_step(run_id, 2, "writer", prompt=base_prompt, status=StepStatus.FAILURE),
            ],
        )
        rule = RuleMatch(
            rule_id="R-IL-001",
            category=FailureCategory.REASONING,
            description="Dropped 4 critical citations from research input",
            severity=RuleSeverity.HIGH,
            agent="writer",
            step=2,
        )
        bundle = AnalysisBundle(
            run_id=run_id,
            primary_cause=FailureCategory.REASONING,
            priority_level=PriorityLevel.P2,
            grounded=False,
            primary_agent="writer",
            rule_matches=[rule],
        )

        fixer = PromptFixer()
        patch = fixer.generate_patch(trace, analysis_bundle=bundle)

        assert patch is not None
        assert isinstance(patch, RemediationPatch)
        assert patch.agent == "writer"
        assert patch.rule_id == "R-IL-001"
        assert patch.constraint_type == "SOURCE_PRESERVATION"
        assert "preserve all source citations" in patch.recommended_constraint
        assert patch.original_prompt_snippet == base_prompt
        assert patch.confidence >= 0.85
        assert patch.regression_risk <= 0.15
        assert "SOURCE_PRESERVATION" in patch.format_summary()

    def test_hallucination_entity_fidelity_patch(self) -> None:
        """Verify R-HL rule match triggers ENTITY_FIDELITY closed-domain constraint."""
        run_id = "fix_hl_01"
        trace = RunTrace(
            run_id=run_id,
            workflow="research_pipeline",
            status=StepStatus.FAILURE,
            timestamp=datetime.now(UTC),
            steps=[
                _make_step(run_id, 1, "researcher"),
                _make_step(run_id, 2, "writer", status=StepStatus.FAILURE),
            ],
        )
        rule = RuleMatch(
            rule_id="R-HL-002",
            category=FailureCategory.REASONING,
            description="Invented 3 ungrounded corporate entity names",
            severity=RuleSeverity.HIGH,
            agent="writer",
            step=2,
        )
        bundle = AnalysisBundle(
            run_id=run_id,
            primary_cause=FailureCategory.REASONING,
            priority_level=PriorityLevel.P2,
            grounded=False,
            primary_agent="writer",
            rule_matches=[rule],
        )

        fixer = PromptFixer()
        patch = fixer.generate_patch(trace, analysis_bundle=bundle)

        assert patch is not None
        assert patch.constraint_type == "ENTITY_FIDELITY"
        assert (
            "Strictly ground your synthesis on the provided context only"
            in patch.recommended_constraint
        )
        assert "Do not introduce external named entities" in patch.recommended_constraint

    def test_contradiction_guard_patch(self) -> None:
        """Verify R-CT rule match triggers CONTRADICTION_GUARD premise constraint."""
        run_id = "fix_ct_01"
        trace = RunTrace(
            run_id=run_id,
            workflow="research_pipeline",
            status=StepStatus.FAILURE,
            timestamp=datetime.now(UTC),
            steps=[
                _make_step(run_id, 1, "researcher"),
                _make_step(run_id, 2, "writer", status=StepStatus.FAILURE),
            ],
        )
        rule = RuleMatch(
            rule_id="R-CT-001",
            category=FailureCategory.REASONING,
            description="Synthesis contradicted research premise regarding Q3 revenue",
            severity=RuleSeverity.HIGH,
            agent="writer",
            step=2,
        )
        bundle = AnalysisBundle(
            run_id=run_id,
            primary_cause=FailureCategory.REASONING,
            priority_level=PriorityLevel.P2,
            grounded=False,
            primary_agent="writer",
            rule_matches=[rule],
        )

        fixer = PromptFixer()
        patch = fixer.generate_patch(trace, analysis_bundle=bundle)

        assert patch is not None
        assert patch.constraint_type == "CONTRADICTION_GUARD"
        assert "Do not contradict any established premise" in patch.recommended_constraint

    def test_workflow_schema_conformance_patch(self) -> None:
        """Verify R-WF rule match triggers SCHEMA_CONFORMANCE state key constraint."""
        run_id = "fix_wf_01"
        trace = RunTrace(
            run_id=run_id,
            workflow="research_pipeline",
            status=StepStatus.FAILURE,
            timestamp=datetime.now(UTC),
            steps=[
                _make_step(run_id, 1, "planner", status=StepStatus.FAILURE),
            ],
        )
        rule = RuleMatch(
            rule_id="R-WF-001",
            category=FailureCategory.WORKFLOW,
            description="Dropped required workflow state key 'citations'",
            severity=RuleSeverity.MEDIUM,
            agent="planner",
            step=1,
        )
        bundle = AnalysisBundle(
            run_id=run_id,
            primary_cause=FailureCategory.WORKFLOW,
            priority_level=PriorityLevel.P3,
            grounded=False,
            primary_agent="planner",
            rule_matches=[rule],
        )

        fixer = PromptFixer()
        patch = fixer.generate_patch(trace, analysis_bundle=bundle)

        assert patch is not None
        assert patch.agent == "planner"
        assert patch.constraint_type == "SCHEMA_CONFORMANCE"
        assert "Preserve all expected dictionary keys" in patch.recommended_constraint

    def test_unified_diff_syntax_and_apply_patch(self) -> None:
        """Verify standard GitHub unified diff syntax generation and apply_patch injection."""
        run_id = "fix_diff_01"
        base_prompt = "You are a synthesis agent.\nSummarize findings."
        trace = RunTrace(
            run_id=run_id,
            workflow="research_pipeline",
            status=StepStatus.FAILURE,
            timestamp=datetime.now(UTC),
            steps=[_make_step(run_id, 1, "writer", prompt=base_prompt, status=StepStatus.FAILURE)],
        )
        rule = RuleMatch(
            rule_id="R-IL-001",
            category=FailureCategory.REASONING,
            description="Omitted citations",
            agent="writer",
            step=1,
        )
        bundle = AnalysisBundle(
            run_id=run_id,
            primary_cause=FailureCategory.REASONING,
            priority_level=PriorityLevel.P2,
            grounded=False,
            primary_agent="writer",
            rule_matches=[rule],
        )

        fixer = PromptFixer()
        patch = fixer.generate_patch(trace, analysis_bundle=bundle)
        assert patch is not None

        diff_text = patch.format_diff()
        assert "--- SYSTEM PROMPT (agent=writer:original)" in diff_text
        assert "+++ SYSTEM PROMPT (agent=writer:remediated)" in diff_text
        assert "@@" in diff_text
        assert "+CRITICAL CONSTRAINT:" in diff_text

        # Test apply_patch method
        remediated = fixer.apply_patch(patch, base_prompt)
        assert base_prompt in remediated
        assert patch.recommended_constraint in remediated

    def test_historical_baseline_comparison_and_scoring(self) -> None:
        """Verify comparison against historical clean runs boosts confidence and lowers risk."""
        failed_id = "fix_hist_fail"
        clean_id = "fix_hist_clean"

        failed_trace = RunTrace(
            run_id=failed_id,
            workflow="research_pipeline",
            status=StepStatus.FAILURE,
            timestamp=datetime.now(UTC),
            steps=[_make_step(failed_id, 1, "writer", status=StepStatus.FAILURE)],
        )
        clean_trace = RunTrace(
            run_id=clean_id,
            workflow="research_pipeline",
            status=StepStatus.SUCCESS,
            timestamp=datetime.now(UTC),
            steps=[_make_step(clean_id, 1, "writer", status=StepStatus.SUCCESS)],
        )

        rule = RuleMatch(
            rule_id="R-IL-001",
            category=FailureCategory.REASONING,
            description="Information loss",
            agent="writer",
            step=1,
        )
        bundle = AnalysisBundle(
            run_id=failed_id,
            primary_cause=FailureCategory.REASONING,
            priority_level=PriorityLevel.P2,
            grounded=False,
            primary_agent="writer",
            rule_matches=[rule],
        )

        fixer = PromptFixer()
        patch = fixer.generate_patch(
            failed_trace,
            analysis_bundle=bundle,
            historical_traces=[clean_trace],
        )

        assert patch is not None
        assert patch.historical_reference_run_id == clean_id
        # Boosted confidence & reduced risk due to clean comparator
        assert patch.confidence >= 0.92
        assert patch.regression_risk <= 0.10

    def test_sqlite_persistence_analysis_bundle_and_api(self, tmp_path: Path, monkeypatch) -> None:
        """Verify DatabaseManager persistence, AnalysisBundle model dump, and GET /api/runs/{id}/verdict."""
        db_file = str(tmp_path / "test_remediation.db")
        db = DatabaseManager(db_path=db_file)
        db.initialize()

        run_id = "fix_api_01"
        trace = RunTrace(
            run_id=run_id,
            workflow="research_pipeline",
            status=StepStatus.FAILURE,
            timestamp=datetime.now(UTC),
            steps=[
                _make_step(
                    run_id,
                    1,
                    "researcher",
                    output="Sources: [1] http://a, [2] http://b. Entities: Alpha, Beta.",
                ),
                _make_step(
                    run_id,
                    2,
                    "writer",
                    status=StepStatus.FAILURE,
                    output="Summary dropping all sources and entities.",
                ),
                _make_step(
                    run_id,
                    3,
                    "verifier",
                    output="APPROVED.",
                ),
            ],
            total_latency_ms=600.0,
            total_tokens=2000,
        )
        db.insert_run(
            run_id=run_id,
            workflow=trace.workflow,
            timestamp=trace.timestamp.isoformat(),
            status=trace.status.value,
            total_latency_ms=trace.total_latency_ms,
            total_tokens=trace.total_tokens,
            schema_version="1.0",
            trace_json=json.dumps(trace.model_dump(mode="json")),
        )

        # Isolate dashboard state DB & cache
        monkeypatch.setattr(dash_state, "get_db", lambda: db)
        dash_state._analysis_cache.pop(run_id, None)

        try:
            state = dash_state.run_full_analysis(run_id, db=db)
            assert state.error is None
            assert state.bundle is not None
            assert state.bundle.remediation_patch is not None

            # Verify SQLite analysis table persistence
            saved_patch = db.get_remediation_patch(run_id)
            assert saved_patch is not None
            assert saved_patch["agent"] == "writer"
            assert "CRITICAL CONSTRAINT" in saved_patch["recommended_constraint"]

            # Verify API /verdict endpoint returns remediation_patch
            with TestClient(fastapi_app) as client:
                resp = client.get(f"/api/runs/{run_id}/verdict")
            assert resp.status_code == 200
            body = resp.json()
            assert body["run_id"] == run_id
            assert body["remediation_patch"] is not None
            assert body["remediation_patch"]["agent"] == "writer"
            assert "diff_patch" in body["remediation_patch"]
        finally:
            dash_state._analysis_cache.pop(run_id, None)
            db.close()
