"""
tests/test_dashboard_dag.py — Day 65: Interactive Visual DAG Dashboard Tests
=============================================================================
Unit and integration tests certifying DAGRenderer, causal graph visualization,
blast radius widgets, and prompt remediation panels (Day 65).

Tests:
  1. DAGRenderer construction and legend HTML generation.
  2. Clean multi-agent pipeline: all nodes rendered with healthy status.
  3. 3-agent cascade: correct root, cascade, and passthrough class assignments.
  4. Label sanitization: quotes, brackets, and linebreaks escaped for Mermaid.
  5. Hierarchical TraceSpan DAG and custom orientation (TD/LR).
  6. Blast radius HTML panel widget rendering and metric binding.
  7. Prompt remediation HTML panel rendering and diff styling.
  8. Dashboard state get_causal_graph integration with database storage.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyzers.causal import (  # noqa: E402
    CausalGraphEngine,
)
from dashboard import state as dash_state  # noqa: E402
from dashboard.dag_renderer import DAGRenderer  # noqa: E402
from schema.models import (  # noqa: E402
    AgentStep,
    AnalysisBundle,
    BlastRadiusReport,
    FailureCategory,
    HandoffState,
    PriorityLevel,
    RemediationPatch,
    RuleMatch,
    RuleSeverity,
    RunTrace,
    StepStatus,
    TokenUsage,
)
from schema.span import SpanType, TraceSpan  # noqa: E402
from storage.db import DatabaseManager  # noqa: E402


def _make_step(
    run_id: str,
    step_num: int,
    agent: str,
    *,
    status: StepStatus = StepStatus.SUCCESS,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    latency_ms: float = 120.0,
) -> AgentStep:
    return AgentStep(
        run_id=run_id,
        step=step_num,
        agent=agent,
        status=status,
        input="input payload",
        output="output payload",
        timestamp=datetime(2026, 11, 15, 12, 0, step_num, tzinfo=UTC),
        tokens=TokenUsage(
            prompt=prompt_tokens,
            completion=completion_tokens,
            total=prompt_tokens + completion_tokens,
        ),
        latency_ms=latency_ms,
        handoff=HandoffState(),
    )


class TestDashboardDAG:
    """Comprehensive test suite for interactive visual DAG rendering."""

    def test_renderer_construction_and_legend(self) -> None:
        """Verify DAGRenderer instantiation and legend HTML generation."""
        renderer = DAGRenderer()
        legend = renderer.get_legend_html()

        assert "Root Cause Origin" in legend
        assert "#ef4444" in legend
        assert "Cascading Failure" in legend
        assert "#f59e0b" in legend
        assert "Escaped QC Verifier" in legend
        assert "#eab308" in legend
        assert "Healthy Node" in legend
        assert "#22c55e" in legend

    def test_clean_pipeline_healthy_nodes(self) -> None:
        """Verify clean execution traces map all nodes to :::healthy with green icons."""
        run_id = "dag_clean_01"
        trace = RunTrace(
            run_id=run_id,
            workflow="research_pipeline",
            status=StepStatus.SUCCESS,
            timestamp=datetime.now(UTC),
            steps=[
                _make_step(run_id, 1, "researcher", latency_ms=150.0),
                _make_step(run_id, 2, "writer", latency_ms=300.0),
                _make_step(run_id, 3, "verifier", latency_ms=100.0),
            ],
        )

        renderer = DAGRenderer()
        mermaid = renderer.render_from_trace(trace, orientation="LR")

        assert "flowchart LR" in mermaid
        assert "classDef healthy" in mermaid
        assert "step_1" in mermaid
        assert "step_2" in mermaid
        assert "step_3" in mermaid

        # All 3 nodes must be styled as healthy
        assert mermaid.count(":::healthy") == 3
        assert ":::root" not in mermaid
        assert ":::cascade" not in mermaid
        assert ":::passthrough" not in mermaid

        # Check edge linkage
        assert "step_1 --> step_2" in mermaid
        assert "step_2 --> step_3" in mermaid

    def test_three_agent_cascade_causal_styling(self) -> None:
        """Verify Researcher (root) -> Writer (cascade) -> Verifier (escaped) causal classes."""
        run_id = "dag_cascade_01"
        trace = RunTrace(
            run_id=run_id,
            workflow="research_pipeline",
            status=StepStatus.FAILURE,
            timestamp=datetime.now(UTC),
            steps=[
                _make_step(run_id, 1, "researcher", latency_ms=200.0),
                _make_step(run_id, 2, "writer", latency_ms=800.0),
                _make_step(run_id, 3, "verifier", latency_ms=400.0),
            ],
        )
        rule = RuleMatch(
            rule_id="R-IL-001",
            category=FailureCategory.REASONING,
            description="Researcher dropped critical sources",
            severity=RuleSeverity.HIGH,
            agent="researcher",
            step=1,
        )
        bundle = AnalysisBundle(
            run_id=run_id,
            primary_cause=FailureCategory.REASONING,
            priority_level=PriorityLevel.P2,
            grounded=False,
            primary_agent="researcher",
            rule_matches=[rule],
        )

        renderer = DAGRenderer()
        mermaid = renderer.render_from_trace(trace, analysis_bundle=bundle)

        # Root cause origin node
        assert "step_1" in mermaid
        assert ":::root" in mermaid
        assert "🔴 researcher" in mermaid
        assert "R-IL-001" in mermaid

        # Cascading failure node
        assert "step_2" in mermaid
        assert ":::cascade" in mermaid
        assert "🟠 writer" in mermaid
        assert "(Cascading Failure)" in mermaid

        # Escaped QC verifier node
        assert "step_3" in mermaid
        assert ":::passthrough" in mermaid
        assert "🟡 verifier" in mermaid
        assert "(Escaped QC Verifier)" in mermaid

    def test_label_sanitization_for_mermaid(self) -> None:
        """Verify quotes, brackets, and newlines are sanitized so Mermaid doesn't syntax error."""
        renderer = DAGRenderer()

        raw = 'Agent "Alpha" [Special] \n with\r\nbreaks'
        clean = renderer.sanitize_label(raw)

        assert '"' not in clean
        assert "#quot;" in clean
        assert "[" not in clean
        assert "&#91;" in clean
        assert "\n" not in clean

    def test_hierarchical_tracespan_dag_and_orientation(self) -> None:
        """Verify OpenInference TraceSpan hierarchical trees support TD top-down orientation."""
        run_id = "dag_span_01"
        root_span = TraceSpan(
            span_id="sp_orch",
            trace_id=run_id,
            parent_span_id=None,
            name="Orchestrator",
            span_type=SpanType.AGENT,
            status=StepStatus.SUCCESS,
            latency_ms=50.0,
        )
        child_tool = TraceSpan(
            span_id="sp_tool",
            trace_id=run_id,
            parent_span_id="sp_orch",
            name="SearchTool",
            span_type=SpanType.TOOL,
            status=StepStatus.ERROR,
            error="API Connection Timeout",
            latency_ms=250.0,
        )
        child_agent = TraceSpan(
            span_id="sp_analyst",
            trace_id=run_id,
            parent_span_id="sp_orch",
            name="DataAnalyst",
            span_type=SpanType.AGENT,
            status=StepStatus.SUCCESS,
            latency_ms=500.0,
        )

        trace = RunTrace(
            run_id=run_id,
            workflow="hierarchical_flow",
            status=StepStatus.ERROR,
            timestamp=datetime.now(UTC),
            spans=[root_span, child_tool, child_agent],
        )

        graph = CausalGraphEngine().build_graph(trace)
        renderer = DAGRenderer()
        mermaid = renderer.render_mermaid(graph, orientation="TD")

        assert "flowchart TD" in mermaid
        assert "sp_orch" in mermaid
        assert "sp_tool" in mermaid
        assert "sp_analyst" in mermaid
        assert "sp_orch --> sp_tool" in mermaid
        assert "sp_orch --> sp_analyst" in mermaid
        assert ":::root" in mermaid  # tool error is identified as root cause

    def test_blast_radius_html_panel(self) -> None:
        """Verify BlastRadiusReport binds correctly to styled HTML card with metrics and progress bar."""
        report = BlastRadiusReport(
            run_id="dag_blast_01",
            root_cause_agent="researcher",
            root_cause_node_id="step_1",
            tainted_steps_ratio=0.6667,
            total_steps=3,
            tainted_steps_count=2,
            wasted_tokens=TokenUsage(prompt=2300, completion=700, total=3000),
            wasted_cost_usd=0.0174,
            wasted_latency_ms=1200.0,
            impacted_agents=["writer", "verifier"],
            pricing_model="gpt-4o",
            summary="Test blast radius summary",
        )

        renderer = DAGRenderer()
        html = renderer.get_blast_radius_html(report)

        assert "$0.0174" in html
        assert "3,000" in html
        assert "1,200" in html
        assert "66.7%" in html
        assert "gpt-4o" in html
        assert "writer, verifier" in html
        assert "width:66%" in html  # progress bar width

    def test_prompt_remediation_html_panel(self) -> None:
        """Verify RemediationPatch renders styled unified diff lines with syntax highlights."""
        patch = RemediationPatch(
            run_id="dag_rem_01",
            agent="writer",
            rule_id="R-IL-001",
            constraint_type="SOURCE_PRESERVATION",
            confidence=0.95,
            regression_risk=0.05,
            explanation="Forces writer to preserve citations.",
            recommended_constraint="CRITICAL CONSTRAINT: Preserve all source citations.",
            diff_patch=(
                "--- SYSTEM PROMPT (agent=writer:original)\n"
                "+++ SYSTEM PROMPT (agent=writer:remediated)\n"
                "@@ -1,2 +1,4 @@\n"
                " You are a writer.\n"
                "+CRITICAL CONSTRAINT: Preserve all source citations.\n"
            ),
        )

        renderer = DAGRenderer()
        html = renderer.get_remediation_panel_html(patch)

        assert "SOURCE_PRESERVATION" in html
        assert "writer" in html
        assert "95%" in html
        assert "5%" in html
        assert "--- SYSTEM PROMPT" in html
        assert "+++ SYSTEM PROMPT" in html
        assert "+CRITICAL CONSTRAINT:" in html

    def test_dashboard_state_causal_graph_integration(self, tmp_path: Path) -> None:
        """Verify dashboard.state.get_causal_graph constructs CausalGraph from database trace."""
        db_file = str(tmp_path / "test_dash_dag.db")
        db = DatabaseManager(db_path=db_file)
        db.initialize()

        run_id = "dash_state_dag_01"
        trace = RunTrace(
            run_id=run_id,
            workflow="research_pipeline",
            status=StepStatus.FAILURE,
            timestamp=datetime.now(UTC),
            steps=[
                _make_step(run_id, 1, "researcher", status=StepStatus.ERROR),
                _make_step(run_id, 2, "writer", status=StepStatus.FAILURE),
            ],
            total_latency_ms=400.0,
            total_tokens=1200,
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

        try:
            graph = dash_state.get_causal_graph(run_id, db=db)
            assert graph is not None
            assert len(graph.nodes) == 2
            assert "step_1" in graph.root_causes
            assert "step_2" in graph.cascades
        finally:
            db.close()
