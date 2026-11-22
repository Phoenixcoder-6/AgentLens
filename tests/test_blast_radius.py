"""
tests/test_blast_radius.py — Day 63: Error Blast Radius & Financial Impact Tests
==================================================================================
Unit and integration tests certifying BlastRadiusCalculator and TokenPricing (Day 63).

Tests:
  1. Clean pipeline run: zero blast radius, zero wasted tokens, $0.00 cost.
  2. Classic 3-agent cascade: Researcher (Root) -> Writer (Cascade) -> Verifier (Escaped).
  3. include_root_cause=True option: includes root node compute in total waste.
  4. Pricing models & presets: default, gpt-4o, gpt-4o-mini, flat_config, custom, from_config().
  5. Parallel fork isolation: healthy parallel branch compute excluded from waste.
  6. Database persistence round-trip: save_blast_radius & get_blast_radius + metrics rows.
  7. End-to-end AnalysisBundle, dashboard.state.run_full_analysis, & API /verdict integration.
  8. Serialization, Pydantic bounds, 60/40 token estimation fallback, and format_summary().
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

from analyzers.causal import (  # noqa: E402
    BlastRadiusCalculator,
    BlastRadiusReport,
    CausalGraphEngine,
    CausalNode,
    CausalNodeState,
    TokenPricing,
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
    total_tokens: int | None = None,
    latency_ms: float = 100.0,
    output: str = "agent output",
    handoff: HandoffState | None = None,
) -> AgentStep:
    tot = total_tokens if total_tokens is not None else (prompt_tokens + completion_tokens)
    return AgentStep(
        run_id=run_id,
        step=step_num,
        agent=agent,
        status=status,
        input="input payload",
        output=output,
        timestamp=datetime(2026, 11, 10, 12, 0, step_num, tzinfo=UTC),
        tokens=TokenUsage(prompt=prompt_tokens, completion=completion_tokens, total=tot),
        latency_ms=latency_ms,
        handoff=handoff or HandoffState(),
    )


class TestBlastRadius:
    """Comprehensive test suite for BlastRadiusCalculator and financial impact tracking."""

    def test_clean_pipeline_zero_blast_radius(self) -> None:
        """Verify clean runs yield 0.0 tainted ratio, 0 wasted tokens, and $0.00 cost."""
        run_id = "br_clean_01"
        trace = RunTrace(
            run_id=run_id,
            workflow="research_pipeline",
            status=StepStatus.SUCCESS,
            timestamp=datetime.now(UTC),
            steps=[
                _make_step(
                    run_id,
                    1,
                    "researcher",
                    prompt_tokens=1000,
                    completion_tokens=400,
                    latency_ms=250.0,
                ),
                _make_step(
                    run_id, 2, "writer", prompt_tokens=1500, completion_tokens=600, latency_ms=400.0
                ),
                _make_step(
                    run_id,
                    3,
                    "verifier",
                    prompt_tokens=800,
                    completion_tokens=200,
                    latency_ms=150.0,
                ),
            ],
        )

        report = BlastRadiusCalculator().calculate(trace)

        assert isinstance(report, BlastRadiusReport)
        assert report.run_id == run_id
        assert report.total_steps == 3
        assert report.tainted_steps_count == 0
        assert report.tainted_steps_ratio == 0.0
        assert report.wasted_tokens.prompt == 0
        assert report.wasted_tokens.completion == 0
        assert report.wasted_tokens.total == 0
        assert report.wasted_cost_usd == 0.0
        assert report.wasted_latency_ms == 0.0
        assert report.impacted_agents == []
        assert "Zero computational waste" in report.format_summary()

    def test_classic_three_agent_cascade_waste(self) -> None:
        """Verify Researcher (root) -> Writer (cascade) -> Verifier (escaped) blast radius math."""
        run_id = "br_cascade_01"
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
                    prompt_tokens=1000,
                    completion_tokens=300,
                    latency_ms=200.0,
                ),
                _make_step(
                    run_id, 2, "writer", prompt_tokens=1500, completion_tokens=500, latency_ms=800.0
                ),
                _make_step(
                    run_id,
                    3,
                    "verifier",
                    prompt_tokens=800,
                    completion_tokens=200,
                    latency_ms=400.0,
                ),
            ],
        )
        rule = RuleMatch(
            rule_id="R-IL-001",
            category=FailureCategory.REASONING,
            description="Researcher omitted key evidence citations",
            severity=RuleSeverity.HIGH,
            agent="researcher",
            step=1,
        )
        graph = CausalGraphEngine().build_graph(trace, rule_matches=[rule])
        report = BlastRadiusCalculator().calculate(trace, causal_graph=graph)

        # Writer + Verifier downstream tainted (2 of 3 steps)
        assert report.root_cause_agent == "researcher"
        assert report.root_cause_node_id == "step_1"
        assert report.total_steps == 3
        assert report.tainted_steps_count == 2
        assert report.tainted_steps_ratio == 0.6667

        # Tokens: writer (1500/500) + verifier (800/200) = 2300 prompt, 700 completion, 3000 total
        assert report.wasted_tokens.prompt == 2300
        assert report.wasted_tokens.completion == 700
        assert report.wasted_tokens.total == 3000

        # Default pricing: $3/1M prompt + $15/1M completion = 0.0069 + 0.0105 = 0.0174 USD
        assert report.wasted_cost_usd == 0.0174
        assert report.wasted_latency_ms == 1200.0
        assert report.impacted_agents == ["writer", "verifier"]

    def test_include_root_cause_flag(self) -> None:
        """Verify include_root_cause=True accounts for origin node tokens and latency as well."""
        run_id = "br_root_inc_01"
        trace = RunTrace(
            run_id=run_id,
            workflow="research_pipeline",
            status=StepStatus.ERROR,
            timestamp=datetime.now(UTC),
            steps=[
                _make_step(
                    run_id,
                    1,
                    "researcher",
                    status=StepStatus.ERROR,
                    prompt_tokens=1000,
                    completion_tokens=0,
                    latency_ms=300.0,
                ),
                _make_step(
                    run_id, 2, "writer", prompt_tokens=1500, completion_tokens=500, latency_ms=500.0
                ),
            ],
        )
        report = BlastRadiusCalculator().calculate(trace, include_root_cause=True)

        assert report.tainted_steps_count == 2
        assert report.tainted_steps_ratio == 1.0
        assert report.wasted_tokens.prompt == 2500
        assert report.wasted_tokens.completion == 500
        assert report.wasted_tokens.total == 3000
        assert report.wasted_latency_ms == 800.0
        assert report.impacted_agents == ["researcher", "writer"]

    def test_custom_pricing_models_and_presets(self) -> None:
        """Verify provider presets, custom TokenPricing, flat rates, and config loading."""
        run_id = "br_pricing_01"
        trace = RunTrace(
            run_id=run_id,
            workflow="research_pipeline",
            status=StepStatus.ERROR,
            timestamp=datetime.now(UTC),
            steps=[
                _make_step(
                    run_id,
                    1,
                    "researcher",
                    status=StepStatus.ERROR,
                    prompt_tokens=500,
                    completion_tokens=100,
                ),
                _make_step(
                    run_id,
                    2,
                    "writer",
                    prompt_tokens=1_000_000,
                    completion_tokens=500_000,
                    latency_ms=2000.0,
                ),
            ],
        )
        calc = BlastRadiusCalculator()

        gpt4o = calc.calculate(trace, pricing=TokenPricing.get_preset("gpt-4o"))
        assert gpt4o.pricing_model == "gpt-4o"
        assert gpt4o.wasted_cost_usd == 7.5  # $2.50 * 1 + $10.00 * 0.5

        mini = calc.calculate(trace, pricing=TokenPricing.get_preset("gpt-4o-mini"))
        assert mini.wasted_cost_usd == 0.45  # $0.15 * 1 + $0.60 * 0.5

        flat = calc.calculate(trace, pricing=TokenPricing.get_preset("flat_config"))
        assert flat.wasted_cost_usd == 7.5  # 1,500,000 * 0.000005

        cfg_pricing = TokenPricing.from_config()
        assert cfg_pricing.flat_cost_per_token == 0.000005
        assert calc.calculate(trace, pricing=cfg_pricing).wasted_cost_usd == 7.5

    def test_parallel_fork_branch_isolation(self) -> None:
        """Verify healthy parallel branch tokens and latency are excluded from blast radius."""
        run_id = "br_parallel_01"
        root_span = TraceSpan(
            span_id="sp_orch",
            trace_id=run_id,
            parent_span_id=None,
            name="Orchestrator",
            span_type=SpanType.AGENT,
            status=StepStatus.SUCCESS,
            tokens=TokenUsage(prompt=200, completion=50, total=250),
            latency_ms=50.0,
        )
        branch_a_root = TraceSpan(
            span_id="sp_img",
            trace_id=run_id,
            parent_span_id="sp_orch",
            name="ImageAnalyzer",
            span_type=SpanType.AGENT,
            status=StepStatus.ERROR,
            error="GPU OOM",
            tokens=TokenUsage(prompt=400, completion=0, total=400),
            latency_ms=120.0,
        )
        branch_a_child = TraceSpan(
            span_id="sp_img_post",
            trace_id=run_id,
            parent_span_id="sp_img",
            name="ImagePostprocessor",
            span_type=SpanType.TOOL,
            status=StepStatus.FAILURE,
            tokens=TokenUsage(prompt=600, completion=400, total=1000),
            latency_ms=350.0,
        )
        healthy_branch = TraceSpan(
            span_id="sp_txt",
            trace_id=run_id,
            parent_span_id="sp_orch",
            name="TextAnalyzer",
            span_type=SpanType.AGENT,
            status=StepStatus.SUCCESS,
            tokens=TokenUsage(prompt=5000, completion=2000, total=7000),
            latency_ms=1800.0,
        )

        trace = RunTrace(
            run_id=run_id,
            workflow="parallel_multimodal",
            status=StepStatus.FAILURE,
            timestamp=datetime.now(UTC),
            spans=[root_span, branch_a_root, branch_a_child, healthy_branch],
        )

        report = BlastRadiusCalculator().calculate(trace)

        assert report.root_cause_agent == "ImageAnalyzer"
        assert report.tainted_steps_count == 1
        assert report.tainted_steps_ratio == 0.25
        assert report.wasted_tokens.total == 1000
        assert report.wasted_latency_ms == 350.0
        assert report.impacted_agents == ["ImagePostprocessor"]
        assert "TextAnalyzer" not in report.impacted_agents

    def test_database_persistence_round_trip(self, tmp_path: Path) -> None:
        """Verify save_blast_radius writes analysis + metrics tables and get_blast_radius retrieves it."""
        db_file = str(tmp_path / "test_blast.db")
        db = DatabaseManager(db_path=db_file)
        db.initialize()

        run_id = "br_db_01"
        db.insert_run(
            run_id=run_id,
            workflow="research_pipeline",
            timestamp=datetime.now(UTC).isoformat(),
            status="FAILURE",
            total_latency_ms=1400.0,
            total_tokens=4300,
            schema_version="1.0",
        )

        report = BlastRadiusReport(
            run_id=run_id,
            root_cause_agent="researcher",
            root_cause_node_id="step_1",
            tainted_steps_ratio=0.6667,
            total_steps=3,
            tainted_steps_count=2,
            wasted_tokens=TokenUsage(prompt=2300, completion=700, total=3000),
            wasted_cost_usd=0.0174,
            wasted_latency_ms=1200.0,
            impacted_agents=["writer", "verifier"],
            pricing_model="default",
            summary="Blast radius summary test",
        )
        db.save_blast_radius(report)

        loaded = db.get_blast_radius(run_id)
        assert loaded is not None
        assert loaded["run_id"] == run_id
        assert loaded["wasted_cost_usd"] == 0.0174
        assert loaded["tainted_steps_ratio"] == 0.6667
        assert loaded["impacted_agents"] == ["writer", "verifier"]

        # Verify individual SQLite metrics rows
        with db.connection() as conn:
            rows = conn.execute(
                "SELECT metric_name, metric_value, metric_unit FROM metrics WHERE run_id = ?",
                (run_id,),
            ).fetchall()
        metric_map = {r["metric_name"]: (r["metric_value"], r["metric_unit"]) for r in rows}
        assert metric_map["wasted_cost_usd"] == (0.0174, "USD")
        assert metric_map["wasted_tokens_total"] == (3000.0, "tokens")
        assert metric_map["wasted_latency_ms"] == (1200.0, "ms")
        assert metric_map["tainted_steps_ratio"] == (0.6667, "ratio")
        db.close()

    def test_analysis_bundle_and_api_verdict_integration(self, tmp_path: Path, monkeypatch) -> None:
        """Verify run_full_analysis attaches BlastRadiusReport and GET /api/runs/{id}/verdict returns it."""
        db_file = str(tmp_path / "test_api_blast.db")
        db = DatabaseManager(db_path=db_file)
        db.initialize()

        run_id = "br_api_01"
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
                    prompt_tokens=1000,
                    completion_tokens=400,
                    latency_ms=200.0,
                    output=(
                        "Key findings from [1] https://example.org/a, [2] https://example.org/b, "
                        "[3] https://example.org/c, [4] https://example.org/d. "
                        "OpenAI, Anthropic, Google DeepMind, Microsoft, and NVIDIA released new benchmarks in 2025."
                    ),
                ),
                _make_step(
                    run_id,
                    2,
                    "writer",
                    prompt_tokens=1500,
                    completion_tokens=500,
                    latency_ms=700.0,
                    output="Brief summary omitting sources and entities.",
                ),
                _make_step(
                    run_id,
                    3,
                    "verifier",
                    prompt_tokens=800,
                    completion_tokens=200,
                    latency_ms=300.0,
                    output="APPROVED: looks fine.",
                ),
            ],
            total_latency_ms=1200.0,
            total_tokens=4400,
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

        # Isolate dashboard state DB & cache for this test
        monkeypatch.setattr(dash_state, "get_db", lambda: db)
        dash_state._analysis_cache.pop(run_id, None)

        state = dash_state.run_full_analysis(run_id, db=db)
        assert state.error is None
        assert state.bundle is not None
        assert state.bundle.blast_radius is not None

        with TestClient(fastapi_app) as client:
            resp = client.get(f"/api/runs/{run_id}/verdict")
        assert resp.status_code == 200
        body = resp.json()
        assert body["run_id"] == run_id
        assert body["blast_radius"] is not None
        assert body["blast_radius"]["run_id"] == run_id
        assert body["blast_radius"]["tainted_steps_ratio"] >= 0.0

        # Cleanup cache entry & db pool
        dash_state._analysis_cache.pop(run_id, None)
        db.close()

    def test_serialization_validation_and_token_fallback(self) -> None:
        """Verify 60/40 prompt/completion fallback when breakdown is absent, ratio clamping, and JSON round-trip."""
        run_id = "br_fallback_01"
        trace = RunTrace(
            run_id=run_id,
            workflow="research_pipeline",
            status=StepStatus.ERROR,
            timestamp=datetime.now(UTC),
            steps=[
                _make_step(
                    run_id,
                    1,
                    "researcher",
                    status=StepStatus.ERROR,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=100,
                ),
                _make_step(
                    run_id,
                    2,
                    "writer",
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=1000,
                    latency_ms=250.0,
                ),
            ],
        )

        # Manually test clamping if an external graph supplies extra nodes beyond trace steps
        extra_graph = CausalGraphEngine().build_graph(trace)
        extra_graph.nodes["extra_cascade"] = CausalNode(
            node_id="extra_cascade",
            name="postprocessor",
            state=CausalNodeState.CASCADING_FAILURE,
            tokens_total=500,
            latency_ms=100.0,
        )
        extra_graph.cascades.append("extra_cascade")

        report = BlastRadiusCalculator().calculate(trace, causal_graph=extra_graph)
        assert 0.0 <= report.tainted_steps_ratio <= 1.0
        # Total = 1000 + 500 = 1500 -> 60% prompt (900), 40% completion (600)
        assert report.wasted_tokens.total == 1500
        assert report.wasted_tokens.prompt == 900
        assert report.wasted_tokens.completion == 600

        summary = report.format_summary()
        assert "Blast Radius Impact" in summary
        assert "Wasted Cost (USD)" in summary

        # AnalysisBundle round-trip
        bundle = AnalysisBundle(
            run_id=run_id,
            primary_cause=FailureCategory.EXECUTION,
            priority_level=PriorityLevel.P2,
            grounded=False,
            primary_agent="researcher",
            blast_radius=report,
        )
        reloaded = AnalysisBundle.model_validate_json(bundle.model_dump_json())
        assert reloaded.blast_radius is not None
        assert reloaded.blast_radius.wasted_tokens.total == 1500
