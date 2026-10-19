"""
tests/test_public_sdk.py — Day 57: Public Developer SDK Test Suite
===================================================================
Validates the top-level AgentLens public API (agentlens):
  1. Multi-agent pipeline tracing in under 5 lines of code.
  2. Decorator invocation variants (@trace_agent vs @trace_agent(role=..., name=...)).
  3. Async coroutine agent tracing (async def with async with trace_run).
  4. Automatic framework inspection (function signature, parameters, docstring).
  5. Trace run expected_output grounded recording.
  6. Agent error capture and exception propagation.
  7. evaluate_run integration (by RunTrace instance and by run_id string).
  8. SDK exception safety (zero interruption to host application).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import agentlens
from agentlens import (
    AgentStep,
    AnalysisBundle,
    FailureCategory,
    PriorityLevel,
    RunTrace,
    SpanType,
    StepStatus,
    TraceSpan,
)


class TestPublicSDK:
    def test_under_5_lines_pipeline(self) -> None:
        """Verify the canonical < 5 lines of code user experience."""

        @agentlens.trace_agent(role="researcher")
        def research(topic: str) -> dict[str, Any]:
            return {"findings": f"Research on {topic}", "source_count": 3, "entity_count": 2}

        @agentlens.trace_agent(role="writer")
        def write(findings: dict[str, Any]) -> dict[str, Any]:
            return {
                "report": f"Report: {findings['findings']}",
                "source_count": 3,
                "entity_count": 2,
            }

        with agentlens.trace_run("minimal_pipeline") as run:
            res = research("Quantum Computing")
            write(res)

        assert len(run.steps) == 2
        assert len(run.spans) == 2
        assert run.steps[0].agent == "research"
        assert run.steps[1].agent == "write"

        bundle = agentlens.evaluate_run(run)
        assert isinstance(bundle, AnalysisBundle)
        assert bundle.priority_level == PriorityLevel.P5
        assert bundle.primary_agent is None

    def test_trace_agent_decorator_variants(self) -> None:
        """Verify bare @trace_agent, @trace_agent(), and @trace_agent(role=..., name=...)."""

        @agentlens.trace_agent
        def step_bare(data: dict[str, Any]) -> dict[str, Any]:
            return {"status": "bare"}

        @agentlens.trace_agent()
        def step_parens(data: dict[str, Any]) -> dict[str, Any]:
            return {"status": "parens"}

        @agentlens.trace_agent(role="auditor", name="compliance_node")
        def step_custom(data: dict[str, Any]) -> dict[str, Any]:
            return {"status": "custom"}

        with agentlens.trace_run("variants_pipeline") as run:
            step_bare({})
            step_parens({})
            step_custom({})

        assert len(run.steps) == 3
        assert run.steps[0].agent == "step_bare"
        assert run.steps[1].agent == "step_parens"
        assert run.steps[2].agent == "compliance_node"

        assert len(run.spans) == 3
        assert run.spans[2].attributes.get("role") == "auditor"

    @pytest.mark.asyncio
    async def test_async_agent_tracing(self) -> None:
        """Verify async def agent nodes and async with trace_run."""

        @agentlens.trace_agent(role="fetcher")
        async def async_fetch(url: str) -> dict[str, Any]:
            return {"url": url, "status_code": 200, "source_count": 1, "entity_count": 1}

        @agentlens.trace_agent(role="parser")
        async def async_parse(payload: dict[str, Any]) -> dict[str, Any]:
            return {"parsed": payload["url"], "source_count": 1, "entity_count": 1}

        async with agentlens.trace_run("async_pipeline") as run:
            fetched = await async_fetch("https://example.com/api")
            await async_parse(fetched)

        assert len(run.steps) == 2
        assert len(run.spans) == 2
        assert run.steps[0].agent == "async_fetch"
        assert run.steps[1].agent == "async_parse"
        assert run.steps[0].latency_ms >= 0.0

    def test_auto_inspection_and_docstring(self) -> None:
        """Verify parameter signature inspection and docstring metadata capture."""

        @agentlens.trace_agent(role="analyst")
        def analyze_market(symbol: str, period_days: int = 30) -> dict[str, Any]:
            """Perform technical market analysis across defined trading period."""
            return {"symbol": symbol, "period": period_days, "signal": "BUY"}

        with agentlens.trace_run("inspection_pipeline") as run:
            analyze_market("NVDA", period_days=60)

        assert len(run.steps) == 1
        step = run.steps[0]
        assert "symbol" in step.input
        assert "NVDA" in step.input
        assert "60" in step.input
        assert "Perform technical market analysis" in step.prompt

    def test_trace_run_with_expected_output(self) -> None:
        """Verify expected_output parameter recording for P1 grounded evaluation."""
        gt_data = {"expected_sentiment": "positive", "confidence": 0.95}

        @agentlens.trace_agent(role="classifier")
        def classify(text: str) -> dict[str, Any]:
            return {"sentiment": "positive"}

        with agentlens.trace_run("grounded_pipeline", expected_output=gt_data) as run:
            classify("AgentLens is incredible!")

        assert run.expected_output is not None
        assert "expected_sentiment" in run.expected_output

    def test_error_capture_and_propagation(self) -> None:
        """Verify exceptions in agent functions are recorded as ERROR steps and re-raised."""

        @agentlens.trace_agent(role="calculator")
        def buggy_division(a: int, b: int) -> float:
            return a / b

        with pytest.raises(ZeroDivisionError):
            with agentlens.trace_run("error_pipeline") as run:
                buggy_division(10, 0)

        assert len(run.steps) == 1
        failed_step = run.steps[0]
        assert failed_step.status == StepStatus.ERROR
        assert "ZeroDivisionError" in str(failed_step.error)

        assert len(run.spans) == 1
        failed_span = run.spans[0]
        assert failed_span.status == StepStatus.ERROR
        assert "ZeroDivisionError" in str(failed_span.error)

    def test_evaluate_run_by_instance_and_run_id(self, tmp_path: Path) -> None:
        """Verify evaluate_run works with RunTrace object and run_id string from database."""
        db_file = tmp_path / "sdk_test.db"
        agentlens.init(db_path=db_file)

        @agentlens.trace_agent(role="researcher")
        def mock_research() -> dict[str, Any]:
            return {"findings": "Normal data", "source_count": 2, "entity_count": 2}

        with agentlens.trace_run("eval_pipeline", sync=True) as run:
            mock_research()

        # 1. Evaluate by RunTrace instance
        bundle_inst = agentlens.evaluate_run(run)
        assert bundle_inst.priority_level == PriorityLevel.P5
        assert bundle_inst.primary_cause == FailureCategory.UNKNOWN

        # 2. Evaluate by run_id string from database
        bundle_id = agentlens.evaluate_run(run.run_id, db_path=db_file)
        assert bundle_id.priority_level == PriorityLevel.P5
        assert bundle_id.run_id == run.run_id

    def test_re_exports_and_public_surface(self) -> None:
        """Verify all essential schema and SDK types are cleanly re-exported."""
        assert hasattr(agentlens, "init")
        assert hasattr(agentlens, "trace_run")
        assert hasattr(agentlens, "trace_agent")
        assert hasattr(agentlens, "get_current_trace")
        assert hasattr(agentlens, "evaluate_run")
        assert hasattr(agentlens, "__version__")

        assert issubclass(RunTrace, object)
        assert issubclass(AgentStep, object)
        assert issubclass(TraceSpan, object)
        assert issubclass(AnalysisBundle, object)
        assert SpanType.AGENT.value == "agent"
