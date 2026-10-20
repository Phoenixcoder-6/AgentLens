"""
AgentLens v2.0 — Public Developer SDK
=====================================
Multi-Agent Failure Attribution, Trace Diffing & Explainability Platform.

Usage in under 5 lines of code:
    import agentlens

    @agentlens.trace_agent(role="researcher")
    def research(topic: str) -> dict:
        return {"findings": f"Notes on {topic}", "source_count": 3}

    @agentlens.trace_agent(role="writer")
    def write(findings: dict) -> dict:
        return {"report": f"Report on {findings}"}

    with agentlens.trace_run("my_pipeline") as run:
        write(research("Quantum Computing"))

    bundle = agentlens.evaluate_run(run)
    print(bundle.primary_cause, bundle.primary_agent)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from analyzers.arbiter import Arbiter
from analyzers.detection.rule_engine import RuleEngine
from capture.session import CaptureSession
from schema.models import (
    AgentStep,
    AnalysisBundle,
    FailureCategory,
    PriorityLevel,
    RunTrace,
    StepStatus,
    TokenUsage,
)
from schema.span import SpanType, TraceSpan
from storage.db import DatabaseManager

from .context_manager import TraceRunContext
from .decorator import trace_agent
from .integrations.autogen import AgentLensAutoGenHook, AgentLensConversationTracer
from .integrations.crewai import AgentLensCrewHook
from .integrations.langchain import AgentLensCallbackHandler

__version__ = "2.0.0"

logger = logging.getLogger("agentlens")

# Global SDK Configuration State
_SDK_CONFIG: dict[str, Any] = {
    "initialized": False,
    "storage_path": None,
    "auto_evaluate": False,
}


def init(
    storage_path: str | Path | None = None,
    db_path: str | Path | None = None,
    auto_evaluate: bool = False,
    async_logging: bool = True,
) -> None:
    """
    Initialize global AgentLens SDK configuration.

    Args:
        storage_path: Path to database file or directory.
        db_path: Explicit SQLite database path.
        auto_evaluate: Automatically run Arbiter analysis when trace_run completes.
        async_logging: Enable non-blocking asynchronous trace telemetry ingestion.
    """
    target_path = db_path or storage_path
    if target_path:
        import os

        os.environ["DB_PATH"] = str(target_path)
        db = DatabaseManager(db_path=Path(target_path))
        db.initialize()
        _SDK_CONFIG["storage_path"] = str(target_path)

    _SDK_CONFIG["auto_evaluate"] = auto_evaluate
    _SDK_CONFIG["initialized"] = True
    logger.info("AgentLens SDK v%s initialized (storage=%s)", __version__, target_path)


def trace_run(
    workflow: str = "default_workflow",
    run_id: str | None = None,
    expected_output: Any = None,
    sync: bool | None = None,
) -> TraceRunContext:
    """
    Context manager managing the lifecycle of an agent execution trace.

    Usage:
        with agentlens.trace_run("document_pipeline") as run:
            agent1()
            agent2()
    """
    return TraceRunContext(
        workflow=workflow,
        run_id=run_id,
        expected_output=expected_output,
        sync=sync,
    )


def get_current_trace() -> RunTrace | None:
    """Return the active RunTrace for the current execution context, or None."""
    return CaptureSession.get_current_trace()


def evaluate_run(
    trace: RunTrace | str | None = None,
    *,
    db_path: str | Path | None = None,
) -> AnalysisBundle:
    """
    Evaluate an agent trace using the Arbiter and deterministic RuleEngine.

    Args:
        trace: RunTrace instance, run_id string, or None (evaluates active trace).
        db_path: Optional path to SQLite database when loading by run_id string.

    Returns:
        AnalysisBundle containing the primary root cause, priority level, and primary agent.
    """
    trace_to_evaluate: RunTrace | None = None

    if isinstance(trace, RunTrace):
        trace_to_evaluate = trace
    elif isinstance(trace, str):
        db = DatabaseManager(db_path=Path(db_path)) if db_path else DatabaseManager()
        run_row = db.get_run(trace)
        if not run_row:
            raise ValueError(f"Run ID '{trace}' not found in database.")
        raw_json = run_row.get("trace_json")
        if not raw_json:
            raise ValueError(f"Run ID '{trace}' contains no trace_json payload.")
        trace_to_evaluate = RunTrace.model_validate_json(raw_json)
    elif trace is None:
        trace_to_evaluate = get_current_trace()

    if trace_to_evaluate is None:
        raise ValueError("No trace provided and no active trace session found.")

    rule_engine = RuleEngine()
    analysis_res = rule_engine.analyze(trace_to_evaluate)
    arbiter = Arbiter()
    bundle = arbiter.run(run_id=trace_to_evaluate.run_id, evidence=analysis_res.evidence)
    return bundle


__all__ = [
    "__version__",
    "init",
    "trace_run",
    "trace_agent",
    "AgentLensCallbackHandler",
    "AgentLensCrewHook",
    "AgentLensAutoGenHook",
    "AgentLensConversationTracer",
    "get_current_trace",
    "evaluate_run",
    "RunTrace",
    "AgentStep",
    "TraceSpan",
    "SpanType",
    "AnalysisBundle",
    "FailureCategory",
    "PriorityLevel",
    "StepStatus",
    "TokenUsage",
]
