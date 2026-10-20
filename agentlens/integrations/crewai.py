"""
agentlens/integrations/crewai.py — CrewAI Swarm Telemetry & Auto-Instrumentation
================================================================================
Day 59: Provides zero-code-change auto-instrumentation for CrewAI swarms.

Intercepts:
  - Crew.kickoff() execution lifecycle (sequential & hierarchical processes)
  - task_callback -> AgentStep and TraceSpan (AGENT / TASK)
  - step_callback -> TraceSpan (TOOL) and step.tool_calls
  - Agent delegation, state handoffs, and token consumption
  - Hierarchical manager-worker call hierarchies into OpenInference DAG spans
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from capture.handoff import HandoffCapture
from normalizer.normalizer import Normalizer
from schema.models import (
    AgentStep,
    AnalysisBundle,
    RunTrace,
    StepStatus,
    TokenUsage,
)
from schema.span import SpanType, TraceSpan
from storage.db import DatabaseManager
from storage.writer import StorageWriter

logger = logging.getLogger("agentlens.integrations.crewai")


def _safe_serialize(obj: Any) -> str:
    """Serialize an object to JSON string with fallback."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, default=str)
    except Exception:
        return str(obj)


def _extract_agent_role(agent_obj: Any) -> str:
    """Extract role name from an Agent object, dict, or string."""
    if agent_obj is None:
        return "crew_agent"
    if isinstance(agent_obj, str):
        return agent_obj
    if hasattr(agent_obj, "role"):
        return str(agent_obj.role)
    if isinstance(agent_obj, dict):
        return str(agent_obj.get("role") or agent_obj.get("name") or "crew_agent")
    return getattr(agent_obj, "name", str(agent_obj))


def _extract_task_output_payload(task_output: Any) -> tuple[str, dict[str, Any], str]:
    """
    Extract (raw_output, output_dict, description) from a CrewAI TaskOutput.
    Supports real CrewAI TaskOutput objects as well as duck-typed mocks/dicts.
    """
    if task_output is None:
        return "", {}, ""

    if isinstance(task_output, dict):
        raw = str(task_output.get("raw") or task_output.get("output") or "")
        desc = str(task_output.get("description") or "")
        json_dict = task_output.get("json_dict")
        if isinstance(json_dict, dict):
            out_dict = json_dict
        else:
            try:
                out_dict = json.loads(raw) if raw.strip().startswith("{") else {"output": raw}
            except Exception:
                out_dict = {"output": raw}
        return raw, out_dict, desc

    # CrewAI TaskOutput object
    raw = (
        getattr(task_output, "raw", None)
        or getattr(task_output, "output", None)
        or str(task_output)
    )
    desc = getattr(task_output, "description", "") or ""
    json_dict = getattr(task_output, "json_dict", None)

    if isinstance(json_dict, dict):
        out_dict = json_dict
    else:
        try:
            out_dict = json.loads(raw) if str(raw).strip().startswith("{") else {"output": raw}
        except Exception:
            out_dict = {"output": raw}

    return str(raw), out_dict, str(desc)


class AgentLensCrewHook:
    """
    Automatic telemetry hook and listener for CrewAI swarms.

    Usage with CrewAI:
        hook = AgentLensCrewHook(workflow="market_research", save_to_db=True)

        crew = Crew(
            agents=[researcher, analyst],
            tasks=[task1, task2],
            step_callback=hook.step_callback,
            task_callback=hook.task_callback,
        )
        crew.kickoff(inputs={"topic": "AI Trends"})

        trace = hook.get_trace()
        bundle = hook.evaluate()

    Or via context manager:
        with hook.monitor_crew(crew):
            crew.kickoff(inputs={"topic": "AI Trends"})
    """

    def __init__(
        self,
        workflow: str = "crewai_swarm",
        run_id: str | None = None,
        save_to_db: bool = False,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.workflow = workflow
        self.run_id = run_id or f"run_{uuid4().hex[:8]}"
        self.save_to_db = save_to_db
        self.tags = tags or []
        self.metadata = metadata or {}

        # Root DAG span
        self._root_span_id = f"span_{uuid4().hex[:8]}"

        # Initialize RunTrace
        self.trace = RunTrace(
            run_id=self.run_id,
            workflow=self.workflow,
            status=StepStatus.SUCCESS,
            timestamp=datetime.now(UTC),
            steps=[],
            spans=[],
        )

        # Iteration indexing per agent role
        self._node_iterations: dict[str, int] = {}

        # In-flight state tracking
        self._active_tools: list[dict[str, Any]] = []
        self._last_state: dict[str, Any] = {}
        self._task_start_times: dict[str, float] = {}
        self._manager_span_id: str | None = None

    def get_trace(self) -> RunTrace:
        """Finalize and return the recorded RunTrace."""
        # Calculate aggregates
        total_lat = sum(s.latency_ms for s in self.trace.steps)
        total_tok = sum(s.tokens.total for s in self.trace.steps)
        self.trace.total_latency_ms = total_lat
        self.trace.total_tokens = total_tok

        # If any step had ERROR/FAILURE, propagate to trace
        if any(s.status in (StepStatus.ERROR, StepStatus.FAILURE) for s in self.trace.steps):
            self.trace.status = StepStatus.FAILURE

        if self.save_to_db:
            try:
                db = DatabaseManager()
                db.initialize()
                normalized = Normalizer().normalize_run(self.trace)
                StorageWriter(db).write_run(
                    normalized,
                    trace_json=self.trace.model_dump_json(indent=2),
                )
            except Exception as exc:
                logger.warning("Failed to auto-save CrewAI trace to database: %s", exc)

        return self.trace

    def get_spans(self) -> list[TraceSpan]:
        """Return all recorded OpenInference trace spans."""
        return self.trace.spans

    def get_steps(self) -> list[AgentStep]:
        """Return all recorded sequential agent steps."""
        return self.trace.steps

    def evaluate(self) -> AnalysisBundle:
        """Run Arbiter diagnostic root-cause analysis on this trace."""
        import agentlens

        return agentlens.evaluate_run(self.get_trace())

    # ─────────────────────────────────────────────────────────────────────────
    # Context Manager for Automatic Crew Wrapping
    # ─────────────────────────────────────────────────────────────────────────

    @contextmanager
    def monitor_crew(self, crew: Any) -> Generator[AgentLensCrewHook, None, None]:
        """
        Context manager that automatically attaches callbacks to a Crew instance,
        monitors execution, and restores original callbacks on exit.
        """
        orig_step_cb = getattr(crew, "step_callback", None)
        orig_task_cb = getattr(crew, "task_callback", None)

        def wrapped_step_cb(step_output: Any) -> Any:
            self.step_callback(step_output)
            if orig_step_cb and callable(orig_step_cb):
                return orig_step_cb(step_output)

        def wrapped_task_cb(task_output: Any) -> Any:
            self.task_callback(task_output)
            if orig_task_cb and callable(orig_task_cb):
                return orig_task_cb(task_output)

        crew.step_callback = wrapped_step_cb
        crew.task_callback = wrapped_task_cb

        # Check if hierarchical process with manager agent
        process = getattr(crew, "process", None)
        process_name = str(process).lower()
        if "hierarchical" in process_name or getattr(crew, "manager_agent", None):
            mgr = getattr(crew, "manager_agent", None)
            mgr_role = _extract_agent_role(mgr) if mgr else "crew_manager"
            self.set_manager_agent(mgr_role)

        try:
            yield self
        finally:
            crew.step_callback = orig_step_cb
            crew.task_callback = orig_task_cb
            self.get_trace()

    # ─────────────────────────────────────────────────────────────────────────
    # Hierarchical Process Delegation
    # ─────────────────────────────────────────────────────────────────────────

    def set_manager_agent(self, manager_role: str = "crew_manager") -> str:
        """
        Establish a parent manager agent span for hierarchical CrewAI processes.
        Subsequent worker task spans will link to this manager as parent_span_id.
        """
        mgr_span_id = f"span_{uuid4().hex[:8]}"
        self._manager_span_id = mgr_span_id

        # Record manager AGENT span
        span = TraceSpan(
            span_id=mgr_span_id,
            trace_id=self.trace.run_id,
            parent_span_id=self._root_span_id,
            name=manager_role,
            span_type=SpanType.AGENT,
            iteration_index=0,
            latency_ms=0.0,
            status=StepStatus.SUCCESS,
            input=_safe_serialize({"process": "hierarchical", "manager": manager_role}),
            output=_safe_serialize({"delegation": "active"}),
        )
        self.trace.spans.append(span)
        return mgr_span_id

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle Callbacks (CrewAI step_callback & task_callback)
    # ─────────────────────────────────────────────────────────────────────────

    def step_callback(self, step_output: Any) -> None:
        """
        CrewAI step_callback hook: captures intermediate agent thoughts, actions,
        and nested tool calls.
        """
        # Determine tool name & inputs
        tool_name = (
            getattr(step_output, "tool", None)
            or getattr(step_output, "tool_name", None)
            or (step_output.get("tool") if isinstance(step_output, dict) else None)
            or "crew_tool"
        )
        tool_input = (
            getattr(step_output, "tool_input", None)
            or (step_output.get("tool_input") if isinstance(step_output, dict) else None)
            or ""
        )
        tool_res = (
            getattr(step_output, "result", None)
            or getattr(step_output, "output", None)
            or (step_output.get("result") if isinstance(step_output, dict) else None)
            or ""
        )
        tool_err = getattr(step_output, "error", None) or (
            step_output.get("error") if isinstance(step_output, dict) else None
        )

        tool_record: dict[str, Any] = {
            "name": str(tool_name),
            "input": tool_input,
            "output": tool_res,
            "latency_ms": 1.0,
        }
        if tool_err:
            tool_record["error"] = str(tool_err)

        self._active_tools.append(tool_record)

        # Record TOOL TraceSpan
        tool_span = TraceSpan(
            span_id=f"span_{uuid4().hex[:8]}",
            trace_id=self.trace.run_id,
            parent_span_id=self._manager_span_id or self._root_span_id,
            name=str(tool_name),
            span_type=SpanType.TOOL,
            latency_ms=1.0,
            status=StepStatus.ERROR if tool_err else StepStatus.SUCCESS,
            error=str(tool_err) if tool_err else None,
            input=_safe_serialize(tool_input),
            output=_safe_serialize(tool_res),
        )
        self.trace.spans.append(tool_span)

    def task_callback(
        self,
        task_output: Any,
        *,
        agent_role: str | None = None,
        latency_ms: float | None = None,
        tokens: TokenUsage | None = None,
        error: str | None = None,
    ) -> AgentStep:
        """
        CrewAI task_callback hook: triggered upon completion of a CrewAI Task.
        Extracts output state, performs state diffing via HandoffCapture,
        and constructs an AgentStep and AGENT TraceSpan.
        """
        raw_output, out_dict, task_desc = _extract_task_output_payload(task_output)

        # Resolve agent role
        if not agent_role:
            raw_agent = getattr(task_output, "agent", None)
            if raw_agent:
                agent_role = _extract_agent_role(raw_agent)
            elif isinstance(task_output, dict) and "agent" in task_output:
                agent_role = _extract_agent_role(task_output["agent"])
            else:
                agent_role = f"agent_{len(self.trace.steps) + 1}"

        iter_idx = self._node_iterations.get(agent_role, 0)
        self._node_iterations[agent_role] = iter_idx + 1

        step_idx = len(self.trace.steps) + 1

        # Use previous output state as input state if available
        input_state = (
            self._last_state if self._last_state else {"task": task_desc or f"Task {step_idx}"}
        )

        # State handoff capture
        capture = HandoffCapture(input_state=input_state)
        capture.record_agent_return(out_dict)
        in_s, filt_s, out_s, diff = capture.finalize()
        self._last_state = out_dict

        # Collect nested tool calls recorded during this task
        tools = list(self._active_tools)
        self._active_tools.clear()

        # Determine status: if error explicitly provided or any tool had error
        has_tool_err = any("error" in t for t in tools)
        status = StepStatus.ERROR if (error or has_tool_err) else StepStatus.SUCCESS

        computed_lat = latency_ms if latency_ms is not None else 10.0
        computed_tokens = tokens or TokenUsage(prompt=0, completion=0, total=0)

        step = AgentStep(
            run_id=self.trace.run_id,
            step=step_idx,
            agent=agent_role,
            status=status,
            input=_safe_serialize(input_state),
            output=raw_output,
            prompt=f"CrewAI Task: {task_desc or agent_role} (iteration {iter_idx})",
            error=error or (tools[0]["error"] if has_tool_err and "error" in tools[0] else None),
            timestamp=datetime.now(UTC),
            latency_ms=computed_lat,
            tokens=computed_tokens,
            tool_calls=tools,
            parent_step=str(step_idx - 1) if step_idx > 1 else None,
        )

        step.handoff.input_state = in_s
        step.handoff.filtered_state = filt_s
        step.handoff.output_state = out_s
        if diff.summary():
            step.prompt = f"{step.prompt} | {diff.summary()}"

        self.trace.steps.append(step)

        # Parent span points to manager if hierarchical, otherwise root span
        parent_span_id = self._manager_span_id or self._root_span_id
        span = TraceSpan(
            span_id=f"span_{uuid4().hex[:8]}",
            trace_id=self.trace.run_id,
            parent_span_id=parent_span_id,
            name=agent_role,
            span_type=SpanType.AGENT,
            iteration_index=iter_idx,
            latency_ms=computed_lat,
            status=status,
            error=step.error,
            input=step.input,
            output=step.output,
            tokens=step.tokens,
        )
        self.trace.spans.append(span)

        return step

    # ─────────────────────────────────────────────────────────────────────────
    # Explicit Manual Hooks (for tests & custom orchestrators)
    # ─────────────────────────────────────────────────────────────────────────

    def on_tool_start(self, tool_name: str, tool_input: Any) -> str:
        """Explicitly record the start of a tool execution."""
        tool_id = f"tool_{uuid4().hex[:8]}"
        self._task_start_times[tool_id] = time.perf_counter()
        return tool_id

    def on_tool_end(
        self,
        tool_name: str,
        output: Any,
        *,
        tool_input: Any = "",
        tool_id: str | None = None,
    ) -> None:
        """Explicitly record successful completion of a tool execution."""
        lat = 1.0
        if tool_id and tool_id in self._task_start_times:
            lat = (time.perf_counter() - self._task_start_times.pop(tool_id)) * 1000.0

        self.step_callback(
            {
                "tool": tool_name,
                "tool_input": tool_input,
                "result": output,
                "latency_ms": lat,
            }
        )

    def on_tool_error(
        self,
        tool_name: str,
        error: BaseException | str,
        *,
        tool_input: Any = "",
        tool_id: str | None = None,
    ) -> None:
        """Explicitly record tool execution failure (triggers Arbiter tool_failure_v1)."""
        lat = 1.0
        if tool_id and tool_id in self._task_start_times:
            lat = (time.perf_counter() - self._task_start_times.pop(tool_id)) * 1000.0

        err_msg = (
            f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else str(error)
        )
        self.step_callback(
            {
                "tool": tool_name,
                "tool_input": tool_input,
                "result": "",
                "error": err_msg,
                "latency_ms": lat,
            }
        )
