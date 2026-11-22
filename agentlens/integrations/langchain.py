"""
agentlens/integrations/langchain.py — LangGraph & LangChain Auto-Instrumentation
================================================================================
Day 58: Provides zero-code-change auto-instrumentation for LangGraph and LangChain.

Intercepts:
  - on_chain_start / on_chain_end / on_chain_error -> AgentStep and TraceSpan (AGENT / CHAIN)
  - on_tool_start / on_tool_end / on_tool_error   -> TraceSpan (TOOL) and step.tool_calls
  - on_llm_start / on_llm_end / on_llm_error       -> TraceSpan (LLM) and step.tokens
  - State handoffs, token consumption, and OpenInference DAG hierarchy
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:

    class BaseCallbackHandler:  # type: ignore[no-redef]
        """Fallback base when langchain-core is not installed."""

        pass


from capture.handoff import HandoffCapture
from schema.models import (
    AgentStep,
    AnalysisBundle,
    RunTrace,
    StepStatus,
    TokenUsage,
)
from schema.span import SpanType, TraceSpan

logger = logging.getLogger("agentlens.integrations.langchain")


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


class AgentLensCallbackHandler(BaseCallbackHandler):
    """
    Automatic telemetry callback handler for LangGraph and LangChain pipelines.

    Usage with LangGraph:
        handler = AgentLensCallbackHandler(workflow="customer_support")
        app.invoke(inputs, config={"callbacks": [handler]})
        trace = handler.get_trace()
        bundle = handler.evaluate()
    """

    def __init__(
        self,
        workflow: str = "langgraph_workflow",
        run_id: str | None = None,
        save_to_db: bool = False,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.workflow = workflow
        self.run_id = run_id or f"run_{uuid4().hex[:8]}"
        self.save_to_db = save_to_db
        self.tags = tags or []
        self.metadata = metadata or {}

        # Initialize RunTrace
        self.trace = RunTrace(
            run_id=self.run_id,
            workflow=self.workflow,
            status=StepStatus.SUCCESS,
            timestamp=datetime.now(UTC),
            steps=[],
            spans=[],
        )

        # In-flight tracking tables
        self._root_run_id: UUID | None = None
        self._active_chains: dict[UUID, dict[str, Any]] = {}
        self._active_tools: dict[UUID, dict[str, Any]] = {}
        self._active_llms: dict[UUID, dict[str, Any]] = {}
        self._active_steps: dict[UUID, AgentStep] = {}
        self._active_captures: dict[UUID, HandoffCapture] = {}

        # Loop iteration counting per node name
        self._node_iterations: dict[str, int] = {}

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
                from normalizer.normalizer import Normalizer
                from storage.db import DatabaseManager
                from storage.writer import StorageWriter

                db = DatabaseManager()
                db.initialize()
                normalized = Normalizer().normalize_run(self.trace)
                StorageWriter(db).write_run(
                    normalized,
                    trace_json=self.trace.model_dump_json(indent=2),
                )
            except Exception as exc:
                logger.warning("Failed to auto-save LangGraph trace to database: %s", exc)

        return self.trace

    def get_spans(self) -> list[TraceSpan]:
        """Return all captured OpenInference trace spans."""
        return self.trace.spans

    def get_steps(self) -> list[AgentStep]:
        """Return all captured sequential agent steps."""
        return self.trace.steps

    def evaluate(self) -> AnalysisBundle:
        """Run Arbiter diagnostic root-cause analysis on this trace."""
        import agentlens

        return agentlens.evaluate_run(self.get_trace())

    # ─────────────────────────────────────────────────────────────────────────
    # Chain / Node Lifecycle Hooks
    # ─────────────────────────────────────────────────────────────────────────

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        meta = metadata or {}
        ser = serialized or {}

        # Identify root graph vs child node
        if parent_run_id is None and self._root_run_id is None:
            self._root_run_id = run_id
            self._active_chains[run_id] = {
                "start_time": time.perf_counter(),
                "name": ser.get("name") or self.workflow,
                "is_root": True,
            }
            return

        # Determine whether this is a graph node or a routing / internal chain
        chain_name = kwargs.get("name") or ser.get("name")
        lg_node = meta.get("langgraph_node")

        if lg_node:
            if chain_name and chain_name != lg_node:
                # Conditional routing edge or branch condition
                is_step = False
                span_type = SpanType.ROUTER
                node_name = chain_name
            else:
                is_step = True
                span_type = SpanType.AGENT
                node_name = lg_node
        else:
            is_step = True
            span_type = SpanType.AGENT
            node_name = (
                meta.get("node_name")
                or ser.get("name")
                or kwargs.get("name")
                or f"agent_{len(self.trace.steps) + 1}"
            )

        if not is_step:
            self._active_chains[run_id] = {
                "start_time": time.perf_counter(),
                "name": node_name,
                "span_type": span_type,
                "parent_run_id": parent_run_id,
                "is_root": False,
                "is_step": False,
                "inputs": inputs,
            }
            return

        iter_idx = self._node_iterations.get(node_name, 0)
        self._node_iterations[node_name] = iter_idx + 1

        step_idx = len(self.trace.steps) + 1
        input_dict = inputs if isinstance(inputs, dict) else {"input": inputs}

        capture = HandoffCapture(input_state=input_dict)
        self._active_captures[run_id] = capture

        step = AgentStep(
            run_id=self.trace.run_id,
            step=step_idx,
            agent=node_name,
            status=StepStatus.SUCCESS,
            input=_safe_serialize(input_dict),
            prompt=f"LangGraph node: {node_name} (iteration {iter_idx})",
            timestamp=datetime.now(UTC),
            parent_step=str(step_idx - 1) if step_idx > 1 else None,
        )
        self._active_steps[run_id] = step

        self._active_chains[run_id] = {
            "start_time": time.perf_counter(),
            "name": node_name,
            "iteration_index": iter_idx,
            "parent_run_id": parent_run_id,
            "is_root": False,
            "is_step": True,
        }

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        chain_info = self._active_chains.pop(run_id, None)
        if not chain_info:
            return

        latency_ms = (time.perf_counter() - chain_info["start_time"]) * 1000.0

        if chain_info.get("is_root"):
            return

        if not chain_info.get("is_step", True):
            # Record router / intermediate TraceSpan
            parent_span_id = str(parent_run_id) if parent_run_id else None
            span = TraceSpan(
                span_id=str(run_id),
                trace_id=self.trace.run_id,
                parent_span_id=parent_span_id,
                name=chain_info["name"],
                span_type=chain_info.get("span_type", SpanType.ROUTER),
                latency_ms=latency_ms,
                status=StepStatus.SUCCESS,
                input=_safe_serialize(chain_info.get("inputs", {})),
                output=_safe_serialize(outputs),
            )
            self.trace.spans.append(span)
            return

        step = self._active_steps.pop(run_id, None)
        capture = self._active_captures.pop(run_id, None)

        if step:
            step.step = len(self.trace.steps) + 1
            if step.step > 1:
                step.parent_step = str(step.step - 1)

            out_dict = outputs if isinstance(outputs, dict) else {"output": outputs}
            step.output = _safe_serialize(out_dict)
            step.latency_ms = latency_ms

            if capture:
                capture.record_agent_return(out_dict)
                in_s, filt_s, out_s, diff = capture.finalize()
                step.handoff.input_state = in_s
                step.handoff.filtered_state = filt_s
                step.handoff.output_state = out_s
                if diff.summary():
                    step.prompt = f"{step.prompt} | {diff.summary()}"

            self.trace.steps.append(step)

            # Record corresponding TraceSpan
            parent_span_id = str(parent_run_id) if parent_run_id else None
            span = TraceSpan(
                span_id=str(run_id),
                trace_id=self.trace.run_id,
                parent_span_id=parent_span_id,
                name=step.agent,
                span_type=SpanType.AGENT,
                iteration_index=chain_info.get("iteration_index", 0),
                latency_ms=latency_ms,
                status=StepStatus.SUCCESS,
                input=step.input,
                output=step.output,
                tokens=step.tokens,
            )
            self.trace.spans.append(span)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        chain_info = self._active_chains.pop(run_id, None)
        if not chain_info:
            return

        latency_ms = (time.perf_counter() - chain_info["start_time"]) * 1000.0
        if chain_info.get("is_root"):
            return

        if not chain_info.get("is_step", True):
            parent_span_id = str(parent_run_id) if parent_run_id else None
            span = TraceSpan(
                span_id=str(run_id),
                trace_id=self.trace.run_id,
                parent_span_id=parent_span_id,
                name=chain_info["name"],
                span_type=chain_info.get("span_type", SpanType.ROUTER),
                latency_ms=latency_ms,
                status=StepStatus.ERROR,
                error=f"{type(error).__name__}: {error}",
                input=_safe_serialize(chain_info.get("inputs", {})),
                output="",
            )
            self.trace.spans.append(span)
            return

        step = self._active_steps.pop(run_id, None)
        capture = self._active_captures.pop(run_id, None)

        if step:
            step.step = len(self.trace.steps) + 1
            if step.step > 1:
                step.parent_step = str(step.step - 1)
            step.status = StepStatus.ERROR
            step.latency_ms = latency_ms
            step.error = f"{type(error).__name__}: {error}"

            if capture:
                capture.record_agent_return({})
                in_s, filt_s, out_s, _ = capture.finalize()
                step.handoff.input_state = in_s
                step.handoff.filtered_state = filt_s
                step.handoff.output_state = out_s

            self.trace.steps.append(step)

            parent_span_id = str(parent_run_id) if parent_run_id else None
            span = TraceSpan(
                span_id=str(run_id),
                trace_id=self.trace.run_id,
                parent_span_id=parent_span_id,
                name=step.agent,
                span_type=SpanType.AGENT,
                iteration_index=chain_info.get("iteration_index", 0),
                latency_ms=latency_ms,
                status=StepStatus.ERROR,
                error=step.error,
                input=step.input,
                output="",
            )
            self.trace.spans.append(span)

    # ─────────────────────────────────────────────────────────────────────────
    # Tool Invocation Lifecycle Hooks
    # ─────────────────────────────────────────────────────────────────────────

    def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        tool_name = (serialized or {}).get("name") or kwargs.get("name") or "tool"
        tool_in = inputs if inputs is not None else input_str

        self._active_tools[run_id] = {
            "name": tool_name,
            "input": tool_in,
            "start_time": time.perf_counter(),
            "parent_run_id": parent_run_id,
        }

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        tool_info = self._active_tools.pop(run_id, None)
        if not tool_info:
            return

        latency_ms = (time.perf_counter() - tool_info["start_time"]) * 1000.0
        tool_name = tool_info["name"]
        tool_input = tool_info["input"]

        # If executed within an active agent step, record in step.tool_calls
        if parent_run_id and parent_run_id in self._active_steps:
            self._active_steps[parent_run_id].tool_calls.append(
                {
                    "name": tool_name,
                    "input": tool_input,
                    "output": output,
                    "latency_ms": latency_ms,
                }
            )

        # Record TOOL span
        parent_span_id = str(parent_run_id) if parent_run_id else None
        span = TraceSpan(
            span_id=str(run_id),
            trace_id=self.trace.run_id,
            parent_span_id=parent_span_id,
            name=tool_name,
            span_type=SpanType.TOOL,
            latency_ms=latency_ms,
            status=StepStatus.SUCCESS,
            input=_safe_serialize(tool_input),
            output=_safe_serialize(output),
        )
        self.trace.spans.append(span)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        tool_info = self._active_tools.pop(run_id, None)
        if not tool_info:
            return

        latency_ms = (time.perf_counter() - tool_info["start_time"]) * 1000.0
        tool_name = tool_info["name"]
        tool_input = tool_info["input"]
        err_msg = f"{type(error).__name__}: {error}"

        if parent_run_id and parent_run_id in self._active_steps:
            self._active_steps[parent_run_id].tool_calls.append(
                {
                    "name": tool_name,
                    "input": tool_input,
                    "error": err_msg,
                    "latency_ms": latency_ms,
                }
            )

        parent_span_id = str(parent_run_id) if parent_run_id else None
        span = TraceSpan(
            span_id=str(run_id),
            trace_id=self.trace.run_id,
            parent_span_id=parent_span_id,
            name=tool_name,
            span_type=SpanType.TOOL,
            latency_ms=latency_ms,
            status=StepStatus.ERROR,
            error=err_msg,
            input=_safe_serialize(tool_input),
            output="",
        )
        self.trace.spans.append(span)

    # ─────────────────────────────────────────────────────────────────────────
    # LLM Lifecycle Hooks
    # ─────────────────────────────────────────────────────────────────────────

    def on_llm_start(
        self,
        serialized: dict[str, Any] | None,
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        llm_name = (serialized or {}).get("name") or "llm"
        self._active_llms[run_id] = {
            "name": llm_name,
            "prompts": prompts,
            "start_time": time.perf_counter(),
            "parent_run_id": parent_run_id,
        }

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        llm_info = self._active_llms.pop(run_id, None)
        if not llm_info:
            return

        latency_ms = (time.perf_counter() - llm_info["start_time"]) * 1000.0
        llm_name = llm_info["name"]

        # Parse token usage from response.llm_output
        p_tok = 0
        c_tok = 0
        llm_out = getattr(response, "llm_output", None) or {}
        if isinstance(llm_out, dict):
            usage = llm_out.get("token_usage") or llm_out.get("usage") or {}
            p_tok = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            c_tok = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)

        # Check generations usage metadata if top-level usage not present
        if not p_tok and not c_tok:
            generations = getattr(response, "generations", [])
            for gen_list in generations:
                for gen in gen_list:
                    gen_info = getattr(gen, "generation_info", {}) or {}
                    usage_meta = gen_info.get("usage_metadata") or {}
                    p_tok += int(usage_meta.get("input_tokens") or 0)
                    c_tok += int(usage_meta.get("output_tokens") or 0)

        # Attribute tokens to active parent agent step
        if parent_run_id and parent_run_id in self._active_steps:
            parent_step = self._active_steps[parent_run_id]
            parent_step.tokens.prompt += p_tok
            parent_step.tokens.completion += c_tok
            parent_step.tokens.total += p_tok + c_tok

        # Record LLM span
        parent_span_id = str(parent_run_id) if parent_run_id else None
        span = TraceSpan(
            span_id=str(run_id),
            trace_id=self.trace.run_id,
            parent_span_id=parent_span_id,
            name=llm_name,
            span_type=SpanType.LLM,
            latency_ms=latency_ms,
            status=StepStatus.SUCCESS,
            tokens=TokenUsage(prompt=p_tok, completion=c_tok, total=p_tok + c_tok),
            input=_safe_serialize(llm_info.get("prompts", [])),
            output=_safe_serialize(getattr(response, "generations", [])),
        )
        self.trace.spans.append(span)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        llm_info = self._active_llms.pop(run_id, None)
        if not llm_info:
            return

        latency_ms = (time.perf_counter() - llm_info["start_time"]) * 1000.0
        llm_name = llm_info["name"]
        err_msg = f"{type(error).__name__}: {error}"

        parent_span_id = str(parent_run_id) if parent_run_id else None
        span = TraceSpan(
            span_id=str(run_id),
            trace_id=self.trace.run_id,
            parent_span_id=parent_span_id,
            name=llm_name,
            span_type=SpanType.LLM,
            latency_ms=latency_ms,
            status=StepStatus.ERROR,
            error=err_msg,
            input=_safe_serialize(llm_info.get("prompts", [])),
            output="",
        )
        self.trace.spans.append(span)
