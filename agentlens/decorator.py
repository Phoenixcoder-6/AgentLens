"""
agentlens/decorator.py — Framework-Agnostic @agentlens.trace_agent Decorator
=============================================================================
Day 57: Provides clean, framework-agnostic instrumentation for agent functions.

Supports:
  - Both @agentlens.trace_agent and @agentlens.trace_agent(role="...", name="...")
  - Synchronous (def) and asynchronous (async def) agent nodes
  - Automatic framework inspection (parameter binding, docstrings, function names)
  - Seamless state transformation and handoff capture via HandoffCapture
  - OpenInference TraceSpan DAG and legacy AgentStep generation
  - Token consumption from pending thread-local/coroutine-local slots
  - Failsafe execution: internal tracing errors never break user code
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar, cast, overload

from capture.handoff import HandoffCapture
from capture.session import CaptureSession
from schema.models import AgentStep, StepStatus, TokenUsage
from schema.span import SpanType, TraceSpan

logger = logging.getLogger("agentlens.sdk.decorator")

F = TypeVar("F", bound=Callable[..., Any])


def _safe_serialize(obj: Any) -> str:
    """Serialize an arbitrary Python object to JSON string with fallback."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, default=str)
    except Exception:
        return str(obj)


def _extract_input_state(
    func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[dict[str, Any], Any]:
    """Inspect arguments and extract state dictionary and full arguments dict."""
    try:
        sig = inspect.signature(func)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
    except Exception:
        arguments = {
            "args": [str(a) for a in args],
            "kwargs": {k: str(v) for k, v in kwargs.items()},
        }

    # Determine primary state dict (e.g. LangGraph / standard state dict)
    state_candidate = None
    if args and isinstance(args[0], dict):
        state_candidate = args[0]
    elif "state" in kwargs and isinstance(kwargs["state"], dict):
        state_candidate = kwargs["state"]
    elif len(arguments) == 1 and isinstance(next(iter(arguments.values())), dict):
        state_candidate = next(iter(arguments.values()))
    else:
        state_candidate = arguments

    return arguments, state_candidate


def _prepare_invocation(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    role: str | None,
    name: str | None,
) -> tuple[Any, HandoffCapture | None, AgentStep | None]:
    """Pre-execution step and span setup."""
    trace = CaptureSession.get_current_trace()
    if not trace:
        return None, None, None

    # Derive agent name
    if name:
        agent_name = name
    else:
        raw_name = func.__name__
        if raw_name.endswith("_node"):
            agent_name = raw_name[:-5]
        elif raw_name.endswith("_agent"):
            agent_name = raw_name[:-6]
        else:
            agent_name = raw_name

    step_idx = len(trace.steps) + 1
    arguments, state_candidate = _extract_input_state(func, args, kwargs)

    capture: HandoffCapture | None = None
    if isinstance(state_candidate, dict):
        capture = HandoffCapture(input_state=state_candidate)

    # Derive prompt / documentation context
    doc = inspect.cleandoc(func.__doc__) if func.__doc__ else ""
    prompt_str = f"Agent: {agent_name} | Role: {role or 'agent'}"
    if doc:
        prompt_str += f" | Doc: {doc.splitlines()[0]}"

    step = AgentStep(
        run_id=trace.run_id,
        step=step_idx,
        agent=agent_name,
        input=_safe_serialize(arguments),
        prompt=prompt_str,
    )

    return trace, capture, step


def _record_completion(
    trace: Any,
    capture: HandoffCapture | None,
    step: AgentStep,
    result: Any,
    latency_ms: float,
    role: str | None = None,
) -> None:
    """Record successful step and trace span."""
    try:
        agent_return = result if isinstance(result, dict) else {}
        if capture:
            capture.record_agent_return(agent_return)
            input_s, filtered_s, output_s, diff = capture.finalize()
            step.handoff.input_state = input_s
            step.handoff.filtered_state = filtered_s
            step.handoff.output_state = output_s
            if diff.summary():
                step.prompt = f"{step.prompt} | {diff.summary()}" if step.prompt else diff.summary()

        step.output = _safe_serialize(result)
        step.latency_ms = latency_ms
        step.status = StepStatus.SUCCESS

        # Consume staged token usage
        pending = CaptureSession.consume_pending_tokens()
        if pending:
            step.tokens = TokenUsage(
                prompt=pending[0],
                completion=pending[1],
                total=pending[0] + pending[1],
            )
        elif isinstance(result, dict) and "usage_metadata" in result:
            usage = result.get("usage_metadata") or {}
            p_tok = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
            c_tok = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
            if p_tok or c_tok:
                step.tokens = TokenUsage(prompt=p_tok, completion=c_tok, total=p_tok + c_tok)

        # Attach step to trace
        CaptureSession.add_step(step)

        # Create OpenInference TraceSpan for DAG alignment
        span_id = f"span_{trace.run_id}_{step.step}"
        parent_span_id = trace.spans[-1].span_id if trace.spans else None
        span = TraceSpan(
            span_id=span_id,
            trace_id=trace.run_id,
            parent_span_id=parent_span_id,
            name=step.agent,
            span_type=SpanType.AGENT,
            input=step.input,
            output=step.output,
            status=StepStatus.SUCCESS,
            attributes={"role": role} if role else {},
        )
        trace.spans.append(span)

    except Exception as exc:
        logger.warning("AgentLens: failed to record step success: %s", exc)


def _record_failure(
    trace: Any,
    capture: HandoffCapture | None,
    step: AgentStep,
    exc: Exception,
    latency_ms: float,
    role: str | None = None,
) -> None:
    """Record failed step and span on exception."""
    try:
        if capture:
            capture.record_agent_return({})
            input_s, filtered_s, output_s, diff = capture.finalize()
            step.handoff.input_state = input_s
            step.handoff.filtered_state = filtered_s
            step.handoff.output_state = output_s

        step.latency_ms = latency_ms
        step.status = StepStatus.ERROR
        step.error = f"{type(exc).__name__}: {exc}"
        CaptureSession.add_step(step)

        span_id = f"span_{trace.run_id}_{step.step}"
        parent_span_id = trace.spans[-1].span_id if trace.spans else None
        span = TraceSpan(
            span_id=span_id,
            trace_id=trace.run_id,
            parent_span_id=parent_span_id,
            name=step.agent,
            span_type=SpanType.AGENT,
            input=step.input,
            output="",
            status=StepStatus.ERROR,
            error=step.error,
            attributes={"role": role} if role else {},
        )
        trace.spans.append(span)

    except Exception as inner_exc:
        logger.warning("AgentLens: failed to record step failure: %s", inner_exc)


@overload
def trace_agent(func: F) -> F: ...


@overload
def trace_agent(
    *,
    role: str | None = None,
    name: str | None = None,
) -> Callable[[F], F]: ...


def trace_agent(
    func: F | None = None,
    *,
    role: str | None = None,
    name: str | None = None,
) -> Any:
    """
    Framework-agnostic decorator for agent functions and workflow nodes.

    Usage:
        @agentlens.trace_agent
        def my_agent(state: dict) -> dict:
            ...

        @agentlens.trace_agent(role="researcher", name="custom_name")
        async def fetch_papers(topic: str) -> dict:
            ...
    """

    def decorator(target_func: F) -> F:
        if inspect.iscoroutinefunction(target_func):

            @functools.wraps(target_func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                trace = None
                capture = None
                step = None
                try:
                    trace, capture, step = _prepare_invocation(
                        target_func, args, kwargs, role, name
                    )
                except Exception as prep_exc:
                    logger.warning("AgentLens: pre-invocation inspection failed: %s", prep_exc)

                if not trace or not step:
                    return await target_func(*args, **kwargs)

                start_t = time.perf_counter()
                try:
                    result = await target_func(*args, **kwargs)
                except Exception as exc:
                    latency = (time.perf_counter() - start_t) * 1000.0
                    _record_failure(trace, capture, step, exc, latency, role)
                    raise

                latency = (time.perf_counter() - start_t) * 1000.0
                _record_completion(trace, capture, step, result, latency, role)
                return result

            return cast(F, async_wrapper)

        @functools.wraps(target_func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            trace = None
            capture = None
            step = None
            try:
                trace, capture, step = _prepare_invocation(target_func, args, kwargs, role, name)
            except Exception as prep_exc:
                logger.warning("AgentLens: pre-invocation inspection failed: %s", prep_exc)

            if not trace or not step:
                return target_func(*args, **kwargs)

            start_t = time.perf_counter()
            try:
                result = target_func(*args, **kwargs)
            except Exception as exc:
                latency = (time.perf_counter() - start_t) * 1000.0
                _record_failure(trace, capture, step, exc, latency, role)
                raise

            latency = (time.perf_counter() - start_t) * 1000.0
            _record_completion(trace, capture, step, result, latency, role)
            return result

        return cast(F, sync_wrapper)

    if func is not None:
        return decorator(func)
    return decorator
