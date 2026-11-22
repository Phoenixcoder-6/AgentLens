"""
capture/tracer.py — @trace_step decorator
==========================================
Day 5: Records every agent call as an AgentStep in the active RunTrace.
Day 6: Properly computes the three-state handoff snapshot using HandoffCapture:
    - input_state    = full LangGraph state BEFORE agent
    - filtered_state = partial dict the agent chose to return
    - output_state   = full merged state AFTER LangGraph applies agent's return
Day 47: Added support for both synchronous and asynchronous (async def) agent nodes
    with contextvars trace isolation.
"""

import functools
import inspect
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime

from capture.handoff import HandoffCapture
from capture.session import CaptureSession
from schema.models import AgentStep, StepStatus


def _prepare_step(func: Callable, args: tuple, kwargs: dict) -> tuple:
    """Pre-execution capture setup common to sync and async runners."""
    trace = CaptureSession.get_current_trace()
    if not trace:
        return None, None, None

    agent_name = func.__name__.replace("_node", "")
    step_idx = len(trace.steps) + 1

    state_in = args[0] if args else kwargs.get("state", {})
    capture = HandoffCapture(input_state=state_in if isinstance(state_in, dict) else {})

    step = AgentStep(
        run_id=trace.run_id,
        step=step_idx,
        agent=agent_name,
        input=json.dumps(state_in, default=str),
        timestamp=datetime.now(UTC),
    )
    return trace, capture, step


def _record_error(capture, step, exc: Exception, latency: float) -> None:
    """Record an error step on agent failure."""
    try:
        if capture is not None and step is not None:
            capture.record_agent_return({})
            input_s, filtered_s, output_s, diff = capture.finalize()

            step.latency_ms = latency
            step.status = StepStatus.ERROR
            step.error = f"{type(exc).__name__}: {exc}"
            step.handoff.input_state = input_s
            step.handoff.filtered_state = filtered_s
            step.handoff.output_state = output_s
            step.prompt = diff.summary()

            CaptureSession.add_step(step)
    except Exception as cap_exc:
        print(f"[trace_step] Warning: error-path capture failed: {cap_exc}")


def _record_success(capture, step, result, latency: float) -> None:
    """Record a successful step with staged tokens and state handoff."""
    try:
        if capture is not None and step is not None:
            agent_return = result if isinstance(result, dict) else {}
            capture.record_agent_return(agent_return)

            input_s, filtered_s, output_s, diff = capture.finalize()

            step.output = json.dumps(result, default=str)
            step.latency_ms = latency
            step.status = StepStatus.SUCCESS
            step.handoff.input_state = input_s
            step.handoff.filtered_state = filtered_s
            step.handoff.output_state = output_s

            pending = CaptureSession.consume_pending_tokens()
            if pending:
                from schema.models import TokenUsage

                step.tokens = TokenUsage(
                    prompt=pending[0],
                    completion=pending[1],
                    total=pending[0] + pending[1],
                )

            step.prompt = diff.summary()
            CaptureSession.add_step(step)
    except Exception as cap_exc:
        print(f"[trace_step] Warning: post-execution capture failed: {cap_exc}")


def trace_step(func: Callable) -> Callable:
    """
    Decorator that wraps an agent node (sync or async) and records an AgentStep.

    Captures:
        - Agent name (derived from function name, "_node" suffix stripped)
        - Wall-clock latency in milliseconds
        - Full input state BEFORE agent runs (handoff.input_state)
        - Partial dict agent returned (handoff.filtered_state)
        - Full merged output state AFTER agent runs (handoff.output_state)
        - Status: SUCCESS or ERROR
        - Error message on exception (step is still recorded — never lost)

    No-op when there is no active CaptureSession (e.g. during unit tests
    that don't start a trace). The wrapped function always returns normally.
    """
    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            trace = None
            capture = None
            step = None
            try:
                trace, capture, step = _prepare_step(func, args, kwargs)
            except Exception as cap_exc:
                print(f"[trace_step] Warning: pre-execution capture failed: {cap_exc}")
                trace = None

            if not trace:
                return await func(*args, **kwargs)

            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
            except Exception as exc:
                latency = (time.perf_counter() - start_time) * 1000.0
                _record_error(capture, step, exc, latency)
                raise

            latency = (time.perf_counter() - start_time) * 1000.0
            _record_success(capture, step, result, latency)
            return result

        return async_wrapper

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        trace = None
        capture = None
        step = None
        try:
            trace, capture, step = _prepare_step(func, args, kwargs)
        except Exception as cap_exc:
            print(f"[trace_step] Warning: pre-execution capture failed: {cap_exc}")
            trace = None

        if not trace:
            return func(*args, **kwargs)

        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        except Exception as exc:
            latency = (time.perf_counter() - start_time) * 1000.0
            _record_error(capture, step, exc, latency)
            raise

        latency = (time.perf_counter() - start_time) * 1000.0
        _record_success(capture, step, result, latency)
        return result

    return sync_wrapper
