"""
capture/tracer.py — @trace_step decorator
==========================================
Day 5: Records every agent call as an AgentStep in the active RunTrace.
Day 6: Properly computes the three-state handoff snapshot using HandoffCapture:
    - input_state    = full LangGraph state BEFORE agent
    - filtered_state = partial dict the agent chose to return
    - output_state   = full merged state AFTER LangGraph applies agent's return
"""

import time
import functools
import json
from datetime import datetime, timezone
from typing import Callable

from schema.models import AgentStep, StepStatus
from capture.session import CaptureSession
from capture.handoff import HandoffCapture


def trace_step(func: Callable) -> Callable:
    """
    Decorator that wraps a LangGraph agent node and records an AgentStep.

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
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # ── No active session → run the function unchanged ─────────────────
        trace = CaptureSession.get_current_trace()
        if not trace:
            return func(*args, **kwargs)

        # ── Metadata ────────────────────────────────────────────────────────
        agent_name = func.__name__.replace("_node", "")
        step_idx   = len(trace.steps) + 1

        # ── Snapshot full state BEFORE agent runs ────────────────────────
        state_in = args[0] if args else kwargs.get("state", {})
        capture  = HandoffCapture(input_state=state_in if isinstance(state_in, dict) else {})

        # ── Build the AgentStep (filled in below) ────────────────────────
        step = AgentStep(
            run_id=trace.run_id,
            step=step_idx,
            agent=agent_name,
            input=json.dumps(state_in, default=str),
            timestamp=datetime.now(timezone.utc),
        )

        start_time = time.perf_counter()

        try:
            # ── Run the actual agent ────────────────────────────────────
            result = func(*args, **kwargs)

            latency = (time.perf_counter() - start_time) * 1000.0

            # ── Record what the agent returned ──────────────────────────
            agent_return = result if isinstance(result, dict) else {}
            capture.record_agent_return(agent_return)

            # ── Finalize the three-state snapshot + diff ─────────────────
            input_s, filtered_s, output_s, diff = capture.finalize()

            # ── Populate the step ────────────────────────────────────────
            step.output              = json.dumps(result, default=str)
            step.latency_ms          = latency
            step.status              = StepStatus.SUCCESS
            step.handoff.input_state    = input_s
            step.handoff.filtered_state = filtered_s   # agent's own return dict
            step.handoff.output_state   = output_s     # full merged state

            # Store the diff summary in metadata for quick querying
            # (full diff object is recoverable by re-running the diff engine)
            step.prompt = diff.summary()   # re-using prompt field for diff log

            CaptureSession.add_step(step)
            return result

        except Exception as exc:
            latency = (time.perf_counter() - start_time) * 1000.0

            # Record partial capture even on error
            capture.record_agent_return({})
            input_s, filtered_s, output_s, diff = capture.finalize()

            step.latency_ms             = latency
            step.status                 = StepStatus.ERROR
            step.error                  = f"{type(exc).__name__}: {exc}"
            step.handoff.input_state    = input_s
            step.handoff.filtered_state = filtered_s
            step.handoff.output_state   = output_s
            step.prompt                 = diff.summary()

            CaptureSession.add_step(step)
            raise

    return wrapper
