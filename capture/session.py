"""
capture/session.py — Trace Session Lifecycle Management
=========================================================
Day 5: Initial single-run singleton implementation.
Day 47: Refactored to utilize `capture.context.ContextVar` backend for complete
async coroutine and multithreaded concurrency isolation, eliminating global
state collisions while retaining 100% backward-compatible class APIs.
"""

import os
import uuid
from datetime import UTC, datetime

from capture.context import (
    consume_pending_tokens,
    get_current_trace,
    get_pending_tokens,
    set_current_trace,
    set_pending_tokens,
)
from schema.models import AgentStep, RunTrace, StepStatus


class _CaptureSessionMeta(type):
    """
    Metaclass providing backward-compatible class property access to
    _current_trace and _pending_tokens mapped to contextvars.
    """

    @property
    def _current_trace(cls) -> RunTrace | None:
        return get_current_trace()

    @_current_trace.setter
    def _current_trace(cls, value: RunTrace | None) -> None:
        set_current_trace(value)

    @property
    def _pending_tokens(cls) -> tuple | None:
        return get_pending_tokens()

    @_pending_tokens.setter
    def _pending_tokens(cls, value: tuple | None) -> None:
        if value is None:
            consume_pending_tokens()
        else:
            set_pending_tokens(value[0], value[1])


class CaptureSession(metaclass=_CaptureSessionMeta):
    """
    Manages the active RunTrace for the current pipeline run.
    Uses contextvars for per-coroutine and per-thread isolation.
    """

    @classmethod
    def set_step_tokens(cls, prompt: int, completion: int) -> None:
        """
        Stage token counts for the step currently being traced in this context.

        Call this inside a pipeline node right after the LLM response:
            response = llm.invoke(messages)
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                CaptureSession.set_step_tokens(
                    prompt=response.usage_metadata.get('input_tokens', 0),
                    completion=response.usage_metadata.get('output_tokens', 0),
                )

        The @trace_step decorator reads and clears this slot when it builds
        the AgentStep — ensuring the correct step gets the correct tokens.
        """
        set_pending_tokens(prompt, completion)

    @classmethod
    def consume_pending_tokens(cls) -> tuple | None:
        """
        Read and clear the pending token slot for the active context.
        Called by @trace_step after the node function returns.
        Returns (prompt, completion) or None if no tokens were staged.
        """
        return consume_pending_tokens()

    @classmethod
    def start_trace(cls, workflow: str, run_id: str | None = None) -> RunTrace:
        """Start a new RunTrace bound to the calling execution context."""
        new_trace: RunTrace | None = None
        try:
            new_trace = RunTrace(
                run_id=run_id or f"run_{uuid.uuid4().hex[:8]}",
                workflow=workflow,
                timestamp=datetime.now(UTC),
                status=StepStatus.SUCCESS,
            )
            set_current_trace(new_trace)
        except Exception as exc:
            print(f"[CaptureSession] Warning: start_trace failed: {exc}")
            set_current_trace(None)
        return new_trace  # type: ignore[return-value]

    @classmethod
    def get_current_trace(cls) -> RunTrace | None:
        """Retrieve the active RunTrace for the current execution context."""
        return get_current_trace()

    @classmethod
    def add_step(cls, step: AgentStep) -> None:
        """Add an AgentStep to the active trace in the current execution context."""
        try:
            trace = get_current_trace()
            if trace:
                from capture.pii_scrubber import PIIScrubber

                PIIScrubber().scrub_step(step)
                trace.steps.append(step)
        except Exception as exc:
            print(f"[CaptureSession] Warning: add_step failed: {exc}")

    @classmethod
    def end_trace(
        cls, status: StepStatus = StepStatus.SUCCESS, error: str | None = None
    ) -> RunTrace | None:
        """Finalize and persist the active RunTrace in the current execution context."""
        trace = get_current_trace()
        if not trace:
            return None

        try:
            if status != StepStatus.SUCCESS:
                trace.status = status

            # Calculate aggregate metrics
            total_latency = 0.0
            total_tokens = 0
            for step in trace.steps:
                total_latency += step.latency_ms
                total_tokens += step.tokens.total
                if step.status in (StepStatus.FAILURE, StepStatus.ERROR):
                    trace.status = step.status
                    if step.error and not error:
                        error = step.error

            trace.total_latency_ms = total_latency
            trace.total_tokens = total_tokens

            # 0. Scrub PII if enabled in config
            try:
                from capture.pii_scrubber import PIIScrubber

                PIIScrubber().scrub_trace(trace)
            except Exception as scrub_exc:
                print(f"[CaptureSession] Warning: PII scrubbing failed: {scrub_exc}")

            # 1. Save full JSON trace to disk (fail-safe)
            cls._save_trace_to_disk(trace)

            # 2. Normalize and persist to SQLite (fail-safe)
            cls._save_trace_to_db(trace)
        except Exception as exc:
            print(f"[CaptureSession] Warning: end_trace failed: {exc}")
        finally:
            set_current_trace(None)

        return trace

    @classmethod
    def _save_trace_to_disk(cls, trace: RunTrace) -> None:
        try:
            from config.config_loader import get

            traces_dir = get("storage", "traces_dir", "data/traces") or "data/traces"
            os.makedirs(traces_dir, exist_ok=True)

            trace.trace_path = f"{traces_dir}/{trace.run_id}.json"

            with open(trace.trace_path, "w", encoding="utf-8") as f:
                f.write(trace.model_dump_json(indent=2))
        except Exception as exc:
            # Disk failure must never crash the pipeline
            print(f"[CaptureSession] Warning: disk trace save failed: {exc}")

    @classmethod
    def _save_trace_to_db(cls, trace: RunTrace) -> None:
        """Normalize the completed RunTrace and write to SQLite."""
        try:
            from normalizer.normalizer import Normalizer
            from storage.db import DatabaseManager
            from storage.writer import StorageWriter

            db = DatabaseManager()
            db.initialize()

            normalized = Normalizer().normalize_run(trace)

            # Read the JSON blob we just saved to disk, or serialize directly on fallback
            trace_json: str | None = None
            if trace.trace_path and os.path.exists(trace.trace_path):
                with open(trace.trace_path, encoding="utf-8") as f:
                    trace_json = f.read()
            else:
                trace_json = trace.model_dump_json(indent=2)

            writer = StorageWriter(db)
            writer.write_run(
                run=normalized,
                trace_json=trace_json,
                trace_path=trace.trace_path,
            )
        except Exception as exc:
            # Storage failure must never crash the pipeline
            print(f"[CaptureSession] Warning: storage write failed: {exc}")
