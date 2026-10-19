"""
agentlens/context_manager.py — Trace Run Context Manager
=========================================================
Day 57: Provides the `trace_run` context manager supporting both synchronous
(`with agentlens.trace_run(...)`) and asynchronous (`async with agentlens.trace_run(...)`)
execution lifecycles.
"""

from __future__ import annotations

import json
from types import TracebackType
from typing import Any

from capture.session import CaptureSession
from schema.models import RunTrace, StepStatus


class TraceRunContext:
    """
    Dual sync/async context manager for managing a RunTrace lifecycle.

    Usage:
        with agentlens.trace_run("my_workflow") as trace:
            step1()
            step2()

        async with agentlens.trace_run("async_workflow") as trace:
            await async_step1()
    """

    def __init__(
        self,
        workflow: str = "default_workflow",
        run_id: str | None = None,
        expected_output: Any = None,
        sync: bool | None = None,
    ) -> None:
        self.workflow = workflow
        self.run_id = run_id
        self.expected_output = expected_output
        self.sync = sync
        self.trace: RunTrace | None = None

    def __enter__(self) -> RunTrace:
        self.trace = CaptureSession.start_trace(workflow=self.workflow, run_id=self.run_id)
        if self.expected_output is not None and self.trace:
            if isinstance(self.expected_output, (dict, list)):
                self.trace.expected_output = json.dumps(self.expected_output)
            else:
                self.trace.expected_output = str(self.expected_output)
        return self.trace

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if not self.trace:
            return

        status = StepStatus.SUCCESS
        error_msg: str | None = None
        if exc_type is not None:
            status = StepStatus.ERROR
            error_msg = f"{exc_type.__name__}: {exc_val}"

        CaptureSession.end_trace(status=status, error=error_msg, sync=self.sync)

    async def __aenter__(self) -> RunTrace:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)
