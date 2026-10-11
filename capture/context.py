"""
capture/context.py — Concurrency-Isolated Execution Context
============================================================
Day 47: Replaces global process-level state with Python `contextvars.ContextVar`.
Provides thread-local and coroutine-local isolation for active RunTraces and
pending step token usage.

Enables safe concurrent execution of multiple agent pipelines in:
- Async environments (FastAPI, asyncio.gather, ASGI)
- Multithreaded environments (ThreadPoolExecutor, background workers)
"""

import contextvars
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from schema.models import RunTrace

# Context variables with default None
_current_trace_var: contextvars.ContextVar[RunTrace | None] = contextvars.ContextVar(
    "agentlens_current_trace", default=None
)
_pending_tokens_var: contextvars.ContextVar[tuple[int, int] | None] = contextvars.ContextVar(
    "agentlens_pending_tokens", default=None
)


def get_current_trace() -> RunTrace | None:
    """Retrieve the active RunTrace for the current execution context."""
    return _current_trace_var.get()


def set_current_trace(trace: RunTrace | None) -> contextvars.Token[RunTrace | None]:
    """
    Set the active RunTrace for the current execution context.
    Returns a Token that can be used to restore the previous context.
    """
    return _current_trace_var.set(trace)


def reset_current_trace(token: contextvars.Token[RunTrace | None]) -> None:
    """Reset the active RunTrace to its prior state using the given token."""
    _current_trace_var.reset(token)


def get_pending_tokens() -> tuple[int, int] | None:
    """Retrieve staged (prompt, completion) tokens for the current execution context."""
    return _pending_tokens_var.get()


def set_pending_tokens(prompt: int, completion: int) -> contextvars.Token[tuple[int, int] | None]:
    """Stage token counts for the step currently being executed in this context."""
    return _pending_tokens_var.set((prompt, completion))


def consume_pending_tokens() -> tuple[int, int] | None:
    """
    Read and clear staged tokens in the current execution context.
    Returns (prompt, completion) or None if no tokens were staged.
    """
    tokens = _pending_tokens_var.get()
    _pending_tokens_var.set(None)
    return tokens


def reset_pending_tokens(token: contextvars.Token[tuple[int, int] | None]) -> None:
    """Reset pending tokens to prior state using the given token."""
    _pending_tokens_var.reset(token)


def clear_context() -> None:
    """Clear both active trace and pending tokens in the current execution context."""
    _current_trace_var.set(None)
    _pending_tokens_var.set(None)


@contextmanager
def trace_context(trace: RunTrace):
    """
    Context manager to scope a RunTrace to a specific code block.

    Example:
        with trace_context(my_trace):
            run_pipeline()
    """
    trace_tok = set_current_trace(trace)
    try:
        yield trace
    finally:
        reset_current_trace(trace_tok)


def run_with_context(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """
    Execute a callable within a copy of the current context.
    Useful for offloading work to a ThreadPoolExecutor while preserving trace context:
        future = executor.submit(run_with_context, worker_func, arg1)
    """
    ctx = contextvars.copy_context()
    return ctx.run(func, *args, **kwargs)
