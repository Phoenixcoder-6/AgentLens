"""
tests/test_concurrency_isolation.py — Day 47 Concurrency Isolation Tests
========================================================================
Validates that ContextVar-backed CaptureSession eliminates state leakage across:
1. Highly concurrent async coroutines (asyncio.gather, 50 parallel pipelines)
2. Concurrent OS threads (ThreadPoolExecutor, 20 parallel threads)
3. Token staging and consumption isolation across interleaved execution
4. Backward compatibility of CaptureSession class properties
"""

import asyncio
import random
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from capture.context import clear_context, get_current_trace, trace_context
from capture.session import CaptureSession
from capture.tracer import trace_step
from schema.models import RunTrace


@pytest.fixture(autouse=True)
def reset_context():
    """Ensure clean execution context before and after each test."""
    clear_context()
    yield
    clear_context()


# ── Sample Traced Nodes ──────────────────────────────────────────────────────


@trace_step
async def async_agent_a_node(state: dict) -> dict:
    await asyncio.sleep(random.uniform(0.001, 0.01))
    token_seed = state.get("token_seed", 10)
    CaptureSession.set_step_tokens(prompt=token_seed, completion=token_seed * 2)
    return {"step_a": "done", "pipeline_id": state.get("pipeline_id")}


@trace_step
async def async_agent_b_node(state: dict) -> dict:
    await asyncio.sleep(random.uniform(0.001, 0.01))
    token_seed = state.get("token_seed", 10)
    CaptureSession.set_step_tokens(prompt=token_seed * 3, completion=token_seed * 4)
    return {"step_b": "done", "pipeline_id": state.get("pipeline_id")}


@trace_step
def sync_agent_node(state: dict) -> dict:
    time.sleep(random.uniform(0.001, 0.008))
    seed = state.get("thread_id", 1)
    CaptureSession.set_step_tokens(prompt=seed * 5, completion=seed * 10)
    return {"thread_result": f"thread_{seed}"}


# ── Async Concurrency Tests ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_async_50_parallel_pipelines_isolation(monkeypatch, tmp_path):
    """
    Simulate 50 parallel async pipelines running concurrently via asyncio.gather.
    Verify that traces, steps, and staged tokens do not cross-contaminate.
    """
    monkeypatch.setenv("AGENTLENS_TRACES_DIR", str(tmp_path))

    async def run_single_pipeline(idx: int) -> RunTrace:
        run_id = f"async_run_{idx:03d}"
        token_seed = (idx + 1) * 100
        state = {"pipeline_id": idx, "token_seed": token_seed}

        trace = CaptureSession.start_trace(workflow="parallel_async_test", run_id=run_id)
        assert trace is not None
        assert trace.run_id == run_id

        # Execute node A
        res_a = await async_agent_a_node(state)
        assert res_a["pipeline_id"] == idx

        # Interleave jitter
        await asyncio.sleep(random.uniform(0.001, 0.005))

        # Execute node B
        res_b = await async_agent_b_node(state)
        assert res_b["pipeline_id"] == idx

        # Finalize trace without DB write
        monkeypatch.setattr(CaptureSession, "_save_trace_to_db", classmethod(lambda cls, t: None))
        final_trace = CaptureSession.end_trace()
        assert final_trace is not None
        return final_trace

    # Launch 50 pipelines concurrently
    results = await asyncio.gather(*[run_single_pipeline(i) for i in range(50)])

    assert len(results) == 50
    seen_run_ids = set()

    for idx, trace in enumerate(results):
        expected_run_id = f"async_run_{idx:03d}"
        token_seed = (idx + 1) * 100

        # Run ID uniqueness and integrity
        assert trace.run_id == expected_run_id
        assert trace.run_id not in seen_run_ids
        seen_run_ids.add(trace.run_id)

        # Step count & ownership
        assert len(trace.steps) == 2, f"Pipeline {idx} has {len(trace.steps)} steps (expected 2)"
        for step in trace.steps:
            assert step.run_id == expected_run_id

        # Step 1 token check
        step_1 = trace.steps[0]
        assert step_1.agent == "async_agent_a"
        assert step_1.tokens.prompt == token_seed
        assert step_1.tokens.completion == token_seed * 2
        assert step_1.tokens.total == token_seed * 3

        # Step 2 token check
        step_2 = trace.steps[1]
        assert step_2.agent == "async_agent_b"
        assert step_2.tokens.prompt == token_seed * 3
        assert step_2.tokens.completion == token_seed * 4
        assert step_2.tokens.total == token_seed * 7

        # Aggregate tokens
        assert trace.total_tokens == (token_seed * 3 + token_seed * 7)

    # After all coroutines finish, active trace must be None
    assert CaptureSession.get_current_trace() is None


# ── Multithreaded Concurrency Tests ──────────────────────────────────────────


def test_multithreading_20_parallel_threads_isolation(monkeypatch, tmp_path):
    """
    Simulate 20 parallel OS threads executing pipelines simultaneously in a ThreadPoolExecutor.
    Verify context isolation across OS threads.
    """
    monkeypatch.setenv("AGENTLENS_TRACES_DIR", str(tmp_path))

    def run_worker_thread(thread_idx: int) -> RunTrace:
        run_id = f"thread_run_{thread_idx:02d}"
        state = {"thread_id": thread_idx + 1}

        trace = CaptureSession.start_trace(workflow="parallel_thread_test", run_id=run_id)
        assert trace is not None

        # Execute sync node twice
        sync_agent_node(state)
        time.sleep(random.uniform(0.001, 0.005))
        sync_agent_node(state)

        monkeypatch.setattr(CaptureSession, "_save_trace_to_db", classmethod(lambda cls, t: None))
        final_trace = CaptureSession.end_trace()
        assert final_trace is not None
        return final_trace

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(run_worker_thread, i) for i in range(20)]
        traces = [f.result() for f in futures]

    assert len(traces) == 20
    seen_run_ids = set()

    for idx, trace in enumerate(traces):
        expected_run_id = f"thread_run_{idx:02d}"
        seed = idx + 1

        assert trace.run_id == expected_run_id
        assert trace.run_id not in seen_run_ids
        seen_run_ids.add(trace.run_id)

        assert len(trace.steps) == 2
        for step in trace.steps:
            assert step.run_id == expected_run_id
            assert step.tokens.prompt == seed * 5
            assert step.tokens.completion == seed * 10

        assert trace.total_tokens == (seed * 15) * 2

    assert CaptureSession.get_current_trace() is None


# ── Interleaved Token Staging Tests ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_interleaved_token_staging_no_cross_leak():
    """
    Test that two asynchronous routines staging tokens in interleaved steps
    never consume or overwrite each other's tokens.
    """
    tokens_captured = {}

    async def coroutine_x():
        CaptureSession.set_step_tokens(111, 222)
        await asyncio.sleep(0.005)  # Let Y run
        tokens_captured["x"] = CaptureSession.consume_pending_tokens()

    async def coroutine_y():
        await asyncio.sleep(0.001)  # Let X stage first
        CaptureSession.set_step_tokens(333, 444)
        await asyncio.sleep(0.010)
        tokens_captured["y"] = CaptureSession.consume_pending_tokens()

    await asyncio.gather(coroutine_x(), coroutine_y())

    assert tokens_captured["x"] == (111, 222)
    assert tokens_captured["y"] == (333, 444)


# ── Trace Scoping & Metaclass Backward Compatibility ────────────────────────


def test_trace_context_manager():
    """Test trace_context manager scopes trace correctly."""
    assert get_current_trace() is None

    mock_trace = RunTrace(run_id="ctx_test_123", workflow="test_ctx")
    with trace_context(mock_trace) as t:
        assert t.run_id == "ctx_test_123"
        assert get_current_trace() is not None
        assert get_current_trace().run_id == "ctx_test_123"

    assert get_current_trace() is None


def test_capture_session_metaclass_backward_compatibility():
    """
    Verify legacy code accessing `CaptureSession._current_trace` directly
    reads and writes through to contextvars without breaking.
    """
    assert CaptureSession._current_trace is None

    test_trace = RunTrace(run_id="meta_run_01", workflow="meta_test")
    CaptureSession._current_trace = test_trace

    assert CaptureSession.get_current_trace() is not None
    assert CaptureSession.get_current_trace().run_id == "meta_run_01"
    assert CaptureSession._current_trace.run_id == "meta_run_01"

    # Test pending tokens backward compatibility
    CaptureSession._pending_tokens = (42, 84)
    assert CaptureSession.consume_pending_tokens() == (42, 84)
    assert CaptureSession._pending_tokens is None

    # Reset
    CaptureSession._current_trace = None
    assert CaptureSession.get_current_trace() is None
