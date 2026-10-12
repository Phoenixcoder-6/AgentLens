"""
tests/test_queue_telemetry.py — Day 48 Telemetry Ring Buffer & Worker Tests
============================================================================
Tests:
1. Bounded ring buffer queue behavior and dropped-span counter on overflow.
2. Background ingestion worker batch flushing triggered by size.
3. Background ingestion worker batch flushing triggered by interval timer.
4. Graceful worker shutdown and flush draining all pending traces.
5. CaptureSession integration with non-blocking ingestion.
6. Benchmark verifying @trace_step & end_trace queue overhead <= 0.5ms.
"""

from __future__ import annotations

import time

import pytest

from capture.context import clear_context
from capture.queue import (
    TraceIngestionQueue,
    TraceIngestionWorker,
    get_global_queue,
    get_global_worker,
    set_async_logging_enabled,
)
from capture.session import CaptureSession
from capture.tracer import trace_step
from schema.models import RunTrace


@pytest.fixture(autouse=True)
def setup_teardown():
    clear_context()
    set_async_logging_enabled(None)
    get_global_queue().clear()
    yield
    clear_context()
    set_async_logging_enabled(None)
    get_global_queue().clear()


def test_queue_bounded_and_drops_on_overflow():
    """Verify queue enforces max_size and records dropped traces on overflow."""
    q = TraceIngestionQueue(max_size=3)
    assert q.size() == 0
    assert q.stats()["dropped"] == 0

    t1 = RunTrace(run_id="run_1", workflow="test")
    t2 = RunTrace(run_id="run_2", workflow="test")
    t3 = RunTrace(run_id="run_3", workflow="test")
    t4 = RunTrace(run_id="run_4", workflow="test")
    t5 = RunTrace(run_id="run_5", workflow="test")

    # Enqueue within capacity
    assert q.enqueue(t1) is True
    assert q.enqueue(t2) is True
    assert q.enqueue(t3) is True
    assert q.size() == 3

    # Enqueue when full -> dropped
    assert q.enqueue(t4) is False
    assert q.enqueue(t5) is False
    assert q.size() == 3

    stats = q.stats()
    assert stats["enqueued"] == 3
    assert stats["dropped"] == 2

    # Dequeue batch
    batch = q.dequeue_batch(max_items=2, timeout=0.01)
    assert len(batch) == 2
    assert q.size() == 1
    assert q.stats()["processed"] == 2

    # Now there is room for 1 more
    assert q.enqueue(t4) is True
    assert q.size() == 2


def test_worker_batch_flushing_by_size():
    """Worker flushes immediately when batch_size is reached, without waiting for interval."""
    persisted: list[str] = []

    def mock_persist(trace: RunTrace):
        persisted.append(trace.run_id)

    q = TraceIngestionQueue(max_size=100)
    worker = TraceIngestionWorker(
        ingestion_queue=q,
        persist_fn=mock_persist,
        flush_interval=5.0,  # Long timeout
        batch_size=5,
    )
    worker.start()

    try:
        for i in range(5):
            q.enqueue(RunTrace(run_id=f"size_run_{i}", workflow="test"))

        # Wait shortly for the thread to process the full batch
        time.sleep(0.05)
        assert len(persisted) == 5
        assert persisted == [f"size_run_{i}" for i in range(5)]
    finally:
        worker.stop(timeout=1.0)


def test_worker_batch_flushing_by_interval():
    """Worker flushes smaller batches when flush_interval timer fires."""
    persisted: list[str] = []

    def mock_persist(trace: RunTrace):
        persisted.append(trace.run_id)

    q = TraceIngestionQueue(max_size=100)
    worker = TraceIngestionWorker(
        ingestion_queue=q,
        persist_fn=mock_persist,
        flush_interval=0.04,  # 40ms interval
        batch_size=50,  # Large batch size
    )
    worker.start()

    try:
        # Enqueue only 2 items (< batch_size 50)
        q.enqueue(RunTrace(run_id="timer_1", workflow="test"))
        q.enqueue(RunTrace(run_id="timer_2", workflow="test"))

        # Wait for timer trigger
        time.sleep(0.10)
        assert len(persisted) == 2
        assert persisted == ["timer_1", "timer_2"]
    finally:
        worker.stop(timeout=1.0)


def test_worker_graceful_shutdown_drains_remaining():
    """Worker.stop() drains all pending items in the queue before terminating."""
    persisted: list[str] = []

    def mock_persist(trace: RunTrace):
        persisted.append(trace.run_id)

    q = TraceIngestionQueue(max_size=100)
    # Don't start worker loop yet
    worker = TraceIngestionWorker(
        ingestion_queue=q,
        persist_fn=mock_persist,
        flush_interval=1.0,
        batch_size=10,
    )

    for i in range(15):
        q.enqueue(RunTrace(run_id=f"drain_{i}", workflow="test"))

    assert q.size() == 15
    worker.start()
    worker.stop(timeout=2.0)

    assert q.size() == 0
    assert len(persisted) == 15


def test_capture_session_nonblocking_queue_integration(monkeypatch):
    """CaptureSession.end_trace enqueues to background queue when async_logging is True."""
    set_async_logging_enabled(True)
    persisted_runs: list[str] = []

    # Mock the persist function on the global worker
    worker = get_global_worker()
    monkeypatch.setattr(worker, "persist_fn", lambda t: persisted_runs.append(t.run_id))

    CaptureSession.start_trace(workflow="integration_test", run_id="async_int_01")
    trace = CaptureSession.end_trace()

    assert trace is not None
    assert trace.run_id == "async_int_01"

    # Flush to ensure background worker processes it
    CaptureSession.flush(timeout=1.0)
    assert "async_int_01" in persisted_runs


def test_overhead_benchmark_under_half_millisecond(monkeypatch):
    """
    Benchmark: measure the time added by @trace_step and end_trace() with non-blocking queue.
    Assert average host execution path overhead is <= 0.5ms per step.
    """
    set_async_logging_enabled(True)
    # Mock worker persistence to prevent actual disk/DB writes during high-throughput loop
    worker = get_global_worker()
    monkeypatch.setattr(worker, "persist_fn", lambda t: None)

    @trace_step
    def bench_agent(state: dict) -> dict:
        return {"val": state["x"] + 1}

    iterations = 50
    total_interception_time = 0.0

    for i in range(iterations):
        t0 = time.perf_counter()
        CaptureSession.start_trace(workflow="bench", run_id=f"bench_{i}")
        res = bench_agent({"x": i})
        _ = CaptureSession.end_trace()
        t1 = time.perf_counter()

        assert res["val"] == i + 1
        total_interception_time += (t1 - t0)

    avg_overhead_ms = (total_interception_time / iterations) * 1000.0
    print(f"\n[Benchmark] Average interception & end_trace overhead: {avg_overhead_ms:.4f} ms")

    # Host execution path must take <= 0.5 ms on average with non-blocking telemetry
    assert avg_overhead_ms <= 0.5, f"Overhead {avg_overhead_ms:.4f}ms exceeded 0.5ms threshold!"
