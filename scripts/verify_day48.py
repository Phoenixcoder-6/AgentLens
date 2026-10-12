"""
scripts/verify_day48.py — Day 48 Verification Suite
===================================================
Automated verification of non-blocking async telemetry ring buffer and background
ingestion worker in AgentLens v2.0:
1. Verify TraceIngestionQueue bounded capacity & statistics initialization
2. Verify non-blocking enqueue and dropped-trace accounting on buffer overflow
3. Verify TraceIngestionWorker batch flushing triggered by size
4. Verify TraceIngestionWorker batch flushing triggered by timer interval (<=100ms)
5. Verify graceful worker shutdown and complete draining of pending traces
6. Verify CaptureSession non-blocking end_trace integration & queue flush
7. Verify host pipeline interception overhead benchmark (<= 0.5ms)
8. Verify zero regressions on Day 43 benchmark (16/20 exact attribution)
"""

from __future__ import annotations

import subprocess
import sys
import time

from capture.context import clear_context
from capture.queue import (
    TraceIngestionQueue,
    TraceIngestionWorker,
    get_global_worker,
    set_async_logging_enabled,
)
from capture.session import CaptureSession
from capture.tracer import trace_step
from schema.models import RunTrace


def check_1_queue_initialization() -> bool:
    print("\n--- Check 1: TraceIngestionQueue Initialization & Metrics ---")
    q = TraceIngestionQueue(max_size=15)
    stats = q.stats()
    if stats["max_size"] != 15 or stats["size"] != 0 or stats["dropped"] != 0:
        print(f"[FAIL] Unexpected initial queue metrics: {stats}")
        return False
    print("[PASS] Queue correctly initializes with bounded capacity and zeroed counters.")
    return True


def check_2_nonblocking_and_overflow() -> bool:
    print("\n--- Check 2: Non-Blocking Enqueue & Dropped Trace Accounting ---")
    q = TraceIngestionQueue(max_size=3)
    traces = [RunTrace(run_id=f"ovf_{i}", workflow="test") for i in range(5)]

    for i in range(3):
        if not q.enqueue(traces[i]):
            print(f"[FAIL] Enqueue failed prematurely for item {i}")
            return False

    # 4th and 5th items must be dropped without throwing
    if q.enqueue(traces[3]):
        print("[FAIL] Queue accepted item beyond capacity")
        return False
    if q.enqueue(traces[4]):
        print("[FAIL] Queue accepted item beyond capacity")
        return False

    stats = q.stats()
    if stats["dropped"] != 2 or stats["enqueued"] != 3:
        print(f"[FAIL] Dropped count mismatch: {stats}")
        return False

    print("[PASS] Non-blocking buffer correctly drops excess traces and increments dropped counter.")
    return True


def check_3_worker_flush_by_size() -> bool:
    print("\n--- Check 3: TraceIngestionWorker Batch Flush by Size ---")
    persisted = []
    q = TraceIngestionQueue(max_size=50)
    worker = TraceIngestionWorker(
        ingestion_queue=q,
        persist_fn=lambda t: persisted.append(t.run_id),
        flush_interval=5.0,  # Long interval to ensure flush is triggered by size
        batch_size=5,
    )
    worker.start()
    try:
        for i in range(5):
            q.enqueue(RunTrace(run_id=f"size_batch_{i}", workflow="test"))

        time.sleep(0.06)
        if len(persisted) != 5:
            print(f"[FAIL] Expected 5 items flushed by batch size, got {len(persisted)}")
            return False
    finally:
        worker.stop(timeout=1.0)

    print("[PASS] Worker flushes batch immediately upon reaching batch_size threshold.")
    return True


def check_4_worker_flush_by_interval() -> bool:
    print("\n--- Check 4: TraceIngestionWorker Batch Flush by Interval ---")
    persisted = []
    q = TraceIngestionQueue(max_size=50)
    worker = TraceIngestionWorker(
        ingestion_queue=q,
        persist_fn=lambda t: persisted.append(t.run_id),
        flush_interval=0.05,  # 50ms interval
        batch_size=50,  # High batch threshold
    )
    worker.start()
    try:
        q.enqueue(RunTrace(run_id="interval_trace_01", workflow="test"))
        q.enqueue(RunTrace(run_id="interval_trace_02", workflow="test"))

        time.sleep(0.12)
        if len(persisted) != 2:
            print(f"[FAIL] Expected 2 items flushed by interval timer, got {len(persisted)}")
            return False
    finally:
        worker.stop(timeout=1.0)

    print("[PASS] Worker flushes pending traces on timer interval (<= 100ms).")
    return True


def check_5_graceful_shutdown_drain() -> bool:
    print("\n--- Check 5: Graceful Worker Shutdown & Queue Drain ---")
    persisted = []
    q = TraceIngestionQueue(max_size=50)
    worker = TraceIngestionWorker(
        ingestion_queue=q,
        persist_fn=lambda t: persisted.append(t.run_id),
        flush_interval=1.0,
        batch_size=10,
    )

    for i in range(12):
        q.enqueue(RunTrace(run_id=f"drain_{i}", workflow="test"))

    worker.start()
    worker.stop(timeout=2.0)

    if q.size() != 0:
        print(f"[FAIL] Queue not empty after stop: {q.size()} remaining")
        return False
    if len(persisted) != 12:
        print(f"[FAIL] Expected 12 items persisted on drain, got {len(persisted)}")
        return False

    print("[PASS] Worker stop successfully drained and persisted all pending traces.")
    return True


def check_6_capture_session_integration() -> bool:
    print("\n--- Check 6: CaptureSession Async Ingestion Integration ---")
    clear_context()
    set_async_logging_enabled(True)
    persisted = []

    worker = get_global_worker()
    orig_persist = worker.persist_fn
    worker.persist_fn = lambda t: persisted.append(t.run_id)

    try:
        CaptureSession.start_trace(workflow="integration_test", run_id="v2_int_99")

        @trace_step
        def sample_node(state: dict) -> dict:
            return {"ok": True}

        sample_node({})
        trace = CaptureSession.end_trace()
        if trace is None or trace.run_id != "v2_int_99":
            print("[FAIL] CaptureSession.end_trace returned invalid trace")
            return False

        CaptureSession.flush(timeout=1.0)
        if "v2_int_99" not in persisted:
            print("[FAIL] Trace was not processed by background worker upon flush")
            return False
    finally:
        worker.persist_fn = orig_persist
        set_async_logging_enabled(None)
        clear_context()

    print("[PASS] CaptureSession seamlessly dispatches to background queue and flushes cleanly.")
    return True


def check_7_interception_overhead_benchmark() -> bool:
    print("\n--- Check 7: Interception Overhead Benchmark (<= 0.5ms) ---")
    clear_context()
    set_async_logging_enabled(True)

    worker = get_global_worker()
    orig_persist = worker.persist_fn
    worker.persist_fn = lambda t: None  # No-op during fast loop

    @trace_step
    def bench_node(state: dict) -> dict:
        return {"val": state["i"]}

    iterations = 50
    total_time = 0.0

    try:
        for i in range(iterations):
            t0 = time.perf_counter()
            CaptureSession.start_trace(workflow="bench", run_id=f"bench_{i:03d}")
            bench_node({"i": i})
            CaptureSession.end_trace()
            t1 = time.perf_counter()
            total_time += (t1 - t0)

        avg_ms = (total_time / iterations) * 1000.0
        print(f"[INFO] Measured average interception + queuing latency: {avg_ms:.4f} ms")

        if avg_ms > 0.5:
            print(f"[FAIL] Interception latency {avg_ms:.4f}ms exceeded 0.5ms threshold")
            return False
    finally:
        worker.persist_fn = orig_persist
        set_async_logging_enabled(None)
        clear_context()

    print(f"[PASS] Interception overhead is {avg_ms:.4f} ms (well within <= 0.5ms gate).")
    return True


def check_8_regression_suite() -> bool:
    print("\n--- Check 8: Day 43 Regression Benchmark ---")
    cmd = [
        sys.executable,
        "scripts/run_day43_regression.py",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[FAIL] Regression benchmark failed with returncode {res.returncode}")
        print(res.stdout)
        print(res.stderr)
        return False

    if "16/20 (80.0%)" not in res.stdout or "Overall Regression Gate" not in res.stdout:
        print("[FAIL] Benchmark output does not match expected format")
        print(res.stdout)
        return False

    print("[PASS] Zero regressions on Day 43 benchmark (16/20 exact matches preserved).")
    return True


def main():
    print("=" * 60)
    print("AgentLens v2.0 - Day 48 Verification Suite")
    print("=" * 60)

    checks = [
        check_1_queue_initialization,
        check_2_nonblocking_and_overflow,
        check_3_worker_flush_by_size,
        check_4_worker_flush_by_interval,
        check_5_graceful_shutdown_drain,
        check_6_capture_session_integration,
        check_7_interception_overhead_benchmark,
        check_8_regression_suite,
    ]

    passed = 0
    for chk in checks:
        if chk():
            passed += 1
        else:
            print(f"\nVerification failed at {chk.__name__}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print(f"DAY 48 VERIFICATION SUCCESSFUL: {passed}/{len(checks)} CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
