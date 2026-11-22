"""
scripts/verify_day47.py — Day 47 Verification Script
===================================================
Automated verification of ContextVar-based concurrency isolation in AgentLens v2.0:
1. Verify capture/context.py exports and clean initialization
2. Verify trace_context context manager scoping
3. Verify CaptureSession class methods route through ContextVar
4. Verify token staging and consumption isolation in concurrent contexts
5. Verify multithreaded context isolation across 20 OS threads
6. Verify asynchronous context isolation across 50 concurrent pipelines
7. Verify backward compatibility with CaptureSession metaclass properties
8. Verify zero regressions on Day 43 attribution benchmark
"""

import asyncio
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from capture.context import (
    clear_context,
    consume_pending_tokens,
    get_current_trace,
    get_pending_tokens,
    set_pending_tokens,
    trace_context,
)
from capture.session import CaptureSession
from capture.tracer import trace_step
from schema.models import AgentStep, RunTrace


def check_1_context_module() -> bool:
    print("\n--- Check 1: Context Module Exports & Defaults ---")
    clear_context()
    if get_current_trace() is not None:
        print("[FAIL] Initial current_trace is not None")
        return False
    if get_pending_tokens() is not None:
        print("[FAIL] Initial pending_tokens is not None")
        return False

    set_pending_tokens(50, 100)
    if get_pending_tokens() != (50, 100):
        print("[FAIL] set_pending_tokens failed to store tokens")
        return False

    consumed = consume_pending_tokens()
    if consumed != (50, 100):
        print(f"[FAIL] consume_pending_tokens returned {consumed}, expected (50, 100)")
        return False
    if get_pending_tokens() is not None:
        print("[FAIL] pending_tokens was not cleared after consumption")
        return False

    print("[PASS] Context module initialized and manages values correctly.")
    return True


def check_2_trace_context_manager() -> bool:
    print("\n--- Check 2: trace_context Manager Scoping ---")
    clear_context()
    dummy = RunTrace(run_id="ctx_scope_001", workflow="test_scope")

    with trace_context(dummy) as active:
        if active.run_id != "ctx_scope_001":
            print("[FAIL] Context manager did not yield trace")
            return False
        current = get_current_trace()
        if current is None or current.run_id != "ctx_scope_001":
            print("[FAIL] get_current_trace inside context does not match")
            return False

    if get_current_trace() is not None:
        print("[FAIL] Current trace was not restored to None after exiting context")
        return False

    print("[PASS] trace_context manager scopes and restores correctly.")
    return True


def check_3_session_delegation() -> bool:
    print("\n--- Check 3: CaptureSession Routing through ContextVar ---")
    clear_context()
    trace = CaptureSession.start_trace(workflow="delegation_test", run_id="del_001")
    if trace is None or trace.run_id != "del_001":
        print("[FAIL] CaptureSession.start_trace failed")
        return False

    current_trace = get_current_trace()
    if current_trace is None or current_trace.run_id != "del_001":
        print("[FAIL] Context did not receive trace from CaptureSession")
        return False

    step = AgentStep(run_id="del_001", step=1, agent="test_agent", input="{}")
    CaptureSession.add_step(step)

    current = CaptureSession.get_current_trace()
    if current is None:
        print("[FAIL] No current trace returned")
        return False

    if len(current.steps) != 1:
        print("[FAIL] Step was not appended to current context trace")
        return False

    # Prevent persistence during this unit check.

    with (
        patch.object(CaptureSession, "_save_trace_to_db", lambda t: None),
        patch.object(CaptureSession, "_save_trace_to_disk", lambda t: None),
    ):
        ended = CaptureSession.end_trace()

    if ended is None or ended.run_id != "del_001":
        print("[FAIL] CaptureSession.end_trace returned invalid trace")
        return False

    if get_current_trace() is not None:
        print("[FAIL] Context was not cleared after end_trace")
        return False
    print("[PASS] CaptureSession delegates trace lifecycle directly to ContextVar.")
    return True


def check_4_token_staging_isolation() -> bool:
    print("\n--- Check 4: Interleaved Token Staging Isolation ---")

    async def run_test():
        records = {}

        async def worker(w_id: int, p: int, c: int):
            CaptureSession.set_step_tokens(p, c)
            await asyncio.sleep(random.uniform(0.002, 0.008))
            records[w_id] = CaptureSession.consume_pending_tokens()

        await asyncio.gather(
            worker(1, 10, 20),
            worker(2, 30, 40),
            worker(3, 50, 60),
            worker(4, 70, 80),
        )

        return (
            records.get(1) == (10, 20)
            and records.get(2) == (30, 40)
            and records.get(3) == (50, 60)
            and records.get(4) == (70, 80)
        )

    ok = asyncio.run(run_test())
    if not ok:
        print("[FAIL] Staged tokens leaked between concurrent async routines")
        return False

    print("[PASS] Token staging is strictly isolated per execution context.")
    return True


def check_5_multithreaded_isolation() -> bool:
    print("\n--- Check 5: Multithreaded Isolation (20 Threads) ---")

    @trace_step
    def thread_step(state: dict) -> dict:
        time.sleep(random.uniform(0.001, 0.005))
        CaptureSession.set_step_tokens(state["tid"] * 10, state["tid"] * 20)
        return {"done": state["tid"]}

    def thread_worker(tid: int):
        run_id = f"thread_{tid:02d}"
        CaptureSession.start_trace("worker_wf", run_id=run_id)
        thread_step({"tid": tid})
        thread_step({"tid": tid})

        return CaptureSession.end_trace()

    with (
        patch.object(CaptureSession, "_save_trace_to_db", lambda t: None),
        patch.object(CaptureSession, "_save_trace_to_disk", lambda t: None),
        ThreadPoolExecutor(max_workers=20) as pool,
    ):
        futures = [pool.submit(thread_worker, i) for i in range(20)]
        traces = [f.result() for f in futures]

    if len(traces) != 20:
        print(f"[FAIL] Expected 20 traces, got {len(traces)}")
        return False

    for idx, trace in enumerate(traces):
        if trace is None:
            print(f"[FAIL] Missing trace for thread {idx}")
            return False
        if trace.run_id != f"thread_{idx:02d}":
            print(f"[FAIL] Run ID mismatch in thread {idx}: {trace.run_id}")
            return False
        if len(trace.steps) != 2:
            print(f"[FAIL] Step count mismatch in thread {idx}: {len(trace.steps)}")
            return False
        for step in trace.steps:
            if step.tokens.prompt != idx * 10 or step.tokens.completion != idx * 20:
                print(f"[FAIL] Token mismatch in thread {idx}: {step.tokens}")
                return False

    print("[PASS] 20 concurrent threads completed with zero trace/step collisions.")
    return True


def check_6_async_pipeline_isolation() -> bool:
    print("\n--- Check 6: Async Pipeline Isolation (50 Pipelines) ---")

    @trace_step
    async def async_node(state: dict) -> dict:
        await asyncio.sleep(random.uniform(0.001, 0.005))
        pid = state["pid"]
        CaptureSession.set_step_tokens(pid * 2, pid * 3)
        return {"result": pid}

    async def run_all():
        async def pipeline_worker(pid: int):
            run_id = f"pipeline_{pid:03d}"
            CaptureSession.start_trace("async_wf", run_id=run_id)
            await async_node({"pid": pid})
            await async_node({"pid": pid})

            return CaptureSession.end_trace()

        return await asyncio.gather(*[pipeline_worker(i) for i in range(50)])

    with (
        patch.object(CaptureSession, "_save_trace_to_db", lambda t: None),
        patch.object(CaptureSession, "_save_trace_to_disk", lambda t: None),
    ):
        traces = asyncio.run(run_all())

    if len(traces) != 50:
        print(f"[FAIL] Expected 50 traces, got {len(traces)}")
        return False

    for idx, trace in enumerate(traces):
        if trace is None:
            print(f"[FAIL] Missing trace for pipeline {idx}")
            return False
        if trace.run_id != f"pipeline_{idx:03d}":
            print(f"[FAIL] Run ID mismatch: {trace.run_id}")
            return False
        if len(trace.steps) != 2:
            print(f"[FAIL] Expected 2 steps in {trace.run_id}, got {len(trace.steps)}")
            return False
        for s in trace.steps:
            if s.tokens.prompt != idx * 2 or s.tokens.completion != idx * 3:
                print(f"[FAIL] Token leak detected in {trace.run_id}: {s.tokens}")
                return False

    print("[PASS] 50 concurrent async pipelines executed with zero cross-talk.")
    return True


def check_7_metaclass_backward_compatibility() -> bool:
    print("\n--- Check 7: Metaclass Backward Compatibility ---")
    clear_context()

    # Direct property read
    if CaptureSession._current_trace is not None:
        print("[FAIL] CaptureSession._current_trace is not None initially")
        return False

    # Direct property assignment
    sample = RunTrace(run_id="legacy_access", workflow="legacy_wf")
    CaptureSession._current_trace = sample

    if get_current_trace() != sample:
        print("[FAIL] Direct assignment did not update context")
        return False
    if CaptureSession._current_trace != sample:
        print("[FAIL] Direct read did not return assigned value")
        return False

    CaptureSession._pending_tokens = (12, 34)
    if get_pending_tokens() != (12, 34):
        print("[FAIL] Direct _pending_tokens assignment did not update context")
        return False

    tokens = CaptureSession.consume_pending_tokens()
    if tokens != (12, 34):
        print(f"[FAIL] consume_pending_tokens returned {tokens}, expected (12, 34)")
        return False

    CaptureSession._current_trace = None
    if get_current_trace() is not None:
        print("[FAIL] Direct None assignment did not clear context")
        return False

    print("[PASS] Metaclass guarantees 100% backward compatibility for legacy callers.")
    return True


def check_8_run_day43_regression() -> bool:
    print("\n--- Check 8: Day 43 Regression Benchmark ---")
    import subprocess

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

    # Verify 16/20 exact match (80%) and 0 regressions
    if "16/20 (80.0%)" not in res.stdout or "Overall Regression Gate" not in res.stdout:
        print("[FAIL] Benchmark output does not match expected format")
        print(res.stdout)
        return False

    print("[PASS] Zero regressions on Day 43 benchmark (16/20 exact matches preserved).")
    return True


def main():
    print("=" * 60)
    print("AgentLens v2.0 - Day 47 Verification Suite")
    print("=" * 60)

    checks = [
        check_1_context_module,
        check_2_trace_context_manager,
        check_3_session_delegation,
        check_4_token_staging_isolation,
        check_5_multithreaded_isolation,
        check_6_async_pipeline_isolation,
        check_7_metaclass_backward_compatibility,
        check_8_run_day43_regression,
    ]

    passed = 0
    for chk in checks:
        if chk():
            passed += 1
        else:
            print(f"\nVerification failed at {chk.__name__}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print(f"DAY 47 VERIFICATION SUCCESSFUL: {passed}/{len(checks)} CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
