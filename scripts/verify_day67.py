"""
scripts/verify_day67.py — Day 67 Ingestion Queue Durability & Backpressure Verification Gate
=============================================================================================
Phase E: Data Durability, Storage Reliability & SRE

Checks:
  1. Disk WAL Creation & Append-Only Serialization.
  2. Crash Recovery & Replay from Local Disk WAL.
  3. Memory Queue Overflow Disk Spillover Activation (Zero-Drop Under Load).
  4. Saturation Detection & High-Watermark Threshold Accounting.
  5. API Backpressure Signaling (HTTP 429 + Retry-After + Saturated Headers).
  6. WAL Chunk File Rotation & Pruning Post-Persistence.
  7. Ingestion Queue Overhead Benchmark (<= 0.8ms per trace).
  8. Static Analysis (Ruff & Mypy 100% Clean) & Unit Test Suite Execution.
"""

# ruff: noqa: E402

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.router import router
from capture.queue import (
    TraceIngestionQueue,
    WALDiskSpooler,
    get_global_queue,
)
from schema.models import RunTrace


def run_checks() -> bool:
    print("=" * 76)
    print("  AGENTLENS v2.1+ — DAY 67 INGESTION DURABILITY & BACKPRESSURE GATE")
    print("  Phase E: Data Durability, Storage Reliability & SRE Hardening")
    print("=" * 76)
    all_passed = True

    temp_dir = Path(tempfile.mkdtemp(prefix="agentlens_verify_day67_"))

    try:
        # ─────────────────────────────────────────────────────────────────────
        # Check 1: Disk WAL Creation & Append-Only Serialization
        # ─────────────────────────────────────────────────────────────────────
        try:
            spool_1 = temp_dir / "spool_1"
            q1 = TraceIngestionQueue(max_size=50, spool_dir=spool_1, wal_enabled=True)
            t1 = RunTrace(run_id="chk1_run_01", workflow="verify_pipeline")
            t2 = RunTrace(run_id="chk1_run_02", workflow="verify_pipeline")

            assert q1.enqueue(t1) is True
            assert q1.enqueue(t2) is True
            assert q1.size() == 2

            wal_files = list(spool_1.glob("wal_*.jsonl"))
            assert len(wal_files) == 1, f"Expected 1 WAL file, found {len(wal_files)}"
            content = wal_files[0].read_text(encoding="utf-8")
            assert "chk1_run_01" in content and "chk1_run_02" in content
            q1.clear()
            print("[PASS] Check 1: Disk WAL Creation & Append-Only Serialization")
        except Exception as exc:
            print(f"[FAIL] Check 1: Disk WAL error: {exc}")
            all_passed = False

        # ─────────────────────────────────────────────────────────────────────
        # Check 2: Crash Recovery & Replay from Local Disk WAL
        # ─────────────────────────────────────────────────────────────────────
        try:
            spool_2 = temp_dir / "spool_2"
            spooler = WALDiskSpooler(spool_dir=spool_2)
            for i in range(10):
                spooler.append_trace(RunTrace(run_id=f"crash_{i}", workflow="recovery_test"))
            spooler.close()

            # Reboot simulation: instantiate new queue pointing to same directory
            q2 = TraceIngestionQueue(max_size=50, spool_dir=spool_2, wal_enabled=True)
            recovered = q2.recover_from_disk()
            assert recovered == 10, f"Expected 10 recovered traces, got {recovered}"
            assert q2.size() == 10

            batch = q2.dequeue_batch(max_items=20)
            assert len(batch) == 10
            assert [t.run_id for t in batch] == [f"crash_{i}" for i in range(10)]
            q2.clear()
            print("[PASS] Check 2: Crash Recovery & Replay from Local Disk WAL (10/10 recovered)")
        except Exception as exc:
            print(f"[FAIL] Check 2: Crash recovery error: {exc}")
            all_passed = False

        # ─────────────────────────────────────────────────────────────────────
        # Check 3: Memory Queue Overflow Disk Spillover Activation
        # ─────────────────────────────────────────────────────────────────────
        try:
            spool_3 = temp_dir / "spool_3"
            q3 = TraceIngestionQueue(
                max_size=5,
                spool_dir=spool_3,
                wal_enabled=False,
                overflow_spill=True,
            )

            # Enqueue 15 traces into a 5-item buffer
            for i in range(15):
                ok = q3.enqueue(RunTrace(run_id=f"burst_{i}", workflow="burst_test"))
                assert ok is True

            stats = q3.stats()
            assert stats["size"] == 5
            assert stats["spilled"] == 10, f"Expected 10 spilled traces, got {stats['spilled']}"
            assert stats["dropped"] == 0, f"Expected 0 dropped traces, got {stats['dropped']}"

            overflow_files = list(spool_3.glob("overflow_*.jsonl"))
            assert len(overflow_files) > 0

            # Recover overflow
            overflow_recovered = (
                q3.spooler.recover_overflow(delete_after_read=True) if q3.spooler else []
            )
            assert len(overflow_recovered) == 10
            q3.clear()
            print(
                "[PASS] Check 3: Memory Queue Overflow Disk Spillover Activation (10 spilled, 0 dropped)"
            )
        except Exception as exc:
            print(f"[FAIL] Check 3: Disk spillover error: {exc}")
            all_passed = False

        # ─────────────────────────────────────────────────────────────────────
        # Check 4: Saturation Detection & High-Watermark Threshold Accounting
        # ─────────────────────────────────────────────────────────────────────
        try:
            q4 = TraceIngestionQueue(max_size=20, high_watermark=0.75)
            assert q4.is_saturated() is False

            for i in range(14):
                q4.enqueue(RunTrace(run_id=f"sat_{i}", workflow="test"))
            assert q4.is_saturated() is False

            # 15th item reaches 75% watermark (15 / 20 = 0.75)
            q4.enqueue(RunTrace(run_id="sat_15", workflow="test"))
            assert q4.is_saturated() is True
            assert q4.stats()["saturated"] is True
            q4.clear()
            print(
                "[PASS] Check 4: Saturation Detection & High-Watermark Accounting (75% threshold)"
            )
        except Exception as exc:
            print(f"[FAIL] Check 4: Saturation accounting error: {exc}")
            all_passed = False

        # ─────────────────────────────────────────────────────────────────────
        # Check 5: API Backpressure Signaling (HTTP 429 + Retry-After)
        # ─────────────────────────────────────────────────────────────────────
        try:
            app = FastAPI()
            app.include_router(router)
            client = TestClient(app)

            # Test queue status endpoint
            status_res = client.get("/queue/status")
            assert status_res.status_code == 200
            assert "saturated" in status_res.json()

            # Temporarily simulate global queue saturation
            g_queue = get_global_queue()
            orig_is_saturated = g_queue.is_saturated
            g_queue.is_saturated = lambda: True  # type: ignore[method-assign]

            try:
                payload = {
                    "resourceSpans": [
                        {
                            "scopeSpans": [
                                {
                                    "spans": [
                                        {
                                            "traceId": "trace_backpressure_gate",
                                            "spanId": "span_gate",
                                            "name": "orchestrator",
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                }
                res = client.post("/v1/traces", json=payload)
                assert res.status_code == 429, f"Expected 429, got {res.status_code}"
                assert res.headers.get("Retry-After") == "5"
                assert res.headers.get("X-AgentLens-Queue-Status") == "saturated"
            finally:
                g_queue.is_saturated = orig_is_saturated  # type: ignore[method-assign]

            print(
                "[PASS] Check 5: API Backpressure Signaling (HTTP 429 + Retry-After: 5 + Status Headers)"
            )
        except Exception as exc:
            print(f"[FAIL] Check 5: API backpressure error: {exc}")
            all_passed = False

        # ─────────────────────────────────────────────────────────────────────
        # Check 6: WAL Chunk File Rotation & Pruning Post-Persistence
        # ─────────────────────────────────────────────────────────────────────
        try:
            spool_6 = temp_dir / "spool_6"
            q6 = TraceIngestionQueue(max_size=50, spool_dir=spool_6, wal_enabled=True)
            assert q6.spooler is not None

            q6.enqueue(RunTrace(run_id="r1", workflow="test"))
            q6.enqueue(RunTrace(run_id="r2", workflow="test"))
            wal_files_1 = list(spool_6.glob("wal_*.jsonl"))
            assert len(wal_files_1) == 1
            first_chunk = wal_files_1[0].name

            # Rotate to next chunk
            q6.spooler._rotate_chunk_locked()
            q6.enqueue(RunTrace(run_id="r3", workflow="test"))

            # Mark r1 and r2 persisted
            pruned = q6.spooler.mark_persisted(["r1", "r2"])
            assert first_chunk in pruned
            assert not (spool_6 / first_chunk).exists()
            q6.clear()
            print("[PASS] Check 6: WAL Chunk File Rotation & Pruning Post-Persistence")
        except Exception as exc:
            print(f"[FAIL] Check 6: WAL pruning error: {exc}")
            all_passed = False

        # ─────────────────────────────────────────────────────────────────────
        # Check 7: Ingestion Queue Overhead Benchmark (<= 0.8ms per trace)
        # ─────────────────────────────────────────────────────────────────────
        try:
            spool_7 = temp_dir / "spool_7"
            q7 = TraceIngestionQueue(
                max_size=1000,
                spool_dir=spool_7,
                wal_enabled=True,
            )
            iterations = 100
            start_perf = time.perf_counter()
            for i in range(iterations):
                q7.enqueue(RunTrace(run_id=f"perf_{i}", workflow="perf_test"))
            end_perf = time.perf_counter()

            avg_overhead_ms = ((end_perf - start_perf) / iterations) * 1000.0
            assert avg_overhead_ms <= 0.8, (
                f"Overhead {avg_overhead_ms:.4f}ms exceeded 0.8ms threshold!"
            )
            q7.clear()
            print(
                f"[PASS] Check 7: Ingestion Queue Overhead Benchmark ({avg_overhead_ms:.4f}ms <= 0.8ms per trace)"
            )
        except Exception as exc:
            print(f"[FAIL] Check 7: Performance benchmark error: {exc}")
            all_passed = False

        # ─────────────────────────────────────────────────────────────────────
        # Check 8: Static Analysis & Unit Test Suite Execution
        # ─────────────────────────────────────────────────────────────────────
        try:
            import os

            py_exe = sys.executable
            sub_env = os.environ.copy()
            anaconda_path = r"C:\Users\Ankita Ghosh\anaconda3;C:\Users\Ankita Ghosh\anaconda3\DLLs;C:\Users\Ankita Ghosh\anaconda3\Library\bin"
            sub_env["PATH"] = anaconda_path + os.pathsep + sub_env.get("PATH", "")

            # 1. Ruff lint check
            targets = [
                "capture/queue.py",
                "api/router.py",
                "tests/test_queue_durability.py",
                "scripts/verify_day67.py",
            ]
            r_chk = subprocess.run(
                [py_exe, "-m", "ruff", "check"] + targets,
                capture_output=True,
                text=True,
                cwd=str(REPO_ROOT),
                env=sub_env,
            )
            assert r_chk.returncode == 0, f"Ruff check failed:\n{r_chk.stdout}\n{r_chk.stderr}"

            # 2. Ruff format check
            r_fmt = subprocess.run(
                [py_exe, "-m", "ruff", "format", "--check"] + targets,
                capture_output=True,
                text=True,
                cwd=str(REPO_ROOT),
                env=sub_env,
            )
            assert r_fmt.returncode == 0, f"Ruff format check failed:\n{r_fmt.stdout}"

            # 3. Mypy strict type check
            my_res = subprocess.run(
                [py_exe, "-m", "mypy"] + targets + ["--ignore-missing-imports"],
                capture_output=True,
                text=True,
                cwd=str(REPO_ROOT),
                env=sub_env,
            )
            assert my_res.returncode == 0, f"Mypy failed:\n{my_res.stdout}\n{my_res.stderr}"

            # 4. Pytest test_queue_durability.py
            pytest_res = subprocess.run(
                [py_exe, "-m", "pytest", "tests/test_queue_durability.py", "-q"],
                capture_output=True,
                text=True,
                cwd=str(REPO_ROOT),
                env=sub_env,
            )
            assert pytest_res.returncode == 0, (
                f"Pytest failed:\n{pytest_res.stdout}\n{pytest_res.stderr}"
            )

            print("[PASS] Check 8: Static Analysis (Ruff & Mypy Clean) & Test Suite Execution")
        except Exception as exc:
            print(f"[FAIL] Check 8: Static analysis / pytest error: {exc}")
            all_passed = False

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("=" * 76)
    if all_passed:
        print(">>> ALL 8 DAY 67 VERIFICATION CHECKS PASSED SUCCESSFULLY! <<<")
        print(">>> INGESTION QUEUE DURABILITY, WAL & BACKPRESSURE CERTIFIED! <<<")
    else:
        print(">>> SOME DAY 67 CHECKS FAILED. SEE ABOVE FOR DETAILS. <<<")
    print("=" * 76)
    return all_passed


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
