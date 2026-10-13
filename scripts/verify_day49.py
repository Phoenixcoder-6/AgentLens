"""
scripts/verify_day49.py — Day 49 Verification Suite
===================================================
Automated verification of SQLite WAL mode, high-concurrency pragmas,
thread-safe connection pooling, and retry backoff in AgentLens v2.0:
1. Verify SQLite WAL mode journal mode configuration
2. Verify high-concurrency pragmas (busy_timeout=10000, synchronous=NORMAL, foreign_keys=ON)
3. Verify connection pool acquisition, bounding, and connection reuse
4. Verify exponential retry backoff on database lock contention
5. Verify 100 simultaneous concurrent writes with zero lock errors
6. Verify concurrent reader non-blocking behavior during active writes
7. Verify transaction rollback integrity on operational error
8. Verify zero regressions on Day 43 benchmark (16/20 exact attribution)
"""

from __future__ import annotations

import random
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from normalizer.normalizer import Normalizer
from schema.models import AgentStep, RunTrace, StepStatus, TokenUsage
from storage.db import DatabaseManager
from storage.writer import StorageWriter


def check_1_wal_mode(tmp_dir: Path) -> bool:
    print("\n--- Check 1: SQLite WAL Mode Configuration ---")
    db_file = str(tmp_dir / "wal_test.db")
    db = DatabaseManager(db_path=db_file)
    db.initialize()

    pragmas = db.get_pragmas()
    if pragmas.get("journal_mode") != "wal":
        print(f"[FAIL] Expected journal_mode='wal', got '{pragmas.get('journal_mode')}'")
        db.close()
        return False

    db.close()
    print("[PASS] SQLite database successfully configured in WAL (Write-Ahead Logging) mode.")
    return True


def check_2_high_concurrency_pragmas(tmp_dir: Path) -> bool:
    print("\n--- Check 2: High-Concurrency Pragmas Verification ---")
    db_file = str(tmp_dir / "pragmas_test.db")
    db = DatabaseManager(db_path=db_file)
    db.initialize()

    pragmas = db.get_pragmas()
    if pragmas.get("busy_timeout") != 10000:
        print(f"[FAIL] Expected busy_timeout=10000, got {pragmas.get('busy_timeout')}")
        db.close()
        return False
    if pragmas.get("synchronous") != 1:
        print(f"[FAIL] Expected synchronous=1 (NORMAL), got {pragmas.get('synchronous')}")
        db.close()
        return False
    if pragmas.get("foreign_keys") != 1:
        print(f"[FAIL] Expected foreign_keys=1 (ON), got {pragmas.get('foreign_keys')}")
        db.close()
        return False

    db.close()
    print("[PASS] High-concurrency pragmas verified (busy_timeout=10000, synchronous=NORMAL, foreign_keys=ON).")
    return True


def check_3_connection_pooling(tmp_dir: Path) -> bool:
    print("\n--- Check 3: Thread-Safe Connection Pool Acquisition & Bounding ---")
    db_file = str(tmp_dir / "pool_test.db")
    db = DatabaseManager(db_path=db_file, pool_size=5)
    db.initialize()

    def worker_fn(idx: int):
        with db.connection() as conn:
            row = conn.execute("SELECT ? AS idx", (idx,)).fetchone()
            time.sleep(0.005)
            return row["idx"]

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker_fn, i) for i in range(20)]
        results = [f.result() for f in futures]

    if len(results) != 20:
        print(f"[FAIL] Expected 20 pool results, got {len(results)}")
        db.close()
        return False

    stats = db.pool.stats()
    if stats["active_connections"] > 5:
        print(f"[FAIL] Pool created {stats['active_connections']} connections, exceeding max 5")
        db.close()
        return False

    db.close()
    print(f"[PASS] Connection pool bounded to {stats['max_connections']} connections and cleanly reused.")
    return True


def check_4_retry_backoff(tmp_dir: Path) -> bool:
    print("\n--- Check 4: Exponential Retry Backoff on Lock Contention ---")
    db_file = str(tmp_dir / "retry_test.db")
    db = DatabaseManager(db_path=db_file)
    db.initialize()

    attempts = 0

    def locked_operation(conn: sqlite3.Connection):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise sqlite3.OperationalError("database is locked")
        conn.execute("INSERT OR REPLACE INTO runs (run_id, workflow, timestamp, status, schema_version) VALUES ('retry_01', 'wf', '2026-01-01', 'SUCCESS', '1.0')")
        return "recovered"

    res = db.execute_with_retry(locked_operation, max_retries=5, base_delay=0.01)
    if res != "recovered" or attempts != 3:
        print(f"[FAIL] Retry backoff failed to recover: res='{res}', attempts={attempts}")
        db.close()
        return False

    run = db.get_run("retry_01")
    if not run:
        print("[FAIL] Record inserted during retry was not persisted")
        db.close()
        return False

    db.close()
    print("[PASS] Exponential retry backoff cleanly handles transient lock contention.")
    return True


def check_5_100_concurrent_writes(tmp_dir: Path) -> bool:
    print("\n--- Check 5: 100 Simultaneous Concurrent Writes Stress Test ---")
    db_file = str(tmp_dir / "stress_100.db")
    db = DatabaseManager(db_path=db_file, pool_size=10)
    db.initialize()
    writer = StorageWriter(db)
    normalizer = Normalizer()

    def write_worker(idx: int) -> str:
        run_id = f"stress_run_{idx:03d}"
        trace = RunTrace(
            run_id=run_id,
            workflow="concurrency_stress",
            status=StepStatus.SUCCESS,
            timestamp=datetime.now(UTC),
            steps=[
                AgentStep(
                    run_id=run_id,
                    step=1,
                    agent="researcher",
                    input='{"topic": "stress"}',
                    output='{"findings": "ok"}',
                    latency_ms=10.0 + random.uniform(1.0, 5.0),
                    status=StepStatus.SUCCESS,
                    tokens=TokenUsage(prompt=100, completion=50, total=150),
                ),
                AgentStep(
                    run_id=run_id,
                    step=2,
                    agent="writer",
                    input='{"findings": "ok"}',
                    output='{"report": "done"}',
                    latency_ms=20.0 + random.uniform(1.0, 5.0),
                    status=StepStatus.SUCCESS,
                    tokens=TokenUsage(prompt=150, completion=70, total=220),
                ),
            ],
        )
        writer.write_run(normalizer.normalize_run(trace))
        return run_id

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(write_worker, i) for i in range(100)]
        _ = [f.result() for f in futures]
    t1 = time.perf_counter()

    counts = db.table_counts()
    if counts["runs"] != 100 or counts["steps"] != 200:
        print(f"[FAIL] Data count mismatch: runs={counts['runs']}, steps={counts['steps']}")
        db.close()
        return False

    elapsed = t1 - t0
    ops_per_sec = 100.0 / elapsed
    print(f"[INFO] 100 full multi-step runs persisted in {elapsed:.2f}s ({ops_per_sec:.1f} runs/sec).")

    db.close()
    print("[PASS] 100 concurrent writes completed with 0 errors and 100% data integrity.")
    return True


def check_6_concurrent_readers(tmp_dir: Path) -> bool:
    print("\n--- Check 6: Concurrent Non-Blocking Readers During Heavy Writes ---")
    db_file = str(tmp_dir / "rw_test.db")
    db = DatabaseManager(db_path=db_file, pool_size=10)
    db.initialize()
    writer = StorageWriter(db)
    normalizer = Normalizer()

    # Seed initial data
    for i in range(5):
        trace = RunTrace(
            run_id=f"seed_{i}",
            workflow="seed",
            status=StepStatus.SUCCESS,
            timestamp=datetime.now(UTC),
            steps=[],
        )
        writer.write_run(normalizer.normalize_run(trace))

    read_count = 0

    def reader_worker():
        nonlocal read_count
        for _ in range(5):
            runs = db.list_runs(limit=10)
            if len(runs) >= 5:
                read_count += 1
            time.sleep(random.uniform(0.001, 0.004))

    def writer_worker(idx: int):
        trace = RunTrace(
            run_id=f"rw_{idx}",
            workflow="rw",
            status=StepStatus.SUCCESS,
            timestamp=datetime.now(UTC),
            steps=[],
        )
        writer.write_run(normalizer.normalize_run(trace))

    with ThreadPoolExecutor(max_workers=20) as executor:
        read_futures = [executor.submit(reader_worker) for _ in range(10)]
        write_futures = [executor.submit(writer_worker, i) for i in range(30)]
        for f in read_futures + write_futures:
            f.result()

    if read_count != 50:
        print(f"[FAIL] Readers encountered lock or missing data: {read_count}/50 succeeded")
        db.close()
        return False

    db.close()
    print("[PASS] Readers read concurrently without blocking or being blocked by writers.")
    return True


def check_7_rollback_integrity(tmp_dir: Path) -> bool:
    print("\n--- Check 7: Transaction Rollback Integrity on Operational Error ---")
    db_file = str(tmp_dir / "rollback_test.db")
    db = DatabaseManager(db_path=db_file)
    db.initialize()

    caught = False
    try:
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, workflow, timestamp, status, schema_version) VALUES ('will_abort', 'wf', '2026-01-01', 'SUCCESS', '1.0')"
            )
            # Intentionally raise to test rollback
            raise RuntimeError("Forced abort for rollback check")
    except RuntimeError:
        caught = True

    if not caught:
        print("[FAIL] Exception was not raised during rollback test")
        db.close()
        return False

    if db.get_run("will_abort") is not None:
        print("[FAIL] Record was committed despite exception in transaction context")
        db.close()
        return False

    db.close()
    print("[PASS] Multi-statement transactions rollback cleanly on error.")
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
    print("AgentLens v2.0 - Day 49 Verification Suite")
    print("=" * 60)

    tmp_dir = Path(tempfile.mkdtemp(prefix="agentlens_day49_"))
    try:
        checks = [
            lambda: check_1_wal_mode(tmp_dir),
            lambda: check_2_high_concurrency_pragmas(tmp_dir),
            lambda: check_3_connection_pooling(tmp_dir),
            lambda: check_4_retry_backoff(tmp_dir),
            lambda: check_5_100_concurrent_writes(tmp_dir),
            lambda: check_6_concurrent_readers(tmp_dir),
            lambda: check_7_rollback_integrity(tmp_dir),
            check_8_regression_suite,
        ]

        passed = 0
        for i, chk in enumerate(checks, 1):
            if chk():
                passed += 1
            else:
                print(f"\nVerification failed at check {i}")
                sys.exit(1)

        print("\n" + "=" * 60)
        print(f"DAY 49 VERIFICATION SUCCESSFUL: {passed}/{len(checks)} CHECKS PASSED")
        print("=" * 60)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
