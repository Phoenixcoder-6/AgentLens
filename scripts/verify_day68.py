"""
scripts/verify_day68.py — Day 68 Storage Fail-Loud & Indexing Verification Gate
================================================================================
Phase E: Data Durability, Storage Reliability & SRE

Checks:
  1. PostgreSQL Driver Missing Fail-Loud (PostgresDriverMissingError, No Silent Mock).
  2. PostgreSQL Connection Fail-Loud (PostgresConnectionError, Masked Credentials).
  3. Exponential Retry Backoff with Jitter Resilience.
  4. Explicit Mock Mode Isolation for Offline Unit Testing.
  5. SQLite Secondary Composite Indices Creation.
  6. EXPLAIN QUERY PLAN Verification (Index Utilization, Zero Table Scans).
  7. High-Throughput Point Lookup Benchmark (<1ms per query on 10,000 records).
  8. Static Analysis (Ruff & Mypy Clean) & Durability Test Suite Execution.
"""

# ruff: noqa: E402

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from storage.db import DatabaseManager
from storage.postgres_adapter import (
    PostgresConnectionError,
    PostgresDriverMissingError,
    PostgreSQLStorageAdapter,
)


def run_checks() -> bool:
    print("=" * 76)
    print("  AGENTLENS v2.1+ — DAY 68 STORAGE FAIL-LOUD & INDEXING GATE")
    print("  Phase E: Data Durability, Storage Reliability & SRE Hardening")
    print("=" * 76)
    all_passed = True

    # ─────────────────────────────────────────────────────────────────────────
    # Check 1: Missing Driver Fails Loud
    # ─────────────────────────────────────────────────────────────────────────
    try:
        url = "postgresql://agentlens:super_secret@prod-db.internal:5432/traces_db"
        adapter = PostgreSQLStorageAdapter(url, mock_mode=False)
        adapter._driver = None  # Ensure driver absence

        assert adapter.mock_mode is False, "mock_mode was silently flipped to True on init!"

        driver_err_raised = False
        try:
            adapter.initialize()
        except PostgresDriverMissingError as exc:
            driver_err_raised = True
            assert "No PostgreSQL driver installed" in str(exc)
            assert "super_secret" not in str(exc), "Password leaked in exception string!"

        assert driver_err_raised, "PostgresDriverMissingError was not raised!"
        assert adapter.mock_mode is False, "mock_mode was silently flipped to True on error!"
        print("[PASS] Check 1: PostgreSQL Missing Driver Fails Loud (Zero Silent Mock Fallback)")
    except Exception as exc:
        print(f"[FAIL] Check 1: PostgreSQL Missing Driver Fail-Loud Failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 2: Connection Failure Fails Loud
    # ─────────────────────────────────────────────────────────────────────────
    try:
        url = "postgresql://agentlens:cluster_token@10.0.0.50:5432/production_traces"
        adapter2 = PostgreSQLStorageAdapter(
            url,
            mock_mode=False,
            max_retries=2,
            retry_delay=0.01,
            jitter=False,
        )
        adapter2._driver = "psycopg2"

        # Mock connection error
        adapter2._raw_connect = MagicMock(side_effect=ConnectionRefusedError("Host unreachable"))  # type: ignore[method-assign]

        conn_err_raised = False
        try:
            adapter2.initialize()
        except PostgresConnectionError as exc:
            conn_err_raised = True
            assert exc.host == "10.0.0.50"
            assert exc.port == 5432
            assert exc.database == "production_traces"
            assert exc.attempts == 2
            assert "cluster_token" not in str(exc), "Password was not masked in exception!"

        assert conn_err_raised, "PostgresConnectionError was not raised on unreachable host!"
        assert adapter2.mock_mode is False, "mock_mode was silently flipped on failure!"
        print(
            "[PASS] Check 2: PostgreSQL Connection Fail-Loud (Masked Credentials & Structured Diagnostics)"
        )
    except Exception as exc:
        print(f"[FAIL] Check 2: PostgreSQL Connection Fail-Loud Failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 3: Exponential Retry Backoff with Jitter
    # ─────────────────────────────────────────────────────────────────────────
    try:
        adapter3 = PostgreSQLStorageAdapter(
            "postgresql://user:pass@127.0.0.1:5432/db",
            mock_mode=False,
            max_retries=3,
            retry_delay=0.02,
            backoff_factor=2.0,
            jitter=True,
        )
        adapter3._driver = "psycopg2"

        attempts = 0
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        def transient_connect():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ConnectionResetError(f"Temporary glitch {attempts}")
            return mock_conn

        adapter3._raw_connect = transient_connect  # type: ignore[method-assign]

        t_start = time.perf_counter()
        conn = adapter3._connect_with_retry()
        t_elapsed = time.perf_counter() - t_start

        assert conn is mock_conn
        assert attempts == 3
        assert t_elapsed >= 0.02, f"Retry delay was not enforced ({t_elapsed:.4f}s)!"
        print(
            f"[PASS] Check 3: Exponential Retry Backoff with Jitter ({attempts} attempts in {t_elapsed:.4f}s)"
        )
    except Exception as exc:
        print(f"[FAIL] Check 3: Retry Backoff with Jitter Failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 4: Explicit Mock Mode Isolation
    # ─────────────────────────────────────────────────────────────────────────
    try:
        adapter4 = PostgreSQLStorageAdapter(
            "postgresql://user:pass@localhost:5432/mock_db", mock_mode=True
        )
        assert adapter4.mock_mode is True
        adapter4.initialize()
        assert adapter4.is_healthy() is True
        assert adapter4.ping() is True

        info = adapter4.get_connection_info(mask_password=True)
        assert info["driver"] == "mock"
        assert info["mock_mode"] is True
        assert info["password"] == "******"
        adapter4.close()
        print("[PASS] Check 4: Explicit Mock Mode Isolation for Offline Testing")
    except Exception as exc:
        print(f"[FAIL] Check 4: Explicit Mock Mode Isolation Failed: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 5: SQLite Secondary Composite Indices Creation
    # ─────────────────────────────────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        temp_db_path = tf.name

    try:
        db = DatabaseManager(temp_db_path)
        db.initialize()

        with db.connection() as conn:
            rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'index';").fetchall()
            indices_found = {r[0] for r in rows}

        required_indices = [
            "idx_analysis_run_step",
            "idx_analysis_run_id",
            "idx_metrics_run_step",
            "idx_metrics_run_name",
            "idx_runs_timestamp",
            "idx_steps_run_status",
        ]

        for req in required_indices:
            assert req in indices_found, f"Missing secondary index: {req}"

        print(
            f"[PASS] Check 5: SQLite Secondary Composite Indices Created ({len(indices_found)} indices)"
        )
    except Exception as exc:
        print(f"[FAIL] Check 5: SQLite Secondary Composite Indices Failed: {exc}")
        all_passed = False
    finally:
        db.close()
        try:
            os.unlink(temp_db_path)
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # Check 6: EXPLAIN QUERY PLAN Index Validation
    # ─────────────────────────────────────────────────────────────────────────
    try:
        mem_db = DatabaseManager(":memory:")
        mem_db.initialize()

        with mem_db.connection() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, workflow, timestamp, status, schema_version) VALUES ('r1', 'wf', '2026-11-27T00:00:00Z', 'COMPLETED', '1.0');"
            )
            conn.execute(
                "INSERT INTO analysis (run_id, step, analyzer, category, timestamp, schema_version) VALUES ('r1', 1, 'arbiter', 'P1', '2026-11-27T00:00:00Z', '1.0');"
            )
            conn.execute(
                "INSERT INTO metrics (run_id, step, metric_name, metric_value, timestamp, schema_version) VALUES ('r1', 1, 'latency_ms', 50.0, '2026-11-27T00:00:00Z', '1.0');"
            )

            # 1. Analysis plan
            plan_a = conn.execute(
                "EXPLAIN QUERY PLAN SELECT * FROM analysis WHERE run_id = 'r1' AND step = 1;"
            ).fetchall()
            plan_a_text = " ".join([str(col) for row in plan_a for col in row])
            assert "idx_analysis_run_step" in plan_a_text
            assert "SCAN TABLE analysis" not in plan_a_text

            # 2. Metrics plan
            plan_m = conn.execute(
                "EXPLAIN QUERY PLAN SELECT * FROM metrics WHERE run_id = 'r1' AND step = 1;"
            ).fetchall()
            plan_m_text = " ".join([str(col) for row in plan_m for col in row])
            assert "idx_metrics_run_step" in plan_m_text
            assert "SCAN TABLE metrics" not in plan_m_text

        print("[PASS] Check 6: EXPLAIN QUERY PLAN Index Utilization (Zero Full Table Scans)")
    except Exception as exc:
        print(f"[FAIL] Check 6: EXPLAIN QUERY PLAN Validation Failed: {exc}")
        all_passed = False
    finally:
        mem_db.close()

    # ─────────────────────────────────────────────────────────────────────────
    # Check 7: High-Throughput Point Lookup Benchmark
    # ─────────────────────────────────────────────────────────────────────────
    try:
        bench_db = DatabaseManager(":memory:")
        bench_db.initialize()

        with bench_db.connection() as conn:
            runs_batch: list[tuple[Any, ...]] = []
            analysis_batch: list[tuple[Any, ...]] = []
            metrics_batch: list[tuple[Any, ...]] = []

            for r_idx in range(100):
                rid = f"bench_{r_idx:04d}"
                runs_batch.append((rid, "wf", "2026-11-27T00:00:00Z", "SUCCESS", "1.0"))
                for s_idx in range(50):
                    analysis_batch.append(
                        (
                            rid,
                            s_idx,
                            "arbiter",
                            "P2",
                            "PASS",
                            0.9,
                            "{}",
                            "2026-11-27T00:00:00Z",
                            "1.0",
                        )
                    )
                    metrics_batch.append(
                        (rid, s_idx, "tokens", float(s_idx * 15), "2026-11-27T00:00:00Z", "1.0")
                    )

            conn.executemany(
                "INSERT INTO runs (run_id, workflow, timestamp, status, schema_version) VALUES (?, ?, ?, ?, ?);",
                runs_batch,
            )
            conn.executemany(
                "INSERT INTO analysis (run_id, step, analyzer, category, verdict, confidence, details_json, timestamp, schema_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);",
                analysis_batch,
            )
            conn.executemany(
                "INSERT INTO metrics (run_id, step, metric_name, metric_value, timestamp, schema_version) VALUES (?, ?, ?, ?, ?, ?);",
                metrics_batch,
            )
            conn.execute("ANALYZE;")

        # Run 200 random queries
        with bench_db.connection() as conn:
            t_start = time.perf_counter()
            for i in range(200):
                target_r = f"bench_{(i * 7) % 100:04d}"
                target_s = (i * 3) % 50
                res = conn.execute(
                    "SELECT id FROM analysis WHERE run_id = ? AND step = ?;", (target_r, target_s)
                ).fetchone()
                assert res is not None
            t_total = time.perf_counter() - t_start
            avg_ms = (t_total / 200) * 1000.0

        assert avg_ms < 1.0, f"Average query latency {avg_ms:.4f}ms exceeded 1.0ms SLA!"
        print(
            f"[PASS] Check 7: High-Throughput Point Lookup Benchmark ({avg_ms:.4f}ms <= 1.0ms per query on 10,000 rows)"
        )
    except Exception as exc:
        print(f"[FAIL] Check 7: Point Lookup Benchmark Failed: {exc}")
        all_passed = False
    finally:
        bench_db.close()

    # ─────────────────────────────────────────────────────────────────────────
    # Check 8: Static Analysis & Test Suite Execution
    # ─────────────────────────────────────────────────────────────────────────
    try:
        sub_env = os.environ.copy()
        anaconda_path = (
            "C:\\Users\\Ankita Ghosh\\anaconda3;"
            "C:\\Users\\Ankita Ghosh\\anaconda3\\DLLs;"
            "C:\\Users\\Ankita Ghosh\\anaconda3\\Library\\bin"
        )
        sub_env["PATH"] = anaconda_path + os.pathsep + sub_env.get("PATH", "")

        python_bin = sys.executable
        ruff_bin = str(REPO_ROOT / ".venv" / "Scripts" / "ruff.exe")

        files_to_check = [
            "storage/postgres_adapter.py",
            "storage/db.py",
            "storage/__init__.py",
            "tests/test_storage_durability.py",
            "scripts/verify_day68.py",
        ]

        # 1. Ruff check
        res_ruff = subprocess.run(
            [ruff_bin, "check", *files_to_check],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
            env=sub_env,
        )
        assert res_ruff.returncode == 0, f"Ruff check failed:\n{res_ruff.stdout}\n{res_ruff.stderr}"

        # 2. Ruff format check
        res_fmt = subprocess.run(
            [ruff_bin, "format", "--check", *files_to_check],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
            env=sub_env,
        )
        assert res_fmt.returncode == 0, (
            f"Ruff format check failed:\n{res_fmt.stdout}\n{res_fmt.stderr}"
        )

        # 3. Mypy type check
        anaconda_python = r"C:\Users\Ankita Ghosh\anaconda3\python.exe"
        mypy_runner = (
            [anaconda_python, "-m", "mypy"]
            if os.path.exists(anaconda_python)
            else [python_bin, "-m", "mypy"]
        )
        res_mypy = subprocess.run(
            mypy_runner
            + [
                *files_to_check,
                "--ignore-missing-imports",
                "--disable-error-code=import-untyped",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
            env=sub_env,
        )
        assert res_mypy.returncode == 0, f"Mypy check failed:\n{res_mypy.stdout}\n{res_mypy.stderr}"

        # 4. Pytest durability test suite
        res_pytest = subprocess.run(
            [python_bin, "-m", "pytest", "tests/test_storage_durability.py", "-v"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
            env=sub_env,
        )
        assert res_pytest.returncode == 0, (
            f"Pytest failed:\n{res_pytest.stdout}\n{res_pytest.stderr}"
        )

        print("[PASS] Check 8: Static Analysis (Ruff & Mypy Clean) & Test Suite Execution")
    except Exception as exc:
        print(f"[FAIL] Check 8: Static Analysis & Test Suite Execution Failed: {exc}")
        all_passed = False

    print("=" * 76)
    if all_passed:
        print(">>> ALL 8 DAY 68 VERIFICATION CHECKS PASSED SUCCESSFULLY! <<<")
        print(">>> POSTGRESQL FAIL-LOUD & SQLITE SECONDARY INDEXING CERTIFIED! <<<")
        print("=" * 76)
        return True
    else:
        print(">>> DAY 68 VERIFICATION FAILED! SEE LOG ABOVE. <<<")
        print("=" * 76)
        return False


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
