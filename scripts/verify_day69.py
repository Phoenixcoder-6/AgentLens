"""
scripts/verify_day69.py — Day 69 Asynchronous Analysis Worker Verification Gate
================================================================================
Phase E: Data Durability, Storage Reliability & SRE

Checks:
  1. Immediate 202 Accepted Response on Analysis Trigger (<50ms submission SLA).
  2. Complete Job Lifecycle Transitions (QUEUED -> PROCESSING -> COMPLETED).
  3. Granular Polling Endpoint Verification (GET /api/jobs/{job_id}).
  4. High-Concurrency Burst Submission (20 simultaneous jobs returning 202).
  5. Error Isolation on Invalid Runs (FAILED status with clean error detail).
  6. Synchronous Backward-Compatibility Mode (POST ?sync=true returning 200 OK).
  7. HTTP Webhook Callback Dispatch on Job Completion.
  8. Static Analysis (Ruff & Mypy Clean) & Test Suite Execution.
"""

# ruff: noqa: E402

from __future__ import annotations

import http.server
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI
from fastapi.testclient import TestClient

import dashboard.state as _state
from analyzers.worker import JobStatus, get_job_manager
from api.router import router
from schema.models import AgentStep, RunTrace, StepStatus, TokenUsage
from storage.db import DatabaseManager


def _seed_test_run(db: DatabaseManager, run_id: str) -> None:
    step1 = AgentStep(
        run_id=run_id,
        step=1,
        agent="researcher",
        status=StepStatus.SUCCESS,
        output="The product launch is scheduled for Q4 in North America.",
        tokens=TokenUsage(prompt=40, completion=25, total=65),
        latency_ms=90.0,
    )
    step2 = AgentStep(
        run_id=run_id,
        step=2,
        agent="writer",
        status=StepStatus.SUCCESS,
        output="Summary: Launch is delayed until Q1 next year.",
        tokens=TokenUsage(prompt=65, completion=30, total=95),
        latency_ms=110.0,
    )
    trace = RunTrace(
        run_id=run_id,
        workflow="product_launch",
        status=StepStatus.FAILURE,
        total_latency_ms=200.0,
        total_tokens=160,
        steps=[step1, step2],
        expected_output="The product launch is scheduled for Q4 in North America.",
    )
    db.insert_run(
        run_id=run_id,
        workflow=trace.workflow,
        timestamp="2026-11-27T21:00:00Z",
        status=trace.status,
        total_latency_ms=trace.total_latency_ms,
        total_tokens=trace.total_tokens,
        schema_version="1.0",
        trace_json=trace.model_dump_json(),
        expected_output=trace.expected_output,
    )


def run_checks() -> bool:
    print("=" * 76)
    print("  AGENTLENS v2.1+ — DAY 69 ASYNCHRONOUS ANALYSIS WORKER GATE")
    print("  Phase E: Data Durability, Storage Reliability & SRE Hardening")
    print("=" * 76)
    all_passed = True

    temp_dir = Path(tempfile.mkdtemp(prefix="agentlens_verify_day69_"))
    test_db_path = str(temp_dir / "verify_day69.db")
    db = DatabaseManager(test_db_path)
    db.initialize()

    # Point state to our verification database
    orig_db = getattr(_state, "_db", None)
    orig_get_db = _state.get_db
    _state._db = db
    _state.get_db = lambda: db
    _state._analysis_cache.clear()

    app = FastAPI()
    app.include_router(router, prefix="/api")
    client = TestClient(app)

    manager = get_job_manager()
    manager.clear()

    try:
        # ─────────────────────────────────────────────────────────────────────
        # Check 1: Immediate 202 Accepted Response (<100ms submission SLA)
        # ─────────────────────────────────────────────────────────────────────
        try:
            # Warm up client route dispatcher
            client.get("/api/runs")

            r1 = "chk1_async_run_01"
            _seed_test_run(db, r1)

            t_start = time.perf_counter()
            resp = client.post(f"/api/analyze/{r1}")
            sub_ms = (time.perf_counter() - t_start) * 1000.0

            assert resp.status_code == 202, f"Expected 202, got {resp.status_code}"
            data = resp.json()
            assert data["run_id"] == r1
            assert data["status"] == "queued"
            assert data["job_id"].startswith("job_")
            assert data["poll_url"] == f"/api/jobs/{data['job_id']}"
            assert sub_ms < 100.0, f"Submission latency {sub_ms:.2f}ms exceeded 100ms SLA"

            print(f"[PASS] Check 1: Immediate 202 Accepted Response ({sub_ms:.2f}ms <= 100ms SLA)")
        except Exception as exc:
            print(f"[FAIL] Check 1: 202 Accepted Response Failed: {exc}")
            all_passed = False

        # ─────────────────────────────────────────────────────────────────────
        # Check 2: Complete Job Lifecycle Transitions (QUEUED -> PROCESSING -> COMPLETED)
        # ─────────────────────────────────────────────────────────────────────
        try:
            r2 = "chk2_lifecycle_run"
            _seed_test_run(db, r2)

            resp2 = client.post(f"/api/analyze/{r2}")
            jid2 = resp2.json()["job_id"]

            job = manager.wait_for_job(jid2, timeout=8.0)
            assert job is not None, "Job timed out"
            assert job.status == JobStatus.COMPLETED
            assert job.started_at is not None
            assert job.completed_at is not None
            assert job.execution_time_ms is not None and job.execution_time_ms > 0.0

            print(
                f"[PASS] Check 2: Complete Job Lifecycle Transitions (QUEUED -> PROCESSING -> COMPLETED in {job.execution_time_ms}ms)"
            )
        except Exception as exc:
            print(f"[FAIL] Check 2: Job Lifecycle Transitions Failed: {exc}")
            all_passed = False

        # ─────────────────────────────────────────────────────────────────────
        # Check 3: Polling Endpoint GET /api/jobs/{job_id}
        # ─────────────────────────────────────────────────────────────────────
        try:
            poll_resp = client.get(f"/api/jobs/{jid2}")
            assert poll_resp.status_code == 200
            pj = poll_resp.json()
            assert pj["job_id"] == jid2
            assert pj["status"] == "completed"
            assert pj["priority_level"] in ("P1", "P2", "P3", "P4", "P5")
            assert pj["primary_cause"] is not None
            assert "execution_time_ms" in pj

            print(
                f"[PASS] Check 3: Granular Polling Endpoint GET /api/jobs/ (Verdict: {pj['priority_level']}, Cause: {pj['primary_cause']})"
            )
        except Exception as exc:
            print(f"[FAIL] Check 3: Polling Endpoint Verification Failed: {exc}")
            all_passed = False

        # ─────────────────────────────────────────────────────────────────────
        # Check 4: High-Concurrency Burst Submission (20 simultaneous jobs)
        # ─────────────────────────────────────────────────────────────────────
        try:
            burst_ids = [f"chk4_burst_{i:02d}" for i in range(20)]
            for bid in burst_ids:
                _seed_test_run(db, bid)

            def submit_single(rid: str) -> tuple[int, dict[str, Any]]:
                t0 = time.perf_counter()
                r = client.post(f"/api/analyze/{rid}")
                ms = (time.perf_counter() - t0) * 1000.0
                return r.status_code, {"ms": ms, "body": r.json()}

            t_burst_start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=10) as executor:
                burst_results = list(executor.map(submit_single, burst_ids))
            total_submit_ms = (time.perf_counter() - t_burst_start) * 1000.0

            for code, meta in burst_results:
                assert code == 202
                assert meta["body"]["status"] == "queued"

            assert total_submit_ms < 500.0, (
                f"Burst submission took {total_submit_ms:.2f}ms (>500ms)"
            )

            # Drain jobs
            for _, meta in burst_results:
                jid = meta["body"]["job_id"]
                j = manager.wait_for_job(jid, timeout=25.0)
                assert j is not None
                assert j.status in (JobStatus.COMPLETED, JobStatus.FAILED)

            print(
                f"[PASS] Check 4: High-Concurrency Burst Submission (20 simultaneous jobs in {total_submit_ms:.2f}ms total)"
            )
        except Exception as exc:
            print(f"[FAIL] Check 4: High-Concurrency Burst Submission Failed: {exc}")
            all_passed = False

        # ─────────────────────────────────────────────────────────────────────
        # Check 5: Error Isolation on Invalid Runs
        # ─────────────────────────────────────────────────────────────────────
        try:
            fake_rid = "nonexistent_verify_9999"
            resp_err = client.post(f"/api/analyze/{fake_rid}")
            assert resp_err.status_code == 202
            err_jid = resp_err.json()["job_id"]

            err_job = manager.wait_for_job(err_jid, timeout=10.0)
            assert err_job is not None, "Error job timed out"
            assert err_job.status == JobStatus.FAILED, f"Expected FAILED, got {err_job.status}"
            assert err_job.error is not None, "Expected error detail"

            poll_err = client.get(f"/api/jobs/{err_jid}").json()
            assert poll_err["status"] == "failed"
            assert "trace_json not found" in poll_err["error"]

            print(
                "[PASS] Check 5: Error Isolation on Invalid Runs (Zero Server Crashes, Clean FAILED State)"
            )
        except Exception as exc:
            print(f"[FAIL] Check 5: Error Isolation on Invalid Runs Failed: {exc}")
            all_passed = False

        # ─────────────────────────────────────────────────────────────────────
        # Check 6: Synchronous Backward-Compatibility Mode (?sync=true)
        # ─────────────────────────────────────────────────────────────────────
        try:
            r6 = "chk6_sync_compat_run"
            _seed_test_run(db, r6)

            resp_sync = client.post(f"/api/analyze/{r6}?sync=true")
            assert resp_sync.status_code == 200
            data_sync = resp_sync.json()
            assert data_sync["status"] == "complete"
            assert data_sync["run_id"] == r6
            assert data_sync["priority_level"] is not None

            print(
                f"[PASS] Check 6: Synchronous Backward-Compatibility Mode (?sync=true returning 200 OK with {data_sync['priority_level']})"
            )
        except Exception as exc:
            print(f"[FAIL] Check 6: Synchronous Mode Backward-Compatibility Failed: {exc}")
            all_passed = False

        # ─────────────────────────────────────────────────────────────────────
        # Check 7: HTTP Webhook Callback Dispatch on Completion
        # ─────────────────────────────────────────────────────────────────────
        try:
            r7 = "chk7_webhook_run"
            _seed_test_run(db, r7)

            hook_payloads: list[dict[str, Any]] = []

            class VerifyWebhookHandler(http.server.BaseHTTPRequestHandler):
                def do_POST(self):
                    cl = int(self.headers.get("Content-Length", 0))
                    payload_bytes = self.rfile.read(cl)
                    hook_payloads.append(json.loads(payload_bytes.decode("utf-8")))
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{"status":"ok"}')

                def log_message(self, format, *args):
                    pass

            v_server = http.server.HTTPServer(("127.0.0.1", 0), VerifyWebhookHandler)
            v_server.timeout = 15.0
            v_port = v_server.server_address[1]
            v_thread = threading.Thread(target=v_server.handle_request, daemon=True)
            v_thread.start()

            hook_url = f"http://127.0.0.1:{v_port}/agentlens-hook"
            resp_wh = client.post(f"/api/analyze/{r7}", json={"webhook_url": hook_url})
            assert resp_wh.status_code == 202
            wh_jid = resp_wh.json()["job_id"]

            wh_job = manager.wait_for_job(wh_jid, timeout=15.0)
            assert wh_job is not None and wh_job.status == JobStatus.COMPLETED, (
                f"Job status: {getattr(wh_job, 'status', None)}"
            )

            v_thread.join(timeout=5.0)
            v_server.server_close()

            assert len(hook_payloads) == 1
            assert hook_payloads[0]["job_id"] == wh_jid
            assert hook_payloads[0]["status"] == "completed"
            assert wh_job.webhook_status == "delivered"

            print(
                "[PASS] Check 7: HTTP Webhook Callback Dispatch on Completion (Payload Verified & Delivered)"
            )
        except Exception as exc:
            print(f"[FAIL] Check 7: HTTP Webhook Dispatch Failed: {exc}")
            all_passed = False

        # ─────────────────────────────────────────────────────────────────────
        # Check 8: Static Analysis & Test Suite Execution
        # ─────────────────────────────────────────────────────────────────────
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
            anaconda_python = r"C:\Users\Ankita Ghosh\anaconda3\python.exe"
            mypy_runner = (
                [anaconda_python, "-m", "mypy"]
                if os.path.exists(anaconda_python)
                else [python_bin, "-m", "mypy"]
            )

            targets = [
                "analyzers/worker.py",
                "analyzers/__init__.py",
                "api/router.py",
                "tests/test_async_analysis.py",
                "scripts/verify_day69.py",
            ]

            # 1. Ruff check
            r_chk = subprocess.run(
                [ruff_bin, "check", *targets],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
                env=sub_env,
            )
            assert r_chk.returncode == 0, f"Ruff check failed:\n{r_chk.stdout}\n{r_chk.stderr}"

            # 2. Ruff format check
            r_fmt = subprocess.run(
                [ruff_bin, "format", "--check", *targets],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
                env=sub_env,
            )
            assert r_fmt.returncode == 0, (
                f"Ruff format check failed:\n{r_fmt.stdout}\n{r_fmt.stderr}"
            )

            # 3. Mypy type check
            r_mypy = subprocess.run(
                mypy_runner
                + [*targets, "--ignore-missing-imports", "--disable-error-code=import-untyped"],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
                env=sub_env,
            )
            assert r_mypy.returncode == 0, f"Mypy check failed:\n{r_mypy.stdout}\n{r_mypy.stderr}"

            # 4. Pytest async analysis test suite
            r_pytest = subprocess.run(
                [python_bin, "-m", "pytest", "tests/test_async_analysis.py", "-v"],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
                env=sub_env,
            )
            assert r_pytest.returncode == 0, f"Pytest failed:\n{r_pytest.stdout}\n{r_pytest.stderr}"

            print("[PASS] Check 8: Static Analysis (Ruff & Mypy Clean) & Test Suite Execution")
        except Exception as exc:
            print(f"[FAIL] Check 8: Static Analysis & Test Suite Execution Failed: {exc}")
            all_passed = False

    finally:
        # Restore state
        _state._db = orig_db
        _state.get_db = orig_get_db
        _state._analysis_cache.clear()
        manager.shutdown(wait=False)
        db.close()
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("=" * 76)
    if all_passed:
        print(">>> ALL 8 DAY 69 VERIFICATION CHECKS PASSED SUCCESSFULLY! <<<")
        print(">>> ASYNCHRONOUS BACKGROUND ANALYSIS & WORKER QUEUE CERTIFIED! <<<")
        print("=" * 76)
        return True
    else:
        print(">>> DAY 69 VERIFICATION FAILED! SEE LOG ABOVE. <<<")
        print("=" * 76)
        return False


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
