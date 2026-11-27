"""
scripts/verify_day70.py — Day 70 Production SRE, Prometheus Metrics & Deployment Gate
======================================================================================
Phase E: Data Durability, Storage Reliability & SRE Final Release Gate

Checks:
  1. Standard Prometheus Text Exposition Syntax (OpenMetrics / Prometheus 0.0.4 standard).
  2. Live Ingestion Queue Telemetry & Saturation Gauges (queue_depth, dropped, spilled).
  3. Background Worker Queue & Active Job Telemetry (active_jobs, completed_jobs, duration).
  4. HTTP Content-Negotiation & JSON Backward-Compatibility (text/plain vs application/json).
  5. Per-Agent Dimensional Label Resolution (latency seconds and total tokens labeled by agent).
  6. Production Auto-Seed Suppression Guard (disabled in production by default).
  7. Explicit Auto-Seed Override Control & Safe Defaults (AGENTLENS_AUTO_SEED overrides).
  8. Static Analysis (Ruff & Mypy Clean) & Test Suite Execution.
"""

# ruff: noqa: E402

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI
from fastapi.testclient import TestClient

import dashboard.state as _state
from analyzers.worker import get_job_manager
from api.metrics import generate_prometheus_metrics
from api.router import health_router, router
from capture.queue import get_global_queue
from dashboard.app import is_auto_seed_enabled
from schema.models import AgentStep, RunTrace, StepStatus, TokenUsage
from storage.db import DatabaseManager


def _seed_test_run(db: DatabaseManager, run_id: str) -> None:
    step1 = AgentStep(
        run_id=run_id,
        step=1,
        agent="researcher",
        status=StepStatus.SUCCESS,
        output="Market analysis complete for Q3.",
        tokens=TokenUsage(prompt=40, completion=25, total=65),
        latency_ms=120.0,
    )
    step2 = AgentStep(
        run_id=run_id,
        step=2,
        agent="writer",
        status=StepStatus.SUCCESS,
        output="Executive summary generated.",
        tokens=TokenUsage(prompt=60, completion=35, total=95),
        latency_ms=160.0,
    )
    trace = RunTrace(
        run_id=run_id,
        workflow="market_analysis",
        status=StepStatus.SUCCESS,
        total_latency_ms=280.0,
        total_tokens=160,
        steps=[step1, step2],
        expected_output="Market analysis complete for Q3.",
    )
    db.insert_run(
        run_id=run_id,
        workflow=trace.workflow,
        timestamp="2026-11-28T01:00:00Z",
        status=trace.status,
        total_latency_ms=trace.total_latency_ms,
        total_tokens=trace.total_tokens,
        schema_version="1.0",
        trace_json=trace.model_dump_json(),
        expected_output=trace.expected_output,
    )
    for s in [step1, step2]:
        db.insert_step(
            run_id=run_id,
            step=s.step,
            agent=s.agent,
            status=str(s.status),
            latency_ms=s.latency_ms,
            tokens_prompt=s.tokens.prompt,
            tokens_completion=s.tokens.completion,
            tokens_total=s.tokens.total,
            diff_summary="no diff",
            timestamp="2026-11-28T01:00:00Z",
            schema_version="1.0",
        )


def run_checks() -> bool:
    print("=" * 76)
    print("  AGENTLENS v2.1+ — DAY 70 PRODUCTION SRE & PROMETHEUS GATE")
    print("  Phase E: Data Durability, Storage Reliability & SRE Final Release Gate")
    print("=" * 76)
    all_passed = True

    temp_dir = Path(tempfile.mkdtemp(prefix="agentlens_verify_day70_"))
    test_db_path = str(temp_dir / "verify_day70.db")
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
    app.include_router(health_router)
    client = TestClient(app)

    manager = get_job_manager()
    manager.clear()

    queue = get_global_queue()
    queue.clear()

    try:
        # ─────────────────────────────────────────────────────────────────────
        # Check 1: Standard Prometheus Text Exposition Syntax
        # ─────────────────────────────────────────────────────────────────────
        try:
            _seed_test_run(db, "chk1_prom_syntax")
            resp = client.get("/api/metrics?format=prometheus")
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
            assert "text/plain" in resp.headers["content-type"]
            assert "version=0.0.4" in resp.headers["content-type"]

            text = resp.text
            required_metric_families = [
                "agentlens_traces_ingested_total",
                "agentlens_queue_depth",
                "agentlens_dropped_spans_total",
                "agentlens_spilled_traces_total",
                "agentlens_active_jobs",
                "agentlens_completed_jobs_total",
                "agentlens_failed_jobs_total",
                "agentlens_analysis_duration_seconds",
                "agentlens_total_runs",
            ]
            for mf in required_metric_families:
                assert f"# HELP {mf}" in text, f"Missing # HELP for {mf}"
                assert f"# TYPE {mf}" in text, f"Missing # TYPE for {mf}"

            # Also verify root scrape path /metrics
            root_resp = client.get("/metrics")
            assert root_resp.status_code == 200
            assert "text/plain" in root_resp.headers["content-type"]

            print(
                "[PASS] Check 1: Standard Prometheus Text Exposition Syntax (9 Metric Families Verified)"
            )
        except Exception as exc:
            print(f"[FAIL] Check 1: Standard Prometheus Syntax Failed: {exc}")
            all_passed = False

        # ─────────────────────────────────────────────────────────────────────
        # Check 2: Live Ingestion Queue Telemetry & Saturation Gauges
        # ─────────────────────────────────────────────────────────────────────
        try:
            q_stats: dict[str, Any] = queue.stats()
            prom_text = generate_prometheus_metrics()

            assert f"agentlens_queue_depth {q_stats['size']}" in prom_text
            assert f"agentlens_dropped_spans_total {q_stats['dropped']}" in prom_text
            assert f"agentlens_spilled_traces_total {q_stats['spilled']}" in prom_text

            print(
                f"[PASS] Check 2: Live Ingestion Queue Telemetry (Depth: {q_stats['size']}, Dropped: {q_stats['dropped']}, Spilled: {q_stats['spilled']})"
            )
        except Exception as exc:
            print(f"[FAIL] Check 2: Ingestion Queue Telemetry Failed: {exc}")
            all_passed = False

        # ─────────────────────────────────────────────────────────────────────
        # Check 3: Background Worker Queue & Active Job Telemetry
        # ─────────────────────────────────────────────────────────────────────
        try:
            r3 = "chk3_worker_job_run"
            _seed_test_run(db, r3)

            job = manager.submit_job(r3)
            # Should reflect active job or completed job
            prom_active = generate_prometheus_metrics()
            assert "agentlens_active_jobs" in prom_active

            completed_job = manager.wait_for_job(job.job_id, timeout=12.0)
            assert completed_job is not None

            prom_completed = generate_prometheus_metrics()
            assert "agentlens_completed_jobs_total" in prom_completed
            assert "agentlens_analysis_duration_seconds" in prom_completed

            print(
                f"[PASS] Check 3: Background Worker Queue Telemetry (Job: {job.job_id}, Duration: {completed_job.execution_time_ms}ms)"
            )
        except Exception as exc:
            print(f"[FAIL] Check 3: Background Worker Telemetry Failed: {exc}")
            all_passed = False

        # ─────────────────────────────────────────────────────────────────────
        # Check 4: HTTP Content-Negotiation & JSON Backward-Compatibility
        # ─────────────────────────────────────────────────────────────────────
        try:
            # text/plain -> Prometheus
            resp_txt = client.get("/api/metrics", headers={"Accept": "text/plain"})
            assert resp_txt.status_code == 200
            assert "text/plain" in resp_txt.headers["content-type"]
            assert "# TYPE agentlens_traces_ingested_total" in resp_txt.text

            # application/json -> JSON MetricsResponse
            resp_json = client.get("/api/metrics", headers={"Accept": "application/json"})
            assert resp_json.status_code == 200
            assert "application/json" in resp_json.headers["content-type"]
            j_data = resp_json.json()
            assert "agents" in j_data
            assert "total_runs" in j_data
            assert isinstance(j_data["agents"], list)

            # Dedicated /api/metrics/prometheus endpoint
            resp_prom = client.get("/api/metrics/prometheus")
            assert resp_prom.status_code == 200
            assert "text/plain" in resp_prom.headers["content-type"]

            print(
                "[PASS] Check 4: HTTP Content-Negotiation & JSON Backward-Compatibility (text/plain & application/json)"
            )
        except Exception as exc:
            print(f"[FAIL] Check 4: Content Negotiation Failed: {exc}")
            all_passed = False

        # ─────────────────────────────────────────────────────────────────────
        # Check 5: Per-Agent Dimensional Label Resolution
        # ─────────────────────────────────────────────────────────────────────
        try:
            r5 = "chk5_agent_labels_run"
            _seed_test_run(db, r5)

            p_text = generate_prometheus_metrics()
            assert 'agentlens_agent_latency_seconds{agent="researcher"}' in p_text
            assert 'agentlens_agent_latency_seconds{agent="writer"}' in p_text
            assert 'agentlens_agent_tokens_total{agent="researcher"}' in p_text
            assert 'agentlens_agent_tokens_total{agent="writer"}' in p_text

            print(
                "[PASS] Check 5: Per-Agent Dimensional Label Resolution (Researcher & Writer Labeled)"
            )
        except Exception as exc:
            print(f"[FAIL] Check 5: Per-Agent Dimensional Labels Failed: {exc}")
            all_passed = False

        # ─────────────────────────────────────────────────────────────────────
        # Check 6: Production Auto-Seed Suppression Guard
        # ─────────────────────────────────────────────────────────────────────
        try:
            orig_env = os.environ.get("ENVIRONMENT")
            orig_ag_env = os.environ.get("AGENTLENS_ENV")
            orig_seed = os.environ.get("AGENTLENS_AUTO_SEED")

            try:
                # Remove explicit override
                os.environ.pop("AGENTLENS_AUTO_SEED", None)

                os.environ["ENVIRONMENT"] = "production"
                assert is_auto_seed_enabled() is False, (
                    "Auto-seed should be False under ENVIRONMENT=production"
                )

                os.environ.pop("ENVIRONMENT", None)
                os.environ["AGENTLENS_ENV"] = "production"
                assert is_auto_seed_enabled() is False, (
                    "Auto-seed should be False under AGENTLENS_ENV=production"
                )

                os.environ["AGENTLENS_ENV"] = "prod"
                assert is_auto_seed_enabled() is False, (
                    "Auto-seed should be False under AGENTLENS_ENV=prod"
                )
            finally:
                if orig_env is not None:
                    os.environ["ENVIRONMENT"] = orig_env
                else:
                    os.environ.pop("ENVIRONMENT", None)
                if orig_ag_env is not None:
                    os.environ["AGENTLENS_ENV"] = orig_ag_env
                else:
                    os.environ.pop("AGENTLENS_ENV", None)
                if orig_seed is not None:
                    os.environ["AGENTLENS_AUTO_SEED"] = orig_seed
                else:
                    os.environ.pop("AGENTLENS_AUTO_SEED", None)

            print(
                "[PASS] Check 6: Production Auto-Seed Suppression Guard (Default False in Production)"
            )
        except Exception as exc:
            print(f"[FAIL] Check 6: Production Auto-Seed Suppression Failed: {exc}")
            all_passed = False

        # ─────────────────────────────────────────────────────────────────────
        # Check 7: Explicit Auto-Seed Override Control & Safe Defaults
        # ─────────────────────────────────────────────────────────────────────
        try:
            orig_seed = os.environ.get("AGENTLENS_AUTO_SEED")
            orig_env = os.environ.get("ENVIRONMENT")
            try:
                os.environ["ENVIRONMENT"] = "production"
                os.environ["AGENTLENS_AUTO_SEED"] = "1"
                assert is_auto_seed_enabled() is True, (
                    "Explicit AGENTLENS_AUTO_SEED=1 should enable seeding"
                )

                os.environ["ENVIRONMENT"] = "development"
                os.environ["AGENTLENS_AUTO_SEED"] = "0"
                assert is_auto_seed_enabled() is False, (
                    "Explicit AGENTLENS_AUTO_SEED=0 should suppress seeding"
                )

                os.environ["AGENTLENS_AUTO_SEED"] = "false"
                assert is_auto_seed_enabled() is False, (
                    "Explicit AGENTLENS_AUTO_SEED=false should suppress seeding"
                )
            finally:
                if orig_seed is not None:
                    os.environ["AGENTLENS_AUTO_SEED"] = orig_seed
                else:
                    os.environ.pop("AGENTLENS_AUTO_SEED", None)
                if orig_env is not None:
                    os.environ["ENVIRONMENT"] = orig_env
                else:
                    os.environ.pop("ENVIRONMENT", None)

            print(
                "[PASS] Check 7: Explicit Auto-Seed Override Control & Safe Defaults (0, false, 1 overrides verified)"
            )
        except Exception as exc:
            print(f"[FAIL] Check 7: Explicit Auto-Seed Override Failed: {exc}")
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
                "api/metrics.py",
                "api/router.py",
                "dashboard/app.py",
                "tests/test_sre_metrics.py",
                "scripts/verify_day70.py",
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

            # 4. Pytest SRE test suite
            r_pytest = subprocess.run(
                [python_bin, "-m", "pytest", "tests/test_sre_metrics.py", "-v"],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
                env=sub_env,
            )
            assert r_pytest.returncode == 0, f"Pytest failed:\n{r_pytest.stdout}\n{r_pytest.stderr}"

            print("[PASS] Check 8: Static Analysis (Ruff & Mypy Clean) & SRE Test Suite Execution")
        except Exception as exc:
            print(f"[FAIL] Check 8: Static Analysis & Test Suite Execution Failed: {exc}")
            all_passed = False

    finally:
        # Restore state
        _state._db = orig_db
        _state.get_db = orig_get_db
        _state._analysis_cache.clear()
        manager.clear()
        queue.clear()
        db.close()
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("=" * 76)
    if all_passed:
        print(">>> ALL 8 DAY 70 VERIFICATION CHECKS PASSED SUCCESSFULLY! <<<")
        print(">>> PHASE E: DATA DURABILITY, STORAGE RELIABILITY & SRE CERTIFIED COMPLETE! <<<")
        print("=" * 76)
        return True
    else:
        print(">>> DAY 70 VERIFICATION FAILED! SEE LOG ABOVE. <<<")
        print("=" * 76)
        return False


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
