"""
scripts/verify_day39.py — Day 39 Verification Script
=====================================================
Verifies all Day 39 deliverables:
  1. config/config.yaml contains capture.fail_safe, capture.retention_days=90,
     and capture.pii_scrubbing with regex patterns
  2. capture/pii_scrubber.py exists and redacts email, phone, SSN, API key, credit card
  3. PIIScrubber scrubs RunTrace / AgentStep handoff states and handles spaCy fallback
  4. Fail-safe capture: @trace_step and CaptureSession never crash the agent when
     capture/storage raises an exception
  5. scripts/cleanup_old_traces.py exists and purges expired JSON + SQLite traces
  6. DatabaseManager.delete_runs_older_than() works and table_counts() preserves 5 keys
  7. README.md documents PII warning ("Traces contain full LLM I/O...") and retention CLI
  8. Unit & integration test suite (tests/test_fail_safe_and_pii.py + tests/test_capture.py) passes
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _check(num: int, title: str, fn) -> bool:
    try:
        detail = fn()
        print(f"  [PASS] Check {num}: {title} -- {detail}")
        return True
    except Exception as exc:
        print(f"  [FAIL] Check {num}: {title} -- {exc}")
        return False


def check_1_config() -> str:
    from config.config_loader import load_config

    load_config.cache_clear()
    cfg = load_config()
    cap = cfg.get("capture", {})
    assert cap.get("fail_safe") is True, "capture.fail_safe must be True"
    assert cap.get("retention_days") == 90, (
        f"Expected retention_days=90, got {cap.get('retention_days')}"
    )
    pii = cap.get("pii_scrubbing", {})
    assert "enabled" in pii, "capture.pii_scrubbing.enabled missing"
    patterns = pii.get("patterns", {})
    for key in ("email", "phone", "ssn", "api_key", "credit_card"):
        assert key in patterns, f"Missing PII pattern '{key}' in config.yaml"
    return f"retention_days={cap['retention_days']}, patterns={list(patterns.keys())}"


def check_2_pii_scrubber_regex() -> str:
    from capture.pii_scrubber import PIIScrubber

    scrubber = PIIScrubber(enabled=True, use_spacy_ner=False)
    sample = (
        "Email user@domain.com, phone (415) 555-2671, SSN 123-45-6789, "
        "key gsk_1234567890abcdefghij, card 4111-2222-3333-4444"
    )
    out = scrubber.scrub_text(sample)
    for token in (
        "[REDACTED_EMAIL]",
        "[REDACTED_PHONE]",
        "[REDACTED_SSN]",
        "[REDACTED_API_KEY]",
        "[REDACTED_CREDIT_CARD]",
    ):
        assert token in out, f"Expected {token} in '{out}'"
    assert "user@domain.com" not in out
    assert "123-45-6789" not in out
    return "All 5 PII pattern categories redacted"


def check_3_pii_scrubber_trace_and_fallback() -> str:
    from capture.pii_scrubber import PIIScrubber
    from schema.models import AgentStep, HandoffState, RunTrace, StepStatus

    scrubber = PIIScrubber(enabled=True, use_spacy_ner=True)
    step = AgentStep(
        run_id="run_verify_39",
        step=1,
        agent="researcher",
        input='{"email": "hidden@test.io"}',
        output='{"ssn": "999-11-2222"}',
        handoff=HandoffState(
            input_state={"email": "hidden@test.io"},
            filtered_state={"ssn": "999-11-2222"},
            output_state={"email": "hidden@test.io", "ssn": "999-11-2222"},
        ),
        timestamp=datetime.now(UTC),
    )
    trace = RunTrace(
        run_id="run_verify_39",
        workflow="verify",
        timestamp=datetime.now(UTC),
        status=StepStatus.SUCCESS,
        steps=[step],
    )
    scrubber.scrub_trace(trace)
    assert step.handoff.input_state["email"] == "[REDACTED_EMAIL]"
    assert step.handoff.filtered_state["ssn"] == "[REDACTED_SSN]"
    return "RunTrace + HandoffState scrubbed with graceful NER fallback"


def check_4_fail_safe_capture() -> str:
    from capture.handoff import HandoffCapture
    from capture.session import CaptureSession
    from capture.tracer import trace_step

    orig_finalize = HandoffCapture.finalize
    orig_save_disk = CaptureSession._save_trace_to_disk
    orig_save_db = CaptureSession._save_trace_to_db
    try:

        def _boom(self):
            raise RuntimeError("Forced capture failure")

        HandoffCapture.finalize = _boom  # type: ignore[method-assign]
        CaptureSession._save_trace_to_disk = classmethod(  # type: ignore[method-assign]
            lambda cls, tr: (_ for _ in ()).throw(OSError("Disk error"))
        )
        CaptureSession._save_trace_to_db = classmethod(  # type: ignore[method-assign]
            lambda cls, tr: (_ for _ in ()).throw(RuntimeError("DB error"))
        )

        CaptureSession.start_trace(workflow="verify_fail_safe")

        @trace_step
        def healthy_agent_node(state):
            return {"status": "completed", "value": state["x"] + 10}

        res = healthy_agent_node({"x": 5})
        assert res == {"status": "completed", "value": 15}
        CaptureSession.end_trace()
        assert CaptureSession.get_current_trace() is None
    finally:
        HandoffCapture.finalize = orig_finalize  # type: ignore[method-assign]
        CaptureSession._save_trace_to_disk = orig_save_disk  # type: ignore[method-assign]
        CaptureSession._save_trace_to_db = orig_save_db  # type: ignore[method-assign]

    return "Agent returned expected result despite forced HandoffCapture/Disk/DB errors"


def check_5_cleanup_utility() -> str:
    from normalizer.normalizer import Normalizer
    from schema.models import RunTrace, StepStatus
    from scripts.cleanup_old_traces import cleanup_old_traces
    from storage.db import DatabaseManager
    from storage.writer import StorageWriter

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        traces_dir = tmp / "traces"
        traces_dir.mkdir()
        db_path = tmp / "test.db"
        db = DatabaseManager(str(db_path))
        db.initialize()
        writer = StorageWriter(db)

        now = datetime.now(UTC)
        for rid, age_days in [("run_expired", 120), ("run_fresh", 10)]:
            ts = now - timedelta(days=age_days)
            tr = RunTrace(run_id=rid, workflow="test", timestamp=ts, status=StepStatus.SUCCESS)
            tfile = traces_dir / f"{rid}.json"
            tjson = tr.model_dump_json(indent=2)
            tfile.write_text(tjson, encoding="utf-8")
            writer.write_run(
                Normalizer().normalize_run(tr), trace_json=tjson, trace_path=str(tfile)
            )

        summary = cleanup_old_traces(
            retention_days=90,
            db_path=str(db_path),
            traces_dir=str(traces_dir),
            dry_run=False,
            now=now,
        )
        assert summary["deleted_files_count"] == 1
        assert summary["deleted_db_runs_count"] == 1
        assert not (traces_dir / "run_expired.json").exists()
        assert (traces_dir / "run_fresh.json").exists()
        assert db.get_run("run_expired") is None
        assert db.get_run("run_fresh") is not None

    return "120-day trace purged and 10-day trace preserved across JSON + SQLite"


def check_6_db_manager_compat() -> str:
    from storage.db import DatabaseManager

    with tempfile.TemporaryDirectory() as tmpdir:
        db = DatabaseManager(str(Path(tmpdir) / "compat.db"))
        db.initialize()
        counts = db.table_counts()
        expected_keys = {"runs", "steps", "analysis", "metrics", "llm_cache"}
        assert set(counts.keys()) == expected_keys, f"Unexpected keys: {set(counts.keys())}"
        assert hasattr(db, "delete_runs_older_than")
    return f"table_counts() keys preserved ({sorted(expected_keys)})"


def check_7_readme_privacy_warning() -> str:
    readme_text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "Traces contain full LLM I/O" in readme_text, "Missing LLM I/O warning in README.md"
    assert "pii_scrubbing" in readme_text, "Missing pii_scrubbing reference in README.md"
    assert "cleanup_old_traces.py" in readme_text, "Missing cleanup_old_traces.py in README.md"
    return "README.md contains privacy warning and cleanup_old_traces.py instructions"


def check_8_pytest_suite() -> str:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_fail_safe_and_pii.py",
            "tests/test_capture.py",
            "-q",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"pytest failed:\n{proc.stdout}\n{proc.stderr}")
    last_line = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()][-1]
    return last_line


def main() -> int:
    print("=" * 70)
    print("Day 39 Verification: Fail-Safe Capture + PII Scrubbing + Retention")
    print("=" * 70)

    checks = [
        (1, "config/config.yaml capture & PII settings", check_1_config),
        (2, "PIIScrubber regex redaction (5 pattern categories)", check_2_pii_scrubber_regex),
        (
            3,
            "PIIScrubber RunTrace/HandoffState & spaCy fallback",
            check_3_pii_scrubber_trace_and_fallback,
        ),
        (4, "Fail-safe capture under forced exceptions", check_4_fail_safe_capture),
        (5, "scripts/cleanup_old_traces.py retention purge", check_5_cleanup_utility),
        (6, "DatabaseManager retention method & table_counts()", check_6_db_manager_compat),
        (7, "README.md privacy warning & retention docs", check_7_readme_privacy_warning),
        (8, "Pytest suite (test_fail_safe_and_pii + test_capture)", check_8_pytest_suite),
    ]

    passed = sum(1 for num, title, fn in checks if _check(num, title, fn))
    print("-" * 70)
    print(f"Result: {passed}/{len(checks)} checks passed.")
    print("=" * 70)
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
