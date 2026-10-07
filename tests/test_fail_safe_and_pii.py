"""
tests/test_fail_safe_and_pii.py — Day 39 Unit & Integration Tests
==================================================================
Covers:
    1. Fail-Safe Capture: forcing exceptions inside CaptureSession, HandoffCapture,
       @trace_step, disk writer, and StorageWriter never crashes or alters the
       underlying multi-agent pipeline execution.
    2. Config-driven PII Scrubber (capture/pii_scrubber.py): regex redaction
       (email, phone, SSN, API key, credit card), recursive dict/list/AgentStep/RunTrace
       scrubbing, optional spaCy NER scrubbing, and graceful fallback.
    3. Trace Retention Policy (scripts/cleanup_old_traces.py + DatabaseManager):
       deleting JSON trace files and cascading SQLite records older than retention_days.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import capture.pii_scrubber as pii_module
from app.pipeline import build_pipeline
from capture.handoff import HandoffCapture
from capture.pii_scrubber import PIIScrubber
from capture.session import CaptureSession
from capture.tracer import trace_step
from normalizer.normalizer import Normalizer
from schema.models import AgentStep, HandoffState, RunTrace, StepStatus
from scripts.cleanup_old_traces import cleanup_old_traces
from scripts.cleanup_old_traces import main as cleanup_main
from storage.db import DatabaseManager
from storage.writer import StorageWriter

# ─────────────────────────────────────────────────────────────────────────────
# 1. Fail-Safe Capture Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_fail_safe_pipeline_completes_when_handoff_finalize_crashes(monkeypatch, tmp_path):
    """
    Integration test: force HandoffCapture.finalize() to raise RuntimeError
    during full 3-agent LangGraph pipeline execution -> pipeline still completes
    normally and returns the exact expected output.
    """
    fake_llm = MagicMock()
    fake_llm.invoke.side_effect = [
        SimpleNamespace(
            content=(
                "SOURCES:\n1. Source A - https://a.example.com\n\n"
                "ENTITIES:\n- EntityAlpha\n\n"
                "KEY FINDINGS:\n- Finding 1"
            ),
            usage_metadata={"input_tokens": 10, "output_tokens": 20},
        ),
        SimpleNamespace(
            content="Introduction\nReport referencing Source A and EntityAlpha.",
            usage_metadata={"input_tokens": 15, "output_tokens": 25},
        ),
        SimpleNamespace(
            content="APPROVED: Report covers all sources and entities.",
            usage_metadata={"input_tokens": 12, "output_tokens": 8},
        ),
    ]
    monkeypatch.setattr("app.pipeline._build_llm", lambda: fake_llm)

    def _exploding_finalize(self):
        raise RuntimeError("Simulated HandoffCapture.finalize explosion")

    monkeypatch.setattr(HandoffCapture, "finalize", _exploding_finalize)
    monkeypatch.setattr(
        CaptureSession,
        "_save_trace_to_disk",
        classmethod(lambda cls, trace: (_ for _ in ()).throw(OSError("Disk full"))),
    )
    monkeypatch.setattr(
        CaptureSession,
        "_save_trace_to_db",
        classmethod(lambda cls, trace: (_ for _ in ()).throw(RuntimeError("DB offline"))),
    )

    CaptureSession.start_trace(workflow="fail_safe_integration_test")
    app = build_pipeline()
    final_state = app.invoke(
        {
            "topic": "Fail-safe capture test",
            "research_findings": "",
            "source_count": 0,
            "entity_count": 0,
            "written_report": "",
            "verification_result": "",
            "verified": False,
            "revision_notes": "",
        }
    )
    trace = CaptureSession.end_trace()

    # Underlying agent pipeline completed and returned full expected output
    assert final_state["verified"] is True
    assert final_state["verification_result"].startswith("APPROVED:")
    assert "EntityAlpha" in final_state["written_report"]
    assert trace is not None
    assert CaptureSession.get_current_trace() is None


def test_fail_safe_pre_execution_capture_exception_does_not_crash_agent(monkeypatch):
    """If pre-execution setup in @trace_step raises, the wrapped agent still runs."""
    CaptureSession.start_trace(workflow="pre_exec_crash")

    def _broken_init(self, input_state):
        raise RuntimeError("HandoffCapture init crashed")

    monkeypatch.setattr(HandoffCapture, "__init__", _broken_init)

    @trace_step
    def sample_agent_node(state):
        return {"answer": state["x"] * 2}

    out = sample_agent_node({"x": 21})
    assert out == {"answer": 42}
    CaptureSession.end_trace()


def test_fail_safe_error_path_preserves_original_agent_exception(monkeypatch):
    """
    If the agent raises ValueError AND the capture layer crashes while recording
    that error, the original ValueError must still propagate unchanged.
    """
    CaptureSession.start_trace(workflow="error_path_crash")

    def _exploding_add_step(cls, step):
        raise RuntimeError("CaptureSession.add_step crashed")

    monkeypatch.setattr(CaptureSession, "add_step", classmethod(_exploding_add_step))

    @trace_step
    def failing_agent_node(state):
        raise ValueError("Original agent failure")

    with pytest.raises(ValueError, match="Original agent failure"):
        failing_agent_node({"input": "test"})

    CaptureSession.end_trace()


def test_fail_safe_disk_and_db_failures_in_end_trace(monkeypatch):
    """
    If os.makedirs / open fails in _save_trace_to_disk and DatabaseManager fails
    in _save_trace_to_db, end_trace() catches both and resets _current_trace.
    """
    CaptureSession.start_trace(workflow="storage_failure_test")

    @trace_step
    def ok_node(state):
        return {"res": "ok"}

    assert ok_node({"in": "val"}) == {"res": "ok"}

    monkeypatch.setattr("os.makedirs", lambda *a, **kw: (_ for _ in ()).throw(OSError("Read-only")))
    monkeypatch.setattr(
        "storage.db.DatabaseManager.initialize",
        lambda self: (_ for _ in ()).throw(RuntimeError("SQLite locked")),
    )

    trace = CaptureSession.end_trace()
    assert trace is not None
    assert CaptureSession.get_current_trace() is None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Config-Driven PII Scrubber Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_pii_scrubber_disabled_leaves_data_unchanged():
    scrubber = PIIScrubber(enabled=False)
    raw = "Contact alice@example.com or call (415) 555-0199, SSN 123-45-6789."
    assert scrubber.scrub_text(raw) == raw
    assert scrubber.scrub_value({"msg": raw}) == {"msg": raw}


def test_pii_scrubber_redacts_email_phone_ssn_apikey_and_credit_card():
    scrubber = PIIScrubber(enabled=True, use_spacy_ner=False)
    raw = (
        "Email: john.doe+test@company.co.uk | "
        "Phone: +1 (800) 555-1234 | "
        "SSN: 123-45-6789 | "
        "Key: gsk_abcdefghijklmnopqrstuvwxyz123456 | "
        "Card: 4111-2222-3333-4444"
    )
    scrubbed = scrubber.scrub_text(raw)

    assert "john.doe+test@company.co.uk" not in scrubbed
    assert "[REDACTED_EMAIL]" in scrubbed
    assert "555-1234" not in scrubbed
    assert "[REDACTED_PHONE]" in scrubbed
    assert "123-45-6789" not in scrubbed
    assert "[REDACTED_SSN]" in scrubbed
    assert "gsk_abcdefghijklmnopqrstuvwxyz123456" not in scrubbed
    assert "[REDACTED_API_KEY]" in scrubbed
    assert "4111-2222-3333-4444" not in scrubbed
    assert "[REDACTED_CREDIT_CARD]" in scrubbed


def test_pii_scrubber_scrubs_trace_and_handoff_states():
    scrubber = PIIScrubber(enabled=True, use_spacy_ner=False)
    step = AgentStep(
        run_id="run_pii_01",
        step=1,
        agent="researcher",
        input='{"user_email": "secret@corp.org", "ssn": "999-88-7777"}',
        output='{"summary": "Called 415-555-9876 for secret@corp.org"}',
        prompt="diff for secret@corp.org",
        error="Error contacting secret@corp.org",
        handoff=HandoffState(
            input_state={"email": "secret@corp.org", "nested": ["999-88-7777"]},
            filtered_state={"phone": "(415) 555-9876"},
            output_state={"email": "secret@corp.org", "phone": "(415) 555-9876"},
        ),
        timestamp=datetime.now(UTC),
    )
    trace = RunTrace(
        run_id="run_pii_01",
        workflow="pii_test",
        timestamp=datetime.now(UTC),
        status=StepStatus.SUCCESS,
        steps=[step],
        expected_output="Expected reply to secret@corp.org",
    )

    scrubber.scrub_trace(trace)

    assert "secret@corp.org" not in trace.expected_output
    assert "[REDACTED_EMAIL]" in trace.expected_output
    assert "[REDACTED_EMAIL]" in step.input
    assert "[REDACTED_SSN]" in step.input
    assert "[REDACTED_PHONE]" in step.output
    assert step.handoff.input_state["email"] == "[REDACTED_EMAIL]"
    assert step.handoff.input_state["nested"] == ["[REDACTED_SSN]"]
    assert step.handoff.filtered_state["phone"] == "[REDACTED_PHONE]"


def test_pii_scrubber_spacy_ner_integration_and_fallback(monkeypatch):
    """Verify optional spaCy NER scrubbing when available and fallback when unavailable."""
    # 1. Mock spaCy NLP doc with PERSON and ORG entities
    text = "Alice Smith works at Acme Corp in London."
    fake_ents = [
        SimpleNamespace(start_char=0, end_char=11, label_="PERSON"),
        SimpleNamespace(start_char=21, end_char=30, label_="ORG"),
        SimpleNamespace(start_char=34, end_char=40, label_="GPE"),
    ]
    monkeypatch.setattr(
        pii_module, "_get_spacy_nlp", lambda: lambda _t: SimpleNamespace(ents=fake_ents)
    )

    scrubber = PIIScrubber(enabled=True, use_spacy_ner=True)
    redacted = scrubber.scrub_text(text)
    assert "[REDACTED_PERSON]" in redacted
    assert "[REDACTED_ORG]" in redacted
    assert "[REDACTED_LOCATION]" in redacted
    assert "Alice Smith" not in redacted

    # 2. Fallback when spaCy is unavailable (returns None)
    monkeypatch.setattr(pii_module, "_get_spacy_nlp", lambda: None)
    fallback_text = scrubber.scrub_text("Reach Alice at alice@acme.com")
    assert fallback_text == "Reach Alice at [REDACTED_EMAIL]"


def test_capture_session_scrubs_pii_when_enabled_in_config(monkeypatch, tmp_path):
    """When capture.pii_scrubbing.enabled is True, CaptureSession redacts PII before disk/DB write."""
    traces_dir = tmp_path / "traces"
    db_path = tmp_path / "pii_test.db"

    def _fake_get(section, key=None, default=None):
        if section == "capture" and key == "pii_scrubbing":
            return {"enabled": True, "use_spacy_ner": False}
        if section == "storage" and key == "traces_dir":
            return str(traces_dir)
        return default

    monkeypatch.setattr("capture.pii_scrubber.get", _fake_get)
    monkeypatch.setattr("config.config_loader.get", _fake_get)
    monkeypatch.setattr("storage.db.DEFAULT_DB_PATH", str(db_path))

    CaptureSession.start_trace(workflow="pii_session_test", run_id="run_pii_session")

    @trace_step
    def researcher_node(state):
        return {"findings": "Found user leak: bob@private.io and SSN 111-22-3333"}

    researcher_node({"topic": "Query from bob@private.io"})
    trace = CaptureSession.end_trace()

    assert trace is not None
    saved_json = (traces_dir / "run_pii_session.json").read_text(encoding="utf-8")
    assert "bob@private.io" not in saved_json
    assert "111-22-3333" not in saved_json
    assert "[REDACTED_EMAIL]" in saved_json
    assert "[REDACTED_SSN]" in saved_json


# ─────────────────────────────────────────────────────────────────────────────
# 3. Trace Retention Policy Tests
# ─────────────────────────────────────────────────────────────────────────────


def _seed_trace_and_db(
    db: DatabaseManager,
    traces_dir: Path,
    run_id: str,
    timestamp: datetime,
) -> Path:
    trace = RunTrace(
        run_id=run_id,
        workflow="retention_test",
        timestamp=timestamp,
        status=StepStatus.SUCCESS,
        steps=[
            AgentStep(
                run_id=run_id,
                step=1,
                agent="researcher",
                input='{"topic": "test"}',
                output='{"findings": "ok"}',
                latency_ms=100.0,
                status=StepStatus.SUCCESS,
                timestamp=timestamp,
            )
        ],
    )
    traces_dir.mkdir(parents=True, exist_ok=True)
    trace_file = traces_dir / f"{run_id}.json"
    trace_json = trace.model_dump_json(indent=2)
    trace_file.write_text(trace_json, encoding="utf-8")

    normalized = Normalizer().normalize_run(trace)
    writer = StorageWriter(db)
    writer.write_run(normalized, trace_json=trace_json, trace_path=str(trace_file))
    db.insert_analysis(
        run_id=run_id,
        analyzer="arbiter",
        timestamp=timestamp.isoformat(),
        schema_version="1.0",
        verdict="PASS",
    )
    db.insert_rule_matches(
        run_id=run_id,
        matches=[{"rule_id": "test_rule", "category": "reasoning", "severity": "LOW"}],
    )
    return trace_file


def test_cleanup_old_traces_dry_run_and_live_execution(tmp_path):
    traces_dir = tmp_path / "traces"
    db_path = tmp_path / "retention.db"
    db = DatabaseManager(str(db_path))
    db.initialize()

    now = datetime(2026, 10, 8, 0, 0, 0, tzinfo=UTC)
    old_ts = now - timedelta(days=120)
    recent_ts = now - timedelta(days=15)

    old_file = _seed_trace_and_db(db, traces_dir, "run_old_120d", old_ts)
    recent_file = _seed_trace_and_db(db, traces_dir, "run_recent_15d", recent_ts)

    # 1. Dry-run: reports 1 file + 1 DB run to delete, but keeps both intact
    dry_summary = cleanup_old_traces(
        retention_days=90,
        db_path=str(db_path),
        traces_dir=str(traces_dir),
        dry_run=True,
        now=now,
    )
    assert dry_summary["deleted_files_count"] == 1
    assert dry_summary["deleted_db_runs_count"] == 1
    assert old_file.exists()
    assert recent_file.exists()
    assert db.get_run("run_old_120d") is not None

    # 2. Live execution: deletes old file and old DB run (plus child tables), keeps recent
    live_summary = cleanup_old_traces(
        retention_days=90,
        db_path=str(db_path),
        traces_dir=str(traces_dir),
        dry_run=False,
        now=now,
    )
    assert live_summary["deleted_files_count"] == 1
    assert live_summary["deleted_db_runs_count"] == 1
    assert not old_file.exists()
    assert recent_file.exists()
    assert db.get_run("run_old_120d") is None
    assert db.get_steps_for_run("run_old_120d") == []
    assert db.get_metrics_for_run("run_old_120d") == []
    assert db.get_run("run_recent_15d") is not None
    assert len(db.get_steps_for_run("run_recent_15d")) == 1


def test_cleanup_cli_main_json_output(tmp_path, capsys):
    traces_dir = tmp_path / "traces"
    db_path = tmp_path / "cli_retention.db"
    db = DatabaseManager(str(db_path))
    db.initialize()

    old_ts = datetime.now(UTC) - timedelta(days=100)
    _seed_trace_and_db(db, traces_dir, "run_cli_old", old_ts)

    rc = cleanup_main(
        [
            "--days",
            "90",
            "--db-path",
            str(db_path),
            "--traces-dir",
            str(traces_dir),
            "--dry-run",
            "--json",
        ]
    )
    assert rc == 0
    captured = json.loads(capsys.readouterr().out)
    assert captured["retention_days"] == 90
    assert captured["dry_run"] is True
    assert captured["deleted_files_count"] == 1
    assert captured["deleted_db_runs_count"] == 1
