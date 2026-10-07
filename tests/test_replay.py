"""
tests/test_replay.py — Day 37: Replay CLI tests
===============================================
Verifies:
  - replay.py CLI flags: --json, --dry-run, --override-topic
  - Semantic exit codes: 0 (PASS), 1 (WARNING), 2 (FAIL), 3 (ERROR)
  - Topic extraction and topic override attribution stability
  - Live pipeline replay and error handling (mocked, no live LLM calls)
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import replay
from replay import (
    EXIT_ERROR,
    EXIT_FAIL,
    EXIT_PASS,
    EXIT_WARNING,
    apply_topic_override,
    execute_replay,
    extract_topic_from_trace,
    load_trace_by_id,
    main,
)


class TestReplayExitCodes:
    """Verify exit codes 0 (PASS), 1 (WARNING), 2 (FAIL), and 3 (ERROR)."""

    def test_pass_run_returns_exit_0(self) -> None:
        payload, code = execute_replay("run_lbl_pass_01", dry_run=True)
        assert code == EXIT_PASS == 0
        assert payload["verdict"] == "PASS"
        assert payload["priority_level"] == "P5"
        assert payload["primary_agent"] is None
        assert payload["rules_fired"] == []

    def test_workflow_violation_returns_exit_1_warning(self) -> None:
        payload, code = execute_replay("run_lbl_workflow_01", dry_run=True)
        assert code == EXIT_WARNING == 1
        assert payload["verdict"] == "WARNING"
        assert payload["priority_level"] == "P3"
        assert payload["primary_cause"] == "workflow"
        assert payload["primary_agent"] == "verifier"

    def test_execution_failure_returns_exit_2_fail(self) -> None:
        payload, code = execute_replay("run_lbl_execution_01", dry_run=True)
        assert code == EXIT_FAIL == 2
        assert payload["verdict"] == "FAIL"
        assert payload["priority_level"] == "P2"
        assert payload["primary_cause"] == "execution"
        assert payload["primary_agent"] == "researcher"
        assert "tool_failure_v1" in payload["rules_fired"]

    def test_missing_run_id_returns_exit_3_error(self) -> None:
        payload, code = execute_replay("run_does_not_exist_999", dry_run=True)
        assert code == EXIT_ERROR == 3
        assert payload["verdict"] == "ERROR"
        assert "not found" in payload["error"].lower()


class TestReplayFlags:
    """Verify --json, --dry-run, and --override-topic CLI flags."""

    def test_json_flag_outputs_machine_readable_json(self, capsys) -> None:
        code = main(["run_lbl_pass_01", "--dry-run", "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)

        assert code == EXIT_PASS
        assert data["run_id"] == "run_lbl_pass_01"
        assert data["verdict"] == "PASS"
        assert data["exit_code"] == 0
        assert data["dry_run"] is True
        assert data["original_topic"] == "History of the Eiffel Tower"

    def test_human_readable_output_in_dry_run(self, capsys) -> None:
        code = main(["run_lbl_reasoning_01", "--dry-run"])
        captured = capsys.readouterr()

        assert code == EXIT_FAIL
        assert "DRY-RUN (no LLM calls)" in captured.out
        assert "run_lbl_reasoning_01" in captured.out
        assert "Rise of AI in America" in captured.out

    def test_override_topic_preserves_attribution_stability(self) -> None:
        payload, code = execute_replay(
            "run_lbl_reasoning_01",
            dry_run=True,
            override_topic="Solid-state lithium-metal batteries",
        )
        assert code == EXIT_FAIL
        assert payload["topic_overridden"] is True
        assert payload["original_topic"] == "Rise of AI in America"
        assert payload["effective_topic"] == "Solid-state lithium-metal batteries"
        assert payload["primary_cause"] == "reasoning"
        assert payload["primary_agent"] == "writer"

    def test_apply_topic_override_updates_trace_steps(self) -> None:
        trace = load_trace_by_id("run_lbl_pass_01")
        assert trace is not None
        assert extract_topic_from_trace(trace) == "History of the Eiffel Tower"

        overridden = apply_topic_override(trace, "CRISPR gene editing")
        assert extract_topic_from_trace(overridden) == "CRISPR gene editing"


class TestReplayLiveModeMocked:
    """Verify live replay path with mocked app.pipeline.run_pipeline."""

    def test_live_replay_invokes_run_pipeline_when_not_dry_run(self) -> None:
        fake_state = {
            "topic": "New live topic",
            "research_findings": "SOURCES:\n- S1\nENTITIES:\n- E1\nKEY FINDINGS:\nFacts",
            "source_count": 5,
            "entity_count": 6,
            "written_report": "Report text",
            "verification_result": "APPROVED",
            "verified": True,
            "revision_notes": "",
        }
        with (
            patch.dict(os.environ, {"GROQ_API_KEY": "gsk_test_live_key"}, clear=False),
            patch("app.pipeline.run_pipeline", return_value=fake_state) as mock_run,
        ):
            payload, code = execute_replay(
                "run_lbl_pass_01",
                dry_run=False,
                override_topic="New live topic",
            )

        mock_run.assert_called_once_with(topic="New live topic")
        assert code == EXIT_PASS
        assert payload["replayed_live"] is True
        assert payload["verdict"] == "PASS"

    def test_live_replay_pipeline_error_returns_exit_3(self) -> None:
        with (
            patch.dict(os.environ, {"GROQ_API_KEY": "gsk_test_live_key"}, clear=False),
            patch("app.pipeline.run_pipeline", side_effect=RuntimeError("Groq timeout")),
        ):
            payload, code = execute_replay("run_lbl_pass_01", dry_run=False)

        assert code == EXIT_ERROR
        assert payload["verdict"] == "ERROR"
        assert "Groq timeout" in payload["error"]

    def test_error_human_output_printed_on_missing_run(self, capsys) -> None:
        code = replay.main(["missing_run_id_404", "--dry-run"])
        captured = capsys.readouterr()
        assert code == EXIT_ERROR
        assert "AgentLens Replay -- ERROR" in captured.out
