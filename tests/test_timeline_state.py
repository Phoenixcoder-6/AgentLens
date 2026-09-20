"""
tests/test_timeline_state.py  —  Day 30 unit tests

Tests the two new state helpers:
    dashboard.state.get_step_handoff_detail()
    dashboard.state.get_timeline_data()

and supporting private helpers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_trace_json(steps: list[dict]) -> str:
    """Build a minimal trace_json string with given steps."""
    return json.dumps({"steps": steps, "workflow": "test", "run_id": "r1"})


def _make_step(agent: str, step: int, handoff: dict | None = None) -> dict:
    return {
        "agent": agent,
        "step": step,
        "latency_ms": 100.0 * step,
        "tokens_total": 200 * step,
        "status": "SUCCESS",
        "handoff": handoff or {},
    }


# ── Test 1: graceful empty return for unknown run ─────────────────────────────

class TestGetStepHandoffDetailEmpty:
    def test_returns_empty_dict_for_missing_run(self):
        """get_step_handoff_detail should return {} for a run not in DB."""
        from dashboard.state import get_step_handoff_detail

        with patch("dashboard.state.get_trace_steps", return_value=[]):
            result = get_step_handoff_detail("nonexistent-run-id", "researcher")

        assert result == {}

    def test_returns_empty_dict_for_missing_agent(self):
        """get_step_handoff_detail returns {} when agent name not in trace."""
        from dashboard.state import get_step_handoff_detail

        steps = [_make_step("researcher", 1)]
        with patch("dashboard.state.get_trace_steps", return_value=steps):
            result = get_step_handoff_detail("some-run", "verifier")

        assert result == {}


# ── Test 2: get_timeline_data returns [] for unknown run ──────────────────────

class TestGetTimelineDataEmpty:
    def test_empty_for_no_trace(self):
        from dashboard.state import get_timeline_data

        with patch("dashboard.state.get_trace_steps", return_value=[]):
            result = get_timeline_data("no-such-run")

        assert result == []


# ── Test 3: diff correctly identifies added keys ──────────────────────────────

class TestDiffExtractionAdded:
    def test_added_keys_detected(self):
        from dashboard.state import get_step_handoff_detail

        handoff = {
            "input_state": {"topic": "AI safety", "sources": []},
            "output_state": {"topic": "AI safety", "sources": ["arxiv:123", "nat:456"], "summary": "..."},
        }
        steps = [_make_step("researcher", 1, handoff=handoff)]

        with patch("dashboard.state.get_trace_steps", return_value=steps), \
             patch("dashboard.state._analysis_cache", {}):
            result = get_step_handoff_detail("run-1", "researcher")

        assert "summary" in result["diff"]["added"], "summary should be added"
        assert result["diff"]["added"] or result["diff"]["modified"], \
            "sources went from [] to populated, should be in added or modified"
        assert result["agent"] == "researcher"
        assert result["step"] == 1


# ── Test 4: diff correctly identifies dropped keys ────────────────────────────

class TestDiffExtractionDropped:
    def test_dropped_keys_detected(self):
        from dashboard.state import get_step_handoff_detail

        handoff = {
            "input_state": {"topic": "AI", "sources": ["s1", "s2"], "entities": ["e1"]},
            "output_state": {"topic": "AI", "sources": ["s1", "s2"], "entities": []},
        }
        steps = [_make_step("writer", 2, handoff=handoff)]

        with patch("dashboard.state.get_trace_steps", return_value=steps), \
             patch("dashboard.state._analysis_cache", {}):
            result = get_step_handoff_detail("run-2", "writer")

        assert "entities" in result["diff"]["dropped"], \
            "entities went from populated to empty — should be dropped"


# ── Test 5: blamed flag only on Arbiter primary_agent ─────────────────────────

class TestBlamedFlag:
    def test_blamed_true_for_primary_agent(self):
        from dashboard.state import get_step_handoff_detail

        steps = [_make_step("researcher", 1), _make_step("writer", 2)]

        # Simulate analysis cache with writer as primary_agent
        mock_bundle = MagicMock()
        mock_bundle.primary_agent = "writer"
        mock_state = MagicMock()
        mock_state.bundle = mock_bundle

        with patch("dashboard.state.get_trace_steps", return_value=steps), \
             patch("dashboard.state._analysis_cache", {"run-3": mock_state}):
            researcher_detail = get_step_handoff_detail("run-3", "researcher")
            writer_detail = get_step_handoff_detail("run-3", "writer")

        assert researcher_detail["blamed"] is False, "researcher should not be blamed"
        assert writer_detail["blamed"] is True, "writer should be blamed"
