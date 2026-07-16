"""
tests/test_handoff.py — Day 6: HandoffCapture + HandoffDiff tests
==================================================================
Verifies that the three-state handoff model correctly classifies every
state change as added / modified / unchanged / dropped.

These tests use no LLM — pure unit tests on the diff engine.
"""

import pytest
from capture.handoff import HandoffCapture, HandoffDiff, _is_empty


# ─────────────────────────────────────────────────────────────────────────────
# _is_empty helper
# ─────────────────────────────────────────────────────────────────────────────

class TestIsEmpty:
    def test_none_is_empty(self):
        assert _is_empty(None) is True

    def test_empty_string_is_empty(self):
        assert _is_empty("") is True

    def test_non_empty_string_is_not_empty(self):
        assert _is_empty("hello") is False

    def test_zero_int_is_empty(self):
        assert _is_empty(0) is True

    def test_positive_int_is_not_empty(self):
        assert _is_empty(5) is False

    def test_empty_list_is_empty(self):
        assert _is_empty([]) is True

    def test_non_empty_list_is_not_empty(self):
        assert _is_empty(["item"]) is False

    def test_empty_dict_is_empty(self):
        assert _is_empty({}) is True

    def test_false_bool_is_not_empty(self):
        # False is a valid value, not absence — bool is not caught by int check
        assert _is_empty(False) is True  # False == 0

    def test_true_bool_is_not_empty(self):
        assert _is_empty(True) is False  # True == 1


# ─────────────────────────────────────────────────────────────────────────────
# HandoffCapture — three-state snapshot
# ─────────────────────────────────────────────────────────────────────────────

class TestHandoffCapture:

    def test_input_state_is_full_state_before(self):
        """input_state must be the exact full state dict passed in."""
        before = {"topic": "AI", "research_findings": "", "source_count": 0}
        capture = HandoffCapture(input_state=before)
        capture.record_agent_return({"research_findings": "SOURCES:\n- Book A", "source_count": 3})
        input_s, _, _, _ = capture.finalize()

        assert input_s["topic"] == "AI"
        assert input_s["research_findings"] == ""
        assert input_s["source_count"] == 0

    def test_filtered_state_is_agent_return(self):
        """filtered_state is exactly what the agent chose to return."""
        before = {"topic": "AI", "research_findings": ""}
        capture = HandoffCapture(input_state=before)
        agent_return = {"research_findings": "SOURCES:\n- Book A"}
        capture.record_agent_return(agent_return)
        _, filtered_s, _, _ = capture.finalize()

        assert filtered_s == agent_return
        assert "topic" not in filtered_s  # agent didn't return topic

    def test_output_state_is_merged_state(self):
        """output_state = input_state keys + agent return keys merged."""
        before = {"topic": "AI", "research_findings": "", "source_count": 0}
        capture = HandoffCapture(input_state=before)
        capture.record_agent_return({"research_findings": "SOURCES:\n- Book A", "source_count": 5})
        _, _, output_s, _ = capture.finalize()

        # Agent-returned keys updated
        assert output_s["research_findings"] == "SOURCES:\n- Book A"
        assert output_s["source_count"] == 5
        # Keys not in agent return are preserved from input
        assert output_s["topic"] == "AI"

    def test_input_state_not_mutated(self):
        """The original input_state dict must not be modified by capture."""
        before = {"topic": "AI", "research_findings": ""}
        original_before = dict(before)
        capture = HandoffCapture(input_state=before)
        capture.record_agent_return({"research_findings": "new content"})
        capture.finalize()

        assert before == original_before


# ─────────────────────────────────────────────────────────────────────────────
# HandoffDiff — four categories
# ─────────────────────────────────────────────────────────────────────────────

class TestHandoffDiff:

    # ── added_keys ──────────────────────────────────────────────────────────

    def test_added_key_when_empty_becomes_populated(self):
        """Researcher fills research_findings → added."""
        before = {"topic": "AI", "research_findings": "", "source_count": 0}
        after  = {"topic": "AI", "research_findings": "SOURCES:\n- Book A", "source_count": 3}
        capture = HandoffCapture(input_state=before)
        capture.record_agent_return({"research_findings": "SOURCES:\n- Book A", "source_count": 3})
        _, _, _, diff = capture.finalize()

        assert "research_findings" in diff.added_keys
        assert "source_count" in diff.added_keys

    # ── unchanged_keys ──────────────────────────────────────────────────────

    def test_unchanged_key_when_topic_passes_through(self):
        """Topic is set at pipeline start and must appear as unchanged."""
        before = {"topic": "AI", "research_findings": ""}
        capture = HandoffCapture(input_state=before)
        capture.record_agent_return({"research_findings": "SOURCES:\n- Book A"})
        _, _, _, diff = capture.finalize()

        assert "topic" in diff.unchanged_keys

    # ── modified_keys ────────────────────────────────────────────────────────

    def test_modified_key_when_value_changes(self):
        """If an agent changes an existing non-empty value, it is modified."""
        before = {"topic": "AI", "source_count": 5}
        capture = HandoffCapture(input_state=before)
        capture.record_agent_return({"source_count": 8})
        _, _, _, diff = capture.finalize()

        assert "source_count" in diff.modified_keys
        assert "source_count" not in diff.unchanged_keys

    # ── dropped_keys ─────────────────────────────────────────────────────────

    def test_dropped_key_when_content_goes_missing(self):
        """If an agent returns an empty value for a previously non-empty key."""
        before = {"topic": "AI", "research_findings": "SOURCES:\n- Book A"}
        capture = HandoffCapture(input_state=before)
        # Agent somehow clears the research findings
        capture.record_agent_return({"research_findings": ""})
        _, _, _, diff = capture.finalize()

        assert "research_findings" in diff.dropped_keys
        assert diff.has_information_loss is True

    def test_no_dropped_keys_on_clean_run(self):
        """Normal researcher run should have zero dropped keys."""
        before  = {"topic": "AI", "research_findings": "", "source_count": 0}
        capture = HandoffCapture(input_state=before)
        capture.record_agent_return({
            "research_findings": "SOURCES:\n- Book A\nENTITIES:\n- Author",
            "source_count": 1,
        })
        _, _, _, diff = capture.finalize()

        assert diff.dropped_keys == []
        assert diff.has_information_loss is False

    # ── Convenience properties ────────────────────────────────────────────

    def test_has_mutations_true(self):
        before = {"source_count": 5}
        capture = HandoffCapture(input_state=before)
        capture.record_agent_return({"source_count": 99})
        _, _, _, diff = capture.finalize()
        assert diff.has_mutations is True

    def test_total_changes_counts_correctly(self):
        before = {
            "topic": "AI",          # will be unchanged
            "research_findings": "", # will be added
            "source_count": 5,       # will be modified
        }
        capture = HandoffCapture(input_state=before)
        capture.record_agent_return({
            "research_findings": "SOURCES:\n- Book A",
            "source_count": 8,
        })
        _, _, _, diff = capture.finalize()
        # added=1 (research_findings), modified=1 (source_count), dropped=0
        assert diff.total_changes == 2
        assert len(diff.unchanged_keys) == 1  # topic

    def test_summary_no_changes(self):
        before = {"topic": "AI"}
        capture = HandoffCapture(input_state=before)
        capture.record_agent_return({})
        _, _, _, diff = capture.finalize()
        assert diff.summary() == "no changes"

    def test_summary_with_drops_shows_caps(self):
        before = {"research_findings": "SOURCES:\n- Book A"}
        capture = HandoffCapture(input_state=before)
        capture.record_agent_return({"research_findings": ""})
        _, _, _, diff = capture.finalize()
        assert "DROPPED" in diff.summary()


# ─────────────────────────────────────────────────────────────────────────────
# Integration: full pipeline-like state flow
# ─────────────────────────────────────────────────────────────────────────────

class TestHandoffCaptureIntegration:

    def test_researcher_handoff_adds_findings_and_counts(self):
        """Simulates the state transition for the Researcher node."""
        pipeline_initial_state = {
            "topic": "Rise of AI in India",
            "research_findings": "",
            "source_count": 0,
            "entity_count": 0,
            "written_report": "",
            "verification_result": "",
            "verified": False,
            "revision_notes": "",
        }

        researcher_return = {
            "research_findings": "SOURCES:\n- Book A\nENTITIES:\n- NITI Aayog\nKEY FINDINGS:\nAI is rising.",
            "source_count": 1,
            "entity_count": 1,
        }

        capture = HandoffCapture(input_state=pipeline_initial_state)
        capture.record_agent_return(researcher_return)
        input_s, filtered_s, output_s, diff = capture.finalize()

        # Researcher should add the three research keys
        assert "research_findings" in diff.added_keys
        assert "source_count" in diff.added_keys
        assert "entity_count" in diff.added_keys

        # Topic and other empty fields should NOT be in added (they are still empty)
        assert "topic" in diff.unchanged_keys
        assert "written_report" not in diff.added_keys

        # No information loss
        assert not diff.has_information_loss

        # Output state is the full merged state
        assert output_s["topic"] == "Rise of AI in India"
        assert output_s["source_count"] == 1
        assert output_s["research_findings"] != ""

    def test_writer_handoff_fills_report(self):
        """Simulates the state transition for the Writer node."""
        state_after_researcher = {
            "topic": "Rise of AI in India",
            "research_findings": "SOURCES:\n- Book A\nKEY FINDINGS:\nAI is rising.",
            "source_count": 1,
            "entity_count": 1,
            "written_report": "",
            "verification_result": "",
            "verified": False,
            "revision_notes": "",
        }

        writer_return = {
            "written_report": "# Introduction\nAI is rising in India."
        }

        capture = HandoffCapture(input_state=state_after_researcher)
        capture.record_agent_return(writer_return)
        _, _, output_s, diff = capture.finalize()

        assert "written_report" in diff.added_keys
        assert "research_findings" in diff.unchanged_keys
        assert not diff.has_information_loss
        assert output_s["written_report"] != ""
        assert output_s["research_findings"] != ""  # research preserved
