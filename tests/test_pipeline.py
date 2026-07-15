# tests/test_pipeline.py
"""
Tests for app/pipeline.py — the LangGraph reference pipeline.

Covers:
  - Pipeline builds without error (graph compiles)
  - PipelineState has all required keys
  - All three nodes are registered in the graph
  - run_pipeline() returns a fully populated state (integration test — skipped if no API key)
  - Verifier output is one of the expected values
  - Source/entity counts are non-negative integers
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.pipeline import build_pipeline, PipelineState


# ── Pipeline structure tests (no API call needed) ─────────────────────────────

def test_pipeline_builds_without_error():
    """The graph must compile successfully."""
    app = build_pipeline()
    assert app is not None


def test_pipeline_state_has_required_keys():
    """All PipelineState keys must be present."""
    required_keys = {
        "topic",
        "research_findings",
        "source_count",
        "entity_count",
        "written_report",
        "verification_result",
        "verified",
        "revision_notes",
    }
    # TypedDict keys are in __annotations__
    state_keys = set(PipelineState.__annotations__.keys())
    assert required_keys == state_keys


def test_pipeline_graph_has_all_three_nodes():
    """All three agent nodes must be registered."""
    app = build_pipeline()
    node_names = set(app.get_graph().nodes.keys())
    assert "researcher" in node_names
    assert "writer" in node_names
    assert "verifier" in node_names


def test_pipeline_graph_has_correct_node_count():
    """Graph should have exactly 5 nodes: __start__, researcher, writer, verifier, __end__."""
    app = build_pipeline()
    nodes = app.get_graph().nodes
    assert len(nodes) == 5


def test_pipeline_graph_edges_are_linear():
    """
    Edges must form a linear chain:
    __start__ → researcher → writer → verifier → __end__
    """
    app = build_pipeline()
    edges = [(e.source, e.target) for e in app.get_graph().edges]

    assert ("__start__", "researcher") in edges
    assert ("researcher", "writer") in edges
    assert ("writer", "verifier") in edges
    assert ("verifier", "__end__") in edges


# ── Integration test (requires GROQ_API_KEY) ──────────────────────────────────

@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") == "gsk_your_key_here",
    reason="GROQ_API_KEY not set — skipping live API integration test",
)
def test_run_pipeline_integration():
    """
    Full end-to-end pipeline run against the real Groq API.
    Verifies all state keys are populated and types are correct.
    Uses a short, well-defined topic to keep latency low.
    """
    from app.pipeline import run_pipeline

    state = run_pipeline(topic="The invention of the World Wide Web by Tim Berners-Lee")

    # All string fields must be non-empty
    assert isinstance(state["topic"], str) and state["topic"]
    assert isinstance(state["research_findings"], str) and state["research_findings"]
    assert isinstance(state["written_report"], str) and state["written_report"]
    assert isinstance(state["verification_result"], str) and state["verification_result"]

    # Numeric fields must be valid
    assert isinstance(state["source_count"], int)
    assert isinstance(state["entity_count"], int)
    assert state["source_count"] >= 0
    assert state["entity_count"] >= 0

    # Verifier must produce a boolean
    assert isinstance(state["verified"], bool)

    # Verification result must start with one of the expected values
    result_upper = state["verification_result"].upper()
    assert result_upper.startswith("APPROVED") or result_upper.startswith("NEEDS_REVISION"), (
        f"Unexpected verification result: {state['verification_result'][:100]}"
    )

    # If not verified, revision_notes must be non-empty
    if not state["verified"]:
        assert state["revision_notes"], "revision_notes should be populated when not verified"
