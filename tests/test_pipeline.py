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

from app.pipeline import (
    PipelineState,
    build_pipeline,
    researcher_node,
    run_pipeline,
    verifier_node,
    writer_node,
)

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

# ── Unit tests for pipeline nodes ─────────────────────────────────────────────


class FakeResponse:
    """Minimal fake LangChain response."""

    def __init__(self, content, usage_metadata=None):
        self.content = content
        self.usage_metadata = usage_metadata


class FakeLLM:
    """Fake LLM that returns a predefined response."""

    def __init__(self, response):
        self.response = response
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        return self.response


def test_researcher_node_counts_sources_and_entities(monkeypatch):
    """Researcher should correctly count sources and entities."""

    response = FakeResponse(
        """SOURCES
- Source One
- Source Two
- Source Three

ENTITIES
- Entity One
- Entity Two

KEY FINDINGS
The research findings are here.
""",
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 50,
        },
    )

    fake_llm = FakeLLM(response)

    monkeypatch.setattr(
        "app.pipeline._build_llm",
        lambda: fake_llm,
    )

    monkeypatch.setattr(
        "app.pipeline.get",
        lambda *args, **kwargs: {
            ("pipeline", "researcher", "min_sources"): 2,
            ("pipeline", "researcher", "max_sources"): 5,
            ("pipeline", "researcher", "min_entities"): 2,
            ("pipeline", "researcher", "depth"): "standard",
        }.get(tuple(args), kwargs.get("default")),
    )

    state = {
        "topic": "Artificial Intelligence",
    }

    result = researcher_node(state)

    assert result["research_findings"]
    assert result["source_count"] == 3
    assert result["entity_count"] == 2


def test_writer_node_returns_written_report(monkeypatch):
    """Writer should return the generated report."""

    response = FakeResponse(
        "This is the final written report.",
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 200,
        },
    )

    fake_llm = FakeLLM(response)

    monkeypatch.setattr(
        "app.pipeline._build_llm",
        lambda: fake_llm,
    )

    monkeypatch.setattr(
        "app.pipeline.get",
        lambda *args, **kwargs: {
            ("pipeline", "writer", "report_sections"): [
                "Introduction",
                "Findings",
                "Conclusion",
            ],
            ("pipeline", "writer", "citation_style"): "APA",
            ("pipeline", "writer", "word_count_target"): 500,
        }.get(tuple(args), kwargs.get("default")),
    )

    state = {
        "topic": "Artificial Intelligence",
        "research_findings": "AI is transforming many industries.",
    }

    result = writer_node(state)

    assert result["written_report"] == "This is the final written report."


def test_verifier_node_approved(monkeypatch):
    """Verifier should mark the report as verified when response starts with APPROVED."""

    response = FakeResponse(
        "APPROVED\nAll checks passed.",
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 20,
        },
    )

    fake_llm = FakeLLM(response)

    monkeypatch.setattr(
        "app.pipeline._build_llm",
        lambda: fake_llm,
    )

    monkeypatch.setattr(
        "app.pipeline.get",
        lambda *args, **kwargs: {
            ("pipeline", "verifier", "strictness"): "high",
            ("pipeline", "verifier", "check_source_coverage"): True,
            ("pipeline", "verifier", "check_entity_coverage"): True,
            ("pipeline", "verifier", "check_no_new_facts"): True,
        }.get(tuple(args), kwargs.get("default")),
    )

    state = {
        "topic": "Artificial Intelligence",
        "research_findings": "Research findings",
        "source_count": 3,
        "entity_count": 2,
        "written_report": "Final report",
    }

    result = verifier_node(state)

    assert result["verification_result"].startswith("APPROVED")
    assert result["verified"] is True
    assert result["revision_notes"] == ""


def test_verifier_node_needs_revision(monkeypatch):
    """Verifier should populate revision notes when report is rejected."""

    response = FakeResponse(
        "NEEDS_REVISION\nMissing source citations.",
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 30,
        },
    )

    fake_llm = FakeLLM(response)

    monkeypatch.setattr(
        "app.pipeline._build_llm",
        lambda: fake_llm,
    )

    monkeypatch.setattr(
        "app.pipeline.get",
        lambda *args, **kwargs: {
            ("pipeline", "verifier", "strictness"): "high",
            ("pipeline", "verifier", "check_source_coverage"): True,
            ("pipeline", "verifier", "check_entity_coverage"): True,
            ("pipeline", "verifier", "check_no_new_facts"): True,
        }.get(tuple(args), kwargs.get("default")),
    )

    state = {
        "topic": "Artificial Intelligence",
        "research_findings": "Research findings",
        "source_count": 3,
        "entity_count": 2,
        "written_report": "Final report",
    }

    result = verifier_node(state)

    assert result["verified"] is False
    assert result["revision_notes"] == result["verification_result"]
    assert "Missing source citations" in result["revision_notes"]


def test_build_llm_requires_api_key(monkeypatch):
    """_build_llm should fail clearly when GROQ_API_KEY is missing."""

    from app.pipeline import _build_llm

    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(OSError, match="GROQ_API_KEY"):
        _build_llm()


def test_run_pipeline_returns_final_state(monkeypatch):
    """run_pipeline should invoke the compiled graph and return its final state."""

    expected_state = {
        "topic": "Test topic",
        "research_findings": "Findings",
        "source_count": 2,
        "entity_count": 1,
        "written_report": "Report",
        "verification_result": "APPROVED",
        "verified": True,
        "revision_notes": "",
    }

    class FakeApp:
        def invoke(self, state):
            return expected_state

    monkeypatch.setattr(
        "app.pipeline.build_pipeline",
        lambda: FakeApp(),
    )

    result = run_pipeline("Test topic")

    assert result == expected_state
