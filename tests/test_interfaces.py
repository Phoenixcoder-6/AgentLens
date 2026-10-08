# tests/test_interfaces.py
"""
Tests for app/interfaces.py — the 4 Protocol interfaces and custom exceptions.

Covers:
  - All interfaces and return types import cleanly
  - Custom exceptions are proper Exception subclasses
  - AnalysisResult and TraceEvent instantiate with defaults
  - Protocol structure is correct (runtime_checkable)
"""

from datetime import UTC

import pytest

from app.interfaces import (
    AnalysisResult,
    Analyzer,
    CaptureError,
    CaptureProvider,
    ExtractionError,
    LLMProvider,
    LLMProviderError,
    StorageError,
    StorageProvider,
    TraceEvent,
)

# ── Import completeness ───────────────────────────────────────────────────────


def test_all_interfaces_importable():
    assert Analyzer is not None
    assert CaptureProvider is not None
    assert StorageProvider is not None
    assert LLMProvider is not None


def test_all_return_types_importable():
    assert AnalysisResult is not None
    assert TraceEvent is not None


def test_all_exceptions_importable():
    assert LLMProviderError is not None
    assert ExtractionError is not None
    assert StorageError is not None
    assert CaptureError is not None


# ── Exception hierarchy ───────────────────────────────────────────────────────


def test_llm_provider_error_is_exception():
    assert issubclass(LLMProviderError, Exception)


def test_extraction_error_is_exception():
    assert issubclass(ExtractionError, Exception)


def test_storage_error_is_exception():
    assert issubclass(StorageError, Exception)


def test_capture_error_is_exception():
    assert issubclass(CaptureError, Exception)


def test_exceptions_can_be_raised_and_caught():
    with pytest.raises(LLMProviderError):
        raise LLMProviderError("Groq API failed after 1 retry")

    with pytest.raises(ExtractionError):
        raise ExtractionError("Malformed JSON in extraction output")


# ── AnalysisResult defaults ───────────────────────────────────────────────────


def test_analysis_result_default_instantiation():
    result = AnalysisResult()
    assert result.evidence == []
    assert result.analyzer_id == ""
    assert result.skipped is False
    assert result.skip_reason == ""


def test_analysis_result_skipped_flag():
    result = AnalysisResult(skipped=True, skip_reason="LLM extraction failed")
    assert result.skipped is True
    assert result.skip_reason == "LLM extraction failed"


def test_analysis_result_with_evidence(sample_evidence_record):
    result = AnalysisResult(
        evidence=[sample_evidence_record],
        analyzer_id="rule_engine",
    )
    assert len(result.evidence) == 1
    assert result.analyzer_id == "rule_engine"


# ── TraceEvent defaults ───────────────────────────────────────────────────────


def test_trace_event_instantiation():
    from datetime import datetime

    event = TraceEvent(
        agent="researcher",
        raw_input="Summarize 10 sources",
        raw_output="Tesla was founded...",
        latency_ms=1400.0,
        timestamp=datetime.now(UTC).isoformat(),
    )
    assert event.agent == "researcher"
    assert event.tool_calls == []
    assert event.metadata == {}


# ── Protocol runtime checkability ─────────────────────────────────────────────


def test_analyzer_is_runtime_checkable():
    """Analyzer is a runtime_checkable Protocol — isinstance() works on it."""

    # A class that implements Analyzer correctly
    class FakeAnalyzer:
        @property
        def analyzer_id(self) -> str:
            return "fake"

        def analyze(self, trace):
            return AnalysisResult()

    fa = FakeAnalyzer()
    assert isinstance(fa, Analyzer)


def test_incomplete_analyzer_fails_isinstance():
    """Class missing analyze() does NOT satisfy Analyzer Protocol."""

    class BrokenAnalyzer:
        @property
        def analyzer_id(self) -> str:
            return "broken"

    assert not isinstance(BrokenAnalyzer(), Analyzer)


# ── Day 41: Concrete Analyzer Interface Validation ────────────────────────────


def _minimal_valid_trace():
    from schema.models import AgentStep, HandoffState, RunTrace, TokenUsage

    return RunTrace(
        run_id="run_day41_minimal",
        workflow="research_summary",
        expected_output="Final verified summary with citations.",
        steps=[
            AgentStep(
                run_id="run_day41_minimal",
                step=1,
                agent="researcher",
                input="Topic: AI safety",
                output="Found 4 sources and 5 entities.",
                latency_ms=120.0,
                tokens=TokenUsage(prompt=50, completion=50, total=100),
                handoff=HandoffState(
                    output_state={
                        "extracted_evidence": {
                            "source_count": 4,
                            "entity_count": 5,
                            "claims": ["Claim A", "Claim B"],
                            "extraction_failed": False,
                        }
                    }
                ),
            ),
            AgentStep(
                run_id="run_day41_minimal",
                step=2,
                agent="writer",
                input="Found 4 sources and 5 entities.",
                output="Draft summary of 4 sources and 5 entities.",
                latency_ms=150.0,
                tokens=TokenUsage(prompt=60, completion=60, total=120),
                handoff=HandoffState(
                    output_state={
                        "extracted_evidence": {
                            "source_count": 4,
                            "entity_count": 5,
                            "claims": ["Claim A", "Claim B"],
                            "extraction_failed": False,
                        }
                    }
                ),
            ),
            AgentStep(
                run_id="run_day41_minimal",
                step=3,
                agent="verifier",
                input="Draft summary of 4 sources and 5 entities.",
                output="Final verified summary with citations.",
                latency_ms=90.0,
                tokens=TokenUsage(prompt=40, completion=40, total=80),
                handoff=HandoffState(
                    output_state={
                        "extracted_evidence": {
                            "source_count": 4,
                            "entity_count": 5,
                            "claims": ["Claim A", "Claim B"],
                            "extraction_failed": False,
                        }
                    }
                ),
            ),
        ],
    )


def _all_concrete_analyzers(tmp_path):
    from unittest.mock import patch

    from analyzers.detection.consistency_validator import ConsistencyValidator
    from analyzers.detection.ground_truth import GroundTruthValidator
    from analyzers.detection.information_loss import InformationLossRule
    from analyzers.detection.rule_engine import RuleEngine
    from analyzers.detection.statistical_detector import StatisticalDetector
    from analyzers.detection.workflow_validator import WorkflowValidator
    from analyzers.diff_engine import DiffEngine, GraphAligner, SemanticSimilarityEngine
    from analyzers.evidence_extraction.extractor import EvidenceExtractor
    from analyzers.metrics_analyzer import MetricsAnalyzer
    from storage.db import DatabaseManager

    db = DatabaseManager(db_path=str(tmp_path / "day41_test.db"))
    with patch.object(EvidenceExtractor, "_build_llm", return_value=None):
        extractor = EvidenceExtractor()

    return [
        EvidenceExtractor if False else extractor,
        DiffEngine(),
        GraphAligner(),
        SemanticSimilarityEngine(),
        MetricsAnalyzer(db=db),
        StatisticalDetector(db=db),
        GroundTruthValidator(),
        RuleEngine(),
        InformationLossRule(),
        WorkflowValidator(),
        ConsistencyValidator(),
    ]


def test_all_concrete_analyzers_implement_analyzer_protocol(tmp_path):
    """Every concrete analyzer satisfies isinstance(analyzer, Analyzer) and has non-empty analyzer_id."""
    analyzers = _all_concrete_analyzers(tmp_path)
    assert len(analyzers) == 11

    for analyzer in analyzers:
        cls_name = type(analyzer).__name__
        assert isinstance(analyzer, Analyzer), f"{cls_name} does not implement Analyzer protocol"
        assert isinstance(analyzer.analyzer_id, str) and len(analyzer.analyzer_id) > 0, (
            f"{cls_name}.analyzer_id must be a non-empty string"
        )
        assert callable(getattr(analyzer, "analyze", None)), f"{cls_name} missing .analyze()"
        assert callable(getattr(analyzer, "run", None)), f"{cls_name} missing .run()"


def test_all_concrete_analyzers_analyze_and_run_minimal_trace(tmp_path, monkeypatch):
    """Instantiating each Analyzer and calling .analyze() and .run() with a minimal valid RunTrace succeeds."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    analyzers = _all_concrete_analyzers(tmp_path)
    trace = _minimal_valid_trace()
    snapshot_before = trace.model_dump()

    for analyzer in analyzers:
        cls_name = type(analyzer).__name__
        res_analyze = analyzer.analyze(trace)
        assert isinstance(res_analyze, AnalysisResult), (
            f"{cls_name}.analyze() returned {type(res_analyze)}"
        )
        assert res_analyze.analyzer_id == analyzer.analyzer_id
        assert isinstance(res_analyze.evidence, list)

        res_run = analyzer.run(trace)
        assert isinstance(res_run, AnalysisResult), f"{cls_name}.run() returned {type(res_run)}"
        assert res_run.analyzer_id == analyzer.analyzer_id
        assert isinstance(res_run.evidence, list)

    # Read-only guarantee: no analyzer may mutate the input RunTrace
    assert trace.model_dump() == snapshot_before


def test_all_concrete_analyzers_graceful_skip_on_empty_trace(tmp_path, monkeypatch):
    """Every concrete Analyzer degrades gracefully (skipped=True, no exception) on an empty RunTrace."""
    from schema.models import RunTrace

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    analyzers = _all_concrete_analyzers(tmp_path)
    empty_trace = RunTrace(run_id="run_empty_day41", workflow="research_summary", steps=[])

    for analyzer in analyzers:
        cls_name = type(analyzer).__name__
        res = analyzer.run(empty_trace)
        assert isinstance(res, AnalysisResult), f"{cls_name}.run(empty) failed"
        assert res.skipped is True, f"{cls_name} should set skipped=True on empty trace"
        assert res.evidence == []


def test_pep561_py_typed_markers_present():
    """All core AgentLens packages ship a PEP 561 py.typed marker file."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    packages = [
        "app",
        "analyzers",
        "capture",
        "config",
        "dashboard",
        "diff_engine",
        "normalizer",
        "schema",
        "storage",
    ]
    for pkg in packages:
        marker = root / pkg / "py.typed"
        assert marker.is_file(), f"Missing PEP 561 marker: {marker}"
