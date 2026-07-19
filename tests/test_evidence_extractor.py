"""
tests/test_evidence_extractor.py — Day 9: Evidence Extractor tests
====================================================================
Unit tests: test ExtractedEvidence model, fallback behaviour, schema_version.
Integration test: real LLM call against a known sample output.

Unit tests run without any API key. Integration test requires GROQ_API_KEY.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from analyzers.evidence_extraction.extractor import ExtractedEvidence, EvidenceExtractor
from schema.models import SCHEMA_VERSION


# ─────────────────────────────────────────────────────────────────────────────
# ExtractedEvidence model
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractedEvidence:

    def test_schema_version_stamped_by_default(self):
        ev = ExtractedEvidence()
        assert ev.schema_version == SCHEMA_VERSION

    def test_default_values(self):
        ev = ExtractedEvidence()
        assert ev.source_count == 0
        assert ev.entity_count == 0
        assert ev.tool_calls == []
        assert ev.extraction_failed is False
        assert ev.error_message is None

    def test_source_count_must_be_non_negative(self):
        with pytest.raises(Exception):
            ExtractedEvidence(source_count=-1)

    def test_entity_count_must_be_non_negative(self):
        with pytest.raises(Exception):
            ExtractedEvidence(entity_count=-1)

    def test_tool_calls_is_list(self):
        ev = ExtractedEvidence(tool_calls=["web_search", "calculator"])
        assert isinstance(ev.tool_calls, list)
        assert len(ev.tool_calls) == 2

    def test_extraction_failed_flag(self):
        ev = ExtractedEvidence(extraction_failed=True, error_message="timeout")
        assert ev.extraction_failed is True
        assert ev.error_message == "timeout"


# ─────────────────────────────────────────────────────────────────────────────
# EvidenceExtractor — unit tests (mocked LLM)
# ─────────────────────────────────────────────────────────────────────────────

class TestEvidenceExtractorUnit:
    """Tests that mock the LLM — no API key needed."""

    def _make_extractor_with_mock(self, mock_response: ExtractedEvidence) -> EvidenceExtractor:
        """Create extractor with mocked LLM returning mock_response."""
        extractor = EvidenceExtractor.__new__(EvidenceExtractor)
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        extractor._llm = mock_llm
        return extractor

    def test_extract_returns_extracted_evidence(self):
        mock_ev = ExtractedEvidence(source_count=5, entity_count=10, tool_calls=[])
        extractor = self._make_extractor_with_mock(mock_ev)
        result = extractor.extract("some output")
        assert isinstance(result, ExtractedEvidence)
        assert result.source_count == 5
        assert result.entity_count == 10

    def test_extract_stamps_schema_version(self):
        mock_ev = ExtractedEvidence(source_count=3, entity_count=7)
        extractor = self._make_extractor_with_mock(mock_ev)
        result = extractor.extract("some output")
        assert result.schema_version == SCHEMA_VERSION

    def test_extract_empty_output_returns_fallback(self):
        extractor = EvidenceExtractor.__new__(EvidenceExtractor)
        extractor._llm = MagicMock()
        result = extractor.extract("")
        assert result.extraction_failed is True
        assert result.source_count == 0

    def test_extract_whitespace_only_returns_fallback(self):
        extractor = EvidenceExtractor.__new__(EvidenceExtractor)
        extractor._llm = MagicMock()
        result = extractor.extract("   \n  ")
        assert result.extraction_failed is True

    def test_extract_llm_exception_returns_fallback(self):
        extractor = EvidenceExtractor.__new__(EvidenceExtractor)
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("API timeout")
        extractor._llm = mock_llm
        result = extractor.extract("some output")
        assert result.extraction_failed is True
        assert "RuntimeError" in result.error_message

    def test_extract_with_tool_calls(self):
        mock_ev = ExtractedEvidence(
            source_count=2,
            entity_count=4,
            tool_calls=["web_search", "calculator"]
        )
        extractor = self._make_extractor_with_mock(mock_ev)
        result = extractor.extract("used web_search and calculator")
        assert result.tool_calls == ["web_search", "calculator"]

    def test_extract_run_processes_all_steps(self):
        mock_ev = ExtractedEvidence(source_count=3, entity_count=6)
        extractor = self._make_extractor_with_mock(mock_ev)

        steps = [
            {"step": 1, "agent": "researcher", "raw_output": "SOURCES:\n- Book A"},
            {"step": 2, "agent": "writer",     "raw_output": "Report text..."},
            {"step": 3, "agent": "verifier",   "raw_output": "APPROVED"},
        ]

        results = extractor.extract_run(steps)
        assert len(results) == 3
        assert 1 in results and 2 in results and 3 in results

    def test_extract_run_returns_correct_step_numbers(self):
        mock_ev = ExtractedEvidence(source_count=1, entity_count=2)
        extractor = self._make_extractor_with_mock(mock_ev)
        steps = [{"step": 1, "agent": "researcher", "raw_output": "output"}]
        results = extractor.extract_run(steps)
        assert results[1].source_count == 1

    def test_long_output_truncated_not_crashes(self):
        """Outputs longer than 4000 chars should be truncated, not crash."""
        mock_ev = ExtractedEvidence(source_count=1, entity_count=1)
        extractor = self._make_extractor_with_mock(mock_ev)
        long_output = "word " * 5000  # 25000 chars
        result = extractor.extract(long_output)
        assert isinstance(result, ExtractedEvidence)
        assert not result.extraction_failed


# ─────────────────────────────────────────────────────────────────────────────
# Integration test — real LLM (requires GROQ_API_KEY)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestEvidenceExtractorIntegration:
    """
    Real LLM call. Requires GROQ_API_KEY in .env

    Run with:
        pytest tests/test_evidence_extractor.py -m integration -v
    """

    SAMPLE_RESEARCHER_OUTPUT = """SOURCES:
- "Artificial Intelligence: A Modern Approach" by Stuart Russell and Peter Norvig (book, 2020)
- "India's AI Strategy" by NITI Aayog (report, 2018)
- "The Future of Work in India" by World Economic Forum (report, 2020)
- "The Rise of AI in India" by McKinsey & Company (article, 2020)
- "India's AI Ecosystem" by KPMG (report, 2020)

ENTITIES:
- NITI Aayog
- Indian Institute of Technology (IIT)
- World Economic Forum (WEF)
- McKinsey & Company
- KPMG
- Narendra Modi
- Bengaluru

KEY FINDINGS:
The rise of AI in India has been significant, driven by government initiatives
and private investments. India aims to contribute $1 trillion to GDP by 2035."""

    def test_real_extraction_source_count(self):
        extractor = EvidenceExtractor()
        result = extractor.extract(self.SAMPLE_RESEARCHER_OUTPUT, agent="researcher")
        assert not result.extraction_failed, f"Extraction failed: {result.error_message}"
        # 5 sources listed — allow ±1 for LLM variance
        assert 4 <= result.source_count <= 6, f"Expected ~5 sources, got {result.source_count}"

    def test_real_extraction_entity_count(self):
        extractor = EvidenceExtractor()
        result = extractor.extract(self.SAMPLE_RESEARCHER_OUTPUT, agent="researcher")
        assert not result.extraction_failed
        # 7 entities listed — allow ±2 for LLM variance
        assert 5 <= result.entity_count <= 9, f"Expected ~7 entities, got {result.entity_count}"

    def test_real_extraction_no_tool_calls(self):
        extractor = EvidenceExtractor()
        result = extractor.extract(self.SAMPLE_RESEARCHER_OUTPUT, agent="researcher")
        assert not result.extraction_failed
        assert isinstance(result.tool_calls, list)
        assert len(result.tool_calls) == 0

    def test_real_extraction_schema_version_stamped(self):
        extractor = EvidenceExtractor()
        result = extractor.extract(self.SAMPLE_RESEARCHER_OUTPUT)
        assert result.schema_version == SCHEMA_VERSION
