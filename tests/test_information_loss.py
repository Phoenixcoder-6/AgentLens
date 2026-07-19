"""
tests/test_information_loss.py — Day 10: Information Loss Rule tests
=====================================================================
Tests for InformationLossRule consuming ExtractedEvidence from Day 9.
No LLM calls here — all evidence is constructed directly.
"""

from __future__ import annotations

import pytest

from analyzers.evidence_extraction.extractor import ExtractedEvidence
from analyzers.detection.information_loss import (
    InformationLossRule,
    InformationLossResult,
    FieldDiff,
)
from schema.models import SCHEMA_VERSION


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_evidence(source_count=8, entity_count=16, tool_calls=None,
                  extraction_failed=False, error_message=None) -> ExtractedEvidence:
    return ExtractedEvidence(
        source_count=source_count,
        entity_count=entity_count,
        tool_calls=tool_calls or [],
        extraction_failed=extraction_failed,
        error_message=error_message,
    )


RUN_ID = "run_test_day10"


# ─────────────────────────────────────────────────────────────────────────────
# FieldDiff
# ─────────────────────────────────────────────────────────────────────────────

class TestFieldDiff:

    def test_dropped_signal_when_delta_negative(self):
        rule = InformationLossRule()
        diff = rule._compute_diff("source_count", researcher_val=8, writer_val=5)
        assert diff.signal == "DROPPED"
        assert diff.delta == -3

    def test_added_signal_when_delta_positive(self):
        rule = InformationLossRule()
        diff = rule._compute_diff("source_count", researcher_val=8, writer_val=11)
        assert diff.signal == "ADDED"
        assert diff.delta == 3

    def test_preserved_signal_when_equal(self):
        rule = InformationLossRule()
        diff = rule._compute_diff("source_count", researcher_val=8, writer_val=8)
        assert diff.signal == "PRESERVED"
        assert diff.delta == 0

    def test_severity_none_when_preserved(self):
        rule = InformationLossRule()
        diff = rule._compute_diff("source_count", 8, 8)
        assert diff.severity == "NONE"

    def test_severity_medium_when_delta_1(self):
        rule = InformationLossRule()
        diff = rule._compute_diff("source_count", 8, 7)
        assert diff.severity == "MEDIUM"

    def test_severity_high_when_delta_3_or_more(self):
        rule = InformationLossRule()
        diff = rule._compute_diff("source_count", 8, 4)
        assert diff.severity == "HIGH"

    def test_field_name_preserved(self):
        rule = InformationLossRule()
        diff = rule._compute_diff("entity_count", 16, 14)
        assert diff.field_name == "entity_count"


# ─────────────────────────────────────────────────────────────────────────────
# InformationLossRule — PASS cases
# ─────────────────────────────────────────────────────────────────────────────

class TestInformationLossPass:

    def setup_method(self):
        self.rule = InformationLossRule()

    def test_exact_match_is_pass(self):
        researcher = make_evidence(source_count=8, entity_count=16)
        writer     = make_evidence(source_count=8, entity_count=16)
        result = self.rule.evaluate(RUN_ID, researcher, writer)
        assert result.verdict == "PASS"

    def test_schema_version_stamped(self):
        researcher = make_evidence(source_count=8, entity_count=16)
        writer     = make_evidence(source_count=8, entity_count=16)
        result = self.rule.evaluate(RUN_ID, researcher, writer)
        assert result.schema_version == SCHEMA_VERSION

    def test_run_id_preserved(self):
        researcher = make_evidence(source_count=8, entity_count=16)
        writer     = make_evidence(source_count=8, entity_count=16)
        result = self.rule.evaluate(RUN_ID, researcher, writer)
        assert result.run_id == RUN_ID

    def test_no_loss_flags_on_pass(self):
        researcher = make_evidence(source_count=8, entity_count=16)
        writer     = make_evidence(source_count=8, entity_count=16)
        result = self.rule.evaluate(RUN_ID, researcher, writer)
        assert result.has_information_loss is False

    def test_confidence_is_1_on_clean_pass(self):
        researcher = make_evidence(source_count=8, entity_count=16)
        writer     = make_evidence(source_count=8, entity_count=16)
        result = self.rule.evaluate(RUN_ID, researcher, writer)
        assert result.confidence == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# InformationLossRule — WARNING cases
# ─────────────────────────────────────────────────────────────────────────────

class TestInformationLossWarning:

    def setup_method(self):
        self.rule = InformationLossRule()

    def test_small_source_drop_is_warning(self):
        researcher = make_evidence(source_count=8, entity_count=16)
        writer     = make_evidence(source_count=7, entity_count=16)  # 1 source dropped
        result = self.rule.evaluate(RUN_ID, researcher, writer)
        assert result.verdict == "WARNING"
        assert result.has_information_loss is True

    def test_small_entity_drop_is_warning(self):
        researcher = make_evidence(source_count=8, entity_count=16)
        writer     = make_evidence(source_count=8, entity_count=15)  # 1 entity dropped
        result = self.rule.evaluate(RUN_ID, researcher, writer)
        assert result.verdict == "WARNING"

    def test_sources_added_is_warning(self):
        """Writer adds MORE sources than Researcher — hallucination risk."""
        researcher = make_evidence(source_count=8, entity_count=16)
        writer     = make_evidence(source_count=11, entity_count=16)  # 3 added
        result = self.rule.evaluate(RUN_ID, researcher, writer)
        assert result.verdict == "WARNING"
        assert result.has_information_gain is True

    def test_entities_added_is_warning(self):
        researcher = make_evidence(source_count=8, entity_count=16)
        writer     = make_evidence(source_count=8, entity_count=19)
        result = self.rule.evaluate(RUN_ID, researcher, writer)
        assert result.verdict == "WARNING"
        assert result.has_information_gain is True

    def test_live_run_data(self):
        """
        Mirrors the real output from the live pipeline run:
            Researcher: source_count=8,  entity_count=16
            Writer:     source_count=11, entity_count=17
        Both counts increased → WARNING (gain, not loss).
        """
        researcher = make_evidence(source_count=8,  entity_count=16)
        writer     = make_evidence(source_count=11, entity_count=17)
        result = self.rule.evaluate(RUN_ID, researcher, writer)
        assert result.verdict == "WARNING"
        assert result.has_information_gain is True
        assert result.has_information_loss is False
        assert result.source_diff.signal == "ADDED"
        assert result.entity_diff.signal == "ADDED"


# ─────────────────────────────────────────────────────────────────────────────
# InformationLossRule — FAIL cases
# ─────────────────────────────────────────────────────────────────────────────

class TestInformationLossFail:

    def setup_method(self):
        self.rule = InformationLossRule()

    def test_large_source_drop_is_fail(self):
        researcher = make_evidence(source_count=8, entity_count=16)
        writer     = make_evidence(source_count=4, entity_count=16)  # 4 dropped (HIGH)
        result = self.rule.evaluate(RUN_ID, researcher, writer)
        assert result.verdict == "FAIL"
        assert result.source_diff.severity == "HIGH"

    def test_large_entity_drop_is_fail(self):
        researcher = make_evidence(source_count=8, entity_count=16)
        writer     = make_evidence(source_count=8, entity_count=10)  # 6 dropped (HIGH)
        result = self.rule.evaluate(RUN_ID, researcher, writer)
        assert result.verdict == "FAIL"
        assert result.entity_diff.severity == "HIGH"

    def test_both_fields_dropped_heavily_is_fail(self):
        researcher = make_evidence(source_count=8, entity_count=16)
        writer     = make_evidence(source_count=2, entity_count=5)
        result = self.rule.evaluate(RUN_ID, researcher, writer)
        assert result.verdict == "FAIL"
        assert result.has_information_loss is True

    def test_confidence_high_on_fail(self):
        researcher = make_evidence(source_count=8, entity_count=16)
        writer     = make_evidence(source_count=2, entity_count=5)
        result = self.rule.evaluate(RUN_ID, researcher, writer)
        # Large drop → confidence should be well above 0.80
        assert result.confidence >= 0.80

    def test_writer_zero_sources_is_fail(self):
        researcher = make_evidence(source_count=8, entity_count=16)
        writer     = make_evidence(source_count=0, entity_count=16)
        result = self.rule.evaluate(RUN_ID, researcher, writer)
        assert result.verdict == "FAIL"


# ─────────────────────────────────────────────────────────────────────────────
# InformationLossRule — extraction failure handling
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractionFailureHandling:

    def setup_method(self):
        self.rule = InformationLossRule()

    def test_researcher_extraction_failed_skips_rule(self):
        researcher = make_evidence(extraction_failed=True, error_message="timeout")
        writer     = make_evidence(source_count=8, entity_count=16)
        result = self.rule.evaluate(RUN_ID, researcher, writer)
        assert result.rule_failed is True
        assert result.verdict == "PASS"   # don't falsely flag when data is bad
        assert result.confidence == 0.0

    def test_writer_extraction_failed_skips_rule(self):
        researcher = make_evidence(source_count=8, entity_count=16)
        writer     = make_evidence(extraction_failed=True, error_message="timeout")
        result = self.rule.evaluate(RUN_ID, researcher, writer)
        assert result.rule_failed is True
        assert result.confidence == 0.0

    def test_both_failed_skips_rule(self):
        researcher = make_evidence(extraction_failed=True)
        writer     = make_evidence(extraction_failed=True)
        result = self.rule.evaluate(RUN_ID, researcher, writer)
        assert result.rule_failed is True

    def test_confidence_zero_on_skip(self):
        researcher = make_evidence(extraction_failed=True, error_message="API timeout")
        writer     = make_evidence(source_count=8, entity_count=16)
        result = self.rule.evaluate(RUN_ID, researcher, writer)
        assert result.confidence == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Scalable confidence — verifies confidence scales with delta magnitude
# ─────────────────────────────────────────────────────────────────────────────

class TestScalableConfidence:
    """
    Verifies that confidence is NOT a fixed lookup but scales with how large
    the delta is relative to the researcher baseline.

    Ranges:
        PASS    → 1.0
        FAIL    → [0.80, 0.99]
        WARNING → [0.60, 0.84]
    """

    def setup_method(self):
        self.rule = InformationLossRule()

    def test_pass_confidence_is_always_1(self):
        researcher = make_evidence(source_count=8, entity_count=16)
        writer     = make_evidence(source_count=8, entity_count=16)
        result = self.rule.evaluate(RUN_ID, researcher, writer)
        assert result.confidence == 1.0

    def test_fail_confidence_in_range(self):
        researcher = make_evidence(source_count=8, entity_count=16)
        writer     = make_evidence(source_count=4, entity_count=16)  # HIGH drop
        result = self.rule.evaluate(RUN_ID, researcher, writer)
        assert result.verdict == "FAIL"
        assert 0.80 <= result.confidence <= 0.99

    def test_warning_confidence_in_range(self):
        researcher = make_evidence(source_count=8, entity_count=16)
        writer     = make_evidence(source_count=7, entity_count=16)  # small drop
        result = self.rule.evaluate(RUN_ID, researcher, writer)
        assert result.verdict == "WARNING"
        assert 0.60 <= result.confidence <= 0.84

    def test_larger_drop_gives_higher_confidence(self):
        """A 4-source drop must give higher confidence than a 1-source drop."""
        researcher = make_evidence(source_count=8, entity_count=16)

        small_drop = self.rule.evaluate(
            RUN_ID,
            researcher,
            make_evidence(source_count=7, entity_count=16),  # drop 1
        )
        large_drop = self.rule.evaluate(
            RUN_ID,
            researcher,
            make_evidence(source_count=4, entity_count=16),  # drop 4
        )
        assert large_drop.confidence > small_drop.confidence

    def test_larger_gain_gives_higher_confidence(self):
        """A 5-source gain must give higher confidence than a 1-source gain."""
        researcher = make_evidence(source_count=8, entity_count=16)

        small_gain = self.rule.evaluate(
            RUN_ID,
            researcher,
            make_evidence(source_count=9,  entity_count=16),  # gain 1
        )
        large_gain = self.rule.evaluate(
            RUN_ID,
            researcher,
            make_evidence(source_count=13, entity_count=16),  # gain 5
        )
        assert large_gain.confidence > small_gain.confidence

    def test_total_drop_gives_near_max_confidence(self):
        """If writer has 0 sources, confidence should be near 0.99."""
        researcher = make_evidence(source_count=8, entity_count=16)
        writer     = make_evidence(source_count=0, entity_count=0)
        result = self.rule.evaluate(RUN_ID, researcher, writer)
        assert result.verdict == "FAIL"
        assert result.confidence >= 0.95

    def test_fail_confidence_always_above_warning_confidence(self):
        """For the same drop ratio, FAIL must always have higher confidence than WARNING."""
        researcher = make_evidence(source_count=8, entity_count=16)

        # WARNING: small drop
        warning_result = self.rule.evaluate(
            RUN_ID, researcher,
            make_evidence(source_count=7, entity_count=16),
        )
        # FAIL: large drop
        fail_result = self.rule.evaluate(
            RUN_ID, researcher,
            make_evidence(source_count=4, entity_count=16),
        )
        assert fail_result.confidence > warning_result.confidence


# ─────────────────────────────────────────────────────────────────────────────
# Summary and result fields
# ─────────────────────────────────────────────────────────────────────────────

class TestResultFields:

    def setup_method(self):
        self.rule = InformationLossRule()

    def test_summary_is_string(self):
        researcher = make_evidence(source_count=8, entity_count=16)
        writer     = make_evidence(source_count=5, entity_count=16)
        result = self.rule.evaluate(RUN_ID, researcher, writer)
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0

    def test_summary_contains_verdict(self):
        researcher = make_evidence(source_count=8, entity_count=16)
        writer     = make_evidence(source_count=5, entity_count=16)
        result = self.rule.evaluate(RUN_ID, researcher, writer)
        assert result.verdict in result.summary

    def test_rule_id_is_set(self):
        researcher = make_evidence(source_count=8, entity_count=16)
        writer     = make_evidence(source_count=8, entity_count=16)
        result = self.rule.evaluate(RUN_ID, researcher, writer)
        assert result.rule_id == "information_loss_v1"

    def test_rule_category_is_workflow(self):
        researcher = make_evidence(source_count=8, entity_count=16)
        writer     = make_evidence(source_count=8, entity_count=16)
        result = self.rule.evaluate(RUN_ID, researcher, writer)
        assert result.rule_category == "workflow"

    def test_timestamp_is_set(self):
        researcher = make_evidence(source_count=8, entity_count=16)
        writer     = make_evidence(source_count=8, entity_count=16)
        result = self.rule.evaluate(RUN_ID, researcher, writer)
        assert result.timestamp is not None
        assert "2026" in result.timestamp or "2025" in result.timestamp
