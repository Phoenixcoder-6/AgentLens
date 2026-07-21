"""
tests/test_arbiter.py — Day 12: Arbiter MVP tests
==================================================
Tests for determine_primary_cause() and Arbiter.run().

Day 12 requirement: "Unit test determinism: same evidence in, same output out, every time."

Test structure:
    TestDeterminism         — core requirement: same input → same output
    TestP2RuleMatch         — P2 evidence fires correct verdict
    TestP5Fallback          — P5 when nothing matches
    TestTieBreak            — multiple P2 rules → tie-break by rule_id ascending
    TestEvidenceConversion  — evidence_from_information_loss() helper
    TestArbiterClass        — Arbiter.run() sorts before deciding
"""

from __future__ import annotations

import random
import pytest
from copy import deepcopy

from schema.models import (
    AnalysisBundle,
    EvidenceRecord,
    EvidenceSource,
    FailureCategory,
    PriorityLevel,
    RuleMatch,
    RuleSeverity,
    SCHEMA_VERSION,
)
from analyzers.arbiter import (
    Arbiter,
    determine_primary_cause,
    evidence_from_information_loss,
    _tiebreak_key,
)
from analyzers.detection.information_loss import InformationLossResult, FieldDiff


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

RUN_ID = "run_test_arbiter"


def make_rule_evidence(
    rule_id: str = "information_loss_v1",
    category: FailureCategory = FailureCategory.REASONING,
    confidence: float = 0.65,
    agent: str = "writer",
) -> EvidenceRecord:
    rule = RuleMatch(
        rule_id=rule_id,
        category=category,
        description=f"Test rule {rule_id}",
        severity=RuleSeverity.MEDIUM,
        agent=agent,
    )
    return EvidenceRecord(
        source=EvidenceSource.RULE_ENGINE,
        description=f"Test P2 evidence for {rule_id}",
        value="WARNING",
        rule_match=rule,
        agent=agent,
        confidence=confidence,
    )


def make_loss_result(
    verdict: str = "WARNING",
    confidence: float = 0.65,
    src_researcher: int = 8,
    src_writer: int = 11,
    ent_researcher: int = 16,
    ent_writer: int = 17,
    rule_failed: bool = False,
) -> InformationLossResult:
    def _signal(r, w):
        if w > r: return "ADDED"
        if w < r: return "DROPPED"
        return "PRESERVED"

    def _severity(delta):
        if delta == 0: return "NONE"
        if delta >= 3: return "HIGH"
        if delta >= 1: return "MEDIUM"
        return "LOW"

    source_diff = FieldDiff(
        field_name="source_count",
        researcher_value=src_researcher,
        writer_value=src_writer,
        delta=src_writer - src_researcher,
        signal=_signal(src_researcher, src_writer),
        severity=_severity(abs(src_writer - src_researcher)),
    )
    entity_diff = FieldDiff(
        field_name="entity_count",
        researcher_value=ent_researcher,
        writer_value=ent_writer,
        delta=ent_writer - ent_researcher,
        signal=_signal(ent_researcher, ent_writer),
        severity=_severity(abs(ent_writer - ent_researcher)),
    )
    return InformationLossResult(
        schema_version=SCHEMA_VERSION,
        run_id=RUN_ID,
        verdict=verdict,
        confidence=confidence,
        source_diff=source_diff,
        entity_diff=entity_diff,
        has_information_gain=(verdict == "WARNING" and src_writer > src_researcher),
        has_information_loss=(verdict == "FAIL"),
        rule_failed=rule_failed,
        summary=(
            f"Verdict: {verdict}\n"
            f"  source_count: {src_researcher}→{src_writer}\n"
            f"  entity_count: {ent_researcher}→{ent_writer}"
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# TestDeterminism — core Day 12 requirement
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism:
    """
    CORE REQUIREMENT: same evidence in → same output out, every time.
    Each scenario runs 10 times; all results must be identical.
    """

    def _assert_deterministic(self, evidence: list, runs: int = 10):
        results = [determine_primary_cause(deepcopy(evidence), RUN_ID) for _ in range(runs)]
        first = results[0]
        for i, r in enumerate(results[1:], start=2):
            assert r.primary_cause   == first.primary_cause,   f"Run {i}: primary_cause differs"
            assert r.priority_level  == first.priority_level,  f"Run {i}: priority_level differs"
            assert r.grounded        == first.grounded,         f"Run {i}: grounded differs"
            assert r.primary_agent   == first.primary_agent,    f"Run {i}: primary_agent differs"

    def test_deterministic_empty_evidence(self):
        self._assert_deterministic([])

    def test_deterministic_single_p2(self):
        self._assert_deterministic([make_rule_evidence("information_loss_v1")])

    def test_deterministic_multiple_p2_rules(self):
        self._assert_deterministic([
            make_rule_evidence("rule_b"),
            make_rule_evidence("rule_a"),
            make_rule_evidence("rule_c"),
        ])

    def test_deterministic_regardless_of_input_order(self):
        """Shuffling input must not change verdict."""
        ev_a = make_rule_evidence("rule_alpha", confidence=0.9)
        ev_b = make_rule_evidence("rule_beta",  confidence=0.5)

        result_ab = determine_primary_cause([ev_a, ev_b], RUN_ID)
        result_ba = determine_primary_cause([ev_b, ev_a], RUN_ID)

        assert result_ab.primary_cause  == result_ba.primary_cause
        assert result_ab.priority_level == result_ba.priority_level
        assert result_ab.primary_agent  == result_ba.primary_agent

    def test_arbiter_run_deterministic_with_shuffled_input(self):
        arbiter = Arbiter()
        evidence = [
            make_rule_evidence("rule_c"),
            make_rule_evidence("rule_a"),
            make_rule_evidence("rule_b"),
        ]
        results = []
        for _ in range(10):
            shuffled = evidence[:]
            random.shuffle(shuffled)
            results.append(arbiter.run(RUN_ID, shuffled))

        first = results[0]
        for r in results[1:]:
            assert r.primary_cause  == first.primary_cause
            assert r.priority_level == first.priority_level
            assert r.primary_agent  == first.primary_agent


# ─────────────────────────────────────────────────────────────────────────────
# TestP2RuleMatch
# ─────────────────────────────────────────────────────────────────────────────

class TestP2RuleMatch:

    def test_p2_priority_on_rule_engine_evidence(self):
        bundle = determine_primary_cause([make_rule_evidence()], RUN_ID)
        assert bundle.priority_level == PriorityLevel.P2

    def test_p2_reasoning_category(self):
        bundle = determine_primary_cause(
            [make_rule_evidence(category=FailureCategory.REASONING)], RUN_ID)
        assert bundle.primary_cause == FailureCategory.REASONING

    def test_p2_workflow_category(self):
        bundle = determine_primary_cause(
            [make_rule_evidence(category=FailureCategory.WORKFLOW)], RUN_ID)
        assert bundle.primary_cause == FailureCategory.WORKFLOW

    def test_p2_not_grounded(self):
        bundle = determine_primary_cause([make_rule_evidence()], RUN_ID)
        assert bundle.grounded is False

    def test_p2_agent_attributed(self):
        bundle = determine_primary_cause([make_rule_evidence(agent="writer")], RUN_ID)
        assert bundle.primary_agent == "writer"

    def test_p2_all_evidence_attached(self):
        bundle = determine_primary_cause(
            [make_rule_evidence("r1"), make_rule_evidence("r2")], RUN_ID)
        assert len(bundle.evidence) == 2

    def test_p2_rule_matches_populated(self):
        bundle = determine_primary_cause([make_rule_evidence()], RUN_ID)
        assert len(bundle.rule_matches) == 1

    def test_p2_run_id_preserved(self):
        bundle = determine_primary_cause([make_rule_evidence()], RUN_ID)
        assert bundle.run_id == RUN_ID

    def test_p2_summary_contains_rule_id(self):
        bundle = determine_primary_cause(
            [make_rule_evidence("information_loss_v1")], RUN_ID)
        assert "information_loss_v1" in bundle.summary


# ─────────────────────────────────────────────────────────────────────────────
# TestP5Fallback
# ─────────────────────────────────────────────────────────────────────────────

class TestP5Fallback:

    def test_p5_on_empty_evidence(self):
        assert determine_primary_cause([], RUN_ID).priority_level == PriorityLevel.P5

    def test_p5_cause_is_unknown(self):
        assert determine_primary_cause([], RUN_ID).primary_cause == FailureCategory.UNKNOWN

    def test_p5_not_grounded(self):
        assert determine_primary_cause([], RUN_ID).grounded is False

    def test_p5_no_primary_agent(self):
        assert determine_primary_cause([], RUN_ID).primary_agent is None

    def test_p5_on_metrics_only_evidence(self):
        """Day 12 MVP: METRICS_ANALYZER evidence falls through to P5."""
        ev = EvidenceRecord(
            source=EvidenceSource.METRICS_ANALYZER,
            description="latency spike",
            confidence=0.9,
        )
        bundle = determine_primary_cause([ev], RUN_ID)
        assert bundle.priority_level == PriorityLevel.P5

    def test_p5_always_returns_bundle(self):
        bundle = determine_primary_cause([], RUN_ID)
        assert isinstance(bundle, AnalysisBundle)


# ─────────────────────────────────────────────────────────────────────────────
# TestTieBreak
# ─────────────────────────────────────────────────────────────────────────────

class TestTieBreak:

    def test_lowest_rule_id_wins(self):
        evidence = [
            make_rule_evidence("rule_c"),
            make_rule_evidence("rule_a"),   # ← wins (alphabetically first)
            make_rule_evidence("rule_b"),
        ]
        bundle = determine_primary_cause(evidence, RUN_ID)
        assert "rule_a" in bundle.summary

    def test_tiebreak_key_no_rule_sorts_last(self):
        ev = EvidenceRecord(
            source=EvidenceSource.RULE_ENGINE,
            description="no rule",
            confidence=0.9,
        )
        assert _tiebreak_key(ev) == "zzz"

    def test_tiebreak_key_with_rule_id(self):
        assert _tiebreak_key(make_rule_evidence("rule_aaa")) == "rule_aaa"

    def test_tiebreak_alphabetical_not_by_confidence(self):
        """Higher confidence must NOT override rule_id tie-break."""
        ev_high = make_rule_evidence("rule_z", confidence=0.99)
        ev_low  = make_rule_evidence("rule_a", confidence=0.10)
        bundle = determine_primary_cause([ev_high, ev_low], RUN_ID)
        assert "rule_a" in bundle.summary   # rule_a wins despite lower confidence


# ─────────────────────────────────────────────────────────────────────────────
# TestEvidenceConversion
# ─────────────────────────────────────────────────────────────────────────────

class TestEvidenceConversion:

    def test_warning_produces_evidence(self):
        ev = evidence_from_information_loss(make_loss_result("WARNING"))
        assert ev is not None and isinstance(ev, EvidenceRecord)

    def test_fail_produces_evidence(self):
        ev = evidence_from_information_loss(make_loss_result("FAIL", src_writer=2))
        assert ev is not None

    def test_pass_produces_none(self):
        assert evidence_from_information_loss(make_loss_result("PASS")) is None

    def test_rule_failed_produces_none(self):
        assert evidence_from_information_loss(make_loss_result(rule_failed=True)) is None

    def test_warning_source_is_rule_engine(self):
        ev = evidence_from_information_loss(make_loss_result("WARNING"))
        assert ev.source == EvidenceSource.RULE_ENGINE

    def test_warning_category_is_reasoning(self):
        ev = evidence_from_information_loss(make_loss_result("WARNING"))
        assert ev.rule_match.category == FailureCategory.REASONING

    def test_fail_category_is_workflow(self):
        ev = evidence_from_information_loss(make_loss_result("FAIL", src_writer=2))
        assert ev.rule_match.category == FailureCategory.WORKFLOW

    def test_confidence_preserved(self):
        ev = evidence_from_information_loss(make_loss_result("WARNING", confidence=0.65))
        assert ev.confidence == 0.65

    def test_rule_id_preserved(self):
        ev = evidence_from_information_loss(make_loss_result("WARNING"))
        assert ev.rule_match.rule_id == "information_loss_v1"

    def test_agent_is_writer(self):
        ev = evidence_from_information_loss(make_loss_result("WARNING"))
        assert ev.agent == "writer"

    def test_end_to_end_warning_reaches_p2_reasoning(self):
        ev = evidence_from_information_loss(make_loss_result("WARNING"))
        bundle = Arbiter().run(RUN_ID, [ev])
        assert bundle.priority_level == PriorityLevel.P2
        assert bundle.primary_cause  == FailureCategory.REASONING

    def test_end_to_end_fail_reaches_p2_workflow(self):
        ev = evidence_from_information_loss(make_loss_result("FAIL", src_writer=2))
        bundle = Arbiter().run(RUN_ID, [ev])
        assert bundle.priority_level == PriorityLevel.P2
        assert bundle.primary_cause  == FailureCategory.WORKFLOW

    def test_end_to_end_pass_reaches_p5(self):
        ev = evidence_from_information_loss(make_loss_result("PASS"))
        bundle = Arbiter().run(RUN_ID, [e for e in [ev] if e is not None])
        assert bundle.priority_level == PriorityLevel.P5


# ─────────────────────────────────────────────────────────────────────────────
# TestArbiterClass
# ─────────────────────────────────────────────────────────────────────────────

class TestArbiterClass:

    def test_run_returns_bundle(self):
        assert isinstance(Arbiter().run(RUN_ID, []), AnalysisBundle)

    def test_run_empty_is_p5(self):
        assert Arbiter().run(RUN_ID, []).priority_level == PriorityLevel.P5

    def test_run_with_p2_gives_p2(self):
        assert Arbiter().run(RUN_ID, [make_rule_evidence()]).priority_level == PriorityLevel.P2

    def test_run_sorts_for_determinism(self):
        evidence = [make_rule_evidence("rule_c"), make_rule_evidence("rule_a")]
        b1 = Arbiter().run(RUN_ID, evidence)
        b2 = Arbiter().run(RUN_ID, list(reversed(evidence)))
        assert b1.primary_cause  == b2.primary_cause
        assert b1.priority_level == b2.priority_level
