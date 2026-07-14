# tests/test_arbiter.py
"""
Tests for analyzers/arbiter.py — Arbiter priority table and pseudocode contracts.

Covers:
  - P1–P5 priority table correctness
  - Priority ordering (P1 highest, P5 lowest)
  - SOURCE_TO_PRIORITY mapping for every EvidenceSource
  - Stub raises NotImplementedError (implementation begins Day 12)
  - Tie-break key function
  - P3 is a distinct entry (not merged into P2) — Day 18a requirement
"""

import pytest
from analyzers.arbiter import (
    SOURCE_TO_PRIORITY,
    PRIORITY_ORDER,
    determine_primary_cause,
    Arbiter,
    _priority_rank,
    _tiebreak_key,
)
from schema import PriorityLevel, EvidenceSource, EvidenceRecord, RuleMatch, FailureCategory, RuleSeverity


# ── Priority table completeness ───────────────────────────────────────────────

def test_all_five_priority_levels_exist():
    assert len(PRIORITY_ORDER) == 5
    for level in [PriorityLevel.P1, PriorityLevel.P2, PriorityLevel.P3,
                  PriorityLevel.P4, PriorityLevel.P5]:
        assert level in PRIORITY_ORDER


def test_p1_is_highest_priority():
    assert PRIORITY_ORDER[0] == PriorityLevel.P1


def test_p5_is_lowest_priority():
    assert PRIORITY_ORDER[4] == PriorityLevel.P5


def test_priority_rank_ordering():
    """Lower rank = higher priority."""
    assert _priority_rank(PriorityLevel.P1) < _priority_rank(PriorityLevel.P2)
    assert _priority_rank(PriorityLevel.P2) < _priority_rank(PriorityLevel.P3)
    assert _priority_rank(PriorityLevel.P3) < _priority_rank(PriorityLevel.P4)
    assert _priority_rank(PriorityLevel.P4) < _priority_rank(PriorityLevel.P5)


# ── SOURCE_TO_PRIORITY mapping ────────────────────────────────────────────────

def test_ground_truth_maps_to_p1():
    assert SOURCE_TO_PRIORITY[EvidenceSource.GROUND_TRUTH] == PriorityLevel.P1


def test_rule_engine_maps_to_p2():
    assert SOURCE_TO_PRIORITY[EvidenceSource.RULE_ENGINE] == PriorityLevel.P2


def test_workflow_validator_maps_to_p3():
    """P3 must be a distinct mapping — Day 18a requirement."""
    assert SOURCE_TO_PRIORITY[EvidenceSource.WORKFLOW_VALIDATOR] == PriorityLevel.P3


def test_consistency_validator_maps_to_p3():
    """Consistency validator is also P3 — secondary cause tier."""
    assert SOURCE_TO_PRIORITY[EvidenceSource.CONSISTENCY_VALIDATOR] == PriorityLevel.P3


def test_metrics_analyzer_maps_to_p4():
    assert SOURCE_TO_PRIORITY[EvidenceSource.METRICS_ANALYZER] == PriorityLevel.P4


def test_p3_is_distinct_from_p2():
    """
    Critical Day 18a requirement:
    P3 (workflow_violation) must be a separate priority from P2 (rule_match).
    They must NOT map to the same PriorityLevel.
    """
    p2 = SOURCE_TO_PRIORITY[EvidenceSource.RULE_ENGINE]
    p3 = SOURCE_TO_PRIORITY[EvidenceSource.WORKFLOW_VALIDATOR]
    assert p2 != p3, "P3 must be distinct from P2 — see Day 18a build plan note"


# ── Tie-break key ─────────────────────────────────────────────────────────────

def test_tiebreak_key_uses_rule_id():
    rule = RuleMatch(
        rule_id="R-WF-001",
        category=FailureCategory.WORKFLOW,
        description="test",
    )
    ev = EvidenceRecord(
        source=EvidenceSource.RULE_ENGINE,
        description="test",
        rule_match=rule,
    )
    assert _tiebreak_key(ev) == "R-WF-001"


def test_tiebreak_key_sorts_ascending():
    """Lower rule_id should sort before higher rule_id (tie-break rule)."""
    rule_a = RuleMatch(rule_id="R-WF-001", category=FailureCategory.WORKFLOW, description="a")
    rule_b = RuleMatch(rule_id="R-WF-005", category=FailureCategory.WORKFLOW, description="b")
    ev_a = EvidenceRecord(source=EvidenceSource.RULE_ENGINE, description="a", rule_match=rule_a)
    ev_b = EvidenceRecord(source=EvidenceSource.RULE_ENGINE, description="b", rule_match=rule_b)
    assert _tiebreak_key(ev_a) < _tiebreak_key(ev_b)


def test_tiebreak_key_no_rule_sorts_last():
    ev = EvidenceRecord(source=EvidenceSource.METRICS_ANALYZER, description="latency spike")
    key = _tiebreak_key(ev)
    assert key == "zzz"  # no rule_id → sorts to end


# ── Stubs raise NotImplementedError ──────────────────────────────────────────

def test_determine_primary_cause_is_stub():
    """Implementation begins Day 12 — stub must raise NotImplementedError."""
    with pytest.raises(NotImplementedError):
        determine_primary_cause([], "run_test")


def test_arbiter_run_is_stub():
    """Arbiter.run() implementation begins Day 12."""
    arbiter = Arbiter(analyzers=[])
    with pytest.raises(NotImplementedError):
        arbiter.run(None)


def test_arbiter_stores_analyzers():
    """Arbiter accepts and stores the analyzers list even before implementation."""
    arbiter = Arbiter(analyzers=["fake_analyzer_1", "fake_analyzer_2"])
    assert len(arbiter.analyzers) == 2
