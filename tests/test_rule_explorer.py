"""
tests/test_rule_explorer.py — Day 32 tests for the rule_matches table,
rule catalog and dashboard.state rule-stats helpers.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analyzers.rule_catalog import RULE_CATALOG, normalize_rule_id  # noqa: E402
from dashboard import state  # noqa: E402
from storage.db import DatabaseManager  # noqa: E402


@pytest.fixture
def db(tmp_path):
    d = DatabaseManager(str(tmp_path / "t.db"))
    d.initialize()
    for rid in ("run-1", "run-2"):
        d.insert_run(rid, "wf", "2026-01-01T00:00:00", "success", 1.0, 1, "1.0")
    return d


def _m(rule_id: str, category: str = "workflow", agent: str = "writer") -> dict:
    return {
        "rule_id": rule_id,
        "rule_version": "1.0.0",
        "category": category,
        "severity": "high",
        "agent": agent,
        "step": 1,
        "description": "d",
    }


def test_initialize_creates_table_and_is_idempotent(db):
    db.initialize()  # second call must not fail
    assert db.rule_match_count() == 0


def test_insert_is_idempotent_per_run(db):
    db.insert_rule_matches("run-1", [_m("hallucination_v1"), _m("tool_failure_v1")])
    db.insert_rule_matches("run-1", [_m("hallucination_v1")])  # re-analysis replaces
    assert db.rule_match_count() == 1


def test_stats_counts_and_example_run(db):
    db.insert_rule_matches("run-1", [_m("hallucination_v1")])
    db.insert_rule_matches("run-2", [_m("hallucination_v1")])
    stats = {s["rule_id"]: s for s in db.get_rule_stats()}
    assert stats["hallucination_v1"]["times_fired"] == 2
    assert stats["hallucination_v1"]["example_run"] == "run-2"  # most recent
    assert stats["hallucination_v1"]["last_triggered"]


def test_get_rule_stats_includes_never_fired_rules(db):
    rows = state.get_rule_stats(db=db)
    assert {r["rule_id"] for r in rows} >= set(RULE_CATALOG)
    assert all(r["times_fired"] == 0 for r in rows)


def test_get_rule_stats_sorted_and_filtered(db):
    db.insert_rule_matches("run-1", [_m("tool_failure_v1", "execution")])
    db.insert_rule_matches("run-2", [_m("tool_failure_v1", "execution")])
    db.insert_rule_matches("run-2", [_m("tool_failure_v1", "execution"), _m("claim_drift_v1")])
    rows = state.get_rule_stats(db=db)
    assert rows[0]["rule_id"] == "tool_failure_v1"  # most fired first
    only_exec = state.get_rule_stats(category="execution", db=db)
    assert only_exec and all(r["category"] == "execution" for r in only_exec)


def test_normalize_statistical_ids():
    assert normalize_rule_id("STAT-LAT-RESEARCHER-001") == "STAT-LAT"
    assert normalize_rule_id("STAT-TOK-WRITER-002") == "STAT-TOK"
    assert normalize_rule_id("hallucination_v1") == "hallucination_v1"


def test_persist_rule_matches_normalizes_and_never_raises(db):
    bundle = SimpleNamespace(
        rule_matches=[
            SimpleNamespace(
                rule_id="STAT-LAT-WRITER-003",
                rule_version="1.0.0",
                category="execution",
                severity="high",
                agent="writer",
                step=3,
                description="slow",
            )
        ]
    )
    assert state.persist_rule_matches("run-1", bundle, db=db) == 1
    assert {s["rule_id"] for s in db.get_rule_stats()} == {"STAT-LAT"}
    # Bad input must not raise
    assert state.persist_rule_matches("run-1", object(), db=db) == 0
