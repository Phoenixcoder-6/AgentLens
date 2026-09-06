"""
tests/test_statistical_detector.py — Day 27: StatisticalDetector tests
=======================================================================
Tests for the per-agent statistical outlier detector.

Approach:
    - Build in-memory SQLite DBs seeded with controlled latency/token data.
    - Verify baseline computation, outlier flagging, source tagging, and
      confidence scaling — all without any LLM calls or network access.
"""

from __future__ import annotations

import pytest

from analyzers.detection.statistical_detector import (
    AgentBaseline,
    StatisticalAnomalyReport,
    StatisticalDetector,
    _confidence_from_z,
    _mean,
    _stddev,
)
from schema.models import (
    SCHEMA_VERSION,
    EvidenceSource,
    PriorityLevel,
)
from storage.db import DatabaseManager

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

TS = "2026-09-06T12:00:00+00:00"


@pytest.fixture
def tmp_db(tmp_path) -> DatabaseManager:
    db = DatabaseManager(db_path=str(tmp_path / "test_stat.db"))
    db.initialize()
    return db


def _seed_run(
    db: DatabaseManager,
    run_id: str,
    latencies: list[float],
    tokens: list[int],
    agents: list[str] | None = None,
) -> None:
    """Seed one run with N steps into the DB."""
    _agents = agents or ["researcher", "writer", "verifier"]
    total_lat = sum(latencies)
    total_tok = sum(tokens)
    db.insert_run(run_id, "test_pipeline", TS, "SUCCESS", total_lat, total_tok, SCHEMA_VERSION)
    for i, (lat, tok) in enumerate(zip(latencies, tokens, strict=False)):
        agent = _agents[i % len(_agents)]
        db.insert_step(
            run_id,
            i + 1,
            agent,
            "SUCCESS",
            lat,
            tok // 3,
            tok - tok // 3 * 2,
            tok,
            "",
            TS,
            SCHEMA_VERSION,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests: stats helpers
# ─────────────────────────────────────────────────────────────────────────────


class TestStatsHelpers:
    def test_mean_empty(self):
        assert _mean([]) == 0.0

    def test_mean_simple(self):
        assert _mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_stddev_single(self):
        assert _stddev([42.0]) == 0.0

    def test_stddev_known(self):
        # Sample stddev (n-1 denominator) of [2, 4, 4, 4, 5, 5, 7, 9]
        # Population stddev = 2.0, but sample stddev = sqrt(sum_sq/(n-1)) ≈ 2.138
        import math as _math

        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        m = sum(values) / len(values)
        expected = _math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))
        assert _stddev(values) == pytest.approx(expected, rel=1e-6)

    def test_confidence_at_threshold(self):
        # Exactly at threshold sigma: confidence = 0.5
        c = _confidence_from_z(2.5, 2.5)
        assert c == pytest.approx(0.5, abs=0.01)

    def test_confidence_above_threshold(self):
        # 1 sigma above threshold → 0.5 + 0.1 = 0.6
        c = _confidence_from_z(3.5, 2.5)
        assert c == pytest.approx(0.6, abs=0.01)

    def test_confidence_capped_at_0_99(self):
        c = _confidence_from_z(100.0, 2.5)
        assert c == pytest.approx(0.99, abs=0.001)


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests: baseline building
# ─────────────────────────────────────────────────────────────────────────────


class TestBaselineBuilding:
    def test_no_baselines_with_insufficient_runs(self, tmp_db):
        """Should return empty baselines when fewer than min_runs_for_baseline."""
        # Default min_runs_for_baseline = 5; seed only 3 runs
        for i in range(3):
            _seed_run(
                tmp_db,
                f"run_{i}",
                [100.0, 200.0],
                [500, 1000],
                agents=["researcher", "writer"],
            )
        detector = StatisticalDetector(tmp_db)
        baselines = detector.get_baselines()
        # With min_runs=5, no agent should have a baseline yet
        assert len(baselines) == 0

    def test_baselines_computed_when_enough_runs(self, tmp_db):
        """Baseline present after >= min_runs_for_baseline runs."""
        for i in range(6):
            _seed_run(
                tmp_db,
                f"run_{i}",
                [100.0],
                [500],
                agents=["researcher"],
            )
        detector = StatisticalDetector(tmp_db)
        baselines = detector.get_baselines()
        assert "researcher" in baselines
        bl = baselines["researcher"]
        assert isinstance(bl, AgentBaseline)
        assert bl.n_runs == 6
        assert bl.latency_mean == pytest.approx(100.0)
        assert bl.token_mean == pytest.approx(500.0)

    def test_baseline_excludes_current_run(self, tmp_db):
        """Baselines should NOT include the run being analyzed."""
        # All historical runs: latency 100 ms
        for i in range(6):
            _seed_run(tmp_db, f"hist_{i}", [100.0], [500], agents=["researcher"])
        # The run under analysis has outlier latency
        _seed_run(tmp_db, "current_run", [9999.0], [500], agents=["researcher"])

        detector = StatisticalDetector(tmp_db)
        report = detector.analyze_run("current_run")
        # Baseline should be based on hist_* runs only, not current
        bl = report.baselines_used.get("researcher")
        assert bl is not None
        assert bl.latency_mean == pytest.approx(100.0, rel=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests: anomaly detection
# ─────────────────────────────────────────────────────────────────────────────


class TestAnomalyDetection:
    def _seed_historical(self, db: DatabaseManager, n: int = 6) -> None:
        """Seed n runs with stable latency=100ms, tokens=500."""
        for i in range(n):
            _seed_run(db, f"hist_{i}", [100.0], [500], agents=["researcher"])

    def test_no_anomaly_for_normal_step(self, tmp_db):
        """Normal latency should produce no anomalies."""
        # Seed with slight variation so stddev > 0
        for i, lat in enumerate([100.0, 105.0, 98.0, 102.0, 99.0, 101.0]):
            _seed_run(tmp_db, f"hist_{i}", [lat], [500], agents=["researcher"])
        # Normal step: latency within 1 sigma of mean, same tokens as historical
        _seed_run(tmp_db, "run_normal", [103.0], [500], agents=["researcher"])

        detector = StatisticalDetector(tmp_db)
        report = detector.analyze_run("run_normal")
        assert isinstance(report, StatisticalAnomalyReport)
        assert not report.has_anomalies
        assert report.anomalies == []

    def test_latency_outlier_flagged(self, tmp_db):
        """Latency far above mean should produce a STATISTICAL_ANOMALY record."""
        # Seed stable latencies with slight variation so stddev > 0
        for i, lat in enumerate([100.0, 105.0, 98.0, 102.0, 99.0, 101.0]):
            _seed_run(tmp_db, f"hist_{i}", [lat], [500], agents=["researcher"])
        # Current run: huge latency spike
        _seed_run(tmp_db, "run_spike", [9000.0], [500], agents=["researcher"])

        detector = StatisticalDetector(tmp_db)
        report = detector.analyze_run("run_spike")

        assert report.has_anomalies
        lat_anomalies = [e for e in report.anomalies if "Latency outlier" in e.description]
        assert len(lat_anomalies) >= 1
        anomaly = lat_anomalies[0]
        assert anomaly.source == EvidenceSource.STATISTICAL_ANOMALY
        assert anomaly.agent == "researcher"
        assert anomaly.step == 1

    def test_token_outlier_flagged(self, tmp_db):
        """Token count far above mean should produce a STATISTICAL_ANOMALY record."""
        for i, tok in enumerate([500, 510, 490, 505, 495, 502]):
            _seed_run(tmp_db, f"hist_{i}", [100.0], [tok], agents=["writer"])
        _seed_run(tmp_db, "run_tokens", [100.0], [50000], agents=["writer"])

        detector = StatisticalDetector(tmp_db)
        report = detector.analyze_run("run_tokens")

        tok_anomalies = [e for e in report.anomalies if "Token outlier" in e.description]
        assert len(tok_anomalies) >= 1
        assert tok_anomalies[0].source == EvidenceSource.STATISTICAL_ANOMALY

    def test_skipped_agents_when_no_baseline(self, tmp_db):
        """Agent with no history should appear in skipped_agents, not raise."""
        # Only seed researcher history
        for i in range(6):
            _seed_run(tmp_db, f"hist_{i}", [100.0], [500], agents=["researcher"])
        # Current run has a brand-new agent "auditor" never seen before
        _seed_run(tmp_db, "run_new_agent", [200.0], [600], agents=["auditor"])

        detector = StatisticalDetector(tmp_db)
        report = detector.analyze_run("run_new_agent")
        assert "auditor" in report.skipped_agents
        assert not report.has_anomalies

    def test_evidence_record_has_rule_match(self, tmp_db):
        """Each anomaly evidence record should have a populated rule_match."""
        for i in range(6):
            _seed_run(tmp_db, f"hist_{i}", [100.0], [500], agents=["researcher"])
        _seed_run(tmp_db, "run_spike", [9000.0], [500], agents=["researcher"])

        detector = StatisticalDetector(tmp_db)
        report = detector.analyze_run("run_spike")

        for e in report.anomalies:
            assert e.rule_match is not None
            assert e.rule_match.rule_id.startswith("STAT-")

    def test_confidence_is_in_valid_range(self, tmp_db):
        """Confidence must be in [0.0, 1.0] for all anomaly records."""
        for i in range(6):
            _seed_run(tmp_db, f"hist_{i}", [100.0], [500], agents=["researcher"])
        _seed_run(tmp_db, "run_spike", [9000.0], [500], agents=["researcher"])

        detector = StatisticalDetector(tmp_db)
        report = detector.analyze_run("run_spike")

        for e in report.anomalies:
            assert 0.0 <= e.confidence <= 1.0

    def test_empty_run_returns_no_anomalies(self, tmp_db):
        """analyze_run on a run with no steps should return empty report."""
        # Just insert the run header, no steps
        tmp_db.insert_run("run_empty", "pipeline", TS, "SUCCESS", 0.0, 0, SCHEMA_VERSION)
        detector = StatisticalDetector(tmp_db)
        report = detector.analyze_run("run_empty")
        assert not report.has_anomalies

    def test_statistical_anomaly_maps_to_p4_in_arbiter(self):
        """EvidenceSource.STATISTICAL_ANOMALY must map to P4 in arbiter priority table."""
        from analyzers.arbiter import SOURCE_TO_PRIORITY

        assert EvidenceSource.STATISTICAL_ANOMALY in SOURCE_TO_PRIORITY
        assert SOURCE_TO_PRIORITY[EvidenceSource.STATISTICAL_ANOMALY] == PriorityLevel.P4
