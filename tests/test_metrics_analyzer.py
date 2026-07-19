"""
tests/test_metrics_analyzer.py — Day 11: Metrics Analyzer tests
================================================================
Tests for MetricsAnalyzer: latency, tokens, execution_time per step,
anomaly detection, and run-level aggregates.

No LLM calls — all data is built directly in the test DB.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from storage.db import DatabaseManager
from analyzers.metrics_analyzer import MetricsAnalyzer, StepMetrics, RunMetrics, _mean, _stddev
from schema.models import SCHEMA_VERSION


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

TS = "2026-07-19T12:00:00+00:00"


@pytest.fixture
def tmp_db(tmp_path) -> DatabaseManager:
    db = DatabaseManager(db_path=str(tmp_path / "test_metrics.db"))
    db.initialize()
    return db


def _seed_run(db: DatabaseManager, run_id: str, latencies: list[float],
              tokens: list[int] = None) -> None:
    """Seed one run with N steps into the DB."""
    agents = ["researcher", "writer", "verifier"]
    total_latency = sum(latencies)
    db.insert_run(run_id, "test_pipeline", TS, "SUCCESS",
                  total_latency, sum(tokens or [0]*len(latencies)), SCHEMA_VERSION)

    for i, latency in enumerate(latencies):
        agent = agents[i % len(agents)]
        tok = tokens[i] if tokens else 0
        db.insert_step(run_id, i + 1, agent, "SUCCESS", latency,
                       tok // 3, tok - tok // 3 * 2, tok,
                       f"added=['key_{i}']", TS, SCHEMA_VERSION)
        db.insert_metric(run_id, i + 1, agent, "latency_ms", latency, "ms", TS, SCHEMA_VERSION)
        db.insert_metric(run_id, i + 1, agent, "execution_time_ms", latency, "ms", TS, SCHEMA_VERSION)
        db.insert_metric(run_id, i + 1, agent, "tokens_total", float(tok), "tokens", TS, SCHEMA_VERSION)


# ─────────────────────────────────────────────────────────────────────────────
# Stats helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestStatsHelpers:

    def test_mean_basic(self):
        assert _mean([1.0, 2.0, 3.0]) == 2.0

    def test_mean_empty(self):
        assert _mean([]) == 0.0

    def test_mean_single(self):
        assert _mean([5.0]) == 5.0

    def test_stddev_basic(self):
        # Sample stddev (n-1 denominator) for [2,4,4,4,5,5,7,9]:
        # mean=5.0, variance=32/7≈4.571, stddev≈2.138
        result = _stddev([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert abs(result - 2.138) < 0.01

    def test_stddev_empty(self):
        assert _stddev([]) == 0.0

    def test_stddev_single(self):
        assert _stddev([5.0]) == 0.0

    def test_stddev_identical_values(self):
        assert _stddev([3.0, 3.0, 3.0]) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# StepMetrics
# ─────────────────────────────────────────────────────────────────────────────

class TestStepMetrics:

    def test_schema_version_stamped(self):
        sm = StepMetrics(
            schema_version=SCHEMA_VERSION, run_id="r", step=1, agent="researcher",
            latency_ms=1000.0, latency_flag=False, latency_threshold_ms=5000.0,
            tokens_prompt=0, tokens_completion=0, tokens_total=0,
            execution_time_ms=1000.0, is_anomalous=False
        )
        assert sm.schema_version == SCHEMA_VERSION

    def test_latency_flag_set_when_above_threshold(self):
        sm = StepMetrics(
            schema_version=SCHEMA_VERSION, run_id="r", step=1, agent="researcher",
            latency_ms=6000.0, latency_flag=True, latency_threshold_ms=5000.0,
            tokens_prompt=0, tokens_completion=0, tokens_total=0,
            execution_time_ms=6000.0, is_anomalous=True,
            anomaly_reasons=["latency exceeded"]
        )
        assert sm.latency_flag is True
        assert sm.is_anomalous is True


# ─────────────────────────────────────────────────────────────────────────────
# MetricsAnalyzer — basic analysis
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricsAnalyzerBasic:

    def test_analyze_run_returns_run_metrics(self, tmp_db):
        _seed_run(tmp_db, "run_a", [2000.0, 3000.0, 1000.0])
        analyzer = MetricsAnalyzer(tmp_db)
        result = analyzer.analyze_run("run_a")
        assert isinstance(result, RunMetrics)

    def test_returns_none_for_unknown_run(self, tmp_db):
        analyzer = MetricsAnalyzer(tmp_db)
        result = analyzer.analyze_run("run_nonexistent")
        assert result is None

    def test_schema_version_stamped_on_run(self, tmp_db):
        _seed_run(tmp_db, "run_a", [2000.0, 3000.0, 1000.0])
        analyzer = MetricsAnalyzer(tmp_db)
        result = analyzer.analyze_run("run_a")
        assert result.schema_version == SCHEMA_VERSION

    def test_schema_version_stamped_on_steps(self, tmp_db):
        _seed_run(tmp_db, "run_a", [2000.0, 3000.0, 1000.0])
        analyzer = MetricsAnalyzer(tmp_db)
        result = analyzer.analyze_run("run_a")
        for step in result.steps:
            assert step.schema_version == SCHEMA_VERSION

    def test_step_count_correct(self, tmp_db):
        _seed_run(tmp_db, "run_a", [2000.0, 3000.0, 1000.0])
        analyzer = MetricsAnalyzer(tmp_db)
        result = analyzer.analyze_run("run_a")
        assert result.step_count == 3

    def test_total_latency_correct(self, tmp_db):
        _seed_run(tmp_db, "run_a", [2000.0, 3000.0, 1000.0])
        analyzer = MetricsAnalyzer(tmp_db)
        result = analyzer.analyze_run("run_a")
        assert result.total_latency_ms == 6000.0

    def test_total_execution_time_equals_total_latency(self, tmp_db):
        _seed_run(tmp_db, "run_a", [2000.0, 3000.0, 1000.0])
        analyzer = MetricsAnalyzer(tmp_db)
        result = analyzer.analyze_run("run_a")
        assert result.total_execution_time_ms == result.total_latency_ms

    def test_per_step_latency_correct(self, tmp_db):
        _seed_run(tmp_db, "run_a", [2772.0, 2795.0, 1274.0])
        analyzer = MetricsAnalyzer(tmp_db)
        result = analyzer.analyze_run("run_a")
        assert result.steps[0].latency_ms == 2772.0
        assert result.steps[1].latency_ms == 2795.0
        assert result.steps[2].latency_ms == 1274.0

    def test_agent_latency_map(self, tmp_db):
        _seed_run(tmp_db, "run_a", [2000.0, 3000.0, 1000.0])
        analyzer = MetricsAnalyzer(tmp_db)
        result = analyzer.analyze_run("run_a")
        assert "researcher" in result.agent_latency
        assert "writer" in result.agent_latency
        assert "verifier" in result.agent_latency

    def test_slowest_step_identified(self, tmp_db):
        _seed_run(tmp_db, "run_a", [2000.0, 5000.0, 1000.0])
        analyzer = MetricsAnalyzer(tmp_db)
        result = analyzer.analyze_run("run_a")
        assert result.slowest_step == "writer"   # 5000ms

    def test_fastest_step_identified(self, tmp_db):
        _seed_run(tmp_db, "run_a", [2000.0, 5000.0, 1000.0])
        analyzer = MetricsAnalyzer(tmp_db)
        result = analyzer.analyze_run("run_a")
        assert result.fastest_step == "verifier"  # 1000ms

    def test_analyze_latest_run(self, tmp_db):
        _seed_run(tmp_db, "run_a", [2000.0, 3000.0, 1000.0])
        analyzer = MetricsAnalyzer(tmp_db)
        result = analyzer.analyze_latest_run()
        assert result is not None
        assert result.run_id == "run_a"

    def test_analyze_latest_run_returns_none_on_empty_db(self, tmp_db):
        analyzer = MetricsAnalyzer(tmp_db)
        result = analyzer.analyze_latest_run()
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# MetricsAnalyzer — latency threshold anomaly
# ─────────────────────────────────────────────────────────────────────────────

class TestLatencyAnomaly:

    def test_no_anomaly_when_below_threshold(self, tmp_db):
        _seed_run(tmp_db, "run_a", [2000.0, 3000.0, 1000.0])
        analyzer = MetricsAnalyzer(tmp_db)
        result = analyzer.analyze_run("run_a")
        assert result.has_anomalies is False
        assert all(not s.is_anomalous for s in result.steps)

    def test_anomaly_when_above_threshold(self, tmp_db):
        # Default threshold is 5000ms from config
        _seed_run(tmp_db, "run_a", [2000.0, 6000.0, 1000.0])  # writer = 6000ms
        analyzer = MetricsAnalyzer(tmp_db)
        result = analyzer.analyze_run("run_a")
        assert result.has_anomalies is True
        assert "writer" in result.anomalous_steps

    def test_latency_flag_set_on_anomalous_step(self, tmp_db):
        _seed_run(tmp_db, "run_a", [2000.0, 7000.0, 1000.0])
        analyzer = MetricsAnalyzer(tmp_db)
        result = analyzer.analyze_run("run_a")
        writer_step = next(s for s in result.steps if s.agent == "writer")
        assert writer_step.latency_flag is True

    def test_anomaly_reason_populated(self, tmp_db):
        _seed_run(tmp_db, "run_a", [2000.0, 7000.0, 1000.0])
        analyzer = MetricsAnalyzer(tmp_db)
        result = analyzer.analyze_run("run_a")
        writer_step = next(s for s in result.steps if s.agent == "writer")
        assert len(writer_step.anomaly_reasons) > 0
        assert "7000" in writer_step.anomaly_reasons[0] or "threshold" in writer_step.anomaly_reasons[0]

    def test_anomalous_steps_list_correct(self, tmp_db):
        _seed_run(tmp_db, "run_a", [6000.0, 7000.0, 1000.0])  # researcher + writer both slow
        analyzer = MetricsAnalyzer(tmp_db)
        result = analyzer.analyze_run("run_a")
        assert "researcher" in result.anomalous_steps
        assert "writer" in result.anomalous_steps
        assert "verifier" not in result.anomalous_steps


# ─────────────────────────────────────────────────────────────────────────────
# MetricsAnalyzer — statistical baseline (requires min_runs_for_baseline runs)
# ─────────────────────────────────────────────────────────────────────────────

class TestStatisticalBaseline:

    def test_no_statistical_anomaly_below_min_runs(self, tmp_db):
        """Statistical detection inactive with fewer than min_runs_for_baseline runs."""
        # Seed only 2 runs (< min_runs_for_baseline = 5)
        _seed_run(tmp_db, "run_a", [2000.0, 3000.0, 1000.0])
        _seed_run(tmp_db, "run_b", [2000.0, 3000.0, 1000.0])
        analyzer = MetricsAnalyzer(tmp_db)
        baseline = analyzer._get_baseline()
        assert baseline["active"] is False

    def test_statistical_baseline_active_with_enough_runs(self, tmp_db):
        for i in range(6):
            _seed_run(tmp_db, f"run_{i}", [2000.0, 3000.0, 1000.0])
        analyzer = MetricsAnalyzer(tmp_db)
        baseline = analyzer._get_baseline()
        assert baseline["active"] is True
        assert "latency_mean" in baseline
        assert "latency_std" in baseline
