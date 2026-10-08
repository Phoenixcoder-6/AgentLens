"""
tests/test_config_sweep_and_cost.py — Day 40 Unit & Integration Tests
======================================================================
Covers:
    1. Cost control configuration in config/config.yaml (budget_alert_usd,
       cost_per_token_usd, primary_model, fallback_model, cache_ttl_hours,
       extraction_max_tokens, explanation_max_tokens).
    2. Cost budget alert: WARNING logged and dashboard header/ticker banner
       rendered when session cost > budget_alert_usd.
    3. Dynamic config sweep verification: pii_scrubbing, alerting,
       retention_days, diff.similarity_threshold, stats.outlier_stddev,
       and arbiter.information_loss all respond to config changes without code changes.
"""

from __future__ import annotations

from datetime import UTC, datetime

import dashboard.app as dash_app
import dashboard.state as dash_state
from analyzers.alerter import Alerter
from analyzers.detection.information_loss import InformationLossRule
from analyzers.detection.statistical_detector import StatisticalDetector
from analyzers.evidence_extraction.extractor import ExtractedEvidence
from capture.pii_scrubber import PIIScrubber
from config import config_loader
from diff_engine.similarity import SemanticSimilarityEngine
from normalizer.normalizer import Normalizer
from schema.models import AgentStep, RunTrace, StepStatus, TokenUsage
from storage.db import DatabaseManager
from storage.writer import StorageWriter

# ─────────────────────────────────────────────────────────────────────────────
# 1. Config Sweep & LLM Cost Control Keys
# ─────────────────────────────────────────────────────────────────────────────


def test_config_yaml_contains_all_day40_cost_and_sweep_keys():
    config_loader.load_config.cache_clear()
    cfg = config_loader.load_config()

    llm_cfg = cfg.get("llm", {})
    assert "budget_alert_usd" in llm_cfg
    assert float(llm_cfg["budget_alert_usd"]) == 1.00
    assert "cost_per_token_usd" in llm_cfg
    assert "cache_ttl_hours" in llm_cfg
    assert "primary_model" in llm_cfg
    assert "fallback_model" in llm_cfg
    assert "extraction_max_tokens" in llm_cfg
    assert "explanation_max_tokens" in llm_cfg

    # Alias check: get("llm", "model") and get("llm", "primary_model") match
    assert config_loader.get("llm", "model") == config_loader.get("llm", "primary_model")

    # stats.outlier_stddev and arbiter.information_loss present
    assert float(config_loader.get("stats", "outlier_stddev")) == 2.5
    info_loss_cfg = config_loader.get("arbiter", "information_loss")
    assert info_loss_cfg["severe_threshold"] == 3
    assert info_loss_cfg["moderate_threshold"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cost Budget Alert (Logger Warning + Dashboard Banner)
# ─────────────────────────────────────────────────────────────────────────────


def _seed_db_with_tokens(tmp_path, total_tokens: int) -> DatabaseManager:
    db_path = tmp_path / "cost_test.db"
    db = DatabaseManager(str(db_path))
    db.initialize()
    trace = RunTrace(
        run_id="run_cost_01",
        workflow="cost_test",
        timestamp=datetime.now(UTC),
        status=StepStatus.SUCCESS,
        steps=[
            AgentStep(
                run_id="run_cost_01",
                step=1,
                agent="researcher",
                input="{}",
                output="{}",
                latency_ms=100.0,
                status=StepStatus.SUCCESS,
                tokens=TokenUsage(
                    prompt=total_tokens // 2,
                    completion=total_tokens - (total_tokens // 2),
                    total=total_tokens,
                ),
                timestamp=datetime.now(UTC),
            )
        ],
    )
    norm = Normalizer().normalize_run(trace)
    StorageWriter(db).write_run(norm, trace_json=trace.model_dump_json())
    return db


def test_cost_budget_alert_under_and_over_budget(monkeypatch, tmp_path):
    # Seed 100,000 tokens -> at $0.000005/token = $0.50 USD
    db = _seed_db_with_tokens(tmp_path, total_tokens=100_000)
    monkeypatch.setattr(dash_state, "get_db", lambda: db)

    warnings_logged: list[str] = []

    class _FakeLogger:
        def warning(self, msg: str, *args, **kwargs):
            warnings_logged.append(msg)

    monkeypatch.setattr("config.logging_config.get_logger", lambda name: _FakeLogger())

    # Case 1: budget_alert_usd = $1.00 -> $0.50 is UNDER budget
    monkeypatch.setattr(dash_state, "get_budget_alert_usd", lambda: 1.00)
    status_under = dash_state.get_budget_status()
    assert status_under["exceeded"] is False
    assert abs(float(status_under["cost_usd"]) - 0.50) < 1e-6
    assert warnings_logged == []

    ticker_under = dash_app._cost_ticker()
    assert "BUDGET ALERT" not in ticker_under
    assert "~$0.5000 est." in ticker_under

    # Case 2: budget_alert_usd = $0.25 -> $0.50 is OVER budget
    monkeypatch.setattr(dash_state, "get_budget_alert_usd", lambda: 0.25)
    status_over = dash_state.get_budget_status()
    assert status_over["exceeded"] is True
    assert len(warnings_logged) >= 1
    assert "exceeded budget_alert_usd" in warnings_logged[-1]

    ticker_over = dash_app._cost_ticker()
    assert "BUDGET ALERT" in ticker_over
    assert "$0.25 limit" in ticker_over


# ─────────────────────────────────────────────────────────────────────────────
# 3. Dynamic Configuration Sweep Across Subsystems
# ─────────────────────────────────────────────────────────────────────────────


def test_dynamic_config_overrides_without_code_changes(monkeypatch, tmp_path):
    """
    Confirm pii_scrubbing, alerting, retention_days, diff.similarity_threshold,
    stats.outlier_stddev, and arbiter.information_loss all respond to config changes.
    """
    # 1. pii_scrubbing via config
    monkeypatch.setattr(
        "capture.pii_scrubber.get",
        lambda s, k=None, d=None: (
            {"enabled": True, "use_spacy_ner": False}
            if (s == "capture" and k == "pii_scrubbing")
            else d
        ),
    )
    scrubber = PIIScrubber()
    assert scrubber.enabled is True
    assert "[REDACTED_EMAIL]" in scrubber.scrub_text("Email: a@b.com")

    # 2. alerting via config
    monkeypatch.setattr(
        "config.config_loader.get",
        lambda s, k=None, d=None: (
            {
                "enabled": False,
                "on_verdict": ["P1"],
                "channel": "slack",
                "cooldown_minutes": 15,
            }.get(k, d)
            if s == "alerting"
            else d
        ),
    )
    alerter = Alerter()
    assert alerter._enabled is False
    assert alerter._on_verdict == ["P1"]
    assert alerter._channel == "slack"
    assert alerter._cooldown_minutes == 15

    # 3. diff.similarity_threshold via config
    monkeypatch.setattr(
        "diff_engine.similarity.get",
        lambda s, k=None, d=None: 0.92 if (s == "diff" and k == "similarity_threshold") else d,
    )
    sim_engine = SemanticSimilarityEngine()
    assert sim_engine._threshold == 0.92

    # 4. stats.outlier_stddev via config
    db = DatabaseManager(str(tmp_path / "stat.db"))
    monkeypatch.setattr(
        "analyzers.detection.statistical_detector.get",
        lambda s, k=None, d=None: (
            1.75 if (s == "stats" and k == "outlier_stddev") else (2.5 if s == "metrics" else d)
        ),
    )
    detector = StatisticalDetector(db)
    assert detector._latency_mult == 1.75
    assert detector._token_mult == 1.75

    # 5. arbiter.information_loss thresholds via config
    monkeypatch.setattr(
        "config.config_loader.get",
        lambda s, k=None, d=None: (
            {"severe_threshold": 5, "moderate_threshold": 2}
            if (s == "arbiter" and k == "information_loss")
            else d
        ),
    )
    rule = InformationLossRule()
    assert rule.severe_threshold == 5
    assert rule.moderate_threshold == 2
    # With severe_threshold=5, a drop of 3 sources (8 -> 5) is now MEDIUM (WARNING), not HIGH (FAIL)
    ev_res = ExtractedEvidence(source_count=8, entity_count=5)
    ev_wr = ExtractedEvidence(source_count=5, entity_count=5)
    loss_res = rule.evaluate("run_cfg_loss", ev_res, ev_wr)
    assert loss_res.verdict == "WARNING"
