"""
scripts/verify_day40.py — Day 40 Verification Script
=====================================================
Verifies all Day 40 deliverables (Configuration Sweep + Cost Controls):
  1. config/config.yaml contains llm cost control keys (budget_alert_usd, cost_per_token_usd,
     primary_model, fallback_model, cache_ttl_hours, extraction_max_tokens, explanation_max_tokens)
  2. config/config.yaml centralizes stats.outlier_stddev, arbiter.information_loss,
     capture.pii_scrubbing, capture.retention_days, alerting, diff.similarity_threshold
  3. InformationLossRule dynamically loads severe_threshold and moderate_threshold from config
  4. StatisticalDetector honors stats.outlier_stddev and metrics.*_stddev_multiplier from config
  5. DatabaseManager resolves storage.db_path from config/config.yaml
  6. Cost budget alert logs WARNING in dashboard/state.py when cost > budget_alert_usd
  7. Dashboard header/ticker renders BUDGET ALERT banner when budget is exceeded
  8. Pytest suite tests/test_config_sweep_and_cost.py passes
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _check(num: int, title: str, fn) -> bool:
    try:
        detail = fn()
        print(f"  [PASS] Check {num}: {title} -- {detail}")
        return True
    except Exception as exc:
        print(f"  [FAIL] Check {num}: {title} -- {exc}")
        return False


def check_1_llm_cost_config() -> str:
    from config import config_loader

    config_loader.load_config.cache_clear()
    llm_cfg = config_loader.get("llm")
    required = (
        "budget_alert_usd",
        "cost_per_token_usd",
        "primary_model",
        "fallback_model",
        "cache_ttl_hours",
        "extraction_max_tokens",
        "explanation_max_tokens",
    )
    for k in required:
        assert k in llm_cfg, f"Missing llm.{k} in config.yaml"
    assert float(llm_cfg["budget_alert_usd"]) == 1.00
    assert config_loader.get("llm", "model") == config_loader.get("llm", "primary_model")
    return f"budget_alert_usd=${llm_cfg['budget_alert_usd']:.2f}, primary_model={llm_cfg['primary_model']}"


def check_2_centralized_config_sections() -> str:
    from config import config_loader

    assert float(config_loader.get("stats", "outlier_stddev")) == 2.5
    assert float(config_loader.get("diff", "similarity_threshold")) == 0.85
    assert int(config_loader.get("capture", "retention_days")) == 90
    assert "enabled" in config_loader.get("capture", "pii_scrubbing")
    assert "enabled" in config_loader.get("alerting")
    info_loss = config_loader.get("arbiter", "information_loss")
    assert info_loss["severe_threshold"] == 3 and info_loss["moderate_threshold"] == 1
    return "stats.outlier_stddev, diff, capture, alerting, arbiter.information_loss verified"


def check_3_information_loss_config() -> str:
    from analyzers.detection.information_loss import InformationLossRule

    rule = InformationLossRule()
    assert rule.severe_threshold == 3
    assert rule.moderate_threshold == 1
    custom_rule = InformationLossRule(severe_threshold=6, moderate_threshold=2)
    assert custom_rule.severe_threshold == 6 and custom_rule.moderate_threshold == 2
    return f"severe={rule.severe_threshold}, moderate={rule.moderate_threshold}"


def check_4_statistical_detector_outlier_config() -> str:
    from analyzers.detection.statistical_detector import StatisticalDetector
    from storage.db import DatabaseManager

    with tempfile.TemporaryDirectory() as tmpdir:
        db = DatabaseManager(str(Path(tmpdir) / "stat_verify.db"))
        det = StatisticalDetector(db)
        assert det._latency_mult == 2.5
        assert det._token_mult == 2.5
    return f"latency_mult={det._latency_mult}, token_mult={det._token_mult}"


def check_5_db_manager_config_path() -> str:
    from config import config_loader
    from storage.db import DatabaseManager

    db = DatabaseManager()
    expected = config_loader.get("storage", "db_path")
    assert db.db_path == expected, f"Expected {expected}, got {db.db_path}"
    return f"db_path='{db.db_path}'"


def check_6_cost_budget_alert_warning() -> str:
    import dashboard.state as dash_state
    from normalizer.normalizer import Normalizer
    from schema.models import AgentStep, RunTrace, StepStatus, TokenUsage
    from storage.db import DatabaseManager
    from storage.writer import StorageWriter

    with tempfile.TemporaryDirectory() as tmpdir:
        db = DatabaseManager(str(Path(tmpdir) / "budget.db"))
        db.initialize()
        tr = RunTrace(
            run_id="run_verify_budget",
            workflow="verify",
            timestamp=datetime.now(UTC),
            status=StepStatus.SUCCESS,
            steps=[
                AgentStep(
                    run_id="run_verify_budget",
                    step=1,
                    agent="researcher",
                    input="{}",
                    output="{}",
                    latency_ms=50.0,
                    status=StepStatus.SUCCESS,
                    tokens=TokenUsage(prompt=150_000, completion=150_000, total=300_000),
                    timestamp=datetime.now(UTC),
                )
            ],
        )
        StorageWriter(db).write_run(Normalizer().normalize_run(tr), trace_json=tr.model_dump_json())

        orig_get_db = dash_state.get_db
        try:
            dash_state.get_db = lambda: db
            status = dash_state.get_budget_status()
            # 300,000 tokens * $0.000005 = $1.50 > $1.00 budget_alert_usd
            assert status["exceeded"] is True
            assert abs(float(status["cost_usd"]) - 1.50) < 1e-6
        finally:
            dash_state.get_db = orig_get_db

    return f"cost=${status['cost_usd']:.2f} > budget=${status['budget_usd']:.2f} -> exceeded=True"


def check_7_dashboard_banner_rendering() -> str:
    import dashboard.app as dash_app
    import dashboard.state as dash_state

    orig_status = dash_state.get_budget_status
    try:
        dash_state.get_budget_status = lambda: {
            "cost_usd": 1.75,
            "budget_usd": 1.00,
            "exceeded": True,
        }
        ticker_html = dash_app._cost_ticker()
        assert "BUDGET ALERT" in ticker_html
        assert "$1.00 limit" in ticker_html
    finally:
        dash_state.get_budget_status = orig_status
    return "Dashboard _cost_ticker() renders BUDGET ALERT badge when exceeded"


def check_8_pytest_suite() -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_config_sweep_and_cost.py", "-q"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"pytest failed:\n{proc.stdout}\n{proc.stderr}")
    last_line = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()][-1]
    return last_line


def main() -> int:
    print("=" * 70)
    print("Day 40 Verification: Configuration Sweep + Cost Controls")
    print("=" * 70)

    checks = [
        (1, "config.yaml LLM cost control settings", check_1_llm_cost_config),
        (
            2,
            "Centralized config sections (stats, diff, capture, alerting)",
            check_2_centralized_config_sections,
        ),
        (3, "InformationLossRule dynamic config thresholds", check_3_information_loss_config),
        (
            4,
            "StatisticalDetector outlier_stddev config",
            check_4_statistical_detector_outlier_config,
        ),
        (5, "DatabaseManager storage.db_path resolution", check_5_db_manager_config_path),
        (
            6,
            "Session cost budget alert warning in dashboard/state.py",
            check_6_cost_budget_alert_warning,
        ),
        (7, "Dashboard header BUDGET ALERT badge rendering", check_7_dashboard_banner_rendering),
        (8, "Pytest suite (tests/test_config_sweep_and_cost.py)", check_8_pytest_suite),
    ]

    passed = sum(1 for num, title, fn in checks if _check(num, title, fn))
    print("-" * 70)
    print(f"Result: {passed}/{len(checks)} checks passed.")
    print("=" * 70)
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
