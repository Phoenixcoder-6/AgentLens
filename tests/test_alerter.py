"""
tests/test_alerter.py — Day 31 unit tests for analyzers.alerter.Alerter

Tests:
  1. Alert not fired when disabled in config
  2. Alert not fired for P5 verdict (below threshold)
  3. Alert fired for P1 verdict (log channel)
  4. Alert suppressed during cooldown window
  5. Alerter.should_alert() returns correct bool
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _make_bundle(priority: str, cause: str = "reasoning") -> MagicMock:
    """Create a minimal mock AnalysisBundle with the given priority and cause."""
    bundle = MagicMock()
    bundle.priority_level = MagicMock()
    bundle.priority_level.value = priority
    bundle.primary_agent = "researcher"
    bundle.primary_cause = MagicMock()
    bundle.primary_cause.value = cause
    bundle.confidence = 0.85
    return bundle


class TestAlerterDisabled:
    def test_does_not_fire_when_disabled(self):
        """Alerter.fire() returns False and does not write logs when disabled."""
        from analyzers.alerter import Alerter

        Alerter.reset_cooldowns()
        with patch("analyzers.alerter.Alerter.__init__", return_value=None) as _:
            alerter = Alerter.__new__(Alerter)
            alerter._enabled = False
            alerter._on_verdict = ["P1", "P2"]
            alerter._channel = "log"
            alerter._webhook_url = ""
            alerter._cooldown_minutes = 60
            alerter._log_dir = "logs"

            result = alerter.fire("run-disabled", _make_bundle("P1"))

        assert result is False, "Should not fire when disabled"


class TestAlerterBelowThreshold:
    def test_does_not_fire_for_p5(self):
        """P5 verdict should not trigger an alert when threshold is P1/P2."""
        from analyzers.alerter import Alerter

        Alerter.reset_cooldowns()
        alerter = Alerter.__new__(Alerter)
        alerter._enabled = True
        alerter._on_verdict = ["P1", "P2"]
        alerter._channel = "log"
        alerter._webhook_url = ""
        alerter._cooldown_minutes = 60
        alerter._log_dir = "logs"

        result = alerter.fire("run-p5", _make_bundle("P5"))
        assert result is False, "P5 should not trigger alert"


class TestAlerterFires:
    def test_fires_and_writes_log_for_p1(self, tmp_path):
        """P1 verdict should fire an alert and write to alerts.log."""
        from analyzers.alerter import Alerter

        Alerter.reset_cooldowns()
        alerter = Alerter.__new__(Alerter)
        alerter._enabled = True
        alerter._on_verdict = ["P1", "P2"]
        alerter._channel = "log"
        alerter._webhook_url = ""
        alerter._cooldown_minutes = 60
        alerter._log_dir = str(tmp_path)

        result = alerter.fire("run-p1", _make_bundle("P1"))

        assert result is True, "P1 should fire"
        alert_file = tmp_path / "alerts.log"
        assert alert_file.exists(), "alerts.log should be created"
        content = alert_file.read_text(encoding="utf-8")
        assert "run-p1" in content
        assert "P1" in content


class TestAlerterCooldown:
    def test_suppressed_during_cooldown(self, tmp_path):
        """Second alert for same run_id within cooldown window should be suppressed."""
        from analyzers.alerter import Alerter

        Alerter.reset_cooldowns()
        alerter = Alerter.__new__(Alerter)
        alerter._enabled = True
        alerter._on_verdict = ["P1", "P2"]
        alerter._channel = "log"
        alerter._webhook_url = ""
        alerter._cooldown_minutes = 60  # 60 minute cooldown
        alerter._log_dir = str(tmp_path)

        # First fire — should succeed
        first = alerter.fire("run-cooldown", _make_bundle("P1"))
        assert first is True, "First alert should fire"

        # Immediate second fire — should be suppressed
        second = alerter.fire("run-cooldown", _make_bundle("P1"))
        assert second is False, "Second alert within cooldown should be suppressed"


class TestShouldAlert:
    def test_should_alert_true_for_threshold(self):
        from analyzers.alerter import Alerter

        alerter = Alerter.__new__(Alerter)
        alerter._enabled = True
        alerter._on_verdict = ["P1", "P2"]

        assert alerter.should_alert("P1") is True
        assert alerter.should_alert("P2") is True
        assert alerter.should_alert("P3") is False
        assert alerter.should_alert("P5") is False

    def test_should_alert_false_when_disabled(self):
        from analyzers.alerter import Alerter

        alerter = Alerter.__new__(Alerter)
        alerter._enabled = False
        alerter._on_verdict = ["P1", "P2"]

        assert alerter.should_alert("P1") is False
