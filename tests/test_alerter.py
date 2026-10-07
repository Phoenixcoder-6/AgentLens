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


# ── Additional coverage tests ────────────────────────────────────────────────


class TestAlerterEdgeCases:
    def test_does_not_fire_when_bundle_is_none(self):
        """No bundle means there is nothing to alert on."""
        from analyzers.alerter import Alerter

        Alerter.reset_cooldowns()

        alerter = Alerter.__new__(Alerter)
        alerter._enabled = True
        alerter._on_verdict = ["P1", "P2"]
        alerter._cooldown_minutes = 60

        assert alerter.fire("run-none", None) is False

    def test_does_not_fire_when_priority_is_missing(self):
        """Bundle without a priority should not trigger an alert."""
        from analyzers.alerter import Alerter

        Alerter.reset_cooldowns()

        alerter = Alerter.__new__(Alerter)
        alerter._enabled = True
        alerter._on_verdict = ["P1", "P2"]
        alerter._cooldown_minutes = 60

        bundle = MagicMock()
        bundle.priority_level = None

        assert alerter.fire("run-no-priority", bundle) is False

    def test_reset_cooldowns_clears_registry(self):
        """reset_cooldowns should remove all cooldown entries."""
        from analyzers.alerter import Alerter

        Alerter.reset_cooldowns()

        alerter = Alerter.__new__(Alerter)
        alerter._enabled = True
        alerter._on_verdict = ["P1"]
        alerter._channel = "log"
        alerter._webhook_url = ""
        alerter._cooldown_minutes = 60
        alerter._log_dir = "logs"

        with patch.object(alerter, "_dispatch"):
            alerter.fire("run-reset", _make_bundle("P1"))

        # Cooldown should exist.
        from analyzers.alerter import _COOLDOWN_REGISTRY

        assert "run-reset" in _COOLDOWN_REGISTRY

        Alerter.reset_cooldowns()

        assert "run-reset" not in _COOLDOWN_REGISTRY

    def test_should_alert_handles_case_exactly(self):
        """Only configured verdict strings should trigger."""
        from analyzers.alerter import Alerter

        alerter = Alerter.__new__(Alerter)
        alerter._enabled = True
        alerter._on_verdict = ["P1", "P2"]

        assert alerter.should_alert("P1") is True
        assert alerter.should_alert("p1") is False
        assert alerter.should_alert("") is False


class TestAlerterDispatch:
    def _make_alerter(self, tmp_path):
        from analyzers.alerter import Alerter

        alerter = Alerter.__new__(Alerter)
        alerter._enabled = True
        alerter._on_verdict = ["P1", "P2"]
        alerter._channel = "log"
        alerter._webhook_url = ""
        alerter._cooldown_minutes = 60
        alerter._log_dir = str(tmp_path)
        return alerter

    def test_build_message_contains_key_information(self, tmp_path):
        """Alert message should contain the important alert information."""
        alerter = self._make_alerter(tmp_path)

        message = alerter._build_message(
            "run-message",
            _make_bundle("P1", "reasoning"),
            "P1",
            "http://localhost:8501",
        )

        assert "run-message" in message
        assert "P1" in message
        assert "researcher" in message
        assert "reasoning" in message
        assert "0.85" in message
        assert "http://localhost:8501/run/run-message" in message

    def test_log_channel_dispatches_to_log(self, tmp_path):
        """Log channel should call _write_log."""
        alerter = self._make_alerter(tmp_path)

        with patch.object(alerter, "_write_log") as write_log:
            alerter._dispatch("test alert", "run-dispatch")

        write_log.assert_called_once_with(
            "test alert",
            "run-dispatch",
        )

    def test_slack_channel_dispatches_to_slack(self, tmp_path):
        """Slack channel should call _send_slack."""
        alerter = self._make_alerter(tmp_path)
        alerter._channel = "slack"

        with patch.object(alerter, "_send_slack") as send_slack:
            alerter._dispatch("test alert", "run-dispatch")

        send_slack.assert_called_once_with("test alert")

    def test_write_log_creates_directory(self, tmp_path):
        """_write_log should create a missing log directory."""
        from analyzers.alerter import Alerter

        log_dir = tmp_path / "nested" / "alerts"

        alerter = Alerter.__new__(Alerter)
        alerter._log_dir = str(log_dir)

        alerter._write_log("hello alert", "run-log")

        assert (log_dir / "alerts.log").exists()

        content = (log_dir / "alerts.log").read_text(encoding="utf-8")

        assert "hello alert" in content
        assert "-" * 60 in content

    def test_slack_without_webhook_falls_back_to_log(self, tmp_path):
        """Slack without a webhook should fall back to local logging."""
        from analyzers.alerter import Alerter

        alerter = Alerter.__new__(Alerter)
        alerter._webhook_url = ""
        alerter._log_dir = str(tmp_path)

        with patch.object(alerter, "_write_log") as write_log:
            alerter._send_slack("fallback message")

        write_log.assert_called_once_with(
            "fallback message",
            run_id="slack-fallback",
        )

    def test_slack_success(self, tmp_path):
        """Successful Slack request should be handled without error."""
        from analyzers.alerter import Alerter

        alerter = Alerter.__new__(Alerter)
        alerter._webhook_url = "https://example.com/webhook"
        alerter._log_dir = str(tmp_path)

        fake_response = MagicMock()
        fake_response.status = 200
        fake_response.__enter__.return_value = fake_response
        fake_response.__exit__.return_value = None

        with patch(
            "urllib.request.urlopen",
            return_value=fake_response,
        ) as urlopen:
            alerter._send_slack("Slack alert")

        urlopen.assert_called_once()

        request = urlopen.call_args.args[0]

        assert request.full_url == "https://example.com/webhook"
        assert request.data is not None
        assert b"Slack alert" in request.data

    def test_slack_non_200_response(self, tmp_path):
        """Non-200 Slack response should log a warning."""
        from analyzers.alerter import Alerter

        alerter = Alerter.__new__(Alerter)
        alerter._webhook_url = "https://example.com/webhook"
        alerter._log_dir = str(tmp_path)

        fake_response = MagicMock()
        fake_response.status = 500
        fake_response.__enter__.return_value = fake_response
        fake_response.__exit__.return_value = None

        with patch(
            "urllib.request.urlopen",
            return_value=fake_response,
        ):
            with patch("analyzers.alerter.logger.warning") as warning:
                alerter._send_slack("Slack alert")

        warning.assert_called_once()

    def test_slack_exception_falls_back_to_log(self, tmp_path):
        """Slack request failures should fall back to local logging."""
        from analyzers.alerter import Alerter

        alerter = Alerter.__new__(Alerter)
        alerter._webhook_url = "https://example.com/webhook"
        alerter._log_dir = str(tmp_path)

        with patch(
            "urllib.request.urlopen",
            side_effect=Exception("network failure"),
        ):
            with patch.object(alerter, "_write_log") as write_log:
                alerter._send_slack("Slack alert")

        write_log.assert_called_once_with(
            "Slack alert",
            run_id="slack-fallback",
        )
