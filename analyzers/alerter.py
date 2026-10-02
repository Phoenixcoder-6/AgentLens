"""
analyzers/alerter.py — AgentLens Alerting System (Day 31)
===========================================================

Fires alerts when a run analysis produces a verdict that meets the
configured severity threshold (P1, P2, etc.).

Channels supported:
  log    — writes to logs/alerts.log (default, always works)
  slack  — HTTP POST to a Slack Incoming Webhook URL

Cooldown:
  The same run_id will not be re-alerted within `cooldown_minutes`
  (tracked in-process; resets on server restart).

Usage (called automatically at end of run_full_analysis):
    from analyzers.alerter import Alerter
    Alerter().fire(run_id, bundle, dashboard_base_url="http://localhost:8080")

Config (config/config.yaml):
    alerting:
      enabled: true
      on_verdict: [P1, P2]
      channel: log          # log | slack
      slack_webhook_url: ""
      cooldown_minutes: 60
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

# ── In-process cooldown tracker ───────────────────────────────────────────────
# Maps run_id → datetime of last alert fired
_COOLDOWN_REGISTRY: dict[str, datetime.datetime] = {}


# ── Alerter ────────────────────────────────────────────────────────────────────


class Alerter:
    """
    Reads alerting config and fires alerts for qualifying verdicts.

    Typical call:
        Alerter().fire(run_id, bundle, dashboard_base_url="http://localhost:8080")
    """

    def __init__(self) -> None:
        try:
            from config.config_loader import get

            self._enabled: bool = bool(get("alerting", "enabled", False))
            self._on_verdict: list[str] = list(get("alerting", "on_verdict", ["P1", "P2"]))
            self._channel: str = str(get("alerting", "channel", "log"))
            self._webhook_url: str = str(get("alerting", "slack_webhook_url", ""))
            self._cooldown_minutes: int = int(get("alerting", "cooldown_minutes", 60))
            self._log_dir: str = str(get("logging", "log_dir", "logs"))
        except Exception:
            # Graceful fallback if config is missing
            self._enabled = False
            self._on_verdict = ["P1", "P2"]
            self._channel = "log"
            self._webhook_url = ""
            self._cooldown_minutes = 60
            self._log_dir = "logs"

    # ── Public API ─────────────────────────────────────────────────────────────

    def fire(
        self,
        run_id: str,
        bundle: object,
        dashboard_base_url: str = "http://localhost:8080",
    ) -> bool:
        """
        Evaluate the bundle and fire an alert if conditions are met.

        Parameters
        ----------
        run_id : str
            The run to alert on.
        bundle : AnalysisBundle
            The Arbiter verdict for this run.
        dashboard_base_url : str
            Base URL for the dashboard link included in the alert.

        Returns
        -------
        bool
            True if an alert was fired, False otherwise.
        """
        if not self._enabled:
            return False
        if bundle is None:
            return False

        verdict = getattr(bundle, "priority_level", None)
        if verdict is None:
            return False
        verdict_str = str(verdict.value) if hasattr(verdict, "value") else str(verdict)

        if verdict_str not in self._on_verdict:
            return False

        if self._in_cooldown(run_id):
            logger.debug("Alert suppressed (cooldown active) for run %s", run_id)
            return False

        msg = self._build_message(run_id, bundle, verdict_str, dashboard_base_url)
        self._dispatch(msg, run_id)
        self._record_cooldown(run_id)
        return True

    def should_alert(self, verdict_str: str) -> bool:
        """Return True if this verdict level would trigger an alert (ignoring cooldown)."""
        return self._enabled and verdict_str in self._on_verdict

    # ── Message building ───────────────────────────────────────────────────────

    def _build_message(
        self,
        run_id: str,
        bundle: object,
        verdict: str,
        base_url: str,
    ) -> str:
        agent = getattr(bundle, "primary_agent", "unknown") or "unknown"
        cause = getattr(bundle, "primary_cause", None)
        # cause_str = cause.value if hasattr(cause, "value") else str(cause or "unknown")
        cause_str = (
            str(cause.value)
            if cause is not None and hasattr(cause, "value")
            else str(cause or "unknown")
        )
        confidence = getattr(bundle, "confidence", None)
        conf_str = f"{confidence:.2f}" if confidence is not None else "n/a"
        url = f"{base_url}/run/{run_id}"
        ts = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M UTC")

        return (
            f"[AgentLens ALERT] {ts}\n"
            f"  Run:        {run_id}\n"
            f"  Verdict:    {verdict}\n"
            f"  Agent:      {agent}\n"
            f"  Cause:      {cause_str}\n"
            f"  Confidence: {conf_str}\n"
            f"  Link:       {url}"
        )

    # ── Dispatch ───────────────────────────────────────────────────────────────

    def _dispatch(self, message: str, run_id: str) -> None:
        if self._channel == "slack":
            self._send_slack(message)
        else:
            self._write_log(message, run_id)

    def _write_log(self, message: str, run_id: str) -> None:
        """Write alert to logs/alerts.log."""
        log_path = Path(self._log_dir) / "alerts.log"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(message + "\n" + "-" * 60 + "\n")
            logger.info("Alert written to %s for run %s", log_path, run_id)
        except Exception as exc:
            logger.warning("Could not write alert log: %s", exc)

    def _send_slack(self, message: str) -> None:
        """POST alert to Slack webhook URL."""
        url = self._webhook_url or os.getenv("SLACK_WEBHOOK_URL", "")
        if not url:
            logger.warning("Slack channel configured but no webhook URL set — falling back to log")
            self._write_log(message, run_id="slack-fallback")
            return

        try:
            import urllib.request

            payload = json.dumps({"text": f"```{message}```"}).encode()
            req = urllib.request.Request(
                url, data=payload, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status != 200:
                    logger.warning("Slack webhook returned HTTP %s", resp.status)
        except Exception as exc:
            logger.warning("Slack webhook failed: %s — writing to log instead", exc)
            self._write_log(message, run_id="slack-fallback")

    # ── Cooldown ───────────────────────────────────────────────────────────────

    def _in_cooldown(self, run_id: str) -> bool:
        last = _COOLDOWN_REGISTRY.get(run_id)
        if last is None:
            return False
        elapsed = (datetime.datetime.now(datetime.UTC) - last).total_seconds() / 60
        return elapsed < self._cooldown_minutes

    def _record_cooldown(self, run_id: str) -> None:
        _COOLDOWN_REGISTRY[run_id] = datetime.datetime.now(datetime.UTC)

    # ── Testing helpers ────────────────────────────────────────────────────────

    @staticmethod
    def reset_cooldowns() -> None:
        """Clear the in-process cooldown registry (for tests)."""
        _COOLDOWN_REGISTRY.clear()
