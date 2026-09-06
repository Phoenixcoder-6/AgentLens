"""
analyzers/detection/statistical_detector.py — Statistical Anomaly Detector (P4)
=================================================================================
Day 27: Computes per-agent latency and token baseline statistics from all
historical runs stored in the database, then flags individual steps whose
metrics exceed configurable stddev thresholds.

Design:
    - Baseline is scoped per-agent, not globally — a "researcher" and a "writer"
      typically have very different latency distributions, so global baselines
      would generate false positives on every run.
    - Minimum run threshold guards against under-powered baselines: no anomalies
      are reported until at least min_runs_for_baseline runs exist per agent.
    - Thresholds come from config.yaml:
        metrics.latency_stddev_multiplier  (default 2.5)
        metrics.token_stddev_multiplier    (default 2.5)
        metrics.min_runs_for_baseline      (default 5)
    - Output: list[EvidenceRecord] with source=STATISTICAL_ANOMALY, priority=P4,
      confidence scaled to 1/(1 + z-score excess) so further outliers rank higher.

Wire into Arbiter:
    EvidenceSource.STATISTICAL_ANOMALY is already mapped to PriorityLevel.P4 in
    analyzers/arbiter.py SOURCE_TO_PRIORITY.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from config.config_loader import get
from config.logging_config import get_logger
from schema.models import (
    EvidenceRecord,
    EvidenceSource,
    FailureCategory,
    RuleMatch,
    RuleSeverity,
)
from storage.db import DatabaseManager

log = get_logger("statistical_detector")

# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AgentBaseline:
    """Per-agent statistical baseline computed from historical runs."""

    agent: str
    n_runs: int  # Number of runs used to build this baseline

    # Latency (ms)
    latency_mean: float
    latency_std: float

    # Tokens (total per step)
    token_mean: float
    token_std: float

    @property
    def active(self) -> bool:
        """True when there are enough runs to trust the baseline."""
        return self.n_runs >= 1  # already filtered upstream; kept for safety


@dataclass
class StatisticalAnomalyReport:
    """Full anomaly detection report for a single run."""

    run_id: str
    baselines_used: dict[str, AgentBaseline] = field(default_factory=dict)
    anomalies: list[EvidenceRecord] = field(default_factory=list)
    skipped_agents: list[str] = field(default_factory=list)  # insufficient history

    @property
    def has_anomalies(self) -> bool:
        return len(self.anomalies) > 0


# ─────────────────────────────────────────────────────────────────────────────
# StatisticalDetector
# ─────────────────────────────────────────────────────────────────────────────


class StatisticalDetector:
    """
    Detects statistical outliers in step latency and token usage.

    Usage:
        db = DatabaseManager()
        detector = StatisticalDetector(db)
        report = detector.analyze_run("run_abc123")
        evidence: list[EvidenceRecord] = report.anomalies
    """

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db
        self._latency_mult = float(get("metrics", "latency_stddev_multiplier", 2.5))
        self._token_mult = float(get("metrics", "token_stddev_multiplier", 2.5))
        self._min_runs = int(get("metrics", "min_runs_for_baseline", 5))

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze_run(self, run_id: str) -> StatisticalAnomalyReport:
        """
        Compute per-agent baselines from all other runs, then flag anomalous
        steps in the given run.

        Returns StatisticalAnomalyReport with evidence list.
        """
        report = StatisticalAnomalyReport(run_id=run_id)

        # 1. Fetch steps for current run
        current_steps = self.db.get_steps_for_run(run_id)
        if not current_steps:
            log.debug("statistical_detector: no steps found for run %s", run_id)
            return report

        # 2. Build per-agent baselines from ALL historical runs (excluding current)
        baselines = self._build_baselines(exclude_run_id=run_id)

        # 3. Flag anomalous steps
        for step in current_steps:
            agent = step.get("agent", "")
            baseline = baselines.get(agent)

            if baseline is None:
                # No baseline for this agent yet
                if agent not in report.skipped_agents:
                    report.skipped_agents.append(agent)
                continue

            report.baselines_used[agent] = baseline
            evidence = self._check_step(step, baseline)
            report.anomalies.extend(evidence)

        if report.has_anomalies:
            log.info(
                "statistical_detector: run %s — %d anomaly(s) detected",
                run_id,
                len(report.anomalies),
            )

        return report

    def get_baselines(self) -> dict[str, AgentBaseline]:
        """Return per-agent baselines from all stored runs (for dashboard/debugging)."""
        return self._build_baselines(exclude_run_id=None)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _build_baselines(self, exclude_run_id: str | None = None) -> dict[str, AgentBaseline]:
        """
        Aggregate steps from all runs (optionally excluding one) into
        per-agent latency and token distributions.

        Returns a dict of agent_name → AgentBaseline.
        Only agents with >= min_runs_for_baseline data points are included.
        """
        all_runs = self.db.list_runs(limit=10_000)

        # group steps by agent
        agent_latencies: dict[str, list[float]] = defaultdict(list)
        agent_tokens: dict[str, list[float]] = defaultdict(list)
        agent_run_ids: dict[str, set[str]] = defaultdict(set)

        for run in all_runs:
            rid = run["run_id"]
            if exclude_run_id and rid == exclude_run_id:
                continue
            steps = self.db.get_steps_for_run(rid)
            for s in steps:
                a = s.get("agent", "")
                if not a:
                    continue
                agent_latencies[a].append(float(s.get("latency_ms", 0.0)))
                agent_tokens[a].append(float(s.get("tokens_total", 0)))
                agent_run_ids[a].add(rid)

        baselines: dict[str, AgentBaseline] = {}
        for agent, latencies in agent_latencies.items():
            n_runs = len(agent_run_ids[agent])
            if n_runs < self._min_runs:
                continue  # not enough history

            tokens = agent_tokens[agent]
            baselines[agent] = AgentBaseline(
                agent=agent,
                n_runs=n_runs,
                latency_mean=_mean(latencies),
                latency_std=_stddev(latencies),
                token_mean=_mean(tokens),
                token_std=_stddev(tokens),
            )

        return baselines

    def _check_step(self, step: dict[str, Any], baseline: AgentBaseline) -> list[EvidenceRecord]:
        """Check one step dict against a baseline and return any EvidenceRecords."""
        evidence: list[EvidenceRecord] = []
        agent = step.get("agent", "")
        step_num = int(step.get("step", 0))
        latency = float(step.get("latency_ms", 0.0))
        tokens = float(step.get("tokens_total", 0))

        # ── Latency outlier ───────────────────────────────────────────────────
        if baseline.latency_std > 0:
            lat_z = (latency - baseline.latency_mean) / baseline.latency_std
            threshold = baseline.latency_mean + self._latency_mult * baseline.latency_std
            flagged = latency > threshold
        else:
            # stddev=0: all historical values identical; any deviation is certain anomaly
            lat_z = float("inf") if latency > baseline.latency_mean else 0.0
            threshold = baseline.latency_mean
            flagged = latency > baseline.latency_mean

        if flagged:
            # Cap z for display when infinite
            display_z = lat_z if lat_z != float("inf") else 99.0
            confidence = _confidence_from_z(display_z, self._latency_mult)
            rule = RuleMatch(
                rule_id=f"STAT-LAT-{agent.upper()}-{step_num:03d}",
                rule_version="1.0.0",
                category=FailureCategory.EXECUTION,
                description=(
                    f"Latency outlier: {latency:.0f}ms is "
                    + (
                        f"{display_z:.1f}\u03c3 above mean "
                        if lat_z != float("inf")
                        else "above the constant baseline "
                    )
                    + f"({baseline.latency_mean:.0f}ms ± {baseline.latency_std:.0f}ms) "
                    f"for agent '{agent}' across {baseline.n_runs} historical runs"
                ),
                severity=RuleSeverity.HIGH if display_z > 3.5 else RuleSeverity.MEDIUM,
                agent=agent,
                step=step_num,
                evidence_detail=(
                    f"z={display_z:.2f}, threshold={threshold:.0f}ms "
                    f"(mean+{self._latency_mult}\u03c3)"
                ),
            )
            evidence.append(
                EvidenceRecord(
                    source=EvidenceSource.STATISTICAL_ANOMALY,
                    description=rule.description,
                    value=latency,
                    rule_match=rule,
                    agent=agent,
                    step=step_num,
                    confidence=confidence,
                )
            )

        # ── Token outlier ─────────────────────────────────────────────────────
        if baseline.token_std > 0:
            tok_z = (tokens - baseline.token_mean) / baseline.token_std
            threshold_tok = baseline.token_mean + self._token_mult * baseline.token_std
            flagged_tok = tokens > threshold_tok
        else:
            tok_z = float("inf") if tokens > baseline.token_mean else 0.0
            threshold_tok = baseline.token_mean
            flagged_tok = tokens > baseline.token_mean

        if flagged_tok:
            display_tok_z = tok_z if tok_z != float("inf") else 99.0
            confidence_tok = _confidence_from_z(display_tok_z, self._token_mult)
            rule = RuleMatch(
                rule_id=f"STAT-TOK-{agent.upper()}-{step_num:03d}",
                rule_version="1.0.0",
                category=FailureCategory.EXECUTION,
                description=(
                    f"Token outlier: {int(tokens)} tokens is "
                    + (
                        f"{display_tok_z:.1f}\u03c3 above mean "
                        if tok_z != float("inf")
                        else "above the constant baseline "
                    )
                    + f"({baseline.token_mean:.0f} \u00b1 {baseline.token_std:.0f}) "
                    f"for agent '{agent}' across {baseline.n_runs} historical runs"
                ),
                severity=RuleSeverity.HIGH if display_tok_z > 3.5 else RuleSeverity.MEDIUM,
                agent=agent,
                step=step_num,
                evidence_detail=(
                    f"z={display_tok_z:.2f}, threshold={threshold_tok:.0f} tokens "
                    f"(mean+{self._token_mult}\u03c3)"
                ),
            )
            evidence.append(
                EvidenceRecord(
                    source=EvidenceSource.STATISTICAL_ANOMALY,
                    description=rule.description,
                    value=int(tokens),
                    rule_match=rule,
                    agent=agent,
                    step=step_num,
                    confidence=confidence_tok,
                )
            )

        return evidence


# ─────────────────────────────────────────────────────────────────────────────
# Stats helpers
# ─────────────────────────────────────────────────────────────────────────────


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stddev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def _confidence_from_z(z: float, threshold_sigma: float) -> float:
    """
    Map z-score excess above the threshold to a confidence in [0.5, 1.0].

    A step exactly at the threshold has confidence 0.5.
    Each additional σ above threshold adds ~0.1, capping at 0.99.
    """
    excess = max(0.0, z - threshold_sigma)
    return min(0.99, 0.50 + 0.10 * excess)
