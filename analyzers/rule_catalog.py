"""
analyzers/rule_catalog.py — Static rule registry (Day 32)
==========================================================

Single source of truth for rule metadata shown in the Rule Explorer.
Fire counts come from the ``rule_matches`` DB table; names, categories and
descriptions come from here, so rules that have never fired are still listed.

Statistical detector rule IDs are parametrized per agent/step
(e.g. ``STAT-LAT-RESEARCHER-001``); ``normalize_rule_id`` collapses them into
the ``STAT-LAT`` / ``STAT-TOK`` families so they aggregate meaningfully.
"""

from __future__ import annotations

from typing import TypedDict


class RuleInfo(TypedDict):
    name: str
    category: str  # execution | reasoning | workflow | verification | unknown
    version: str
    description: str
    source: str  # which analyzer owns the rule


RULE_CATALOG: dict[str, RuleInfo] = {
    "missing_tool_output_v1": {
        "name": "Missing Tool Output",
        "category": "execution",
        "version": "1.0.0",
        "description": "A tool was called but no output or error was recorded.",
        "source": "rule_engine",
    },
    "tool_failure_v1": {
        "name": "Tool Failure",
        "category": "execution",
        "version": "1.0.0",
        "description": "A tool call returned an error or exception.",
        "source": "rule_engine",
    },
    "researcher_quality_v1": {
        "name": "Researcher Source Shortfall",
        "category": "reasoning",
        "version": "1.0.0",
        "description": "Researcher cited fewer sources than the configured minimum.",
        "source": "rule_engine",
    },
    "hallucination_v1": {
        "name": "Writer Hallucination",
        "category": "reasoning",
        "version": "1.0.0",
        "description": "Writer introduced more entities than the researcher supplied.",
        "source": "rule_engine",
    },
    "verifier_passthrough_v1": {
        "name": "Verifier Passthrough",
        "category": "verification",
        "version": "1.0.0",
        "description": "Verifier approved output that failed consistency checks.",
        "source": "consistency_validator",
    },
    "claim_drift_v1": {
        "name": "Claim Drift",
        "category": "verification",
        "version": "1.0.0",
        "description": "Claims changed between research and final output.",
        "source": "consistency_validator",
    },
    "skipped_step_v1": {
        "name": "Skipped Step",
        "category": "workflow",
        "version": "1.0.0",
        "description": "A required agent step did not run.",
        "source": "workflow_validator",
    },
    "wrong_order_v1": {
        "name": "Wrong Step Order",
        "category": "workflow",
        "version": "1.0.0",
        "description": "Agents executed in an unexpected order.",
        "source": "workflow_validator",
    },
    "information_loss_v1": {
        "name": "Information Loss",
        "category": "workflow",
        "version": "1.0.0",
        "description": "Evidence dropped between researcher and writer handoff.",
        "source": "information_loss",
    },
    "gt_mismatch_v1": {
        "name": "Ground-Truth Mismatch",
        "category": "reasoning",
        "version": "1.0.0",
        "description": "Output differs from the expected output (P1 evidence).",
        "source": "ground_truth",
    },
    "STAT-LAT": {
        "name": "Latency Outlier",
        "category": "execution",
        "version": "1.0.0",
        "description": "Step latency exceeded mean + N*stddev of the agent baseline.",
        "source": "statistical_detector",
    },
    "STAT-TOK": {
        "name": "Token Outlier",
        "category": "execution",
        "version": "1.0.0",
        "description": "Step token usage exceeded mean + N*stddev of the agent baseline.",
        "source": "statistical_detector",
    },
}

CATEGORIES = ["execution", "reasoning", "workflow", "verification", "unknown"]


def normalize_rule_id(rule_id: str) -> str:
    """Collapse parametrized statistical IDs into their family; pass others through."""
    if rule_id.startswith("STAT-LAT"):
        return "STAT-LAT"
    if rule_id.startswith("STAT-TOK"):
        return "STAT-TOK"
    return rule_id


def get_rule_info(rule_id: str) -> RuleInfo | None:
    """Catalog entry for a (possibly parametrized) rule ID, or None if unknown."""
    return RULE_CATALOG.get(normalize_rule_id(rule_id))
