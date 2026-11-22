"""
analyzers/remediation — Automated Prompt Auto-Diff & Self-Healing Engine
========================================================================
Day 64: Automated prompt remediation generating targeted constraints,
GitHub-style unified diffs, and regression avoidance scoring.
"""

from __future__ import annotations

from .prompt_fixer import (
    ConstraintTemplate,
    PromptFixer,
    RemediationPatch,
)

__all__ = [
    "ConstraintTemplate",
    "PromptFixer",
    "RemediationPatch",
]
