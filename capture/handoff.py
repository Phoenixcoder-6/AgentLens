"""
capture/handoff.py — Handoff State Diff Engine
================================================
Day 6: Computes the structured diff between state BEFORE and AFTER
an agent runs. This is what makes information-loss detection possible.

The three-state model:
    input_state    = full LangGraph state dict BEFORE agent runs
    filtered_state = partial dict the agent CHOSE to return (its decisions)
    output_state   = full merged state AFTER LangGraph applies agent's return

The diff model:
    added_keys     = keys that went from empty → populated (agent introduced)
    modified_keys  = keys present in both, but value changed
    unchanged_keys = keys present in both, value identical
    dropped_keys   = keys that had content before but are empty/missing after

Why this matters:
    If Researcher finds 14 entities and Writer's output_state still has 14
    → no information loss.
    If Writer's output_state has 0 entities referenced
    → dropped_keys fires → WORKFLOW failure candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Value helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_empty(value: Any) -> bool:
    """
    True when a state value is considered 'absent':
        None, empty string, zero int/float, empty list/dict.
    """
    if value is None:
        return True
    if isinstance(value, (str, list, dict)):
        return len(value) == 0
    if isinstance(value, (int, float)):
        return value == 0
    return False


def _values_equal(a: Any, b: Any) -> bool:
    """Strict equality — no coercion."""
    return a == b


# ─────────────────────────────────────────────────────────────────────────────
# HandoffDiff — the structured diff result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HandoffDiff:
    """
    Structured diff between the state BEFORE and AFTER an agent ran.

    Fields
    ------
    added_keys     : Keys that were empty in input but populated in output.
                     These are what the agent *contributed*.
    modified_keys  : Keys present in both states but with different values.
                     These are what the agent *changed* (mutation).
    unchanged_keys : Keys with identical values in both states.
                     These passed through untouched.
    dropped_keys   : Keys that had content in input but are empty in output.
                     These are *information loss* — the critical failure signal.
    """
    added_keys: list[str] = field(default_factory=list)
    modified_keys: list[str] = field(default_factory=list)
    unchanged_keys: list[str] = field(default_factory=list)
    dropped_keys: list[str] = field(default_factory=list)

    # ── Convenience properties ──────────────────────────────────────────────

    @property
    def has_information_loss(self) -> bool:
        """True when any key with content in input was dropped by the agent."""
        return len(self.dropped_keys) > 0

    @property
    def has_mutations(self) -> bool:
        """True when the agent changed pre-existing state values."""
        return len(self.modified_keys) > 0

    @property
    def total_changes(self) -> int:
        return len(self.added_keys) + len(self.modified_keys) + len(self.dropped_keys)

    def summary(self) -> str:
        parts = []
        if self.added_keys:
            parts.append(f"added={self.added_keys}")
        if self.modified_keys:
            parts.append(f"modified={self.modified_keys}")
        if self.dropped_keys:
            parts.append(f"DROPPED={self.dropped_keys}")
        if not parts:
            return "no changes"
        return ", ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# HandoffCapture — the main class
# ─────────────────────────────────────────────────────────────────────────────

class HandoffCapture:
    """
    Computes the three-state snapshot and structured diff for a single
    agent handoff.

    Usage (called inside @trace_step):
        capture = HandoffCapture(input_state=state_before)
        result  = agent_fn(state_before)          # run the agent
        capture.record_agent_return(result)        # record what agent returned
        handoff_state, diff = capture.finalize()   # get structured result
    """

    def __init__(self, input_state: dict[str, Any]) -> None:
        # Snapshot the full state BEFORE the agent runs.
        # Deep-copy strings/primitives; shallow-copy dicts since state
        # values are strings, ints, bools — no nested mutable objects.
        self._input_state: dict[str, Any] = dict(input_state)
        self._agent_return: dict[str, Any] = {}

    def record_agent_return(self, agent_return: dict[str, Any]) -> None:
        """
        Called immediately after the agent function returns.
        Stores the partial dict the agent chose to emit.
        This becomes filtered_state — the agent's visible decision.
        """
        self._agent_return = dict(agent_return) if agent_return else {}

    def finalize(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], HandoffDiff]:
        """
        Returns (input_state, filtered_state, output_state, diff).

        output_state is computed by merging input_state with the agent's
        return dict — exactly how LangGraph merges state between nodes.
        """
        # LangGraph merge: output = input overridden by agent's return values
        output_state: dict[str, Any] = {**self._input_state, **self._agent_return}

        diff = self._compute_diff(self._input_state, output_state)

        return (
            self._input_state,      # what agent received
            self._agent_return,     # what agent chose to return (filtered view)
            output_state,           # full merged state going to next agent
            diff,
        )

    # ── Internal diff logic ─────────────────────────────────────────────────

    @staticmethod
    def _compute_diff(before: dict[str, Any], after: dict[str, Any]) -> HandoffDiff:
        """
        Computes the four-category diff between two full state snapshots.

        Rules:
        - added    : key was empty/missing in before, has content in after
        - modified : key had content in before AND after, but values differ
        - unchanged: key had identical values in both
        - dropped  : key had content in before, is empty/missing in after
        """
        diff = HandoffDiff()

        all_keys = set(before.keys()) | set(after.keys())

        for key in sorted(all_keys):
            before_val = before.get(key)
            after_val  = after.get(key)

            before_empty = _is_empty(before_val)
            after_empty  = _is_empty(after_val)

            if before_empty and not after_empty:
                # Key went from absent to populated → agent added it
                diff.added_keys.append(key)

            elif not before_empty and after_empty:
                # Key had content but is now empty → information loss
                diff.dropped_keys.append(key)

            elif not before_empty and not after_empty:
                if _values_equal(before_val, after_val):
                    diff.unchanged_keys.append(key)
                else:
                    diff.modified_keys.append(key)

            # else: both empty → ignore (key existed in both but was always empty)

        return diff
