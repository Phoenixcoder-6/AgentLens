"""
diff_engine/aligner.py — Day 24: Graph Alignment Engine
=========================================================
Aligns two pipeline execution traces (Run A and Run B) by agent identity
and parent-child topology rather than fixed array indices.

Features:
  - Sequence & Graph Alignment: Matches corresponding agent nodes across runs.
  - Handles Missing & Extra Steps: Discovers when an agent was skipped in Run B
    or inserted in Run A, assigning explicit AlignmentStatus:
        MATCHED       : Step exists in both traces for this agent.
        MISSING_IN_A  : Step exists in Trace B but missing in Trace A.
        MISSING_IN_B  : Step exists in Trace A but missing in Trace B.
  - Graph-based matching: Considers parent_step / child_step relationships
    and agent names so index shifts do not cause false mismatches.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from schema.models import AgentStep, RunTrace

# ─────────────────────────────────────────────────────────────────────────────
# Alignment Models
# ─────────────────────────────────────────────────────────────────────────────


class AlignmentStatus(StrEnum):
    """Status of step alignment between Trace A and Trace B."""

    MATCHED = "MATCHED"
    MISSING_IN_A = "MISSING_IN_A"
    MISSING_IN_B = "MISSING_IN_B"


class AlignedStepPair(BaseModel):
    """
    Represents an aligned pair of agent steps from Trace A and Trace B.

    If status is MATCHED, both step_a and step_b are populated.
    If MISSING_IN_A, step_a is None (step exists only in Trace B).
    If MISSING_IN_B, step_b is None (step exists only in Trace A).
    """

    agent: str = Field(description="Agent name associated with this aligned pair")
    status: AlignmentStatus = Field(description="MATCHED, MISSING_IN_A, or MISSING_IN_B")
    step_a: AgentStep | None = Field(default=None, description="Step from Trace A (if present)")
    step_b: AgentStep | None = Field(default=None, description="Step from Trace B (if present)")
    step_index_a: int | None = Field(default=None, description="1-indexed step number in Trace A")
    step_index_b: int | None = Field(default=None, description="1-indexed step number in Trace B")

    @property
    def is_matched(self) -> bool:
        return self.status == AlignmentStatus.MATCHED

    @property
    def is_missing(self) -> bool:
        return self.status in (AlignmentStatus.MISSING_IN_A, AlignmentStatus.MISSING_IN_B)


class GraphAlignmentResult(BaseModel):
    """
    Top-level container for the alignment of two execution traces.
    """

    run_id_a: str = Field(description="Run ID of Trace A (baseline)")
    run_id_b: str = Field(description="Run ID of Trace B (comparison)")
    pairs: list[AlignedStepPair] = Field(default_factory=list)
    matched_count: int = Field(default=0)
    missing_in_a_count: int = Field(default=0)
    missing_in_b_count: int = Field(default=0)
    is_fully_aligned: bool = Field(
        default=True,
        description="True if all steps matched 1-to-1 with no missing steps",
    )
    alignment_summary: str = Field(default="", description="Human-readable summary of alignment")

    def get_matched_pairs(self) -> list[AlignedStepPair]:
        """Return only the matched step pairs."""
        return [p for p in self.pairs if p.status == AlignmentStatus.MATCHED]

    def get_missing_steps(self) -> list[AlignedStepPair]:
        """Return all missing step pairs (either in A or in B)."""
        return [p for p in self.pairs if p.is_missing]

    def get_pairs_by_agent(self, agent: str) -> list[AlignedStepPair]:
        """Return all pairs involving a specific agent name."""
        return [p for p in self.pairs if p.agent == agent]


# ─────────────────────────────────────────────────────────────────────────────
# GraphAligner Engine
# ─────────────────────────────────────────────────────────────────────────────


class GraphAligner:
    """
    Aligns two RunTrace objects by matching agent identities and parent/child sequence topology.
    """

    @staticmethod
    def align_traces(trace_a: RunTrace, trace_b: RunTrace) -> GraphAlignmentResult:
        """
        Align trace_a and trace_b using Needleman-Wunsch global sequence alignment
        on agent identities and parent-child sequence hints.

        Args:
            trace_a: Baseline RunTrace
            trace_b: Comparison RunTrace

        Returns:
            GraphAlignmentResult containing aligned step pairs and summary stats.
        """
        steps_a = trace_a.steps or []
        steps_b = trace_b.steps or []

        aligned_pairs = GraphAligner._needleman_wunsch_alignment(steps_a, steps_b)

        matched_count = sum(1 for p in aligned_pairs if p.status == AlignmentStatus.MATCHED)
        missing_in_a = sum(1 for p in aligned_pairs if p.status == AlignmentStatus.MISSING_IN_A)
        missing_in_b = sum(1 for p in aligned_pairs if p.status == AlignmentStatus.MISSING_IN_B)
        is_fully = missing_in_a == 0 and missing_in_b == 0

        # Build human-readable summary
        summary_parts = [
            f"Aligned {matched_count} matched steps between '{trace_a.run_id}' and '{trace_b.run_id}'."
        ]
        if missing_in_a > 0:
            missing_a_agents = [
                p.agent for p in aligned_pairs if p.status == AlignmentStatus.MISSING_IN_A
            ]
            summary_parts.append(f"Missing in Trace A: {missing_a_agents}.")
        if missing_in_b > 0:
            missing_b_agents = [
                p.agent for p in aligned_pairs if p.status == AlignmentStatus.MISSING_IN_B
            ]
            summary_parts.append(f"Missing in Trace B: {missing_b_agents}.")

        summary = " ".join(summary_parts)

        return GraphAlignmentResult(
            run_id_a=trace_a.run_id,
            run_id_b=trace_b.run_id,
            pairs=aligned_pairs,
            matched_count=matched_count,
            missing_in_a_count=missing_in_a,
            missing_in_b_count=missing_in_b,
            is_fully_aligned=is_fully,
            alignment_summary=summary,
        )

    @staticmethod
    def _needleman_wunsch_alignment(
        seq_a: list[AgentStep], seq_b: list[AgentStep]
    ) -> list[AlignedStepPair]:
        """
        Perform Needleman-Wunsch global sequence alignment between seq_a and seq_b.
        Scoring:
          Match: +2 if agents match (+1 bonus if parent/child relations match)
          Mismatch: -2
          Gap: -1
        """
        m, n = len(seq_a), len(seq_b)
        GAP_PENALTY = -1

        # Initialize DP matrix and traceback matrix
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        traceback = [[0] * (n + 1) for _ in range(m + 1)]  # 1: Diag, 2: Up (Gap B), 3: Left (Gap A)

        for i in range(1, m + 1):
            dp[i][0] = i * GAP_PENALTY
            traceback[i][0] = 2  # Up
        for j in range(1, n + 1):
            dp[0][j] = j * GAP_PENALTY
            traceback[0][j] = 3  # Left

        for i in range(1, m + 1):
            step_a = seq_a[i - 1]
            for j in range(1, n + 1):
                step_b = seq_b[j - 1]

                # Calculate match / mismatch score
                if step_a.agent == step_b.agent:
                    score = 2
                    # Bonus for parent/child relation compatibility
                    if (
                        step_a.parent_step is not None
                        and step_b.parent_step is not None
                        and step_a.parent_step == step_b.parent_step
                    ):
                        score += 1
                else:
                    score = -2

                diag = dp[i - 1][j - 1] + score
                up = dp[i - 1][j] + GAP_PENALTY
                left = dp[i][j - 1] + GAP_PENALTY

                best = max(diag, up, left)
                dp[i][j] = best

                if best == diag:
                    traceback[i][j] = 1  # Diag
                elif best == up:
                    traceback[i][j] = 2  # Up (Gap in B => step in A only)
                else:
                    traceback[i][j] = 3  # Left (Gap in A => step in B only)

        # Reconstruct alignment from traceback matrix
        i, j = m, n
        pairs_reversed: list[AlignedStepPair] = []

        while i > 0 or j > 0:
            if i > 0 and j > 0 and traceback[i][j] == 1:
                step_a = seq_a[i - 1]
                step_b = seq_b[j - 1]
                if step_a.agent == step_b.agent:
                    pairs_reversed.append(
                        AlignedStepPair(
                            agent=step_a.agent,
                            status=AlignmentStatus.MATCHED,
                            step_a=step_a,
                            step_b=step_b,
                            step_index_a=step_a.step,
                            step_index_b=step_b.step,
                        )
                    )
                else:
                    # Treat mismatch as Gap in B then Gap in A for cleaner reporting
                    pairs_reversed.append(
                        AlignedStepPair(
                            agent=step_b.agent,
                            status=AlignmentStatus.MISSING_IN_A,
                            step_a=None,
                            step_b=step_b,
                            step_index_a=None,
                            step_index_b=step_b.step,
                        )
                    )
                    pairs_reversed.append(
                        AlignedStepPair(
                            agent=step_a.agent,
                            status=AlignmentStatus.MISSING_IN_B,
                            step_a=step_a,
                            step_b=None,
                            step_index_a=step_a.step,
                            step_index_b=None,
                        )
                    )
                i -= 1
                j -= 1
            elif i > 0 and (j == 0 or traceback[i][j] == 2):
                step_a = seq_a[i - 1]
                pairs_reversed.append(
                    AlignedStepPair(
                        agent=step_a.agent,
                        status=AlignmentStatus.MISSING_IN_B,
                        step_a=step_a,
                        step_b=None,
                        step_index_a=step_a.step,
                        step_index_b=None,
                    )
                )
                i -= 1
            else:
                step_b = seq_b[j - 1]
                pairs_reversed.append(
                    AlignedStepPair(
                        agent=step_b.agent,
                        status=AlignmentStatus.MISSING_IN_A,
                        step_a=None,
                        step_b=step_b,
                        step_index_a=None,
                        step_index_b=step_b.step,
                    )
                )
                j -= 1

        pairs_reversed.reverse()
        return pairs_reversed
