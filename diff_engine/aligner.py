"""
diff_engine/aligner.py — Day 24 / Day 53: Graph Alignment Engine
=================================================================
Aligns two pipeline execution traces (Run A and Run B) by agent identity,
reflection loop iteration index, and parent-child DAG topology rather
than fixed array indices.

Features:
  - Sequence & Graph Alignment: Matches corresponding agent nodes across runs.
  - Reflection Loop Alignment (Day 53): Pairs nodes matching (agent, iteration_index).
    Isolates missing iterations when Run A has N iterations and Run B has M iterations.
  - Parallel Fork Diffing (Day 53): Aligns sibling branches independently by parent
    topology and agent role, resolving non-deterministic parallel completion orders.
  - OpenInference Span DAG Support (Day 53): Supports traces with hierarchical spans
    or legacy linear steps seamlessly.
  - Handles Missing & Extra Steps:
        MATCHED       : Step exists in both traces for this agent and iteration.
        MISSING_IN_A  : Step exists in Trace B but missing in Trace A.
        MISSING_IN_B  : Step exists in Trace A but missing in Trace B.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

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
    Represents an aligned pair of agent steps/spans from Trace A and Trace B.

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
    iteration_index_a: int | None = Field(
        default=None, description="0-indexed loop iteration index in Trace A"
    )
    iteration_index_b: int | None = Field(
        default=None, description="0-indexed loop iteration index in Trace B"
    )
    span_id_a: str | None = Field(
        default=None, description="Span ID in Trace A (if aligned via spans)"
    )
    span_id_b: str | None = Field(
        default=None, description="Span ID in Trace B (if aligned via spans)"
    )

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

    def get_pairs_by_iteration(self, iteration_index: int) -> list[AlignedStepPair]:
        """Return all pairs belonging to a specific reflection loop iteration."""
        return [
            p
            for p in self.pairs
            if p.iteration_index_a == iteration_index or p.iteration_index_b == iteration_index
        ]

    def get_loop_iterations(self) -> set[int]:
        """Return the set of all reflection loop iteration indices present in the alignment."""
        iters = set()
        for p in self.pairs:
            if p.iteration_index_a is not None:
                iters.add(p.iteration_index_a)
            if p.iteration_index_b is not None:
                iters.add(p.iteration_index_b)
        return iters


# ─────────────────────────────────────────────────────────────────────────────
# Internal Alignment Item & Topology Canonicalizer
# ─────────────────────────────────────────────────────────────────────────────


class _AlignItem:
    """Internal normalized representation of a step or span for DAG alignment."""

    def __init__(
        self,
        agent: str,
        iteration_index: int,
        step_index: int | None,
        parent_id: str | None,
        parent_agent: str | None,
        depth: int,
        step_obj: AgentStep | None = None,
        span_id: str | None = None,
    ) -> None:
        self.agent = agent
        self.iteration_index = iteration_index
        self.step_index = step_index
        self.parent_id = parent_id
        self.parent_agent = parent_agent
        self.depth = depth
        self.step_obj = step_obj
        self.span_id = span_id

    def __repr__(self) -> str:
        return (
            f"_AlignItem(agent={self.agent!r}, iter={self.iteration_index}, "
            f"step={self.step_index}, parent={self.parent_id!r}, depth={self.depth})"
        )


def _build_items_from_steps(steps: list[AgentStep], run_id: str = "") -> list[_AlignItem]:
    """Convert a sequence of AgentSteps into normalized _AlignItems."""
    if not steps:
        return []

    step_by_id: dict[str, AgentStep] = {str(s.step): s for s in steps}
    agent_counts: dict[str, int] = {}
    items: list[_AlignItem] = []

    for i, step in enumerate(steps):
        agent_name = step.agent

        # Determine reflection loop iteration index
        explicit_iter = getattr(step, "iteration_index", None)
        if explicit_iter is not None and explicit_iter > 0:
            iter_idx = explicit_iter
            agent_counts[agent_name] = max(agent_counts.get(agent_name, 0), iter_idx + 1)
        else:
            iter_idx = agent_counts.get(agent_name, 0)
            agent_counts[agent_name] = iter_idx + 1

        # Resolve parent step relationship
        parent_agent: str | None = None
        parent_id: str | None = None
        depth = 0

        if step.parent_step is not None:
            parent_id = str(step.parent_step)
            if parent_id in step_by_id:
                parent_agent = step_by_id[parent_id].agent
                # Calculate depth by walking parent pointers
                curr_pid: str | None = parent_id
                visited: set[str] = set()
                while curr_pid and curr_pid in step_by_id and curr_pid not in visited:
                    visited.add(curr_pid)
                    depth += 1
                    curr_pid = (
                        str(step_by_id[curr_pid].parent_step)
                        if step_by_id[curr_pid].parent_step is not None
                        else None
                    )
        elif i > 0:
            # Default sequential hint for linear workflows
            parent_agent = steps[i - 1].agent
            depth = i

        items.append(
            _AlignItem(
                agent=agent_name,
                iteration_index=iter_idx,
                step_index=step.step,
                parent_id=parent_id,
                parent_agent=parent_agent,
                depth=depth,
                step_obj=step,
                span_id=None,
            )
        )

    return items


def _build_items_from_trace(trace: RunTrace) -> list[_AlignItem]:
    """Extract normalized _AlignItems from a RunTrace (using steps or spans) without mutating trace."""
    steps = trace.steps
    spans = trace.spans

    if not steps and spans:
        from schema.span import spans_to_steps

        steps = spans_to_steps(spans, trace.run_id)

    if not steps:
        return []

    # Map span IDs to steps if spans are present
    span_by_step: dict[int, Any] = {}
    span_by_id: dict[str, Any] = {}
    if spans:
        for span in spans:
            span_by_id[span.span_id] = span
            step_attr = span.attributes.get("step") if hasattr(span, "attributes") else None
            if step_attr is not None:
                span_by_step[int(step_attr)] = span

    items = _build_items_from_steps(steps, run_id=trace.run_id)

    # Enrich with span metadata if available
    for item in items:
        if item.step_index is not None and item.step_index in span_by_step:
            span = span_by_step[item.step_index]
            item.span_id = span.span_id
            if span.parent_span_id and not item.parent_id:
                item.parent_id = span.parent_span_id
                if span.parent_span_id in span_by_id:
                    item.parent_agent = span_by_id[span.parent_span_id].name

    return items


def _build_items_from_spans(spans: list[Any], run_id: str = "") -> list[_AlignItem]:
    """Extract normalized _AlignItems directly from a list of TraceSpans."""
    from schema.span import SpanType

    span_by_id = {s.span_id: s for s in spans}
    depth_cache: dict[str, int] = {}

    def get_depth(sid: str) -> int:
        if sid in depth_cache:
            return depth_cache[sid]
        s = span_by_id.get(sid)
        if not s or not s.parent_span_id:
            depth_cache[sid] = 0
            return 0
        depth_cache[sid] = 1 + get_depth(s.parent_span_id)
        return depth_cache[sid]

    items: list[_AlignItem] = []
    for s in spans:
        # Child tool spans are embedded in agent spans, skip in top-level agent alignment
        if getattr(s, "span_type", None) == SpanType.TOOL and s.parent_span_id:
            continue

        parent_agent: str | None = None
        if s.parent_span_id and s.parent_span_id in span_by_id:
            parent_agent = span_by_id[s.parent_span_id].name

        step_idx = s.attributes.get("step") if hasattr(s, "attributes") else None
        depth = get_depth(s.span_id)

        items.append(
            _AlignItem(
                agent=s.name,
                iteration_index=getattr(s, "iteration_index", 0),
                step_index=step_idx,
                parent_id=s.parent_span_id,
                parent_agent=parent_agent,
                depth=depth,
                step_obj=None,
                span_id=s.span_id,
            )
        )

    return items


def _canonicalize_forks(items: list[_AlignItem]) -> list[_AlignItem]:
    """
    Reorder sibling items in parallel execution forks into canonical order
    (by agent name and iteration_index) so that non-deterministic execution
    orders across runs do not cause false alignment gaps.

    Preserves original step_index, step_obj, span_id, and parent links.
    """
    if len(items) <= 1:
        return list(items)

    result: list[_AlignItem] = []
    i = 0
    n = len(items)

    while i < n:
        curr = items[i]
        # Sibling fork group requires a shared parent identifier
        parent_key = curr.parent_id or (curr.parent_agent if curr.depth > 0 else None)

        if parent_key is None:
            result.append(curr)
            i += 1
            continue

        # Look ahead for contiguous items sharing the same parent and depth
        sibling_group = [curr]
        j = i + 1
        while j < n:
            next_item = items[j]
            next_parent_key = next_item.parent_id or (
                next_item.parent_agent if next_item.depth > 0 else None
            )
            if next_parent_key == parent_key and next_item.depth == curr.depth:
                sibling_group.append(next_item)
                j += 1
            else:
                break

        if len(sibling_group) > 1:
            # Parallel fork detected! Sort siblings canonically by (agent, iteration_index)
            sibling_group.sort(key=lambda item: (item.agent, item.iteration_index))
            result.extend(sibling_group)
            i = j
        else:
            result.append(curr)
            i += 1

    return result


# ─────────────────────────────────────────────────────────────────────────────
# GraphAligner Engine
# ─────────────────────────────────────────────────────────────────────────────


class GraphAligner:
    """
    Aligns two RunTrace objects by matching agent identities, reflection loop
    iteration indices, and parent/child DAG sequence topology.
    """

    @staticmethod
    def align_traces(trace_a: RunTrace, trace_b: RunTrace) -> GraphAlignmentResult:
        """
        Align trace_a and trace_b using DAG-aware Needleman-Wunsch global alignment
        on (agent, iteration_index) coordinates and parent-child sequence topology.

        Args:
            trace_a: Baseline RunTrace
            trace_b: Comparison RunTrace

        Returns:
            GraphAlignmentResult containing aligned step pairs and summary stats.
        """
        items_a = _build_items_from_trace(trace_a)
        items_b = _build_items_from_trace(trace_b)

        # Canonicalize parallel forks within each trace before alignment
        items_a = _canonicalize_forks(items_a)
        items_b = _canonicalize_forks(items_b)

        aligned_pairs = GraphAligner._needleman_wunsch_alignment(items_a, items_b)

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
                f"{p.agent}[iter {p.iteration_index_b}]"
                if p.iteration_index_b is not None and p.iteration_index_b > 0
                else p.agent
                for p in aligned_pairs
                if p.status == AlignmentStatus.MISSING_IN_A
            ]
            summary_parts.append(f"Missing in Trace A: {missing_a_agents}.")
        if missing_in_b > 0:
            missing_b_agents = [
                f"{p.agent}[iter {p.iteration_index_a}]"
                if p.iteration_index_a is not None and p.iteration_index_a > 0
                else p.agent
                for p in aligned_pairs
                if p.status == AlignmentStatus.MISSING_IN_B
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
    def align_spans(
        spans_a: list[Any],
        spans_b: list[Any],
        run_id_a: str = "run_a",
        run_id_b: str = "run_b",
    ) -> GraphAlignmentResult:
        """
        Directly align two lists of TraceSpans using DAG topology and iteration indices.
        """
        items_a = _canonicalize_forks(_build_items_from_spans(spans_a, run_id=run_id_a))
        items_b = _canonicalize_forks(_build_items_from_spans(spans_b, run_id=run_id_b))

        aligned_pairs = GraphAligner._needleman_wunsch_alignment(items_a, items_b)

        matched_count = sum(1 for p in aligned_pairs if p.status == AlignmentStatus.MATCHED)
        missing_in_a = sum(1 for p in aligned_pairs if p.status == AlignmentStatus.MISSING_IN_A)
        missing_in_b = sum(1 for p in aligned_pairs if p.status == AlignmentStatus.MISSING_IN_B)
        is_fully = missing_in_a == 0 and missing_in_b == 0

        summary = f"Aligned {matched_count} matched spans between '{run_id_a}' and '{run_id_b}'."

        return GraphAlignmentResult(
            run_id_a=run_id_a,
            run_id_b=run_id_b,
            pairs=aligned_pairs,
            matched_count=matched_count,
            missing_in_a_count=missing_in_a,
            missing_in_b_count=missing_in_b,
            is_fully_aligned=is_fully,
            alignment_summary=summary,
        )

    @staticmethod
    def _needleman_wunsch_alignment(seq_a: list[Any], seq_b: list[Any]) -> list[AlignedStepPair]:
        """
        Perform Needleman-Wunsch global sequence alignment between seq_a and seq_b.
        Accepts either list of _AlignItem or list of AgentStep.

        Scoring:
          Match: +3 if agents and loop iterations match
                 +1 bonus if parent agents match
                 +1 bonus if parent ID or depth matches
          Iteration mismatch: -4 (penalizes aligning across reflection iterations)
          Agent mismatch: -3
          Gap penalty: -1
        """
        if not seq_a and not seq_b:
            return []

        # Convert to _AlignItem if raw AgentSteps were supplied
        items_a: list[_AlignItem]
        if seq_a and isinstance(seq_a[0], _AlignItem):
            items_a = seq_a  # type: ignore[assignment]
        else:
            items_a = _build_items_from_steps(seq_a)  # type: ignore[arg-type]

        items_b: list[_AlignItem]
        if seq_b and isinstance(seq_b[0], _AlignItem):
            items_b = seq_b  # type: ignore[assignment]
        else:
            items_b = _build_items_from_steps(seq_b)  # type: ignore[arg-type]

        m, n = len(items_a), len(items_b)
        GAP_PENALTY = -1

        # Initialize DP matrix and traceback matrix
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        traceback = [[0] * (n + 1) for _ in range(m + 1)]  # 1: Diag, 2: Up (Gap B), 3: Left (Gap A)

        for i in range(1, m + 1):
            dp[i][0] = i * GAP_PENALTY
            traceback[i][0] = 2  # Up (Gap in B)
        for j in range(1, n + 1):
            dp[0][j] = j * GAP_PENALTY
            traceback[0][j] = 3  # Left (Gap in A)

        for i in range(1, m + 1):
            it_a = items_a[i - 1]
            for j in range(1, n + 1):
                it_b = items_b[j - 1]

                # Match / mismatch calculation
                if it_a.agent == it_b.agent:
                    if it_a.iteration_index == it_b.iteration_index:
                        score = 3
                        # Bonus for matching parent agent
                        if (
                            it_a.parent_agent is not None
                            and it_b.parent_agent is not None
                            and it_a.parent_agent == it_b.parent_agent
                        ):
                            score += 1
                        # Bonus for matching parent ID or depth
                        if (
                            it_a.parent_id is not None
                            and it_b.parent_id is not None
                            and it_a.parent_id == it_b.parent_id
                        ):
                            score += 1
                        elif it_a.depth == it_b.depth:
                            score += 1
                    else:
                        # Iteration index mismatch in loops
                        score = -4
                else:
                    score = -3

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
                it_a = items_a[i - 1]
                it_b = items_b[j - 1]
                if it_a.agent == it_b.agent and it_a.iteration_index == it_b.iteration_index:
                    pairs_reversed.append(
                        AlignedStepPair(
                            agent=it_a.agent,
                            status=AlignmentStatus.MATCHED,
                            step_a=it_a.step_obj,
                            step_b=it_b.step_obj,
                            step_index_a=it_a.step_index,
                            step_index_b=it_b.step_index,
                            iteration_index_a=it_a.iteration_index,
                            iteration_index_b=it_b.iteration_index,
                            span_id_a=it_a.span_id,
                            span_id_b=it_b.span_id,
                        )
                    )
                else:
                    # Mismatch -> emit Gap in A then Gap in B for clean reporting
                    pairs_reversed.append(
                        AlignedStepPair(
                            agent=it_b.agent,
                            status=AlignmentStatus.MISSING_IN_A,
                            step_a=None,
                            step_b=it_b.step_obj,
                            step_index_a=None,
                            step_index_b=it_b.step_index,
                            iteration_index_a=None,
                            iteration_index_b=it_b.iteration_index,
                            span_id_a=None,
                            span_id_b=it_b.span_id,
                        )
                    )
                    pairs_reversed.append(
                        AlignedStepPair(
                            agent=it_a.agent,
                            status=AlignmentStatus.MISSING_IN_B,
                            step_a=it_a.step_obj,
                            step_b=None,
                            step_index_a=it_a.step_index,
                            step_index_b=None,
                            iteration_index_a=it_a.iteration_index,
                            iteration_index_b=None,
                            span_id_a=it_a.span_id,
                            span_id_b=None,
                        )
                    )
                i -= 1
                j -= 1
            elif i > 0 and (j == 0 or traceback[i][j] == 2):
                it_a = items_a[i - 1]
                pairs_reversed.append(
                    AlignedStepPair(
                        agent=it_a.agent,
                        status=AlignmentStatus.MISSING_IN_B,
                        step_a=it_a.step_obj,
                        step_b=None,
                        step_index_a=it_a.step_index,
                        step_index_b=None,
                        iteration_index_a=it_a.iteration_index,
                        iteration_index_b=None,
                        span_id_a=it_a.span_id,
                        span_id_b=None,
                    )
                )
                i -= 1
            else:
                it_b = items_b[j - 1]
                pairs_reversed.append(
                    AlignedStepPair(
                        agent=it_b.agent,
                        status=AlignmentStatus.MISSING_IN_A,
                        step_a=None,
                        step_b=it_b.step_obj,
                        step_index_a=None,
                        step_index_b=it_b.step_index,
                        iteration_index_a=None,
                        iteration_index_b=it_b.iteration_index,
                        span_id_a=None,
                        span_id_b=it_b.span_id,
                    )
                )
                j -= 1

        pairs_reversed.reverse()
        return pairs_reversed

    @property
    def analyzer_id(self) -> str:
        return "diff_engine"

    def __init__(self, baseline_trace: RunTrace | None = None) -> None:
        self.baseline_trace = baseline_trace

    def analyze(self, trace: RunTrace, baseline_trace: RunTrace | None = None) -> Any:
        """
        Run Analyzer-compatible graph alignment analysis on a RunTrace.
        """
        from app.interfaces import AnalysisResult
        from schema.models import (
            EvidenceRecord,
            EvidenceSource,
            FailureCategory,
            RuleMatch,
            RuleSeverity,
        )

        try:
            if not trace or (not trace.steps and not trace.spans):
                return AnalysisResult(
                    skipped=True,
                    skip_reason="No steps or spans in trace",
                    analyzer_id=self.analyzer_id,
                )

            ref_trace = baseline_trace or self.baseline_trace or trace
            alignment = self.align_traces(ref_trace, trace)
            evidence: list[EvidenceRecord] = []

            for pair in alignment.get_missing_steps():
                iter_suffix = ""
                if pair.iteration_index_a is not None and pair.iteration_index_a > 0:
                    iter_suffix = f" (iteration {pair.iteration_index_a})"
                elif pair.iteration_index_b is not None and pair.iteration_index_b > 0:
                    iter_suffix = f" (iteration {pair.iteration_index_b})"

                desc = f"Agent '{pair.agent}' alignment status: {pair.status.value}{iter_suffix}"
                rule = RuleMatch(
                    rule_id=f"DIFF-ALIGN-{pair.agent.upper()}",
                    rule_version="1.0.0",
                    category=FailureCategory.WORKFLOW,
                    description=desc,
                    severity=RuleSeverity.MEDIUM,
                    agent=pair.agent,
                    step=pair.step_index_b or pair.step_index_a,
                )
                evidence.append(
                    EvidenceRecord(
                        source=EvidenceSource.DIFF_ENGINE,
                        description=desc,
                        value=pair.status.value,
                        rule_match=rule,
                        agent=pair.agent,
                        step=pair.step_index_b or pair.step_index_a,
                        confidence=1.0,
                    )
                )

            return AnalysisResult(
                evidence=evidence,
                analyzer_id=self.analyzer_id,
                skipped=False,
            )
        except Exception as exc:
            return AnalysisResult(
                skipped=True,
                skip_reason=f"GraphAligner error: {exc}",
                analyzer_id=self.analyzer_id,
            )

    def run(self, trace: RunTrace, baseline_trace: RunTrace | None = None) -> Any:
        """Standard Analyzer execution alias."""
        return self.analyze(trace, baseline_trace=baseline_trace)
