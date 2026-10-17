"""
schema/span.py — OpenInference Span DAG Schema & Utilities
==========================================================
AgentLens v2.0 (Day 52): Universal hierarchical DAG span representation
for complex multi-agent workflows, parallel forks, and reflection loops.

Compatible with OpenInference conventions:
    - spans have span_id, trace_id, and parent_span_id
    - span_type distinguishes agent, tool, chain, llm, and router invocations
    - iteration_index explicitly tracks cyclic reflection passes (A -> B -> A)
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_serializer

from schema.models import SCHEMA_VERSION, StepStatus, TokenUsage


class SpanType(StrEnum):
    """Categorization of an execution span in a multi-agent DAG."""

    AGENT = "agent"
    TOOL = "tool"
    CHAIN = "chain"
    LLM = "llm"
    ROUTER = "router"


class TraceSpan(BaseModel):
    """
    OpenInference-compatible hierarchical span in a workflow DAG.

    Enables modeling of:
        - Parent-child call hierarchies (e.g. Agent calling Tool)
        - Parallel execution branches (multiple agents sharing a parent)
        - Iterative reflection cycles (same agent executing with incremented iteration_index)
    """

    schema_version: str = Field(default=SCHEMA_VERSION)
    span_id: str = Field(
        default_factory=lambda: f"span_{uuid.uuid4().hex[:8]}",
        description="Unique identifier for this span",
    )
    trace_id: str = Field(
        description="Run ID linking this span to its parent RunTrace",
    )
    parent_span_id: str | None = Field(
        default=None,
        description="Span ID of the parent in the DAG / tree hierarchy",
    )
    name: str = Field(
        description="Logical name of the agent, tool, or chain (e.g. 'researcher', 'web_search')",
    )
    span_type: SpanType = Field(
        default=SpanType.AGENT,
        description="Type of execution node",
    )
    iteration_index: int = Field(
        default=0,
        description="Pass index for reflection loops (0 for 1st execution, 1 for 2nd, etc.)",
    )

    # Timing & Performance
    start_time: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Start timestamp",
    )
    end_time: datetime | None = Field(
        default=None,
        description="End timestamp (if completed)",
    )
    latency_ms: float = Field(
        default=0.0,
        description="Execution wall-clock latency in milliseconds",
    )

    # Execution State
    status: StepStatus = Field(
        default=StepStatus.SUCCESS,
        description="Execution outcome",
    )
    error: str | None = Field(
        default=None,
        description="Error message if status is FAILURE or ERROR",
    )
    input: Any = Field(
        default="",
        description="Input payload or arguments passed to this span",
    )
    output: Any = Field(
        default="",
        description="Output payload returned by this span",
    )
    tokens: TokenUsage = Field(
        default_factory=TokenUsage,
        description="Token consumption breakdown",
    )

    # OpenInference Extensibility
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary OpenInference standard and custom metadata attributes",
    )

    @field_serializer("start_time")
    def serialize_start_time(self, v: datetime) -> str:
        return v.isoformat()

    @field_serializer("end_time")
    def serialize_end_time(self, v: datetime | None) -> str | None:
        return v.isoformat() if v is not None else None

    model_config = {}


# ─────────────────────────────────────────────────────────────────────────────
# DAG & Tree Utilities
# ─────────────────────────────────────────────────────────────────────────────


def get_root_spans(spans: list[TraceSpan]) -> list[TraceSpan]:
    """Return all root spans in the trace (those without a parent_span_id)."""
    return [s for s in spans if s.parent_span_id is None]


def get_child_spans(spans: list[TraceSpan], parent_span_id: str) -> list[TraceSpan]:
    """Return all immediate children of a given parent span."""
    return [s for s in spans if s.parent_span_id == parent_span_id]


def calculate_span_depth(spans: list[TraceSpan], span_id: str) -> int:
    """Calculate tree depth of a span (0 for root spans)."""
    lookup = {s.span_id: s for s in spans}
    depth = 0
    curr = lookup.get(span_id)
    visited = set()
    while curr and curr.parent_span_id and curr.parent_span_id not in visited:
        visited.add(curr.span_id)
        depth += 1
        curr = lookup.get(curr.parent_span_id)
    return depth


def build_span_tree(spans: list[TraceSpan]) -> list[dict[str, Any]]:
    """
    Construct a nested hierarchical tree dictionary from a flat list of spans.
    Useful for visualization and Cytoscape/DAG rendering.
    """
    children_map: dict[str | None, list[TraceSpan]] = {}
    for s in spans:
        children_map.setdefault(s.parent_span_id, []).append(s)

    def _build_subtree(parent_id: str | None) -> list[dict[str, Any]]:
        nodes = []
        for span in children_map.get(parent_id, []):
            node_dict = span.model_dump()
            node_dict["children"] = _build_subtree(span.span_id)
            nodes.append(node_dict)
        return nodes

    return _build_subtree(None)


def find_reflection_cycles(spans: list[TraceSpan]) -> dict[str, list[TraceSpan]]:
    """
    Find agents involved in reflection loops.
    Returns mapping from agent name to list of spans executed with iteration_index >= 0
    for agents that executed more than once.
    """
    agent_spans: dict[str, list[TraceSpan]] = {}
    for s in spans:
        if s.span_type == SpanType.AGENT:
            agent_spans.setdefault(s.name, []).append(s)

    # Keep only agents with multiple executions or iteration_index > 0
    cycles = {}
    for name, s_list in agent_spans.items():
        if len(s_list) > 1 or any(s.iteration_index > 0 for s in s_list):
            s_list.sort(key=lambda s: s.iteration_index)
            cycles[name] = s_list
    return cycles


# ─────────────────────────────────────────────────────────────────────────────
# Bidirectional Converters: AgentStep <-> TraceSpan
# ─────────────────────────────────────────────────────────────────────────────


def steps_to_spans(steps: list[Any], run_id: str = "") -> list[TraceSpan]:
    """
    Convert a linear sequence of legacy AgentSteps into hierarchical TraceSpans.

    Automatically computes:
        - iteration_index for repeated agent executions in reflection loops
        - parent_span_id linking sequential steps or explicit parent_step links
        - child tool spans for step.tool_calls
    """
    spans: list[TraceSpan] = []
    agent_counts: dict[str, int] = {}
    step_id_map: dict[str, str] = {}
    prev_span_id: str | None = None

    for step in steps:
        eff_run_id = getattr(step, "run_id", "") or run_id
        agent_name = getattr(step, "agent", "agent")
        step_num = getattr(step, "step", 1)

        explicit_iter = getattr(step, "iteration_index", None)
        if explicit_iter is not None and explicit_iter > 0:
            iter_idx = explicit_iter
            agent_counts[agent_name] = max(agent_counts.get(agent_name, 0), iter_idx + 1)
        else:
            iter_idx = agent_counts.get(agent_name, 0)
            agent_counts[agent_name] = iter_idx + 1

        span_id = f"span_{eff_run_id}_{step_num}"
        step_id_map[str(step_num)] = span_id

        parent_id: str | None
        parent_step = getattr(step, "parent_step", None)
        if parent_step and str(parent_step) in step_id_map:
            parent_id = step_id_map[str(parent_step)]
        else:
            parent_id = prev_span_id

        node_type = str(getattr(step, "node_type", "llm")).lower()
        span_type = SpanType.TOOL if "tool" in node_type else SpanType.AGENT

        handoff = getattr(step, "handoff", None)
        handoff_dict = (
            handoff.model_dump() if (handoff is not None and hasattr(handoff, "model_dump")) else {}
        )

        tokens = getattr(step, "tokens", None)
        tok_usage = tokens if isinstance(tokens, TokenUsage) else TokenUsage()

        agent_span = TraceSpan(
            span_id=span_id,
            trace_id=eff_run_id,
            parent_span_id=parent_id,
            name=agent_name,
            span_type=span_type,
            iteration_index=iter_idx,
            start_time=getattr(step, "timestamp", datetime.now(UTC)),
            latency_ms=getattr(step, "latency_ms", 0.0),
            status=getattr(step, "status", StepStatus.SUCCESS),
            error=getattr(step, "error", None),
            input=getattr(step, "input", ""),
            output=getattr(step, "output", ""),
            tokens=tok_usage,
            attributes={
                "step": step_num,
                "prompt": getattr(step, "prompt", ""),
                "system_prompt": getattr(step, "system_prompt", ""),
                "model": getattr(step, "model", ""),
                "handoff": handoff_dict,
                "expected_output": getattr(step, "expected_output", None),
            },
        )
        spans.append(agent_span)
        prev_span_id = span_id

        # Convert tool calls to child spans
        tool_calls = getattr(step, "tool_calls", []) or []
        for t_idx, tc in enumerate(tool_calls):
            t_name = tc.get("name") or tc.get("tool") or f"tool_{t_idx + 1}"
            tool_span = TraceSpan(
                span_id=f"span_{eff_run_id}_{step_num}_t{t_idx + 1}",
                trace_id=eff_run_id,
                parent_span_id=agent_span.span_id,
                name=t_name,
                span_type=SpanType.TOOL,
                iteration_index=0,
                start_time=agent_span.start_time,
                status=StepStatus.SUCCESS if not tc.get("error") else StepStatus.FAILURE,
                error=tc.get("error"),
                input=tc.get("input") or tc.get("args") or "",
                output=tc.get("output") or tc.get("result") or "",
                attributes={"tool_index": t_idx + 1},
            )
            spans.append(tool_span)

    return spans


def spans_to_steps(spans: list[TraceSpan], run_id: str = "") -> list[Any]:
    """
    Convert a list of TraceSpans back into sequential AgentSteps.

    Preserves full backward compatibility with v1.0 downstream analyzers.
    Embeds child tool spans into the parent agent's tool_calls field.
    """
    from schema.models import AgentStep, HandoffState, NodeType

    tools_by_parent: dict[str, list[dict[str, Any]]] = {}
    for s in spans:
        if s.span_type == SpanType.TOOL and s.parent_span_id:
            tools_by_parent.setdefault(s.parent_span_id, []).append(
                {
                    "name": s.name,
                    "input": s.input,
                    "output": s.output,
                    "error": s.error,
                    "latency_ms": s.latency_ms,
                }
            )

    steps: list[AgentStep] = []
    step_idx = 1
    prev_step_str: str | None = None

    for s in spans:
        # Child tool spans become entries in tool_calls, not separate AgentSteps
        if s.span_type == SpanType.TOOL and s.parent_span_id:
            continue

        eff_run_id = s.trace_id or run_id
        tool_calls = tools_by_parent.get(s.span_id, [])

        handoff_dict = s.attributes.get("handoff", {})
        handoff = HandoffState(**handoff_dict) if handoff_dict else HandoffState()

        step_obj = AgentStep(
            schema_version=s.schema_version,
            run_id=eff_run_id,
            step=s.attributes.get("step", step_idx),
            agent=s.name,
            node_type=NodeType.TOOL if s.span_type == SpanType.TOOL else NodeType.LLM,
            input=str(s.input) if not isinstance(s.input, str) else s.input,
            output=str(s.output) if not isinstance(s.output, str) else s.output,
            expected_output=s.attributes.get("expected_output"),
            prompt=s.attributes.get("prompt", ""),
            system_prompt=s.attributes.get("system_prompt", ""),
            model=s.attributes.get("model", ""),
            latency_ms=s.latency_ms,
            tokens=s.tokens,
            tool_calls=tool_calls,
            handoff=handoff,
            parent_step=prev_step_str,
            iteration_index=s.iteration_index,
            status=s.status,
            error=s.error,
            timestamp=s.start_time,
        )
        steps.append(step_obj)
        prev_step_str = str(step_obj.step)
        step_idx += 1

    return steps
