"""
normalizer/otlp_normalizer.py — OpenTelemetry (OTLP) & OpenInference Normalizer
================================================================================
Day 60: Converts standard OpenTelemetry (OTLP/HTTP) and OpenInference JSON
payloads into canonical AgentLens RunTrace, AgentStep, and TraceSpan models.

Standards Supported:
  - OpenTelemetry Traces v1 OTLP JSON (resourceSpans -> scopeSpans -> spans)
  - OpenInference Semantic Conventions:
      openinference.span.kind (AGENT, TOOL, CHAIN, LLM, ROUTER, RETRIEVER)
      input.value / output.value
      llm.token_count.prompt / completion / total
      tool.name / tool.parameters / tool.output
      iteration.index / openinference.iteration_index
      exception.message / error.message
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from capture.handoff import HandoffCapture
from schema.models import (
    AgentStep,
    RunTrace,
    StepStatus,
    TokenUsage,
)
from schema.span import SpanType, TraceSpan

logger = logging.getLogger("agentlens.normalizer.otlp")


def _safe_serialize(obj: Any) -> str:
    """Serialize an object to JSON string with fallback."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, default=str)
    except Exception:
        return str(obj)


def _parse_attr_value(val: Any) -> Any:
    """Parse OTLP AnyValue protobuf-as-JSON or direct primitive."""
    if isinstance(val, dict):
        if "stringValue" in val:
            return val["stringValue"]
        if "intValue" in val:
            try:
                return int(val["intValue"])
            except Exception:
                return val["intValue"]
        if "doubleValue" in val:
            return float(val["doubleValue"])
        if "boolValue" in val:
            return bool(val["boolValue"])
        if "arrayValue" in val:
            values = val["arrayValue"].get("values", [])
            return [_parse_attr_value(v) for v in values]
        if "kvlistValue" in val:
            kvs = val["kvlistValue"].get("values", [])
            return {kv.get("key", ""): _parse_attr_value(kv.get("value")) for kv in kvs}
        if len(val) == 1:
            return next(iter(val.values()))
    return val


def _extract_attributes(raw_attrs: Any) -> dict[str, Any]:
    """Normalize OTLP attributes (list of {key, value} or flat dict) into a dict."""
    if not raw_attrs:
        return {}
    if isinstance(raw_attrs, dict):
        return {str(k): _parse_attr_value(v) for k, v in raw_attrs.items()}
    if isinstance(raw_attrs, list):
        res: dict[str, Any] = {}
        for item in raw_attrs:
            if isinstance(item, dict):
                k = item.get("key")
                if k:
                    res[str(k)] = _parse_attr_value(item.get("value"))
        return res
    return {}


def _nano_to_datetime(nano_val: Any) -> datetime:
    """Convert nanoseconds to UTC datetime."""
    if not nano_val:
        return datetime.now(UTC)
    try:
        nano_int = int(nano_val)
        sec = nano_int / 1_000_000_000.0
        return datetime.fromtimestamp(sec, tz=UTC)
    except Exception:
        return datetime.now(UTC)


def _calc_latency_ms(start_nano: Any, end_nano: Any) -> float:
    """Calculate latency in milliseconds from OTLP timestamps."""
    try:
        s = int(start_nano)
        e = int(end_nano)
        if e >= s:
            return max(0.01, (e - s) / 1_000_000.0)
    except Exception:
        pass
    return 1.0


def _map_span_type(kind_attr: str | None, span_name: str, attrs: dict[str, Any]) -> SpanType:
    """Map OpenInference / OTLP attributes to AgentLens SpanType."""
    if kind_attr:
        k = kind_attr.strip().upper()
        if k in ("AGENT", "AGENT_SPAN"):
            return SpanType.AGENT
        if k in ("TOOL", "FUNCTION", "RETRIEVER", "EMBEDDING", "RERANKER"):
            return SpanType.TOOL
        if k in ("LLM", "CHAT_MODEL", "COMPLETION"):
            return SpanType.LLM
        if k in ("ROUTER", "CONDITION", "BRANCH"):
            return SpanType.ROUTER
        if k in ("CHAIN", "WORKFLOW", "GRAPH"):
            return SpanType.CHAIN

    # Attribute-based inference
    if "tool.name" in attrs or "tool_name" in attrs:
        return SpanType.TOOL
    if (
        "llm.model_name" in attrs
        or "llm.token_count.prompt" in attrs
        or "llm.token_count.total" in attrs
    ):
        return SpanType.LLM
    if "router.branch" in attrs or "route" in span_name.lower():
        return SpanType.ROUTER

    # Name-based heuristic
    low_name = span_name.lower()
    if any(kw in low_name for kw in ("tool", "search", "fetch", "query", "exec")):
        return SpanType.TOOL
    if any(kw in low_name for kw in ("llm", "gpt", "claude", "gemini", "chat")):
        return SpanType.LLM

    return SpanType.AGENT


class OTLPNormalizer:
    """
    Normalizes OpenTelemetry / OpenInference JSON trace payloads into AgentLens RunTrace models.
    """

    def normalize_otlp_payload(self, payload: dict[str, Any]) -> list[RunTrace]:
        """
        Parse a full OTLP JSON payload (with resourceSpans -> scopeSpans -> spans)
        and construct canonical RunTrace objects.
        """
        resource_spans = payload.get("resourceSpans", [])
        if not resource_spans and "spans" in payload:
            # Flat span payload support for lightweight clients
            resource_spans = [{"scopeSpans": [{"spans": payload["spans"]}]}]

        # Group raw spans by traceId
        traces_by_id: dict[str, list[dict[str, Any]]] = {}
        resource_attrs_by_trace: dict[str, dict[str, Any]] = {}

        for r_span in resource_spans:
            r_attrs = _extract_attributes((r_span.get("resource") or {}).get("attributes"))
            for s_span in r_span.get("scopeSpans", []):
                for span in s_span.get("spans", []):
                    trace_id = str(span.get("traceId") or f"trace_{uuid4().hex[:8]}")
                    traces_by_id.setdefault(trace_id, []).append(span)
                    if trace_id not in resource_attrs_by_trace:
                        resource_attrs_by_trace[trace_id] = r_attrs

        run_traces: list[RunTrace] = []
        for trace_id, raw_spans in traces_by_id.items():
            r_attrs = resource_attrs_by_trace.get(trace_id, {})
            run_trace = self._build_run_trace(trace_id, raw_spans, r_attrs)
            run_traces.append(run_trace)

        return run_traces

    def _build_run_trace(
        self,
        trace_id: str,
        raw_spans: list[dict[str, Any]],
        resource_attrs: dict[str, Any],
    ) -> RunTrace:
        """Construct one RunTrace from a list of raw OTLP spans belonging to the same trace."""
        # 1. Resolve workflow name
        workflow = (
            resource_attrs.get("service.name")
            or resource_attrs.get("service.namespace")
            or "otlp_workflow"
        )

        # 2. Convert all raw spans into TraceSpan objects
        converted_spans: list[TraceSpan] = []
        spans_by_id: dict[str, TraceSpan] = {}
        child_spans_by_parent: dict[str, list[TraceSpan]] = {}

        min_start_nano: int | None = None
        max_end_nano: int | None = None

        has_trace_error = False

        for raw in raw_spans:
            s_id = str(raw.get("spanId") or f"span_{uuid4().hex[:8]}")
            p_id = raw.get("parentSpanId")
            p_span_id = str(p_id) if p_id else None
            name = str(raw.get("name") or "step")

            attrs = _extract_attributes(raw.get("attributes"))

            start_nano = raw.get("startTimeUnixNano", 0)
            end_nano = raw.get("endTimeUnixNano", 0)

            try:
                s_nano = int(start_nano)
                if min_start_nano is None or s_nano < min_start_nano:
                    min_start_nano = s_nano
            except Exception:
                pass

            try:
                e_nano = int(end_nano)
                if max_end_nano is None or e_nano > max_end_nano:
                    max_end_nano = e_nano
            except Exception:
                pass

            lat_ms = _calc_latency_ms(start_nano, end_nano)

            # Determine span type
            kind_val = attrs.get("openinference.span.kind")
            span_type = _map_span_type(str(kind_val) if kind_val else None, name, attrs)

            # Iteration index
            iter_idx = int(
                attrs.get("iteration.index") or attrs.get("openinference.iteration_index") or 0
            )

            # Token usage
            p_tok = int(
                attrs.get("llm.token_count.prompt") or attrs.get("llm.token_count.input") or 0
            )
            c_tok = int(
                attrs.get("llm.token_count.completion") or attrs.get("llm.token_count.output") or 0
            )
            t_tok = int(attrs.get("llm.token_count.total") or (p_tok + c_tok))
            tokens = TokenUsage(prompt=p_tok, completion=c_tok, total=t_tok)

            # Status & error
            status_obj = raw.get("status") or {}
            status_code = status_obj.get("code")
            status_msg = status_obj.get("message")
            err_msg = attrs.get("error.message") or attrs.get("exception.message") or status_msg

            is_error = status_code == 2 or bool(err_msg)
            if is_error:
                has_trace_error = True

            span_status = StepStatus.ERROR if is_error else StepStatus.SUCCESS

            # Input / Output
            inp = (
                attrs.get("input.value")
                or attrs.get("tool.parameters")
                or attrs.get("tool.input")
                or ""
            )
            out = attrs.get("output.value") or attrs.get("tool.output") or ""

            trace_span = TraceSpan(
                span_id=s_id,
                trace_id=trace_id,
                parent_span_id=p_span_id,
                name=name,
                span_type=span_type,
                iteration_index=iter_idx,
                latency_ms=lat_ms,
                status=span_status,
                error=str(err_msg) if err_msg else None,
                input=_safe_serialize(inp),
                output=_safe_serialize(out),
                tokens=tokens,
            )

            converted_spans.append(trace_span)
            spans_by_id[s_id] = trace_span
            if p_span_id:
                child_spans_by_parent.setdefault(p_span_id, []).append(trace_span)

        # 3. Reconstruct AgentSteps from spans
        agent_steps: list[AgentStep] = []
        agent_spans = [s for s in converted_spans if s.span_type == SpanType.AGENT]

        # If no AGENT spans were explicitly declared, use root or non-tool spans as agent steps
        if not agent_spans:
            agent_spans = [
                s for s in converted_spans if s.span_type not in (SpanType.TOOL, SpanType.LLM)
            ]
        if not agent_spans:
            agent_spans = converted_spans

        last_state: dict[str, Any] = {}

        for idx, a_span in enumerate(agent_spans, start=1):
            # Gather child tools & llm calls
            child_spans = child_spans_by_parent.get(a_span.span_id, [])
            tool_spans = [cs for cs in child_spans if cs.span_type == SpanType.TOOL]
            llm_spans = [cs for cs in child_spans if cs.span_type == SpanType.LLM]

            tool_calls = [
                {
                    "name": ts.name,
                    "input": ts.input,
                    "output": ts.output,
                    "latency_ms": ts.latency_ms,
                    **({"error": ts.error} if ts.error else {}),
                }
                for ts in tool_spans
            ]

            # Aggregate tokens
            total_p = a_span.tokens.prompt + sum(ls.tokens.prompt for ls in llm_spans)
            total_c = a_span.tokens.completion + sum(ls.tokens.completion for ls in llm_spans)
            total_t = a_span.tokens.total + sum(ls.tokens.total for ls in llm_spans)
            step_tokens = TokenUsage(prompt=total_p, completion=total_c, total=total_t)

            # Parse state dicts for handoff capture
            input_dict = self._parse_state_dict(a_span.input)
            output_dict = self._parse_state_dict(a_span.output)

            capture = HandoffCapture(input_state=input_dict if input_dict else last_state)
            capture.record_agent_return(output_dict)
            in_s, filt_s, out_s, diff = capture.finalize()
            last_state = output_dict

            prompt = f"OTLP Span: {a_span.name} (iteration {a_span.iteration_index})"
            if diff.summary():
                prompt = f"{prompt} | {diff.summary()}"

            step = AgentStep(
                run_id=trace_id,
                step=idx,
                agent=a_span.name,
                status=a_span.status,
                input=a_span.input,
                output=a_span.output,
                prompt=prompt,
                error=a_span.error,
                timestamp=datetime.now(UTC),
                latency_ms=a_span.latency_ms,
                tokens=step_tokens,
                tool_calls=tool_calls,
                parent_step=str(idx - 1) if idx > 1 else None,
            )

            step.handoff.input_state = in_s
            step.handoff.filtered_state = filt_s
            step.handoff.output_state = out_s

            agent_steps.append(step)

        total_latency = sum(s.latency_ms for s in agent_steps)
        if min_start_nano and max_end_nano and max_end_nano >= min_start_nano:
            total_latency = (max_end_nano - min_start_nano) / 1_000_000.0

        total_tokens = sum(s.tokens.total for s in agent_steps)

        return RunTrace(
            run_id=trace_id,
            workflow=workflow,
            status=StepStatus.FAILURE if has_trace_error else StepStatus.SUCCESS,
            timestamp=_nano_to_datetime(min_start_nano),
            total_latency_ms=total_latency,
            total_tokens=total_tokens,
            steps=agent_steps,
            spans=converted_spans,
        )

    @staticmethod
    def _parse_state_dict(val: str | Any) -> dict[str, Any]:
        """Attempt to parse string payload into dictionary for HandoffCapture."""
        if isinstance(val, dict):
            return val
        if not val or not isinstance(val, str):
            return {}
        trimmed = val.strip()
        if trimmed.startswith("{") and trimmed.endswith("}"):
            try:
                parsed = json.loads(trimmed)
                return parsed if isinstance(parsed, dict) else {"value": parsed}
            except Exception:
                pass
        return {"value": val}
