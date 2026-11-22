"""
agentlens/integrations/autogen.py — Microsoft AutoGen Swarm Telemetry & Turn Tracer
==================================================================================
Day 59: Provides conversational turn tracing for AutoGen multi-agent swarms.

Intercepts:
  - Conversational turns between ConversableAgent, AssistantAgent, and UserProxyAgent
  - GroupChat round-robin and manager-selected speaker transitions
  - Speaker cycling with iteration_index tracking (A -> B -> A)
  - Python code block extraction, execution interception, and tool spans
  - Code execution error detection (exitcode: 1 / Traceback) -> Arbiter tool_failure_v1
  - State handoffs between conversational turns
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from capture.handoff import HandoffCapture
from normalizer.normalizer import Normalizer
from schema.models import (
    AgentStep,
    AnalysisBundle,
    RunTrace,
    StepStatus,
    TokenUsage,
)
from schema.span import SpanType, TraceSpan
from storage.db import DatabaseManager
from storage.writer import StorageWriter

logger = logging.getLogger("agentlens.integrations.autogen")

# Pattern to extract code blocks: ```python ... ```
CODE_BLOCK_PATTERN = re.compile(r"```([a-zA-Z0-9_\-\+]*)\n(.*?)```", re.DOTALL)


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


def _extract_agent_name(agent_obj: Any) -> str:
    """Extract name from an AutoGen agent object, dict, or string."""
    if agent_obj is None:
        return "autogen_agent"
    if isinstance(agent_obj, str):
        return agent_obj
    if hasattr(agent_obj, "name"):
        return str(agent_obj.name)
    if isinstance(agent_obj, dict):
        return str(agent_obj.get("name") or agent_obj.get("role") or "autogen_agent")
    return str(agent_obj)


def _extract_message_content(message: Any) -> tuple[str, dict[str, Any]]:
    """Extract text content and dict payload from an AutoGen message."""
    if message is None:
        return "", {}
    if isinstance(message, str):
        try:
            parsed = (
                json.loads(message) if message.strip().startswith("{") else {"content": message}
            )
            return message, parsed
        except Exception:
            return message, {"content": message}
    if isinstance(message, dict):
        content = str(message.get("content") or "")
        return content, message
    # Message object with content attribute
    content = getattr(message, "content", str(message))
    return str(content), {"content": content}


def _detect_code_blocks(content: str) -> list[tuple[str, str]]:
    """Detect fenced code blocks in message content: list of (language, code)."""
    return CODE_BLOCK_PATTERN.findall(content)


def _detect_execution_failure(content: str) -> str | None:
    """Detect if message content indicates a code execution failure."""
    lowered = content.lower()
    if "exitcode: 1" in lowered or "exitcode: 2" in lowered:
        return content.strip()
    if "traceback (most recent call last)" in lowered:
        return content.strip()
    if "syntaxerror" in lowered or "nameerror" in lowered or "typeerror" in lowered:
        return content.strip()
    return None


class AgentLensAutoGenHook:
    """
    Automatic telemetry hook and conversation tracer for AutoGen multi-agent swarms.

    Usage with AutoGen:
        hook = AgentLensAutoGenHook(workflow="support_group_chat", save_to_db=True)

        # Register hook on AutoGen agents:
        hook.register(user_proxy)
        hook.register(assistant)

        # Or monitor group chat:
        user_proxy.initiate_chat(assistant, message="Solve problem")

        trace = hook.get_trace()
        bundle = hook.evaluate()

    Or via context manager:
        with hook.monitor_conversation([user_proxy, assistant]):
            user_proxy.initiate_chat(assistant, message="Solve problem")
    """

    def __init__(
        self,
        workflow: str = "autogen_conversation",
        run_id: str | None = None,
        save_to_db: bool = False,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.workflow = workflow
        self.run_id = run_id or f"run_{uuid4().hex[:8]}"
        self.save_to_db = save_to_db
        self.tags = tags or []
        self.metadata = metadata or {}

        # Root DAG span
        self._root_span_id = f"span_{uuid4().hex[:8]}"

        # Initialize RunTrace
        self.trace = RunTrace(
            run_id=self.run_id,
            workflow=self.workflow,
            status=StepStatus.SUCCESS,
            timestamp=datetime.now(UTC),
            steps=[],
            spans=[],
        )

        # Track iteration index per speaker
        self._speaker_iterations: dict[str, int] = {}

        # In-flight conversational state
        self._last_content: str = ""
        self._last_state: dict[str, Any] = {}
        self._registered_agents: list[Any] = []

    def get_trace(self) -> RunTrace:
        """Finalize and return the recorded RunTrace."""
        # Calculate aggregates
        total_lat = sum(s.latency_ms for s in self.trace.steps)
        total_tok = sum(s.tokens.total for s in self.trace.steps)
        self.trace.total_latency_ms = total_lat
        self.trace.total_tokens = total_tok

        # If any step had ERROR/FAILURE, propagate to trace
        if any(s.status in (StepStatus.ERROR, StepStatus.FAILURE) for s in self.trace.steps):
            self.trace.status = StepStatus.FAILURE

        if self.save_to_db:
            try:
                db = DatabaseManager()
                db.initialize()
                normalized = Normalizer().normalize_run(self.trace)
                StorageWriter(db).write_run(
                    normalized,
                    trace_json=self.trace.model_dump_json(indent=2),
                )
            except Exception as exc:
                logger.warning("Failed to auto-save AutoGen trace to database: %s", exc)

        return self.trace

    def get_spans(self) -> list[TraceSpan]:
        """Return all recorded OpenInference trace spans."""
        return self.trace.spans

    def get_steps(self) -> list[AgentStep]:
        """Return all recorded sequential agent steps."""
        return self.trace.steps

    def evaluate(self) -> AnalysisBundle:
        """Run Arbiter diagnostic root-cause analysis on this trace."""
        import agentlens

        return agentlens.evaluate_run(self.get_trace())

    # ─────────────────────────────────────────────────────────────────────────
    # AutoGen Registration & Hook Handlers
    # ─────────────────────────────────────────────────────────────────────────

    def register(self, agent: Any) -> None:
        """
        Register AgentLens hooks onto an AutoGen ConversableAgent instance.
        Uses AutoGen's register_hook API if available.
        """
        self._registered_agents.append(agent)

        if hasattr(agent, "register_hook"):

            def _before_send(
                sender: Any,
                message: Any,
                recipient: Any,
                silent: bool = False,
            ) -> Any:
                self.record_message(sender=sender, recipient=recipient, message=message)
                return message

            try:
                agent.register_hook("process_message_before_send", _before_send)
            except Exception as exc:
                logger.debug(
                    "Failed to register process_message_before_send on agent %s: %s", agent, exc
                )

    def register_group_chat(self, groupchat: Any) -> None:
        """Register all agents in an AutoGen GroupChat."""
        agents = getattr(groupchat, "agents", [])
        for agent in agents:
            self.register(agent)

    @contextmanager
    def monitor_conversation(
        self, agents: list[Any]
    ) -> Generator[AgentLensAutoGenHook, None, None]:
        """
        Context manager monitoring an AutoGen conversation across a list of agents.
        """
        for a in agents:
            self.register(a)

        try:
            yield self
        finally:
            self.get_trace()

    # ─────────────────────────────────────────────────────────────────────────
    # Turn Recording Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def record_message(
        self,
        sender: Any,
        recipient: Any,
        message: Any,
        *,
        latency_ms: float | None = None,
        tokens: TokenUsage | None = None,
    ) -> AgentStep:
        """
        Callback triggered when a message is sent between AutoGen agents.
        Extracts conversational turn, code blocks, state diffs, and error signals.
        """
        speaker_name = _extract_agent_name(sender)
        recipient_name = _extract_agent_name(recipient)
        content, msg_dict = _extract_message_content(message)

        return self.on_turn(
            speaker=speaker_name,
            message=content,
            recipient=recipient_name,
            payload=msg_dict,
            latency_ms=latency_ms,
            tokens=tokens,
        )

    def on_turn(
        self,
        speaker: str,
        message: str,
        *,
        recipient: str | None = None,
        payload: dict[str, Any] | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        latency_ms: float | None = None,
        tokens: TokenUsage | None = None,
        error: str | None = None,
    ) -> AgentStep:
        """
        Record a conversational turn by a speaker.
        Automatically detects embedded code blocks, execution failures,
        and constructs an AgentStep and AGENT TraceSpan.
        """
        iter_idx = self._speaker_iterations.get(speaker, 0)
        self._speaker_iterations[speaker] = iter_idx + 1

        step_idx = len(self.trace.steps) + 1
        recip_str = recipient or "group"

        # Check for code execution blocks in message
        detected_tools: list[dict[str, Any]] = list(tool_calls or [])
        code_blocks = _detect_code_blocks(message)
        for lang, code in code_blocks:
            tool_name = f"code_exec_{lang}" if lang else "python_exec"
            detected_tools.append(
                {
                    "name": tool_name,
                    "input": code.strip(),
                    "output": "Code block generated",
                    "latency_ms": 1.0,
                }
            )

        # Detect execution errors
        exec_err = error or _detect_execution_failure(message)
        if exec_err:
            if detected_tools:
                detected_tools[-1]["error"] = exec_err
            else:
                detected_tools.append(
                    {
                        "name": "code_execution",
                        "input": self._last_content or "code_block",
                        "output": message,
                        "error": exec_err,
                        "latency_ms": 1.0,
                    }
                )

        status = StepStatus.ERROR if exec_err else StepStatus.SUCCESS

        # Input state is previous conversational turn
        input_state = (
            self._last_state
            if self._last_state
            else {"turn": "initial", "message": self._last_content}
        )
        output_state = (
            payload if payload else {"speaker": speaker, "message": message, "recipient": recip_str}
        )

        # HandoffCapture for conversational state transition
        capture = HandoffCapture(input_state=input_state)
        capture.record_agent_return(output_state)
        in_s, filt_s, out_s, diff = capture.finalize()

        self._last_content = message
        self._last_state = output_state

        computed_lat = latency_ms if latency_ms is not None else 10.0
        computed_tokens = tokens or TokenUsage(prompt=0, completion=0, total=0)

        step = AgentStep(
            run_id=self.trace.run_id,
            step=step_idx,
            agent=speaker,
            status=status,
            input=_safe_serialize(input_state),
            output=message,
            prompt=f"AutoGen Turn: {speaker} -> {recip_str} (iteration {iter_idx})",
            error=exec_err,
            timestamp=datetime.now(UTC),
            latency_ms=computed_lat,
            tokens=computed_tokens,
            tool_calls=detected_tools,
            parent_step=str(step_idx - 1) if step_idx > 1 else None,
        )

        step.handoff.input_state = in_s
        step.handoff.filtered_state = filt_s
        step.handoff.output_state = out_s
        if diff.summary():
            step.prompt = f"{step.prompt} | {diff.summary()}"

        self.trace.steps.append(step)

        # Record corresponding AGENT TraceSpan
        agent_span_id = f"span_{uuid4().hex[:8]}"
        span = TraceSpan(
            span_id=agent_span_id,
            trace_id=self.trace.run_id,
            parent_span_id=self._root_span_id,
            name=speaker,
            span_type=SpanType.AGENT,
            iteration_index=iter_idx,
            latency_ms=computed_lat,
            status=status,
            error=exec_err,
            input=step.input,
            output=step.output,
            tokens=step.tokens,
        )
        self.trace.spans.append(span)

        # Record nested TOOL spans for code blocks or tool calls
        for t in detected_tools:
            tool_span = TraceSpan(
                span_id=f"span_{uuid4().hex[:8]}",
                trace_id=self.trace.run_id,
                parent_span_id=agent_span_id,
                name=t.get("name", "tool"),
                span_type=SpanType.TOOL,
                latency_ms=float(t.get("latency_ms", 1.0)),
                status=StepStatus.ERROR if "error" in t else StepStatus.SUCCESS,
                error=t.get("error"),
                input=_safe_serialize(t.get("input")),
                output=_safe_serialize(t.get("output")),
            )
            self.trace.spans.append(tool_span)

        return step


# Backward compatibility alias
AgentLensConversationTracer = AgentLensAutoGenHook
