"""
agentlens/integrations — Framework Auto-Instrumentation Integrations
=====================================================================
Day 58: LangGraph & LangChain (AgentLensCallbackHandler)
Day 59: CrewAI (AgentLensCrewHook) & AutoGen (AgentLensAutoGenHook)
Day 60: OpenTelemetry / OTLP (planned)
"""

from __future__ import annotations

from .autogen import AgentLensAutoGenHook, AgentLensConversationTracer
from .crewai import AgentLensCrewHook
from .langchain import AgentLensCallbackHandler

__all__ = [
    "AgentLensCallbackHandler",
    "AgentLensCrewHook",
    "AgentLensAutoGenHook",
    "AgentLensConversationTracer",
]
