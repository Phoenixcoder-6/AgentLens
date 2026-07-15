"""
app/pipeline.py — AgentLens Reference Pipeline
===============================================
A three-agent LangGraph pipeline used as the reference workflow for
AgentLens observability and failure attribution.

Pipeline:
    Researcher  →  Writer  →  Verifier

Task: Given a topic, research it, write a structured report, and verify
      the report is accurate and complete.

Day 4 note:
    No capture or observability yet — this is a clean pipeline that just
    runs and produces output. The @trace_step decorator (Day 5) will be
    layered on top without touching this logic.

All configuration (model, temperature, max_tokens) comes from config.yaml.
Nothing is hardcoded here.
"""

from __future__ import annotations

import os
import sys
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.config_loader import get
from capture.tracer import trace_step

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline State
# ─────────────────────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    """
    Shared state that flows between all agents in the pipeline.

    Each agent reads from this state and writes its output back into it.
    The HandoffState capture (Day 6) will snapshot this dict before and
    after each agent call to detect information loss.
    """
    topic: str                    # The research topic / input question
    research_findings: str        # Researcher output — key facts and sources
    source_count: int             # Number of sources the researcher cited
    entity_count: int             # Number of entities the researcher identified
    written_report: str           # Writer output — structured report
    verification_result: str      # Verifier output — APPROVED / NEEDS_REVISION
    verified: bool                # True if verifier approved the report
    revision_notes: str           # Verifier's notes if report needs revision


# ─────────────────────────────────────────────────────────────────────────────
# LLM Client (loaded from config.yaml — nothing hardcoded)
# ─────────────────────────────────────────────────────────────────────────────

def _build_llm() -> ChatGroq:
    """Build the LLM client from config.yaml settings."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found. "
            "Copy .env.example to .env and set your key."
        )
    return ChatGroq(
        model=get("llm", "model"),
        temperature=get("llm", "temperature"),
        max_tokens=get("llm", "max_tokens"),
        api_key=api_key,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent Nodes
# ─────────────────────────────────────────────────────────────────────────────

@trace_step
def researcher_node(state: PipelineState) -> dict:
    """
    Researcher Agent
    ----------------
    Receives a topic and produces structured research findings:
    key facts, named entities, and cited sources.

    Output written to state:
        research_findings : prose summary of research
        source_count      : how many sources were consulted
        entity_count      : how many entities were identified

    All targets read from config.yaml — pipeline.researcher
    """
    llm = _build_llm()

    min_sources  = get("pipeline", "researcher", "min_sources")
    max_sources  = get("pipeline", "researcher", "max_sources")
    min_entities = get("pipeline", "researcher", "min_entities")
    depth        = get("pipeline", "researcher", "depth")

    system_prompt = (
        f"You are a meticulous research agent. Given a topic, you produce "
        f"structured research findings with clearly cited sources, key facts, "
        f"and named entities.\n\n"
        f"Requirements:\n"
        f"- Cite between {min_sources} and {max_sources} sources (books, papers, courses, articles)\n"
        f"- Identify at least {min_entities} named entities (people, organizations, dates, places, concepts)\n"
        f"- Research depth: {depth}\n\n"
        f"Format your response EXACTLY as:\n"
        f"SOURCES (list each on a new line starting with '- '):\n"
        f"ENTITIES (list each on a new line starting with '- '):\n"
        f"KEY FINDINGS (detailed prose):"
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Research this topic thoroughly: {state['topic']}"),
    ])

    findings = response.content

    # Count sources and entities from the structured response
    source_count = findings.count("\n- ", 0, findings.find("ENTITIES")) if "ENTITIES" in findings else findings.count("\n- ")
    entity_lines = findings[findings.find("ENTITIES"):findings.find("KEY FINDINGS")] if "ENTITIES" in findings and "KEY FINDINGS" in findings else ""
    entity_count = entity_lines.count("\n- ")

    return {
        "research_findings": findings,
        "source_count": max(source_count, 0),
        "entity_count": max(entity_count, 0),
    }


@trace_step
def writer_node(state: PipelineState) -> dict:
    """
    Writer Agent
    ------------
    Receives research findings and produces a well-structured written report.
    Must reference all sources and entities from the research.

    Output written to state:
        written_report : the full report text

    All settings read from config.yaml — pipeline.writer
    """
    llm = _build_llm()

    sections       = get("pipeline", "writer", "report_sections")
    citation_style = get("pipeline", "writer", "citation_style")
    word_count     = get("pipeline", "writer", "word_count_target")
    sections_str   = ", ".join(sections)

    system_prompt = (
        f"You are a professional report writer. Given research findings, "
        f"write a clear, well-structured report of approximately {word_count} words.\n\n"
        f"Required sections: {sections_str}\n"
        f"Citation style: {citation_style}\n\n"
        f"Rules:\n"
        f"- Reference ALL cited sources from the research\n"
        f"- Mention ALL named entities from the research\n"
        f"- Do NOT invent new facts — only use what is in the research findings\n"
        f"- Every factual claim must have a citation"
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"Write a report on: {state['topic']}\n\n"
                f"Using these research findings:\n{state['research_findings']}"
            )
        ),
    ])

    return {
        "written_report": response.content,
    }


@trace_step
def verifier_node(state: PipelineState) -> dict:
    """
    Verifier Agent
    --------------
    Reviews the written report against the research findings and topic.
    Checks for accuracy, completeness, and source/entity coverage.

    Output written to state:
        verification_result : "APPROVED" or "NEEDS_REVISION"
        verified            : True if approved
        revision_notes      : Explanation of issues (empty if approved)

    All settings read from config.yaml — pipeline.verifier
    """
    llm = _build_llm()

    strictness           = get("pipeline", "verifier", "strictness")
    check_sources        = get("pipeline", "verifier", "check_source_coverage")
    check_entities       = get("pipeline", "verifier", "check_entity_coverage")
    check_no_new_facts   = get("pipeline", "verifier", "check_no_new_facts")

    checks = []
    checks.append(f"1. Accuracy — does the report contain any facts NOT in the research? (check_no_new_facts={check_no_new_facts})")
    if check_sources:
        checks.append("2. Source coverage — are ALL cited sources from the research mentioned?")
    if check_entities:
        checks.append("3. Entity coverage — are ALL named entities from the research mentioned?")
    checks.append("4. Completeness — does the report cover all key findings?")
    checks_str = "\n".join(checks)

    system_prompt = (
        f"You are a fact-checking and verification agent operating at strictness level: {strictness}.\n"
        f"Review the written report against the original research findings.\n\n"
        f"Checks to perform:\n{checks_str}\n\n"
        f"Strictness guide:\n"
        f"  lenient   → only flag clear hallucinations\n"
        f"  standard  → flag missing sources and hallucinations\n"
        f"  strict    → flag any imprecision, added detail, or missing coverage\n\n"
        f"Respond with exactly one of:\n"
        f"APPROVED: [one sentence explaining what was verified]\n"
        f"NEEDS_REVISION: [specific bullet list of every issue found]"
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"Topic: {state['topic']}\n\n"
                f"Original Research Findings:\n{state['research_findings']}\n\n"
                f"Written Report to Verify:\n{state['written_report']}"
            )
        ),
    ])

    result = response.content.strip()
    approved = result.upper().startswith("APPROVED")

    return {
        "verification_result": result,
        "verified": approved,
        "revision_notes": "" if approved else result,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Graph Construction
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline() -> StateGraph:
    """
    Build and compile the Researcher → Writer → Verifier pipeline.

    Returns a compiled LangGraph app ready to invoke.
    The graph is a simple linear chain — no conditional edges yet.
    """
    graph = StateGraph(PipelineState)

    # Register nodes
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("verifier", verifier_node)

    # Wire the edges: START → researcher → writer → verifier → END
    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "verifier")
    graph.add_edge("verifier", END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Public run function
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(topic: str) -> PipelineState:
    """
    Run the full pipeline on a given topic and return the final state.

    Args:
        topic: The research topic, e.g. "The history of the Apollo space program"

    Returns:
        The final PipelineState with all agent outputs populated.
    """
    app = build_pipeline()

    initial_state: PipelineState = {
        "topic": topic,
        "research_findings": "",
        "source_count": 0,
        "entity_count": 0,
        "written_report": "",
        "verification_result": "",
        "verified": False,
        "revision_notes": "",
    }

    final_state = app.invoke(initial_state)
    return final_state
