"""
scripts/generate_labeled_set.py
================================
Day 15 — Generates the 20-run labeled test set used for all future validation passes.

Creates synthetic traces for each failure category by crafting precise
source/entity count patterns that deterministically trigger each rule.
No LLM calls required — traces are fully synthetic and reproducible.

Run from project root:
    python scripts/generate_labeled_set.py

Output:
    - sample_data/labeled_traces/run_lbl_*.json  (20 trace files)
    - sample_data/labels.json                    (ground truth labels)
    - Imports all 20 runs into the DB
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from typing import Any

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.db import DatabaseManager

SCHEMA_VERSION = "1.0"
WORKFLOW = "research_report_pipeline"
TRACES_DIR = "sample_data/labeled_traces"
LABELS_PATH = "sample_data/labels.json"
DB_PATH = "data/agentlens.db"


# ─────────────────────────────────────────────────────────────────────────────
# Trace builder helpers
# ─────────────────────────────────────────────────────────────────────────────


def _ts(offset_seconds: float = 0.0) -> str:
    """ISO timestamp."""
    return datetime.now(UTC).isoformat()


def _make_step(
    run_id: str,
    step: int,
    agent: str,
    topic: str,
    source_count: int,
    entity_count: int,
    written_report: str = "",
    verification_result: str = "",
    verified: bool = False,
    tool_calls: list[dict] = None,
    status: str = "SUCCESS",
    error: str | None = None,
    latency_ms: float = 1200.0,
) -> dict[str, Any]:
    """Build one agent step matching the existing trace schema exactly."""
    tool_calls = tool_calls or []

    input_state: dict[str, Any] = {
        "topic": topic,
        "research_findings": "",
        "source_count": 0,
        "entity_count": 0,
        "written_report": written_report,
        "verification_result": verification_result,
        "verified": verified,
        "revision_notes": "",
    }

    output_state: dict[str, Any] = {
        "source_count": source_count,
        "entity_count": entity_count,
    }

    if agent == "researcher":
        output_state["research_findings"] = _make_research_text(topic, source_count, entity_count)
    elif agent == "writer":
        output_state["written_report"] = _make_report_text(topic, source_count, entity_count)
    elif agent == "verifier":
        output_state["verification_result"] = verification_result
        output_state["verified"] = verified

    filtered = {k: v for k, v in output_state.items()}

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "step": step,
        "agent": agent,
        "node_type": "llm",
        "input": json.dumps(input_state),
        "output": json.dumps(output_state),
        "expected_output": None,
        "prompt": f"added={list(output_state.keys())}",
        "system_prompt": "",
        "model": "llama3-8b-8192",
        "generation_params": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 2048, "seed": None},
        "latency_ms": latency_ms,
        "tokens": {"prompt": 200, "completion": 400, "total": 600},
        "tool_calls": tool_calls,
        "handoff": {
            "input_state": input_state,
            "filtered_state": filtered,
            "output_state": {**input_state, **output_state},
        },
        "parent_step": step - 1 if step > 1 else None,
        "child_step": step + 1,
        "status": status,
        "error": error,
        "timestamp": _ts(),
    }


def _make_research_text(topic: str, sources: int, entities: int) -> str:
    src_lines = "\n".join(f"- Source {i + 1} on {topic}" for i in range(sources))
    ent_lines = "\n".join(f"- Entity_{i + 1}" for i in range(entities))
    return f"SOURCES:\n{src_lines}\n\nENTITIES:\n{ent_lines}\n\nKEY FINDINGS:\nResearch findings for {topic}."


def _make_report_text(topic: str, sources: int, entities: int) -> str:
    return (
        f"**Report on {topic}**\n\n"
        f"Based on {sources} sources and {entities} entities, "
        f"this report covers the key aspects of {topic}.\n\n"
        f"**Conclusion**: The analysis shows significant findings about {topic}."
    )


def _make_trace(
    run_id: str,
    topic: str,
    steps: list[dict],
    total_latency_ms: float,
) -> dict[str, Any]:
    """Assemble a full trace dict."""
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "workflow": WORKFLOW,
        "timestamp": _ts(),
        "steps": steps,
        "total_latency_ms": total_latency_ms,
        "total_tokens": len(steps) * 600,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 20 labeled trace specifications
# ─────────────────────────────────────────────────────────────────────────────

LABELED_SPECS = [
    # ── 4× PASS ──────────────────────────────────────────────────────────────
    {
        "label_id": "lbl_pass_01",
        "topic": "History of the Eiffel Tower",
        "category": "pass",
        "expected_verdict": "PASS",
        "expected_primary_agent": None,
        "notes": "Researcher and writer have identical source/entity counts. No rule should fire.",
        "researcher_sources": 8,
        "researcher_entities": 10,
        "writer_sources": 8,
        "writer_entities": 10,
        "include_verifier": True,
        "verified": True,
        "verifier_sources": 8,
        "verifier_entities": 10,
        "tool_calls": [],
        "exec_failure": False,
        "skip_step": None,
    },
    {
        "label_id": "lbl_pass_02",
        "topic": "How vaccines work",
        "category": "pass",
        "expected_verdict": "PASS",
        "expected_primary_agent": None,
        "notes": "Minor 1-entity variation — within tolerance. Pipeline runs cleanly.",
        "researcher_sources": 7,
        "researcher_entities": 9,
        "writer_sources": 7,
        "writer_entities": 9,
        "include_verifier": True,
        "verified": True,
        "verifier_sources": 7,
        "verifier_entities": 9,
        "tool_calls": [],
        "exec_failure": False,
        "skip_step": None,
    },
    {
        "label_id": "lbl_pass_03",
        "topic": "The water cycle explained",
        "category": "pass",
        "expected_verdict": "PASS",
        "expected_primary_agent": None,
        "notes": "Low source/entity run — small topic, still balanced across agents.",
        "researcher_sources": 4,
        "researcher_entities": 6,
        "writer_sources": 4,
        "writer_entities": 6,
        "include_verifier": True,
        "verified": True,
        "verifier_sources": 4,
        "verifier_entities": 6,
        "tool_calls": [],
        "exec_failure": False,
        "skip_step": None,
    },
    {
        "label_id": "lbl_pass_04",
        "topic": "The French Revolution",
        "category": "pass",
        "expected_verdict": "PASS",
        "expected_primary_agent": None,
        "notes": "High-source run, all agents faithful. Verifier flags nothing.",
        "researcher_sources": 12,
        "researcher_entities": 18,
        "writer_sources": 12,
        "writer_entities": 18,
        "include_verifier": True,
        "verified": True,
        "verifier_sources": 12,
        "verifier_entities": 18,
        "tool_calls": [],
        "exec_failure": False,
        "skip_step": None,
    },
    # ── 4× REASONING FAILURE (hallucination — writer adds entities) ──────────
    {
        "label_id": "lbl_reasoning_01",
        "topic": "Rise of AI in America",
        "category": "reasoning_failure",
        "expected_verdict": "WARNING",
        "expected_primary_agent": "writer",
        "notes": "Writer introduced 14 extra entities not in research (hallucination). Classic information_gain signal.",
        "researcher_sources": 8,
        "researcher_entities": 10,
        "writer_sources": 11,
        "writer_entities": 24,
        "include_verifier": True,
        "verified": True,
        "verifier_sources": 11,
        "verifier_entities": 24,
        "tool_calls": [],
        "exec_failure": False,
        "skip_step": None,
    },
    {
        "label_id": "lbl_reasoning_02",
        "topic": "Quantum computing applications",
        "category": "reasoning_failure",
        "expected_verdict": "WARNING",
        "expected_primary_agent": "writer",
        "notes": "Writer nearly doubles entity count — large hallucination signal.",
        "researcher_sources": 6,
        "researcher_entities": 8,
        "writer_sources": 10,
        "writer_entities": 18,
        "include_verifier": True,
        "verified": True,
        "verifier_sources": 10,
        "verifier_entities": 18,
        "tool_calls": [],
        "exec_failure": False,
        "skip_step": None,
    },
    {
        "label_id": "lbl_reasoning_03",
        "topic": "Climate change and renewable energy",
        "category": "reasoning_failure",
        "expected_verdict": "WARNING",
        "expected_primary_agent": "writer",
        "notes": "Moderate hallucination — writer adds 8 entities beyond research.",
        "researcher_sources": 5,
        "researcher_entities": 7,
        "writer_sources": 8,
        "writer_entities": 15,
        "include_verifier": True,
        "verified": True,
        "verifier_sources": 8,
        "verifier_entities": 15,
        "tool_calls": [],
        "exec_failure": False,
        "skip_step": None,
    },
    {
        "label_id": "lbl_reasoning_04",
        "topic": "Space exploration history",
        "category": "reasoning_failure",
        "expected_verdict": "WARNING",
        "expected_primary_agent": "writer",
        "notes": "Writer adds sources and entities — both dimensions inflated.",
        "researcher_sources": 7,
        "researcher_entities": 9,
        "writer_sources": 11,
        "writer_entities": 20,
        "include_verifier": True,
        "verified": True,
        "verifier_sources": 11,
        "verifier_entities": 20,
        "tool_calls": [],
        "exec_failure": False,
        "skip_step": None,
    },
    # ── 4× EXECUTION FAILURE (tool call error / missing output) ─────────────
    {
        "label_id": "lbl_execution_01",
        "topic": "Latest stock market trends",
        "category": "execution_failure",
        "expected_verdict": "FAIL",
        "expected_primary_agent": "researcher",
        "notes": "Researcher tool call returns error — web search API timed out. Result: 0 sources found.",
        "researcher_sources": 0,
        "researcher_entities": 0,
        "writer_sources": 0,
        "writer_entities": 0,
        "include_verifier": False,
        "verified": False,
        "verifier_sources": 0,
        "verifier_entities": 0,
        "tool_calls": [
            {
                "name": "web_search",
                "args": {"query": "stock market trends 2024"},
                "result": None,
                "error": "TimeoutError: Request timed out after 30s",
            }
        ],
        "exec_failure": True,
        "skip_step": None,
    },
    {
        "label_id": "lbl_execution_02",
        "topic": "Real-time weather in Mumbai",
        "category": "execution_failure",
        "expected_verdict": "FAIL",
        "expected_primary_agent": "researcher",
        "notes": "Tool call made but returned empty result. Researcher step completes but no data captured.",
        "researcher_sources": 0,
        "researcher_entities": 1,
        "writer_sources": 0,
        "writer_entities": 1,
        "include_verifier": False,
        "verified": False,
        "verifier_sources": 0,
        "verifier_entities": 0,
        "tool_calls": [
            {"name": "weather_api", "args": {"city": "Mumbai"}, "result": "", "error": None}
        ],
        "exec_failure": True,
        "skip_step": None,
    },
    {
        "label_id": "lbl_execution_03",
        "topic": "Current cryptocurrency prices",
        "category": "execution_failure",
        "expected_verdict": "FAIL",
        "expected_primary_agent": "researcher",
        "notes": "Multiple tool calls — all return auth error. API key expired.",
        "researcher_sources": 0,
        "researcher_entities": 0,
        "writer_sources": 0,
        "writer_entities": 0,
        "include_verifier": False,
        "verified": False,
        "verifier_sources": 0,
        "verifier_entities": 0,
        "tool_calls": [
            {
                "name": "crypto_api",
                "args": {"coin": "BTC"},
                "result": None,
                "error": "AuthError: Invalid API key",
            },
            {
                "name": "crypto_api",
                "args": {"coin": "ETH"},
                "result": None,
                "error": "AuthError: Invalid API key",
            },
        ],
        "exec_failure": True,
        "skip_step": None,
    },
    {
        "label_id": "lbl_execution_04",
        "topic": "Live sports scores",
        "category": "execution_failure",
        "expected_verdict": "FAIL",
        "expected_primary_agent": "researcher",
        "notes": "Tool call declared but output field is null — capture failure, output state incomplete.",
        "researcher_sources": 2,
        "researcher_entities": 3,
        "writer_sources": 2,
        "writer_entities": 3,
        "include_verifier": False,
        "verified": False,
        "verifier_sources": 0,
        "verifier_entities": 0,
        "tool_calls": [
            {"name": "sports_api", "args": {"league": "IPL"}, "result": None, "error": None}
        ],
        "exec_failure": True,
        "skip_step": None,
    },
    # ── 4× WORKFLOW FAILURE (missing/wrong-order steps) ─────────────────────
    {
        "label_id": "lbl_workflow_01",
        "topic": "Benefits of meditation",
        "category": "workflow_failure",
        "expected_verdict": "FAIL",
        "expected_primary_agent": "verifier",
        "notes": "Verifier step entirely absent from trace. Workflow incomplete.",
        "researcher_sources": 8,
        "researcher_entities": 10,
        "writer_sources": 8,
        "writer_entities": 10,
        "include_verifier": False,
        "verified": False,
        "verifier_sources": 0,
        "verifier_entities": 0,
        "tool_calls": [],
        "exec_failure": False,
        "skip_step": "verifier",
    },
    {
        "label_id": "lbl_workflow_02",
        "topic": "History of the Roman Empire",
        "category": "workflow_failure",
        "expected_verdict": "FAIL",
        "expected_primary_agent": "verifier",
        "notes": "Verifier skipped — writer output never checked. High-risk for unverified claims.",
        "researcher_sources": 10,
        "researcher_entities": 14,
        "writer_sources": 10,
        "writer_entities": 14,
        "include_verifier": False,
        "verified": False,
        "verifier_sources": 0,
        "verifier_entities": 0,
        "tool_calls": [],
        "exec_failure": False,
        "skip_step": "verifier",
    },
    {
        "label_id": "lbl_workflow_03",
        "topic": "Introduction to machine learning",
        "category": "workflow_failure",
        "expected_verdict": "FAIL",
        "expected_primary_agent": "researcher",
        "notes": "Researcher step absent — writer has no research input to work from.",
        "researcher_sources": 0,
        "researcher_entities": 0,
        "writer_sources": 5,
        "writer_entities": 8,
        "include_verifier": True,
        "verified": False,
        "verifier_sources": 5,
        "verifier_entities": 8,
        "tool_calls": [],
        "exec_failure": False,
        "skip_step": "researcher",
    },
    {
        "label_id": "lbl_workflow_04",
        "topic": "Principles of sustainable agriculture",
        "category": "workflow_failure",
        "expected_verdict": "FAIL",
        "expected_primary_agent": "verifier",
        "notes": "Pipeline terminates after writer — verifier never runs. Incomplete workflow.",
        "researcher_sources": 6,
        "researcher_entities": 9,
        "writer_sources": 6,
        "writer_entities": 9,
        "include_verifier": False,
        "verified": False,
        "verifier_sources": 0,
        "verifier_entities": 0,
        "tool_calls": [],
        "exec_failure": False,
        "skip_step": "verifier",
    },
    # ── 4× VERIFICATION FAILURE (verifier passes hallucinated content) ───────
    {
        "label_id": "lbl_verification_01",
        "topic": "The history of the internet",
        "category": "verification_failure",
        "expected_verdict": "FAIL",
        "expected_primary_agent": "verifier",
        "notes": "Writer hallucinated 12 entities; verifier approved without catching any. Verifier entity count matches writer's inflated count.",
        "researcher_sources": 6,
        "researcher_entities": 8,
        "writer_sources": 9,
        "writer_entities": 20,
        "include_verifier": True,
        "verified": True,
        "verifier_sources": 9,
        "verifier_entities": 20,
        "tool_calls": [],
        "exec_failure": False,
        "skip_step": None,
    },
    {
        "label_id": "lbl_verification_02",
        "topic": "Blockchain technology fundamentals",
        "category": "verification_failure",
        "expected_verdict": "FAIL",
        "expected_primary_agent": "verifier",
        "notes": "Verifier rubber-stamps writer output — approved=True despite entity inflation of 15.",
        "researcher_sources": 5,
        "researcher_entities": 7,
        "writer_sources": 8,
        "writer_entities": 22,
        "include_verifier": True,
        "verified": True,
        "verifier_sources": 8,
        "verifier_entities": 22,
        "tool_calls": [],
        "exec_failure": False,
        "skip_step": None,
    },
    {
        "label_id": "lbl_verification_03",
        "topic": "Future of autonomous vehicles",
        "category": "verification_failure",
        "expected_verdict": "FAIL",
        "expected_primary_agent": "verifier",
        "notes": "Always-approve verifier pattern — verifier never rejects anything regardless of content quality.",
        "researcher_sources": 7,
        "researcher_entities": 10,
        "writer_sources": 12,
        "writer_entities": 25,
        "include_verifier": True,
        "verified": True,
        "verifier_sources": 12,
        "verifier_entities": 25,
        "tool_calls": [],
        "exec_failure": False,
        "skip_step": None,
    },
    {
        "label_id": "lbl_verification_04",
        "topic": "Impact of social media on mental health",
        "category": "verification_failure",
        "expected_verdict": "FAIL",
        "expected_primary_agent": "verifier",
        "notes": "Verifier accepts inflated source count without flagging the discrepancy from research.",
        "researcher_sources": 4,
        "researcher_entities": 6,
        "writer_sources": 11,
        "writer_entities": 18,
        "include_verifier": True,
        "verified": True,
        "verifier_sources": 11,
        "verifier_entities": 18,
        "tool_calls": [],
        "exec_failure": False,
        "skip_step": None,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Build traces from specs
# ─────────────────────────────────────────────────────────────────────────────


def build_trace(spec: dict) -> tuple[str, dict]:
    """Build a full trace dict from a spec. Returns (run_id, trace_dict)."""
    run_id = f"run_{spec['label_id']}"
    topic = spec["topic"]
    steps = []
    latency_total = 0.0

    skip = spec.get("skip_step")

    # ── Researcher step ───────────────────────────────────────────────────────
    if skip != "researcher":
        exec_fail = spec.get("exec_failure", False)
        r_status = "ERROR" if exec_fail else "SUCCESS"
        r_error = "ToolExecutionError: tool call failed or returned empty" if exec_fail else None

        step = _make_step(
            run_id=run_id,
            step=1,
            agent="researcher",
            topic=topic,
            source_count=spec["researcher_sources"],
            entity_count=spec["researcher_entities"],
            tool_calls=spec.get("tool_calls", []),
            status=r_status,
            error=r_error,
            latency_ms=1200.0,
        )
        steps.append(step)
        latency_total += 1200.0

    # ── Writer step ───────────────────────────────────────────────────────────
    step = _make_step(
        run_id=run_id,
        step=len(steps) + 1,
        agent="writer",
        topic=topic,
        source_count=spec["writer_sources"],
        entity_count=spec["writer_entities"],
        latency_ms=2000.0,
    )
    steps.append(step)
    latency_total += 2000.0

    # ── Verifier step ─────────────────────────────────────────────────────────
    if spec.get("include_verifier", True) and skip != "verifier":
        v_result = "APPROVED" if spec.get("verified") else "NEEDS_REVISION"
        step = _make_step(
            run_id=run_id,
            step=len(steps) + 1,
            agent="verifier",
            topic=topic,
            source_count=spec.get("verifier_sources", spec["writer_sources"]),
            entity_count=spec.get("verifier_entities", spec["writer_entities"]),
            verification_result=v_result,
            verified=spec.get("verified", False),
            latency_ms=800.0,
        )
        steps.append(step)
        latency_total += 800.0

    trace = _make_trace(run_id, topic, steps, latency_total)
    return run_id, trace


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    os.makedirs(TRACES_DIR, exist_ok=True)
    db = DatabaseManager(DB_PATH)
    db.initialize()

    labels = []
    generated = 0

    print(f"\n{'=' * 60}")
    print("AgentLens — Day 15: Labeled Test Set Generator")
    print(f"{'=' * 60}")

    for spec in LABELED_SPECS:
        run_id, trace = build_trace(spec)

        # Save trace JSON
        trace_path = os.path.join(TRACES_DIR, f"{run_id}.json")
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2)

        # Insert into DB
        total_latency = trace["total_latency_ms"]
        total_tokens = trace["total_tokens"]
        trace_json = json.dumps(trace)

        db.insert_run(
            run_id=run_id,
            workflow=WORKFLOW,
            timestamp=trace["timestamp"],
            status="SUCCESS" if spec["category"] != "execution_failure" else "ERROR",
            total_latency_ms=total_latency,
            total_tokens=total_tokens,
            schema_version=SCHEMA_VERSION,
            trace_path=trace_path,
            trace_json=trace_json,
        )

        # Insert steps
        for step_data in trace["steps"]:
            db.insert_step(
                run_id=run_id,
                step=step_data["step"],
                agent=step_data["agent"],
                status=step_data["status"],
                latency_ms=step_data["latency_ms"],
                tokens_prompt=step_data["tokens"]["prompt"],
                tokens_completion=step_data["tokens"]["completion"],
                tokens_total=step_data["tokens"]["total"],
                diff_summary="",
                error=step_data.get("error"),
                timestamp=step_data["timestamp"],
                schema_version=SCHEMA_VERSION,
            )

        # Build label record
        labels.append(
            {
                "run_id": run_id,
                "label_id": spec["label_id"],
                "topic": spec["topic"],
                "category": spec["category"],
                "expected_verdict": spec["expected_verdict"],
                "expected_primary_agent": spec["expected_primary_agent"],
                "notes": spec["notes"],
                "researcher_sources": spec["researcher_sources"],
                "researcher_entities": spec["researcher_entities"],
                "writer_sources": spec["writer_sources"],
                "writer_entities": spec["writer_entities"],
                "include_verifier": spec.get("include_verifier", True),
                "has_tool_calls": len(spec.get("tool_calls", [])) > 0,
                "skip_step": spec.get("skip_step"),
                "frozen": True,  # never regenerate — used as fixed validation baseline
            }
        )

        cat_emoji = {
            "pass": "✅",
            "reasoning_failure": "🟡",
            "execution_failure": "🔴",
            "verification_failure": "🔵",
            "workflow_failure": "🟠",
        }
        print(
            f"  {cat_emoji.get(spec['category'], '⚪')} {run_id}  [{spec['category']}]  {spec['topic']}"
        )
        generated += 1

    # Save labels.json
    labels_data = {
        "generated_at": datetime.now(UTC).isoformat(),
        "version": "1.0.0",
        "total": len(labels),
        "frozen": True,
        "note": "DO NOT regenerate this file after Day 15. This is the fixed validation baseline used for all future accuracy measurements.",
        "category_counts": {
            "pass": sum(1 for lbl in labels if lbl["category"] == "pass"),
            "reasoning_failure": sum(1 for lbl in labels if lbl["category"] == "reasoning_failure"),
            "execution_failure": sum(1 for lbl in labels if lbl["category"] == "execution_failure"),
            "workflow_failure": sum(1 for lbl in labels if lbl["category"] == "workflow_failure"),
            "verification_failure": sum(
                1 for lbl in labels if lbl["category"] == "verification_failure"
            ),
        },
        "runs": labels,
    }

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels_data, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  Generated {generated}/20 labeled traces")
    print(f"  Trace files : {TRACES_DIR}/")
    print(f"  Labels file : {LABELS_PATH}")
    print(f"  DB updated  : {DB_PATH}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
