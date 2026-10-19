"""
sample_data/generate_v2_traces.py — Day 56: Phase B Multi-Agent v2 Trace Generator
==================================================================================
Constructs 5 canonical Phase B multi-agent test traces containing reflection loops,
parallel DAG execution forks, entity-swapping substitution attacks, and semantic
contradictions.

The 5 Canonical v2 Traces:
  1. trace_v2_loop_reflection_pass:
     Cyclic self-correction loop (A -> B -> A -> B) resolving to PASS on iteration 2 (P5).
  2. trace_v2_parallel_fork_divergence:
     Parallel fork with 2 analysts where one branch diverges/drops data (P2).
  3. trace_v2_entity_substitution_swap:
     1-to-1 entity replacement attack (3 in, 3 out) detected by hallucination_v2 (P2).
  4. trace_v2_semantic_contradiction:
     Direct relation / negation inversion (failed -> met) detected by contradiction_v1 (P2).
  5. trace_v2_loop_deadlock_unresolved:
     Non-converging cyclic loop exceeding max iterations with uncorrected quality failure (P2).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from schema.models import (
    SCHEMA_VERSION,
    AgentStep,
    HandoffState,
    RunTrace,
    StepStatus,
    TokenUsage,
)
from schema.span import SpanType, TraceSpan, steps_to_spans

logger = logging.getLogger("agentlens.sample_data.v2")

DEFAULT_V2_OUTPUT_DIR = Path("sample_data/v2_traces")


def _make_step(
    run_id: str,
    step_num: int,
    agent: str,
    output_dict: dict[str, Any],
    input_state: dict[str, Any],
    filtered_state: dict[str, Any],
    output_state: dict[str, Any],
    latency_ms: float = 300.0,
    tokens_prompt: int = 150,
    tokens_completion: int = 120,
    tool_calls: list[dict[str, Any]] | None = None,
    ts: datetime | None = None,
    parent_step: str | None = None,
) -> AgentStep:
    timestamp = ts or datetime.now(UTC)
    return AgentStep(
        run_id=run_id,
        step=step_num,
        agent=agent,
        status=StepStatus.SUCCESS,
        latency_ms=latency_ms,
        tokens=TokenUsage(
            prompt=tokens_prompt,
            completion=tokens_completion,
            total=tokens_prompt + tokens_completion,
        ),
        input=json.dumps(input_state),
        output=json.dumps(output_dict),
        prompt=f"Execute {agent} step {step_num} for run {run_id}",
        tool_calls=tool_calls or [],
        handoff=HandoffState(
            input_state=input_state,
            filtered_state=filtered_state,
            output_state=output_state,
        ),
        parent_step=parent_step or (str(step_num - 1) if step_num > 1 else None),
        timestamp=timestamp,
        schema_version=SCHEMA_VERSION,
    )


def build_v2_trace_specs() -> list[dict[str, Any]]:
    """Build specifications for the 5 canonical Phase B v2 traces."""
    base_time = datetime(2026, 10, 19, 10, 0, 0, tzinfo=UTC)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Reflection Loop PASS (P5)
    # ─────────────────────────────────────────────────────────────────────────
    r1_id = "trace_v2_loop_reflection_pass"
    s1_in = {"topic": "Quantum Teleportation Protocols"}
    s1_out = {
        "draft": "Initial overview of quantum teleportation using EPR pairs without fidelity bounds.",
        "source_count": 3,
        "entity_count": 4,
    }
    s2_out = {
        "approved": False,
        "feedback": "REJECT: Missing Bell-state measurement details and fidelity benchmarks.",
        "source_count": 3,
    }
    s3_out = {
        "draft": (
            "Revised teleportation protocol incorporating Bell-state measurements, "
            "achieving 99.2% state fidelity across a 100km fiber link."
        ),
        "source_count": 5,
        "entity_count": 6,
    }
    s4_out = {
        "approved": True,
        "feedback": "APPROVED: Fidelity metrics and Bell measurements successfully verified.",
        "source_count": 5,
    }

    step1 = _make_step(
        r1_id,
        1,
        "researcher",
        s1_out,
        s1_in,
        s1_out,
        s1_out,
        ts=base_time,
    )
    step2 = _make_step(
        r1_id,
        2,
        "reviewer",
        s2_out,
        s1_out,
        s2_out,
        s2_out,
        ts=base_time + timedelta(seconds=1),
    )
    step3 = _make_step(
        r1_id,
        3,
        "researcher",
        s3_out,
        s2_out,
        s3_out,
        s3_out,
        ts=base_time + timedelta(seconds=2),
    )
    step4 = _make_step(
        r1_id,
        4,
        "reviewer",
        s4_out,
        s3_out,
        s4_out,
        s4_out,
        ts=base_time + timedelta(seconds=3),
    )

    spans_r1 = [
        TraceSpan(
            span_id="span_r1_1",
            trace_id=r1_id,
            name="researcher",
            span_type=SpanType.AGENT,
            iteration_index=0,
            output=json.dumps(s1_out),
        ),
        TraceSpan(
            span_id="span_r1_2",
            trace_id=r1_id,
            parent_span_id="span_r1_1",
            name="reviewer",
            span_type=SpanType.AGENT,
            iteration_index=0,
            output=json.dumps(s2_out),
        ),
        TraceSpan(
            span_id="span_r1_3",
            trace_id=r1_id,
            parent_span_id="span_r1_2",
            name="researcher",
            span_type=SpanType.AGENT,
            iteration_index=1,
            output=json.dumps(s3_out),
        ),
        TraceSpan(
            span_id="span_r1_4",
            trace_id=r1_id,
            parent_span_id="span_r1_3",
            name="reviewer",
            span_type=SpanType.AGENT,
            iteration_index=1,
            output=json.dumps(s4_out),
        ),
    ]

    trace_loop_pass = RunTrace(
        run_id=r1_id,
        workflow="iterative_reflection_pipeline",
        status=StepStatus.SUCCESS,
        total_latency_ms=1200.0,
        total_tokens=1100,
        timestamp=base_time,
        steps=[step1, step2, step3, step4],
        spans=spans_r1,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Parallel Fork Divergence (P2)
    # ─────────────────────────────────────────────────────────────────────────
    r2_id = "trace_v2_parallel_fork_divergence"
    t2_time = base_time + timedelta(minutes=10)
    p_coord_in = {"task": "Analyze Clean Energy Sector Q3"}
    p_coord_out = {"plan": "Fork parallel analysis: Market Analyst & Financial Analyst"}
    p_mkt_out = {
        "market_analysis": "EV adoption grew 28% in EU with Tesla, BYD, and Volkswagen leading.",
        "source_count": 4,
        "entity_count": 5,
    }
    p_fin_out = {
        "financial_analysis": "Data fetch error; failed to retrieve balance sheet metrics.",
        "source_count": 0,
        "entity_count": 0,
    }
    p_syn_out = {
        "final_report": "Summary based only on market adoption due to missing financial statements.",
        "source_count": 4,
        "entity_count": 5,
    }

    s2_1 = _make_step(
        r2_id, 1, "coordinator", p_coord_out, p_coord_in, p_coord_out, p_coord_out, ts=t2_time
    )
    s2_2 = _make_step(
        r2_id,
        2,
        "market_analyst",
        p_mkt_out,
        p_coord_out,
        p_mkt_out,
        p_mkt_out,
        parent_step="1",
        ts=t2_time + timedelta(seconds=1),
    )
    s2_3 = _make_step(
        r2_id,
        3,
        "financial_analyst",
        p_fin_out,
        p_coord_out,
        p_fin_out,
        p_fin_out,
        tool_calls=[{"name": "fetch_financials", "error": "API rate limit 429"}],
        parent_step="1",
        ts=t2_time + timedelta(seconds=1),
    )
    s2_4 = _make_step(
        r2_id,
        4,
        "synthesizer",
        p_syn_out,
        {**p_mkt_out, **p_fin_out},
        p_syn_out,
        p_syn_out,
        parent_step="2,3",
        ts=t2_time + timedelta(seconds=2),
    )

    trace_fork = RunTrace(
        run_id=r2_id,
        workflow="parallel_analyst_pipeline",
        status=StepStatus.SUCCESS,
        total_latency_ms=980.0,
        total_tokens=950,
        timestamp=t2_time,
        steps=[s2_1, s2_2, s2_3, s2_4],
        spans=steps_to_spans([s2_1, s2_2, s2_3, s2_4], r2_id),
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Entity Substitution Attack (P2 - hallucination_v2)
    # ─────────────────────────────────────────────────────────────────────────
    r3_id = "trace_v2_entity_substitution_swap"
    t3_time = base_time + timedelta(minutes=20)
    e1_in = {"topic": "cloud infrastructure investments"}
    e1_out = {
        "research_findings": "Microsoft, Amazon, and Google announced major cloud infrastructure investments.",
        "source_count": 3,
        "entity_count": 3,
    }
    e2_out = {
        "written_report": ("Pfizer, Moderna, and BioNTech announced major clinical trials."),
        "source_count": 3,
        "entity_count": 3,
    }
    e3_out = {
        "verified": True,
        "approved": True,
        "verification_result": "APPROVED: All 3 entities matched.",
    }

    s3_1 = _make_step(r3_id, 1, "researcher", e1_out, e1_in, e1_out, e1_out, ts=t3_time)
    s3_2 = _make_step(
        r3_id, 2, "writer", e2_out, e1_out, e2_out, e2_out, ts=t3_time + timedelta(seconds=1)
    )
    s3_3 = _make_step(
        r3_id, 3, "verifier", e3_out, e2_out, e3_out, e3_out, ts=t3_time + timedelta(seconds=2)
    )

    trace_sub = RunTrace(
        run_id=r3_id,
        workflow="research_report_pipeline",
        status=StepStatus.SUCCESS,
        total_latency_ms=880.0,
        total_tokens=820,
        timestamp=t3_time,
        steps=[s3_1, s3_2, s3_3],
        spans=steps_to_spans([s3_1, s3_2, s3_3], r3_id),
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Semantic Relation Contradiction (P2 - contradiction_v1)
    # ─────────────────────────────────────────────────────────────────────────
    r4_id = "trace_v2_semantic_contradiction"
    t4_time = base_time + timedelta(minutes=30)
    c1_in = {"trial": "NCT-0391"}
    c1_out = {
        "research_findings": "Clinical study NCT-0391 failed to meet its primary safety endpoints in adult patients.",
        "source_count": 2,
        "entity_count": 2,
    }
    c2_out = {
        "written_report": (
            "Clinical study NCT-0391 successfully met its primary safety endpoints in adult patients."
        ),
        "source_count": 2,
        "entity_count": 2,
    }
    c3_out = {
        "verified": False,
        "approved": False,
        "verification_result": "REJECTED: Finding directly contradicts upstream findings.",
    }

    s4_1 = _make_step(r4_id, 1, "researcher", c1_out, c1_in, c1_out, c1_out, ts=t4_time)
    s4_2 = _make_step(
        r4_id, 2, "writer", c2_out, c1_out, c2_out, c2_out, ts=t4_time + timedelta(seconds=1)
    )
    s4_3 = _make_step(
        r4_id, 3, "verifier", c3_out, c2_out, c3_out, c3_out, ts=t4_time + timedelta(seconds=2)
    )

    trace_contra = RunTrace(
        run_id=r4_id,
        workflow="research_report_pipeline",
        status=StepStatus.SUCCESS,
        total_latency_ms=850.0,
        total_tokens=790,
        timestamp=t4_time,
        steps=[s4_1, s4_2, s4_3],
        spans=steps_to_spans([s4_1, s4_2, s4_3], r4_id),
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Non-Converging Deadlock Loop (P2 - quality failure)
    # ─────────────────────────────────────────────────────────────────────────
    r5_id = "trace_v2_loop_deadlock_unresolved"
    t5_time = base_time + timedelta(minutes=40)
    d_steps: list[AgentStep] = []
    d_spans: list[TraceSpan] = []

    for it in range(3):
        res_dict = {
            "draft": f"Attempt {it + 1}: empty literature search.",
            "source_count": 0,
            "entity_count": 1,
        }
        rev_dict = {
            "approved": False,
            "feedback": f"REJECTED iteration {it}: insufficient sources.",
            "source_count": 0,
        }
        st1 = _make_step(
            r5_id,
            it * 2 + 1,
            "researcher",
            res_dict,
            {},
            res_dict,
            res_dict,
            ts=t5_time + timedelta(seconds=it * 2),
        )
        st2 = _make_step(
            r5_id,
            it * 2 + 2,
            "reviewer",
            rev_dict,
            res_dict,
            rev_dict,
            rev_dict,
            ts=t5_time + timedelta(seconds=it * 2 + 1),
        )
        d_steps.extend([st1, st2])
        d_spans.append(
            TraceSpan(
                span_id=f"span_d_{it}_1",
                trace_id=r5_id,
                name="researcher",
                span_type=SpanType.AGENT,
                iteration_index=it,
                output=json.dumps(res_dict),
            )
        )
        d_spans.append(
            TraceSpan(
                span_id=f"span_d_{it}_2",
                trace_id=r5_id,
                name="reviewer",
                span_type=SpanType.AGENT,
                iteration_index=it,
                output=json.dumps(rev_dict),
            )
        )

    trace_deadlock = RunTrace(
        run_id=r5_id,
        workflow="iterative_reflection_pipeline",
        status=StepStatus.FAILURE,
        total_latency_ms=1800.0,
        total_tokens=1600,
        timestamp=t5_time,
        steps=d_steps,
        spans=d_spans,
    )

    return [
        {
            "run_id": r1_id,
            "category": "loop_reflection",
            "description": "Cyclic self-correction loop resolving to PASS on iteration 2 (P5)",
            "expected_verdict": "PASS",
            "expected_priority": "P5",
            "primary_agent": None,
            "trace": trace_loop_pass,
        },
        {
            "run_id": r2_id,
            "category": "parallel_fork",
            "description": "Parallel analyst fork where financial analyst encounters tool crash (P2)",
            "expected_verdict": "FAIL",
            "expected_priority": "P2",
            "primary_agent": "financial_analyst",
            "trace": trace_fork,
        },
        {
            "run_id": r3_id,
            "category": "entity_substitution",
            "description": "Entity substitution swap (3 tech -> 3 pharma) caught by hallucination_v2 (P2)",
            "expected_verdict": "FAIL",
            "expected_priority": "P2",
            "primary_agent": "writer",
            "trace": trace_sub,
        },
        {
            "run_id": r4_id,
            "category": "semantic_contradiction",
            "description": "Semantic contradiction (failed -> met) caught by contradiction_v1 (P2)",
            "expected_verdict": "FAIL",
            "expected_priority": "P2",
            "primary_agent": "writer",
            "trace": trace_contra,
        },
        {
            "run_id": r5_id,
            "category": "loop_deadlock",
            "description": "Non-converging reflection loop caught by researcher_quality_v1 (P2)",
            "expected_verdict": "FAIL",
            "expected_priority": "P2",
            "primary_agent": "researcher",
            "trace": trace_deadlock,
        },
    ]


def generate_v2_traces(
    output_dir: Path | None = None,
    seed_db: bool = False,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """
    Generate the 5 canonical Phase B v2 traces, save them as JSON blobs,
    and produce an evaluation manifest.
    """
    out_dir = Path(output_dir or DEFAULT_V2_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = build_v2_trace_specs()
    trace_records: list[dict[str, Any]] = []

    for spec in specs:
        run_id = spec["run_id"]
        trace = spec["trace"]
        trace_file = out_dir / f"{run_id}.json"

        # Serialize RunTrace with spans and steps
        trace_json = trace.model_dump_json(indent=2)
        trace_file.write_text(trace_json, encoding="utf-8")

        trace_records.append(
            {
                "run_id": run_id,
                "category": spec["category"],
                "description": spec["description"],
                "file": str(trace_file.name),
                "steps_count": len(trace.steps),
                "spans_count": len(trace.spans),
                "expected_verdict": spec["expected_verdict"],
                "expected_priority": spec["expected_priority"],
                "primary_agent": spec["primary_agent"],
            }
        )

    manifest: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "total_traces": len(trace_records),
        "traces": trace_records,
    }

    manifest_file = out_dir / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Generated %d v2 traces into %s", len(trace_records), out_dir)

    if seed_db:
        from storage.db import DatabaseManager

        db = DatabaseManager(db_path=db_path) if db_path else DatabaseManager()
        db.initialize()
        for spec in specs:
            trace = spec["trace"]
            try:
                db.insert_run(
                    run_id=trace.run_id,
                    workflow=trace.workflow,
                    timestamp=trace.timestamp.isoformat(),
                    status=trace.status.value,
                    total_latency_ms=trace.total_latency_ms,
                    total_tokens=trace.total_tokens,
                    schema_version=trace.schema_version,
                    trace_path=str(out_dir / f"{trace.run_id}.json"),
                    trace_json=trace.model_dump_json(),
                    expected_output=trace.expected_output,
                )
                for s in trace.steps:
                    db.insert_step(
                        run_id=trace.run_id,
                        step=s.step,
                        agent=s.agent,
                        status=s.status.value,
                        latency_ms=s.latency_ms,
                        tokens_prompt=s.tokens.prompt,
                        tokens_completion=s.tokens.completion,
                        tokens_total=s.tokens.total,
                        diff_summary=s.prompt or "",
                        timestamp=s.timestamp.isoformat(),
                        schema_version=s.schema_version,
                        error=s.error,
                    )
            except Exception as e:
                logger.warning("Failed to seed run %s into database: %s", spec["run_id"], e)

    return manifest


if __name__ == "__main__":
    manifest = generate_v2_traces(seed_db=False)
    print(f"Generated {manifest['total_traces']} Phase B v2 traces.")
