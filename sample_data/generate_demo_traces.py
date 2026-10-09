"""
sample_data/generate_demo_traces.py -- Day 44 Zero-Setup Sample Data Generator & Seeder
=======================================================================================
Generates a rich set of zero-setup demo traces covering all core AgentLens modes:
  1. `demo_normal_pass_01`               (normal:    PASS / P5, clean 3-agent run)
  2. `demo_grounded_p1_01`               (grounded:  FAIL / P1, ground-truth contradiction)
  3. `demo_heuristic_p2_hallucination`   (heuristic: FAIL / P2, writer entity/source inflation)
  4. `demo_heuristic_p3_workflow`        (heuristic: FAIL / P3, skipped verifier step)
  5. `demo_diff_pair_baseline_a`         (diff_pair: PASS / P5, baseline run A for Diff View)
  6. `demo_diff_pair_diverged_b`         (diff_pair: FAIL / P2, diverged run B for Diff View)

Writes JSON trace files and `manifest.json` to `sample_data/demo_traces/` and optionally
seeds the SQLite database (`runs`, `steps`, `analysis`, `rule_matches`) so the dashboard
can be explored immediately with zero setup or API keys.

Usage:
    python -m sample_data.generate_demo_traces
    agentlens-seed
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analyzers.arbiter import Arbiter  # noqa: E402
from analyzers.detection.consistency_validator import ConsistencyValidator  # noqa: E402
from analyzers.detection.ground_truth import GroundTruthValidator  # noqa: E402
from analyzers.detection.information_loss import InformationLossRule  # noqa: E402
from analyzers.detection.rule_engine import RuleEngine  # noqa: E402
from analyzers.detection.workflow_validator import WorkflowValidator  # noqa: E402
from dashboard.state import persist_rule_matches  # noqa: E402
from normalizer.normalizer import Normalizer  # noqa: E402
from schema.models import (  # noqa: E402
    SCHEMA_VERSION,
    AgentStep,
    EvidenceRecord,
    HandoffState,
    RunTrace,
    StepStatus,
    TokenUsage,
)
from storage.db import DatabaseManager  # noqa: E402

DEMO_TRACES_DIR = Path(__file__).resolve().parent / "demo_traces"
MANIFEST_PATH = DEMO_TRACES_DIR / "manifest.json"


def _step(
    run_id: str,
    step_num: int,
    agent: str,
    output_dict: dict[str, Any],
    input_state: dict[str, Any],
    filtered_state: dict[str, Any],
    output_state: dict[str, Any],
    latency_ms: float = 310.0,
    tokens_prompt: int = 180,
    tokens_completion: int = 140,
    tool_calls: list[dict[str, Any]] | None = None,
    ts: datetime | None = None,
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
        prompt=f"Execute {agent} step for run {run_id}",
        tool_calls=tool_calls or [],
        handoff=HandoffState(
            input_state=input_state,
            filtered_state=filtered_state,
            output_state=output_state,
        ),
        timestamp=timestamp,
        schema_version=SCHEMA_VERSION,
    )


def build_demo_specs() -> list[dict[str, Any]]:
    """Construct the 6 canonical Day 44 demo traces (normal, grounded, heuristic, diff pairs)."""
    base_time = datetime(2026, 10, 9, 12, 0, 0, tzinfo=UTC)

    # 1. Normal PASS (P5)
    run_normal_id = "demo_normal_pass_01"
    topic_normal = "History of the Apollo Space Program"
    s1_in = {"topic": topic_normal}
    s1_out_dict = {
        "research_findings": (
            "Apollo 11 landed on the Moon on July 20, 1969. Neil Armstrong, Buzz Aldrin, "
            "and Michael Collins crewed the mission launched by Saturn V from Kennedy Space Center."
        ),
        "source_count": 6,
        "entity_count": 9,
        "claims": ["Apollo 11 landed July 20, 1969", "Launched aboard Saturn V"],
        "references": ["NASA-SP-4205", "Apollo-11-Mission-Report"],
    }
    s1_out_state = {**s1_in, **s1_out_dict}
    s2_out_dict = {
        "written_report": (
            "The Apollo Space Program achieved its primary milestone on July 20, 1969 when "
            "Apollo 11 astronauts Neil Armstrong and Buzz Aldrin walked on the lunar surface, "
            "supported by Command Module Pilot Michael Collins and the Saturn V launch vehicle."
        ),
        "source_count": 6,
        "entity_count": 9,
        "claims": ["Apollo 11 landed July 20, 1969", "Launched aboard Saturn V"],
        "references": ["NASA-SP-4205", "Apollo-11-Mission-Report"],
    }
    s2_out_state = {**s1_out_state, **s2_out_dict}
    s3_out_dict = {
        "verified": True,
        "approved": True,
        "verification_result": "APPROVED: All 6 sources and 9 entities match upstream research.",
        "revision_notes": "",
        "source_count": 6,
        "entity_count": 9,
    }
    s3_out_state = {**s2_out_state, **s3_out_dict}

    trace_normal = RunTrace(
        run_id=run_normal_id,
        workflow="research_report_pipeline",
        status=StepStatus.SUCCESS,
        total_latency_ms=940.0,
        total_tokens=960,
        timestamp=base_time,
        steps=[
            _step(
                run_normal_id,
                1,
                "researcher",
                s1_out_dict,
                s1_in,
                s1_out_dict,
                s1_out_state,
                latency_ms=320.0,
                tool_calls=[
                    {"name": "nasa_archive_search", "output": "6 primary sources retrieved"}
                ],
                ts=base_time,
            ),
            _step(
                run_normal_id,
                2,
                "writer",
                s2_out_dict,
                s1_out_state,
                s2_out_dict,
                s2_out_state,
                latency_ms=380.0,
                ts=base_time + timedelta(seconds=1),
            ),
            _step(
                run_normal_id,
                3,
                "verifier",
                s3_out_dict,
                s2_out_state,
                s3_out_dict,
                s3_out_state,
                latency_ms=240.0,
                ts=base_time + timedelta(seconds=2),
            ),
        ],
    )

    # 2. Grounded P1 Failure (Ground-Truth Contradiction)
    run_grounded_id = "demo_grounded_p1_01"
    topic_grounded = "The History of the Eiffel Tower"
    expected_gt = json.dumps(
        {
            "year_completed": "1889",
            "architect": "Gustave Eiffel",
            "location": "Paris",
        }
    )
    g1_in = {"topic": topic_grounded}
    g1_out_dict = {
        "research_findings": (
            "Designed by Gustave Eiffel in Paris and completed in 1889 for the World's Fair."
        ),
        "source_count": 5,
        "entity_count": 7,
        "year_completed": "1889",
        "architect": "Gustave Eiffel",
        "location": "Paris",
    }
    g1_out_state = {**g1_in, **g1_out_dict}
    g2_out_dict = {
        "written_report": (
            "The Eiffel Tower was completed in 1925 in Berlin by architect Antoni Gaudi, "
            "fabricated with 18 unverified claims."
        ),
        "source_count": 9,
        "entity_count": 19,
        "year_completed": "1925",
        "architect": "Antoni Gaudi",
        "location": "Berlin",
    }
    g2_out_state = {**g1_out_state, **g2_out_dict}
    g3_out_dict = {
        "verified": False,
        "approved": False,
        "verification_result": "REJECTED: Factual contradiction against ground truth (1889/Paris).",
        "revision_notes": "Writer replaced 1889/Gustave Eiffel/Paris with 1925/Antoni Gaudi/Berlin.",
        "source_count": 9,
        "entity_count": 19,
    }
    g3_out_state = {**g2_out_state, **g3_out_dict}

    trace_grounded = RunTrace(
        run_id=run_grounded_id,
        workflow="research_report_pipeline",
        status=StepStatus.SUCCESS,
        total_latency_ms=1120.0,
        total_tokens=1080,
        expected_output=expected_gt,
        timestamp=base_time + timedelta(minutes=5),
        steps=[
            _step(
                run_grounded_id,
                1,
                "researcher",
                g1_out_dict,
                g1_in,
                g1_out_dict,
                g1_out_state,
                latency_ms=340.0,
                tool_calls=[{"name": "historical_lookup", "output": "Completed 1889 in Paris"}],
                ts=base_time + timedelta(minutes=5),
            ),
            _step(
                run_grounded_id,
                2,
                "writer",
                g2_out_dict,
                g1_out_state,
                g2_out_dict,
                g2_out_state,
                latency_ms=490.0,
                ts=base_time + timedelta(minutes=5, seconds=1),
            ),
            _step(
                run_grounded_id,
                3,
                "verifier",
                g3_out_dict,
                g2_out_state,
                g3_out_dict,
                g3_out_state,
                latency_ms=290.0,
                ts=base_time + timedelta(minutes=5, seconds=2),
            ),
        ],
    )

    # 3. Heuristic P2 Failure (Hallucination & Information Gain)
    run_heur_p2_id = "demo_heuristic_p2_hallucination"
    topic_heur_p2 = "CRISPR Gene Editing Clinical Applications"
    h1_in = {"topic": topic_heur_p2}
    h1_out_dict = {
        "research_findings": "Casgevy approved for sickle cell disease; 4 clinical sources and 5 entities documented.",
        "source_count": 4,
        "entity_count": 5,
    }
    h1_out_state = {**h1_in, **h1_out_dict}
    h2_out_dict = {
        "written_report": (
            "Writer fabricated 11 additional unverified pharmaceutical trials and 6 extra citations "
            "not present in upstream research findings."
        ),
        "source_count": 10,
        "entity_count": 16,
    }
    h2_out_state = {**h1_out_state, **h2_out_dict}
    h3_out_dict = {
        "verified": False,
        "approved": False,
        "verification_result": "REJECTED: Entity count inflated from 5 to 16 (+11 entities).",
        "revision_notes": "Remove unverified clinical trial entities.",
        "source_count": 10,
        "entity_count": 16,
    }
    h3_out_state = {**h2_out_state, **h3_out_dict}

    trace_heuristic_p2 = RunTrace(
        run_id=run_heur_p2_id,
        workflow="research_report_pipeline",
        status=StepStatus.SUCCESS,
        total_latency_ms=1050.0,
        total_tokens=1020,
        timestamp=base_time + timedelta(minutes=10),
        steps=[
            _step(
                run_heur_p2_id,
                1,
                "researcher",
                h1_out_dict,
                h1_in,
                h1_out_dict,
                h1_out_state,
                latency_ms=310.0,
                tool_calls=[{"name": "pubmed_search", "output": "4 clinical trial records"}],
                ts=base_time + timedelta(minutes=10),
            ),
            _step(
                run_heur_p2_id,
                2,
                "writer",
                h2_out_dict,
                h1_out_state,
                h2_out_dict,
                h2_out_state,
                latency_ms=460.0,
                ts=base_time + timedelta(minutes=10, seconds=1),
            ),
            _step(
                run_heur_p2_id,
                3,
                "verifier",
                h3_out_dict,
                h2_out_state,
                h3_out_dict,
                h3_out_state,
                latency_ms=280.0,
                ts=base_time + timedelta(minutes=10, seconds=2),
            ),
        ],
    )

    # 4. Heuristic P3 Failure (Skipped Verifier Step)
    run_heur_p3_id = "demo_heuristic_p3_workflow"
    topic_heur_p3 = "Global Semiconductor Supply Chain Resilience"
    w1_in = {"topic": topic_heur_p3}
    w1_out_dict = {
        "research_findings": "TSMC, ASML, and Samsung account for primary EUV lithography capacity.",
        "source_count": 5,
        "entity_count": 6,
    }
    w1_out_state = {**w1_in, **w1_out_dict}
    w2_out_dict = {
        "written_report": "Analysis of EUV lithography bottlenecks across TSMC, ASML, and Samsung.",
        "source_count": 5,
        "entity_count": 6,
    }
    w2_out_state = {**w1_out_state, **w2_out_dict}

    trace_heuristic_p3 = RunTrace(
        run_id=run_heur_p3_id,
        workflow="research_report_pipeline",
        status=StepStatus.SUCCESS,
        total_latency_ms=670.0,
        total_tokens=640,
        timestamp=base_time + timedelta(minutes=15),
        steps=[
            _step(
                run_heur_p3_id,
                1,
                "researcher",
                w1_out_dict,
                w1_in,
                w1_out_dict,
                w1_out_state,
                latency_ms=330.0,
                tool_calls=[{"name": "industry_db_query", "output": "5 supply chain reports"}],
                ts=base_time + timedelta(minutes=15),
            ),
            _step(
                run_heur_p3_id,
                2,
                "writer",
                w2_out_dict,
                w1_out_state,
                w2_out_dict,
                w2_out_state,
                latency_ms=340.0,
                ts=base_time + timedelta(minutes=15, seconds=1),
            ),
        ],
    )

    # 5 & 6. Diff Pair (Run A = Baseline vs. Run B = Diverged at Writer)
    run_diff_a_id = "demo_diff_pair_baseline_a"
    run_diff_b_id = "demo_diff_pair_diverged_b"
    topic_diff = "Quantum Computing Fault-Tolerant Error Correction"

    da1_in = {"topic": topic_diff}
    da1_out = {
        "research_findings": (
            "Surface codes, topological qubits, neutral atom arrays, and magic state distillation "
            "reduce logical error rates below threshold across 6 studies and 8 hardware entities."
        ),
        "source_count": 6,
        "entity_count": 8,
    }
    da1_state = {**da1_in, **da1_out}
    da2_out = {
        "written_report": (
            "Fault-tolerant quantum computing relies on surface code lattices, topological qubits, "
            "neutral atom arrays, and magic state distillation across all 8 documented platforms."
        ),
        "source_count": 6,
        "entity_count": 8,
    }
    da2_state = {**da1_state, **da2_out}
    da3_out = {
        "verified": True,
        "approved": True,
        "verification_result": "APPROVED: Complete coverage of all 6 sources and 8 entities.",
        "revision_notes": "",
        "source_count": 6,
        "entity_count": 8,
    }
    da3_state = {**da2_state, **da3_out}

    trace_diff_a = RunTrace(
        run_id=run_diff_a_id,
        workflow="research_report_pipeline",
        status=StepStatus.SUCCESS,
        total_latency_ms=890.0,
        total_tokens=910,
        timestamp=base_time + timedelta(minutes=20),
        steps=[
            _step(
                run_diff_a_id,
                1,
                "researcher",
                da1_out,
                da1_in,
                da1_out,
                da1_state,
                latency_ms=300.0,
                tool_calls=[
                    {"name": "arxiv_search", "output": "6 quantum error correction papers"}
                ],
                ts=base_time + timedelta(minutes=20),
            ),
            _step(
                run_diff_a_id,
                2,
                "writer",
                da2_out,
                da1_state,
                da2_out,
                da2_state,
                latency_ms=350.0,
                ts=base_time + timedelta(minutes=20, seconds=1),
            ),
            _step(
                run_diff_a_id,
                3,
                "verifier",
                da3_out,
                da2_state,
                da3_out,
                da3_state,
                latency_ms=240.0,
                ts=base_time + timedelta(minutes=20, seconds=2),
            ),
        ],
    )

    # Run B has identical researcher output, but writer drops 5 entities & 4 sources and diverges in text
    db2_out = {
        "written_report": (
            "Short summary of classical bit-flip parity checks only; omitted surface codes, "
            "magic state distillation, and neutral atom platforms."
        ),
        "source_count": 2,
        "entity_count": 3,
    }
    db2_state = {**da1_state, **db2_out}
    db3_out = {
        "verified": False,
        "approved": False,
        "verification_result": "REJECTED: Writer dropped 4 sources and 5 entities during synthesis.",
        "revision_notes": "Restore missing surface code and magic state distillation sections.",
        "source_count": 2,
        "entity_count": 3,
    }
    db3_state = {**db2_state, **db3_out}

    trace_diff_b = RunTrace(
        run_id=run_diff_b_id,
        workflow="research_report_pipeline",
        status=StepStatus.SUCCESS,
        total_latency_ms=1340.0,
        total_tokens=780,
        timestamp=base_time + timedelta(minutes=25),
        steps=[
            _step(
                run_diff_b_id,
                1,
                "researcher",
                da1_out,
                da1_in,
                da1_out,
                da1_state,
                latency_ms=305.0,
                tool_calls=[
                    {"name": "arxiv_search", "output": "6 quantum error correction papers"}
                ],
                ts=base_time + timedelta(minutes=25),
            ),
            _step(
                run_diff_b_id,
                2,
                "writer",
                db2_out,
                da1_state,
                db2_out,
                db2_state,
                latency_ms=760.0,
                ts=base_time + timedelta(minutes=25, seconds=1),
            ),
            _step(
                run_diff_b_id,
                3,
                "verifier",
                db3_out,
                db2_state,
                db3_out,
                db3_state,
                latency_ms=275.0,
                ts=base_time + timedelta(minutes=25, seconds=2),
            ),
        ],
    )

    return [
        {
            "run_id": run_normal_id,
            "category": "normal",
            "description": "Clean 3-agent run resolving to PASS (P5)",
            "expected_output": None,
            "trace": trace_normal,
        },
        {
            "run_id": run_grounded_id,
            "category": "grounded",
            "description": "Ground-truth factual contradiction resolving to FAIL (P1, grounded=True)",
            "expected_output": json.loads(expected_gt),
            "trace": trace_grounded,
        },
        {
            "run_id": run_heur_p2_id,
            "category": "heuristic",
            "description": "Heuristic reasoning failure (hallucination_v1 + information_loss_v1, P2)",
            "expected_output": None,
            "trace": trace_heuristic_p2,
        },
        {
            "run_id": run_heur_p3_id,
            "category": "heuristic",
            "description": "Heuristic workflow failure (skipped_step_v1 on verifier, P3)",
            "expected_output": None,
            "trace": trace_heuristic_p3,
        },
        {
            "run_id": run_diff_a_id,
            "category": "diff_pair",
            "pair_id": "diff_pair_01",
            "pair_role": "baseline_run_a",
            "partner_run_id": run_diff_b_id,
            "description": "Baseline run A for side-by-side Diff View comparison",
            "expected_output": None,
            "trace": trace_diff_a,
        },
        {
            "run_id": run_diff_b_id,
            "category": "diff_pair",
            "pair_id": "diff_pair_01",
            "pair_role": "diverged_run_b",
            "partner_run_id": run_diff_a_id,
            "description": "Diverged run B (diverges at writer with severe information loss) for Diff View",
            "expected_output": None,
            "trace": trace_diff_b,
        },
    ]


def generate_demo_traces(
    output_dir: Path | None = None,
    seed_db: bool = True,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Write all 6 Day 44 demo traces + `manifest.json` to `sample_data/demo_traces/`
    and optionally seed them (with full analysis verdicts) into SQLite.
    """
    target_dir = output_dir or DEMO_TRACES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    specs = build_demo_specs()
    normalizer = Normalizer()
    gt_val = GroundTruthValidator()
    rule_engine = RuleEngine()
    info_loss_rule = InformationLossRule()
    workflow_val = WorkflowValidator()
    consistency_val = ConsistencyValidator()
    arbiter = Arbiter()

    db: DatabaseManager | None = None
    if seed_db:
        db = DatabaseManager(db_path=Path(db_path)) if db_path else DatabaseManager()
        db.initialize()

    manifest_entries: list[dict[str, Any]] = []

    for item in specs:
        trace: RunTrace = item["trace"]

        # Run deterministic analyzer suite on the demo trace
        normalizer.normalize_run(trace)
        all_evidence: list[EvidenceRecord] = []

        gt_res = gt_val.analyze(trace)
        if not gt_res.skipped:
            all_evidence.extend(gt_res.evidence)

        re_res = rule_engine.analyze(trace)
        if not re_res.skipped:
            all_evidence.extend(re_res.evidence)

        il_res = info_loss_rule.analyze(trace)
        if not il_res.skipped:
            all_evidence.extend(il_res.evidence)

        wf_res = workflow_val.analyze(trace)
        if not wf_res.skipped:
            all_evidence.extend(wf_res.evidence)

        cv_res = consistency_val.analyze(trace)
        if not cv_res.skipped:
            all_evidence.extend(cv_res.evidence)

        bundle = arbiter.run(run_id=trace.run_id, evidence=all_evidence)
        verdict = "PASS" if bundle.priority_level.value == "P5" else "FAIL"
        confidence = max((e.confidence for e in all_evidence), default=1.0) if all_evidence else 1.0
        rules_fired = [m.rule_id for m in bundle.rule_matches]

        # Write trace JSON file
        trace_file = target_dir / f"{trace.run_id}.json"
        payload = json.loads(trace.model_dump_json())
        payload["demo_metadata"] = {
            "category": item["category"],
            "description": item["description"],
            "pair_id": item.get("pair_id"),
            "pair_role": item.get("pair_role"),
            "partner_run_id": item.get("partner_run_id"),
            "verdict": verdict,
            "priority_level": bundle.priority_level.value,
            "primary_cause": bundle.primary_cause.value,
            "primary_agent": bundle.primary_agent,
            "grounded": bundle.grounded,
            "rules_fired": rules_fired,
        }
        trace_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        # Seed into SQLite database if requested
        if db is not None:
            db.insert_run(
                run_id=trace.run_id,
                workflow=trace.workflow,
                timestamp=trace.timestamp.isoformat(),
                status=trace.status.value,
                total_latency_ms=trace.total_latency_ms,
                total_tokens=trace.total_tokens,
                schema_version=trace.schema_version,
                trace_path=str(trace_file),
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
            db.insert_analysis(
                run_id=trace.run_id,
                analyzer="arbiter",
                timestamp=datetime.now(UTC).isoformat(),
                schema_version=SCHEMA_VERSION,
                category=bundle.primary_cause.value,
                verdict=verdict,
                confidence=confidence,
                details=json.loads(bundle.model_dump_json()),
            )
            persist_rule_matches(trace.run_id, bundle, db=db)

        manifest_entries.append(
            {
                "run_id": trace.run_id,
                "file": f"{trace.run_id}.json",
                "category": item["category"],
                "description": item["description"],
                "pair_id": item.get("pair_id"),
                "pair_role": item.get("pair_role"),
                "partner_run_id": item.get("partner_run_id"),
                "verdict": verdict,
                "priority_level": bundle.priority_level.value,
                "primary_cause": bundle.primary_cause.value,
                "primary_agent": bundle.primary_agent,
                "grounded": bundle.grounded,
                "rules_fired": rules_fired,
            }
        )

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "total_traces": len(manifest_entries),
        "categories": {
            "normal": [e["run_id"] for e in manifest_entries if e["category"] == "normal"],
            "grounded": [e["run_id"] for e in manifest_entries if e["category"] == "grounded"],
            "heuristic": [e["run_id"] for e in manifest_entries if e["category"] == "heuristic"],
            "diff_pair": [e["run_id"] for e in manifest_entries if e["category"] == "diff_pair"],
        },
        "diff_pairs": [
            {
                "pair_id": "diff_pair_01",
                "run_a": "demo_diff_pair_baseline_a",
                "run_b": "demo_diff_pair_diverged_b",
                "expected_first_divergence": "writer",
            }
        ],
        "traces": manifest_entries,
    }
    manifest_file = target_dir / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Day 44 zero-setup demo traces and seed SQLite database."
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Write JSON files to sample_data/demo_traces/ without seeding SQLite.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Optional custom SQLite DB path to seed.",
    )
    args = parser.parse_args()

    manifest = generate_demo_traces(seed_db=not args.no_db, db_path=args.db_path)
    print("=" * 72)
    print("AgentLens -- Day 44 Zero-Setup Sample Data Generator")
    print("=" * 72)
    for entry in manifest["traces"]:
        print(
            f"  [{entry['priority_level']}] {entry['run_id']:<32} | "
            f"cat={entry['category']:<9} | verdict={entry['verdict']:<4} | "
            f"grounded={str(entry['grounded']):<5} | agent={str(entry['primary_agent'])}"
        )
    print("-" * 72)
    print(f"  Generated {manifest['total_traces']} demo traces -> {DEMO_TRACES_DIR}")
    if not args.no_db:
        print("  Seeded SQLite database with runs, steps, analysis bundles, and rule matches.")
    print("=" * 72)


if __name__ == "__main__":
    main()
