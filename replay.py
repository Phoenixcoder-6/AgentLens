"""
replay.py — AgentLens Replay & CI Gating CLI (Day 37)
=====================================================
Re-runs or deterministically evaluates a captured multi-agent workflow trace
by ``run_id``, supports topic overrides for attribution-stability testing, and
returns semantic exit codes for CI/CD pipeline gating.

Usage:
    python replay.py <run_id>
    python replay.py <run_id> --dry-run
    python replay.py <run_id> --json
    python replay.py <run_id> --override-topic "Quantum error correction"
    python replay.py <run_id> --dry-run --json --override-topic "New topic"

Exit Codes:
    0 = PASS     (P5 — no failures or anomalies detected)
    1 = WARNING  (P3 / P4 — workflow violation, statistical outlier, or medium warning)
    2 = FAIL     (P1 / P2 — ground-truth mismatch or high/critical rule failure)
    3 = ERROR    (run_id not found, malformed trace, or runtime execution error)
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyzers.arbiter import Arbiter, evidence_from_information_loss  # noqa: E402
from analyzers.detection.consistency_validator import ConsistencyValidator  # noqa: E402
from analyzers.detection.ground_truth import GroundTruthValidator  # noqa: E402
from analyzers.detection.information_loss import InformationLossRule  # noqa: E402
from analyzers.detection.rule_engine import RuleEngine, _prestructured_evidence  # noqa: E402
from analyzers.detection.workflow_validator import WorkflowValidator  # noqa: E402
from config import config_loader  # noqa: E402
from normalizer.normalizer import Normalizer  # noqa: E402
from schema.models import (  # noqa: E402
    AgentStep,
    AnalysisBundle,
    EvidenceRecord,
    PriorityLevel,
    RuleSeverity,
    RunTrace,
    StepStatus,
)
from storage.db import DatabaseManager  # noqa: E402

EXIT_PASS = 0
EXIT_WARNING = 1
EXIT_FAIL = 2
EXIT_ERROR = 3


def _coerce_raw_trace(raw: dict[str, Any]) -> RunTrace:
    """Coerce a raw trace dict into a validated RunTrace model."""
    data = copy.deepcopy(raw)
    for step in data.get("steps", []):
        if step.get("parent_step") is not None:
            step["parent_step"] = str(step["parent_step"])
        if step.get("child_step") is not None:
            step["child_step"] = str(step["child_step"])
    return RunTrace.model_validate(data)


def load_trace_by_id(run_id: str, db_path: str | None = None) -> RunTrace | None:
    """Locate and load a RunTrace by run_id from SQLite DB or trace directories."""
    # 1. Check SQLite database
    try:
        resolved_db = db_path or config_loader.get("storage", "db_path", "data/agentlens.db")
        db_file = Path(resolved_db)
        if not db_file.is_absolute():
            db_file = ROOT / db_file
        if db_file.exists():
            db = DatabaseManager(db_path=str(db_file))
            db.initialize()
            row = db.get_run(run_id)
            if row and row.get("trace_json"):
                raw = json.loads(row["trace_json"])
                return _coerce_raw_trace(raw)
    except Exception:
        pass

    # 2. Check data/traces/<run_id>.json and sample_data/labeled_traces/<run_id>.json
    search_dirs = [
        ROOT / "data" / "traces",
        ROOT / "sample_data" / "labeled_traces",
    ]
    for folder in search_dirs:
        candidate = folder / f"{run_id}.json"
        if candidate.exists():
            raw = json.loads(candidate.read_text(encoding="utf-8"))
            return _coerce_raw_trace(raw)

    return None


def extract_topic_from_trace(trace: RunTrace) -> str:
    """Extract the original input topic from a RunTrace."""
    for step in trace.steps:
        if step.handoff:
            for state_dict in (
                step.handoff.input_state,
                step.handoff.output_state,
                step.handoff.filtered_state,
            ):
                if isinstance(state_dict, dict) and state_dict.get("topic"):
                    return str(state_dict["topic"])
        if step.input:
            raw_in = step.input.strip()
            if raw_in.startswith("{"):
                try:
                    parsed = json.loads(raw_in)
                    if isinstance(parsed, dict) and parsed.get("topic"):
                        return str(parsed["topic"])
                except Exception:
                    pass
    return trace.workflow or "unknown_topic"


def apply_topic_override(trace: RunTrace, new_topic: str) -> RunTrace:
    """Return a copy of the trace with the overridden topic applied to step inputs/handoffs."""
    cloned = trace.model_copy(deep=True)
    for step in cloned.steps:
        if step.input and step.input.strip().startswith("{"):
            try:
                data = json.loads(step.input)
                if isinstance(data, dict):
                    data["topic"] = new_topic
                    step.input = json.dumps(data)
            except Exception:
                pass
        if step.handoff:
            if isinstance(step.handoff.input_state, dict) and "topic" in step.handoff.input_state:
                step.handoff.input_state["topic"] = new_topic
            if isinstance(step.handoff.output_state, dict) and "topic" in step.handoff.output_state:
                step.handoff.output_state["topic"] = new_topic
    return cloned


def _build_trace_from_pipeline_state(
    run_id: str,
    workflow: str,
    state: dict[str, Any],
) -> RunTrace:
    """Construct a RunTrace from a completed PipelineState dict."""
    src_count = int(state.get("source_count", 0) or 0)
    ent_count = int(state.get("entity_count", 0) or 0)
    verified = bool(state.get("verified", True))
    ver_result = str(state.get("verification_result", "APPROVED"))

    res_out = json.dumps(
        {
            "source_count": src_count,
            "entity_count": ent_count,
            "research_findings": str(state.get("research_findings", "")),
        }
    )
    wr_out = json.dumps(
        {
            "source_count": src_count,
            "entity_count": ent_count,
            "written_report": str(state.get("written_report", "")),
        }
    )
    ver_out = json.dumps(
        {
            "source_count": src_count,
            "entity_count": ent_count,
            "verified": verified,
            "verification_result": ver_result,
        }
    )

    steps = [
        AgentStep(
            run_id=run_id,
            step=1,
            agent="researcher",
            input=json.dumps({"topic": state.get("topic", "")}),
            output=res_out,
            status=StepStatus.SUCCESS,
        ),
        AgentStep(
            run_id=run_id,
            step=2,
            agent="writer",
            input=json.dumps({"topic": state.get("topic", "")}),
            output=wr_out,
            status=StepStatus.SUCCESS,
        ),
        AgentStep(
            run_id=run_id,
            step=3,
            agent="verifier",
            input=json.dumps({"topic": state.get("topic", "")}),
            output=ver_out,
            status=StepStatus.SUCCESS,
        ),
    ]
    return RunTrace(run_id=run_id, workflow=workflow, steps=steps)


def analyze_trace(trace: RunTrace, *, dry_run: bool = False) -> AnalysisBundle:
    """Run the deterministic AgentLens analysis pipeline on a RunTrace."""
    normalizer = Normalizer()
    gt_validator = GroundTruthValidator()
    rule_engine = RuleEngine()
    info_loss_rule = InformationLossRule()
    workflow_validator = WorkflowValidator()
    consistency_validator = ConsistencyValidator()
    arbiter = Arbiter()

    normalizer.normalize_run(trace)
    evidence: list[EvidenceRecord] = []

    def _collect() -> None:
        gt_res = gt_validator.analyze(trace)
        if not gt_res.skipped:
            evidence.extend(gt_res.evidence)

        re_res = rule_engine.analyze(trace)
        if not re_res.skipped:
            evidence.extend(re_res.evidence)

        res_steps = [s for s in trace.steps if s.agent == "researcher"]
        wr_steps = [s for s in trace.steps if s.agent == "writer"]
        if res_steps and wr_steps:
            res_ev = _prestructured_evidence(res_steps[-1])
            wr_ev = _prestructured_evidence(wr_steps[-1])
            if res_ev is not None and wr_ev is not None:
                loss_res = info_loss_rule.evaluate(
                    run_id=trace.run_id,
                    researcher_evidence=res_ev,
                    writer_evidence=wr_ev,
                )
                loss_ev = evidence_from_information_loss(loss_res)
                if loss_ev is not None:
                    evidence.append(loss_ev)

        wf_res = workflow_validator.analyze(trace)
        if not wf_res.skipped:
            evidence.extend(wf_res.evidence)

        cv_res = consistency_validator.analyze(trace)
        if not cv_res.skipped:
            evidence.extend(cv_res.evidence)

    if dry_run:
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}, clear=False):
            _collect()
    else:
        _collect()

    return arbiter.run(run_id=trace.run_id, evidence=evidence)


def resolve_verdict_and_exit_code(bundle: AnalysisBundle) -> tuple[str, int]:
    """Map an AnalysisBundle priority/severity to (verdict_str, exit_code).

    Exit codes:
      0 = PASS     (P5)
      1 = WARNING  (P3, P4, or P2 where all rule matches are MEDIUM/LOW severity)
      2 = FAIL     (P1 or P2 with HIGH/CRITICAL severity)
    """
    pri = bundle.priority_level
    if pri == PriorityLevel.P5:
        return "PASS", EXIT_PASS

    if pri in (PriorityLevel.P3, PriorityLevel.P4):
        return "WARNING", EXIT_WARNING

    if pri == PriorityLevel.P2 and bundle.rule_matches:
        severities = {rm.severity for rm in bundle.rule_matches}
        if severities.issubset({RuleSeverity.LOW, RuleSeverity.MEDIUM}):
            return "WARNING", EXIT_WARNING

    return "FAIL", EXIT_FAIL


def execute_replay(
    run_id: str,
    *,
    dry_run: bool = False,
    override_topic: str | None = None,
    db_path: str | None = None,
) -> tuple[dict[str, Any], int]:
    """Core replay execution returning (result_payload, exit_code)."""
    try:
        original_trace = load_trace_by_id(run_id, db_path=db_path)
    except Exception as exc:
        return (
            {
                "run_id": run_id,
                "verdict": "ERROR",
                "exit_code": EXIT_ERROR,
                "error": f"Failed to load trace '{run_id}': {exc}",
            },
            EXIT_ERROR,
        )

    if original_trace is None:
        return (
            {
                "run_id": run_id,
                "verdict": "ERROR",
                "exit_code": EXIT_ERROR,
                "error": f"Run '{run_id}' not found in database or trace directories.",
            },
            EXIT_ERROR,
        )

    original_topic = extract_topic_from_trace(original_trace)
    effective_topic = override_topic if override_topic is not None else original_topic
    planned_agents = [s.agent for s in original_trace.steps] or [
        "researcher",
        "writer",
        "verifier",
    ]

    replayed_live = False
    target_trace = original_trace

    if override_topic is not None:
        target_trace = apply_topic_override(original_trace, override_topic)

    # Live execution when not in dry_run and GROQ_API_KEY is configured
    api_key = os.getenv("GROQ_API_KEY", "")
    if not dry_run and api_key and api_key != "gsk_your_key_here":
        try:
            from app.pipeline import run_pipeline

            state = run_pipeline(topic=effective_topic)
            target_trace = _build_trace_from_pipeline_state(
                run_id=f"{run_id}_replay",
                workflow=original_trace.workflow,
                state=dict(state),
            )
            replayed_live = True
        except Exception as exc:
            return (
                {
                    "run_id": run_id,
                    "verdict": "ERROR",
                    "exit_code": EXIT_ERROR,
                    "error": f"Live pipeline replay failed: {exc}",
                },
                EXIT_ERROR,
            )

    try:
        bundle = analyze_trace(target_trace, dry_run=dry_run)
        verdict, exit_code = resolve_verdict_and_exit_code(bundle)
        confidence = (
            max((e.confidence for e in bundle.evidence), default=1.0) if bundle.evidence else 1.0
        )
        rules_fired = [rm.rule_id for rm in bundle.rule_matches]

        payload: dict[str, Any] = {
            "run_id": run_id,
            "replay_run_id": target_trace.run_id,
            "workflow": original_trace.workflow,
            "dry_run": dry_run,
            "replayed_live": replayed_live,
            "original_topic": original_topic,
            "effective_topic": effective_topic,
            "topic_overridden": override_topic is not None,
            "planned_steps": planned_agents,
            "verdict": verdict,
            "exit_code": exit_code,
            "priority_level": bundle.priority_level.value,
            "primary_cause": bundle.primary_cause.value,
            "primary_agent": bundle.primary_agent,
            "confidence": round(float(confidence), 4),
            "grounded": bundle.grounded,
            "rules_fired": rules_fired,
            "summary": bundle.summary,
        }
        return payload, exit_code
    except Exception as exc:
        return (
            {
                "run_id": run_id,
                "verdict": "ERROR",
                "exit_code": EXIT_ERROR,
                "error": f"Replay analysis failed: {exc}",
            },
            EXIT_ERROR,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="replay.py",
        description="Replay an AgentLens run by run_id and output its deterministic verdict.",
    )
    parser.add_argument(
        "run_id",
        type=str,
        help="The run_id to replay (e.g. run_lbl_pass_01).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output verdict as machine-readable JSON (for CI scripting).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Show what the pipeline would do and evaluate deterministically without calling LLMs.",
    )
    parser.add_argument(
        "--override-topic",
        type=str,
        default=None,
        dest="override_topic",
        help="Replay the workflow with a different input topic to test attribution stability.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        dest="db_path",
        help="Optional override path to the SQLite database.",
    )
    return parser


def _print_human_output(payload: dict[str, Any]) -> None:
    if payload.get("verdict") == "ERROR":
        print("=" * 70)
        print(f"AgentLens Replay -- ERROR (exit_code={payload['exit_code']})")
        print("=" * 70)
        print(f"  Run ID : {payload['run_id']}")
        print(f"  Error  : {payload.get('error')}")
        print("=" * 70)
        return

    mode_label = "DRY-RUN (no LLM calls)" if payload["dry_run"] else "STANDARD REPLAY"
    print("=" * 70)
    print(f"AgentLens Replay -- {mode_label}")
    print("=" * 70)
    print(f"  Run ID          : {payload['run_id']}")
    print(f"  Workflow        : {payload['workflow']}")
    print(f"  Original Topic  : {payload['original_topic']}")
    if payload["topic_overridden"]:
        print(f"  Override Topic  : {payload['effective_topic']}")
    print(f"  Planned Steps   : {' -> '.join(payload['planned_steps'])}")
    print("-" * 70)
    print(f"  Verdict         : {payload['verdict']} (exit_code={payload['exit_code']})")
    print(f"  Priority Level  : {payload['priority_level']}")
    print(f"  Primary Cause   : {payload['primary_cause']}")
    print(f"  Primary Agent   : {payload['primary_agent'] or 'none'}")
    print(f"  Confidence      : {payload['confidence']:.2f}")
    print(f"  Grounded (P1)   : {payload['grounded']}")
    print(f"  Rules Fired     : {payload['rules_fired']}")
    print(f"  Summary         : {payload['summary']}")
    print("=" * 70)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else EXIT_ERROR
        return code if code == 0 else EXIT_ERROR

    payload, exit_code = execute_replay(
        args.run_id,
        dry_run=args.dry_run,
        override_topic=args.override_topic,
        db_path=args.db_path,
    )

    if args.json_output:
        print(json.dumps(payload, indent=2))
    else:
        _print_human_output(payload)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
