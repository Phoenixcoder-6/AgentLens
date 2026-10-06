"""
scripts/run_day34_validation.py — Day 34 Labeled Set Validation Runner
=======================================================================
Runs all 20 frozen labeled traces from Day 15 (sample_data/labels.json)
through the complete AgentLens analysis pipeline:
  Normalizer -> GroundTruthValidator -> RuleEngine -> InformationLossRule
  -> WorkflowValidator -> ConsistencyValidator -> StatisticalDetector -> Arbiter

Records for each run:
  verdict, priority_level, primary_cause, primary_agent, confidence,
  grounded, rules_fired, and summary.

Outputs:
  validation/results_day34.json

Run from project root:
    conda run -n agentlens python scripts/run_day34_validation.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analyzers.arbiter import Arbiter, evidence_from_information_loss  # noqa: E402
from analyzers.detection.consistency_validator import ConsistencyValidator  # noqa: E402
from analyzers.detection.ground_truth import GroundTruthValidator  # noqa: E402
from analyzers.detection.information_loss import InformationLossRule  # noqa: E402
from analyzers.detection.rule_engine import RuleEngine, _prestructured_evidence  # noqa: E402
from analyzers.detection.statistical_detector import StatisticalDetector  # noqa: E402
from analyzers.detection.workflow_validator import WorkflowValidator  # noqa: E402
from dashboard.state import persist_rule_matches  # noqa: E402
from normalizer.normalizer import Normalizer  # noqa: E402
from schema.models import (  # noqa: E402
    SCHEMA_VERSION,
    AgentStep,
    EvidenceRecord,
    PriorityLevel,
    RunTrace,
)
from storage.db import DatabaseManager  # noqa: E402

LABELS_PATH = _ROOT / "sample_data" / "labels.json"
TRACES_DIR = _ROOT / "sample_data" / "labeled_traces"
OUTPUT_PATH = _ROOT / "validation" / "results_day34.json"


def load_labeled_trace(run_id: str) -> tuple[RunTrace, dict[str, Any]]:
    """Load a labeled trace JSON file into a validated RunTrace object."""
    path = TRACES_DIR / f"{run_id}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))

    steps: list[AgentStep] = []
    for s in raw.get("steps", []):
        step_dict = dict(s)
        if step_dict.get("parent_step") is not None:
            step_dict["parent_step"] = str(step_dict["parent_step"])
        if step_dict.get("child_step") is not None:
            step_dict["child_step"] = str(step_dict["child_step"])

        # Normalize tool_calls so both 'output' and 'result' keys are preserved
        norm_tools = []
        for tc in step_dict.get("tool_calls", []):
            norm_tools.append(
                {
                    "name": tc.get("name", ""),
                    "args": tc.get("args", {}),
                    "output": tc.get("output") or tc.get("result") or "",
                    "error": tc.get("error") or "",
                }
            )
        step_dict["tool_calls"] = norm_tools
        steps.append(AgentStep(**step_dict))

    trace = RunTrace(
        schema_version=raw.get("schema_version", SCHEMA_VERSION),
        run_id=raw.get("run_id", run_id),
        workflow=raw.get("workflow", "research_report_pipeline"),
        steps=steps,
        total_latency_ms=float(raw.get("total_latency_ms", 0.0)),
        total_tokens=int(raw.get("total_tokens", 0)),
        expected_output=raw.get("expected_output"),
    )
    return trace, raw


def _seed_trace_into_db(db: DatabaseManager, trace: RunTrace, raw: dict[str, Any]) -> None:
    """Insert a labeled run and its steps into a DatabaseManager instance."""
    db.insert_run(
        run_id=trace.run_id,
        workflow=trace.workflow,
        timestamp=raw.get("timestamp", datetime.now(UTC).isoformat()),
        status=str(trace.status.value),
        total_latency_ms=trace.total_latency_ms,
        total_tokens=trace.total_tokens,
        schema_version=trace.schema_version,
        trace_path=str(TRACES_DIR / f"{trace.run_id}.json"),
        trace_json=json.dumps(raw),
        expected_output=trace.expected_output,
    )
    for st in trace.steps:
        db.insert_step(
            run_id=trace.run_id,
            step=st.step,
            agent=st.agent,
            status=str(st.status.value),
            latency_ms=st.latency_ms,
            tokens_prompt=st.tokens.prompt,
            tokens_completion=st.tokens.completion,
            tokens_total=st.tokens.total,
            diff_summary="",
            error=st.error,
            timestamp=st.timestamp.isoformat(),
            schema_version=st.schema_version,
        )


def run_validation() -> dict[str, Any]:
    """Execute all 20 labeled traces through the full pipeline and write results_day34.json."""
    labels_doc = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    label_runs: list[dict[str, Any]] = labels_doc["runs"]

    # Isolated DB for StatisticalDetector so baseline reflects the 20 labeled runs cleanly
    tmp_dir = tempfile.mkdtemp()
    val_db = DatabaseManager(str(Path(tmp_dir) / "val_day34.db"))
    val_db.initialize()

    # Main project DB so rule_matches table is populated for /rules
    main_db = DatabaseManager()
    main_db.initialize()

    loaded: list[tuple[dict[str, Any], RunTrace]] = []
    for item in label_runs:
        run_id = item["run_id"]
        trace, raw = load_labeled_trace(run_id)
        _seed_trace_into_db(val_db, trace, raw)
        _seed_trace_into_db(main_db, trace, raw)
        loaded.append((item, trace))

    normalizer = Normalizer()
    gt_eval = GroundTruthValidator()
    rule_engine = RuleEngine()
    info_loss_rule = InformationLossRule()
    workflow_val = WorkflowValidator()
    consistency_val = ConsistencyValidator()
    stat_detector = StatisticalDetector(val_db)
    arbiter = Arbiter()

    results_list: list[dict[str, Any]] = []
    priority_counts: dict[str, int] = {"P1": 0, "P2": 0, "P3": 0, "P4": 0, "P5": 0}

    print("\n" + "=" * 72)
    print("AgentLens -- Day 34: Running 20 Labeled Traces Through Full Pipeline")
    print("=" * 72)

    for item, trace in loaded:
        run_id = trace.run_id
        # 1. Normalize
        _ = normalizer.normalize_run(trace)

        # 2. Run all deterministic & statistical detectors
        evidence: list[EvidenceRecord] = []

        gt_res = gt_eval.analyze(trace)
        if not gt_res.skipped:
            evidence.extend(gt_res.evidence)

        re_res = rule_engine.analyze(trace)
        if not re_res.skipped:
            evidence.extend(re_res.evidence)

        # Information loss check (researcher -> writer)
        res_steps = [s for s in trace.steps if s.agent == "researcher"]
        wr_steps = [s for s in trace.steps if s.agent == "writer"]
        res_ev = _prestructured_evidence(res_steps[-1]) if res_steps else None
        wr_ev = _prestructured_evidence(wr_steps[-1]) if wr_steps else None
        if res_ev is not None and wr_ev is not None:
            loss_res = info_loss_rule.evaluate(
                run_id=run_id, researcher_evidence=res_ev, writer_evidence=wr_ev
            )
            loss_ev = evidence_from_information_loss(loss_res)
            if loss_ev is not None:
                evidence.append(loss_ev)

        wv_res = workflow_val.analyze(trace)
        if not wv_res.skipped:
            evidence.extend(wv_res.evidence)

        cv_res = consistency_val.analyze(trace)
        if not cv_res.skipped:
            evidence.extend(cv_res.evidence)

        stat_rep = stat_detector.analyze_run(run_id)
        if stat_rep.anomalies:
            evidence.extend(stat_rep.anomalies)

        # 3. Resolve verdict via Arbiter
        bundle = arbiter.run(run_id=run_id, evidence=evidence)
        persist_rule_matches(run_id, bundle, db=main_db)

        pri = bundle.priority_level.value
        priority_counts[pri] = priority_counts.get(pri, 0) + 1

        verdict = "PASS" if bundle.priority_level == PriorityLevel.P5 else "FAIL"
        confidence = max((e.confidence for e in evidence), default=1.0) if evidence else 1.0
        rules_fired = [m.rule_id for m in bundle.rule_matches]

        rec = {
            "run_id": run_id,
            "label_id": item["label_id"],
            "topic": item["topic"],
            "label_category": item["category"],
            "expected_verdict": item["expected_verdict"],
            "expected_primary_agent": item["expected_primary_agent"],
            "verdict": verdict,
            "priority_level": pri,
            "primary_cause": bundle.primary_cause.value,
            "primary_agent": bundle.primary_agent,
            "confidence": round(float(confidence), 4),
            "grounded": bool(bundle.grounded),
            "rules_fired": rules_fired,
            "summary": bundle.summary,
        }
        results_list.append(rec)

        agent_str = bundle.primary_agent or "none"
        print(
            f"  [{pri}] {run_id:<24} | verdict={verdict:<4} "
            f"| cause={bundle.primary_cause.value:<12} | agent={agent_str:<10} "
            f"| rules={rules_fired}"
        )

    output_doc = {
        "generated_at": datetime.now(UTC).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "total_runs": len(results_list),
        "priority_counts": priority_counts,
        "pass_count": sum(1 for r in results_list if r["verdict"] == "PASS"),
        "fail_count": sum(1 for r in results_list if r["verdict"] == "FAIL"),
        "runs": results_list,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output_doc, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"  Saved {len(results_list)}/20 validation results -> {OUTPUT_PATH}")
    print(f"  Priority breakdown: {priority_counts}")
    print("=" * 72 + "\n")
    return output_doc


if __name__ == "__main__":
    run_validation()
