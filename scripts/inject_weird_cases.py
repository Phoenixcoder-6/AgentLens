"""
scripts/inject_weird_cases.py — Weird Case Generator
================================─────────────────────
Generates 2 "weird" test cases to demonstrate AgentLens diagnosing:
1. Verifier latency & token spike (P4 Statistical Outlier -> verifier primary agent)
2. Researcher empty handoff failure (P2 Information Loss -> researcher primary agent)
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

from analyzers.detection import StatisticalDetector
from dashboard.state import run_full_analysis
from schema.models import (
    AgentStep,
    HandoffState,
    RunTrace,
    StepStatus,
    TokenUsage,
)
from storage.db import DatabaseManager

db = DatabaseManager()
db.initialize()


def create_verifier_spike_run():
    """Case 1: Verifier suffers a 45-second latency spike (P4 Statistical Outlier)."""
    run_id = f"run_weird_verifier_{uuid.uuid4().hex[:4]}"
    now = datetime.now(UTC)

    steps = [
        AgentStep(
            step_id=1,
            agent="researcher",
            status=StepStatus.SUCCESS,
            latency_ms=1200.0,
            tokens=TokenUsage(prompt=300, completion=200, total=500),
            handoff=HandoffState(
                input_state={"topic": "Quantum Encryption Key Distribution"},
                output_state={
                    "research_findings": "Quantum Key Distribution (QKD) relies on photons. Key protocol: BB84 created by Bennett and Brassard in 1984. 3 sources cited."
                },
            ),
        ),
        AgentStep(
            step_id=2,
            agent="writer",
            status=StepStatus.SUCCESS,
            latency_ms=1800.0,
            tokens=TokenUsage(prompt=400, completion=400, total=800),
            handoff=HandoffState(
                input_state={
                    "research_findings": "Quantum Key Distribution (QKD) relies on photons. Key protocol: BB84 created by Bennett and Brassard in 1984. 3 sources cited."
                },
                output_state={
                    "written_report": "Quantum Key Distribution (QKD) uses photon polarization to establish secure keys. Designed by Bennett and Brassard in 1984 (BB84 protocol). Cited 3 sources."
                },
            ),
        ),
        AgentStep(
            step_id=3,
            agent="verifier",
            status=StepStatus.SUCCESS,
            latency_ms=45000.0,  # 45 SECONDS! HUGE ANOMALY!
            tokens=TokenUsage(prompt=8000, completion=4000, total=12000),  # 12K TOKENS!
            handoff=HandoffState(
                input_state={
                    "written_report": "Quantum Key Distribution (QKD) uses photon polarization..."
                },
                output_state={"verification_result": "APPROVED", "verified": True},
            ),
        ),
    ]

    trace = RunTrace(
        run_id=run_id,
        workflow="research_report_pipeline",
        timestamp=now,
        status=StepStatus.SUCCESS,
        steps=steps,
        total_latency_ms=48000.0,
        total_tokens=13300,
    )

    # Save to DB
    db.save_run(
        run_id=run_id,
        workflow="research_report_pipeline",
        timestamp=now.isoformat(),
        status="success",
    )
    for s in steps:
        db.save_step(
            run_id=run_id,
            step=s.step_id,
            agent=s.agent,
            status=s.status.value,
            latency_ms=s.latency_ms,
            tokens_prompt=s.tokens.prompt,
            tokens_completion=s.tokens.completion,
            tokens_total=s.tokens.total,
        )

    # Save trace_json
    trace_dict = trace.model_dump(mode="json")
    db.update_run_trace(run_id, json.dumps(trace_dict))
    return run_id


def create_researcher_failure_run():
    """Case 2: Researcher fails and returns 0 sources/0 findings."""
    run_id = f"run_weird_researcher_{uuid.uuid4().hex[:4]}"
    now = datetime.now(UTC)

    steps = [
        AgentStep(
            step_id=1,
            agent="researcher",
            status=StepStatus.FAILURE,
            error="API Connection Error: Search Engine Timed Out",
            latency_ms=500.0,
            tokens=TokenUsage(prompt=100, completion=0, total=100),
            handoff=HandoffState(
                input_state={"topic": "Autonomous Mars Rover Navigation"},
                output_state={"research_findings": "", "source_count": 0, "entity_count": 0},
            ),
        ),
        AgentStep(
            step_id=2,
            agent="writer",
            status=StepStatus.SUCCESS,
            latency_ms=2100.0,
            tokens=TokenUsage(prompt=300, completion=500, total=800),
            handoff=HandoffState(
                input_state={"research_findings": ""},
                output_state={
                    "written_report": "Mars rovers use cameras and lidar for autonomous navigation across Martian terrain. NASA Curiosity and Perseverance rovers use AutoNav algorithm."
                },
            ),
        ),
        AgentStep(
            step_id=3,
            agent="verifier",
            status=StepStatus.SUCCESS,
            latency_ms=1500.0,
            tokens=TokenUsage(prompt=400, completion=200, total=600),
            handoff=HandoffState(
                input_state={"written_report": "Mars rovers use cameras and lidar..."},
                output_state={"verification_result": "UNVERIFIED", "verified": False},
            ),
        ),
    ]

    trace = RunTrace(
        run_id=run_id,
        workflow="research_report_pipeline",
        timestamp=now,
        status=StepStatus.FAILURE,
        steps=steps,
        total_latency_ms=4100.0,
        total_tokens=1500,
    )

    db.save_run(
        run_id=run_id,
        workflow="research_report_pipeline",
        timestamp=now.isoformat(),
        status="failure",
    )
    for s in steps:
        db.save_step(
            run_id=run_id,
            step=s.step_id,
            agent=s.agent,
            status=s.status.value,
            latency_ms=s.latency_ms,
            tokens_prompt=s.tokens.prompt,
            tokens_completion=s.tokens.completion,
            tokens_total=s.tokens.total,
        )

    trace_dict = trace.model_dump(mode="json")
    db.update_run_trace(run_id, json.dumps(trace_dict))
    return run_id


def main():
    print("Generating Weird Test Cases...\n")

    run1 = create_verifier_spike_run()
    print(f"[WEIRD CASE 1] Created run: {run1}")
    detector = StatisticalDetector(db)
    report1 = detector.analyze_run(run1)
    print(f"  Detector Anomalies: {len(report1.anomalies)}")
    for a in report1.anomalies:
        print(f"  👉 [{a.agent.upper()}] {a.description}")

    print()
    run2 = create_researcher_failure_run()
    print(f"[WEIRD CASE 2] Created run: {run2}")
    st2 = run_full_analysis(run2)
    if st2.bundle:
        print(f"  👉 Primary Cause : {st2.bundle.primary_cause.value}")
        print(f"  👉 Primary Agent : {st2.bundle.primary_agent}")
        print(f"  👉 Priority      : {st2.bundle.priority_level.value}")


if __name__ == "__main__":
    main()
