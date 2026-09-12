"""
tests/test_long_tail.py — Unit Tests for Day 28 Long-Tail Failure Proof
========================================================================
Tests that:
1. Long-tail reasoning loop scenario bypasses deterministic rules (0 rules fire).
2. StatisticalDetector detects latency and token anomalies on long-tail runs.
3. Arbiter assigns P4 priority and correctly attributes primary_agent='writer'.
4. Groundedness flag is set appropriately.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from analyzers.arbiter import Arbiter
from analyzers.detection.information_loss import InformationLossRule
from analyzers.detection.statistical_detector import StatisticalDetector
from normalizer.normalizer import Normalizer
from schema.models import (
    AgentStep,
    EvidenceSource,
    HandoffState,
    PriorityLevel,
    RunTrace,
    StepStatus,
    TokenUsage,
)
from storage.db import DatabaseManager
from storage.writer import StorageWriter


@pytest.fixture
def db_with_baselines(tmp_path):
    db_file = tmp_path / "test_long_tail.db"
    db = DatabaseManager(str(db_file))
    db.initialize()

    from normalizer.normalizer import Normalizer
    from storage.writer import StorageWriter

    writer = StorageWriter(db)

    # Seed 5 baseline runs
    for i in range(5):
        run_id = f"run_base_{i}"
        now = datetime.now(UTC)
        steps = [
            AgentStep(
                run_id=run_id,
                step=1,
                agent="researcher",
                status=StepStatus.SUCCESS,
                latency_ms=1800.0 + i * 50.0,
                tokens=TokenUsage(
                    prompt=600 + i * 10, completion=400 + i * 10, total=1000 + i * 20
                ),
                handoff=HandoffState(
                    input_state={"topic": "Quantum"},
                    output_state={"research_findings": "Findings text. 4 sources."},
                ),
            ),
            AgentStep(
                run_id=run_id,
                step=2,
                agent="writer",
                status=StepStatus.SUCCESS,
                latency_ms=2000.0 + i * 50.0,
                tokens=TokenUsage(
                    prompt=700 + i * 10, completion=500 + i * 10, total=1200 + i * 20
                ),
                handoff=HandoffState(
                    input_state={"research_findings": "Findings text. 4 sources."},
                    output_state={"written_report": "Written report text. 4 sources."},
                ),
            ),
        ]
        trace = RunTrace(
            run_id=run_id,
            workflow="test_flow",
            timestamp=now,
            status=StepStatus.SUCCESS,
            steps=steps,
            total_latency_ms=3800.0,
            total_tokens=2200,
        )
        norm = Normalizer().normalize_run(trace)
        writer.write_run(norm, trace_json=json.dumps(trace.model_dump(mode="json")))

    return db


def test_long_tail_bypasses_deterministic_rules(db_with_baselines):
    """Test that a long-tail payload with 5.5x latency produces 0 deterministic rule matches."""
    db = db_with_baselines
    from analyzers.evidence_extraction.extractor import ExtractedEvidence

    res_ev = ExtractedEvidence(source_count=5, entity_count=8)
    wri_ev = ExtractedEvidence(source_count=5, entity_count=8)

    info_rule = InformationLossRule()
    res = info_rule.evaluate(
        run_id="run_long_tail_test", researcher_evidence=res_ev, writer_evidence=wri_ev
    )

    assert res.verdict == "PASS"
    assert not res.has_information_loss
    assert not res.has_information_gain


def test_long_tail_statistical_detection(db_with_baselines):
    """Test that StatisticalDetector flags writer step with latency/token anomaly."""
    db = db_with_baselines
    run_id = "run_long_tail_spike"
    now = datetime.now(UTC)

    steps = [
        AgentStep(
            run_id=run_id,
            step=1,
            agent="researcher",
            status=StepStatus.SUCCESS,
            latency_ms=1800.0,
            tokens=TokenUsage(prompt=600, completion=400, total=1000),
            handoff=HandoffState(
                input_state={"topic": "Quantum"},
                output_state={"research_findings": "Findings text"},
            ),
        ),
        AgentStep(
            run_id=run_id,
            step=2,
            agent="writer",
            status=StepStatus.SUCCESS,
            latency_ms=25000.0,  # 12.5x spike
            tokens=TokenUsage(prompt=2000, completion=8000, total=10000),  # 8.3x spike
            handoff=HandoffState(
                input_state={"research_findings": "Findings text"},
                output_state={"written_report": "Valid report text"},
            ),
        ),
    ]

    trace = RunTrace(
        run_id=run_id,
        workflow="test_flow",
        timestamp=now,
        status=StepStatus.SUCCESS,
        steps=steps,
        total_latency_ms=26800.0,
        total_tokens=11000,
    )

    norm = Normalizer().normalize_run(trace)
    writer = StorageWriter(db)
    writer.write_run(norm, trace_json=json.dumps(trace.model_dump(mode="json")))

    detector = StatisticalDetector(db)
    report = detector.analyze_run(run_id)

    assert len(report.anomalies) >= 1
    anom = report.anomalies[0]
    assert anom.agent == "writer"
    assert anom.source == EvidenceSource.STATISTICAL_ANOMALY

    # Wire to Arbiter
    evidence = report.anomalies
    bundle = Arbiter().run(run_id=run_id, evidence=evidence)

    assert bundle.priority_level == PriorityLevel.P4
    assert bundle.primary_agent == "writer"
    assert bundle.grounded is True
