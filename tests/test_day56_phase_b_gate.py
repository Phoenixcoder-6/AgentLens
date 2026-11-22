"""
tests/test_day56_phase_b_gate.py — Day 56: Phase B Verification Gate Tests
===========================================================================
Unit and integration tests certifying Phase B (Universal Core Verification &
Accuracy Gate, Days 52-56) in AgentLens v2.0.

Tests:
  1. Multi-agent v2 trace manifest and structure validation.
  2. Multi-agent reflection loop trace (iteration_index, align_spans, PASS/P5).
  3. Parallel DAG fork trace (divergent branch detection, financial_analyst crash).
  4. Entity substitution trace (EntityTracker 1-to-1 swap, hallucination_v2).
  5. Semantic relation contradiction trace (NLIEngine polar inversion, contradiction_v1).
  6. Deadlock unresolved loop trace (cyclic non-convergence, researcher_quality_v1).
  7. Automated Phase B accuracy evaluation runner integration (>= 90% accuracy).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analyzers.arbiter import Arbiter
from analyzers.detection.rule_engine import RuleEngine
from analyzers.semantic.entity_tracker import EntityTracker
from analyzers.semantic.nli_engine import get_nli_engine
from diff_engine import align_spans
from sample_data.generate_v2_traces import generate_v2_traces
from schema.models import RunTrace
from scripts.evaluate_phase_b_accuracy import run_phase_b_evaluation


@pytest.fixture(scope="module")
def v2_traces_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    temp_dir = tmp_path_factory.mktemp("v2_traces")
    generate_v2_traces(output_dir=temp_dir)
    return temp_dir


class TestDay56PhaseBGate:
    def test_v2_traces_manifest_and_generation(self, v2_traces_dir: Path) -> None:
        manifest_file = v2_traces_dir / "manifest.json"
        assert manifest_file.exists()

        data = json.loads(manifest_file.read_text(encoding="utf-8"))
        assert data["total_traces"] == 5
        assert len(data["traces"]) == 5

        categories = {t["category"] for t in data["traces"]}
        expected_categories = {
            "loop_reflection",
            "parallel_fork",
            "entity_substitution",
            "semantic_contradiction",
            "loop_deadlock",
        }
        assert categories == expected_categories

        for record in data["traces"]:
            trace_path = v2_traces_dir / record["file"]
            assert trace_path.exists()
            trace = RunTrace.model_validate_json(trace_path.read_text(encoding="utf-8"))
            assert trace.run_id == record["run_id"]
            assert len(trace.steps) == record["steps_count"]
            assert len(trace.spans) == record["spans_count"]

    def test_v2_loop_reflection_trace(self, v2_traces_dir: Path) -> None:
        trace_file = v2_traces_dir / "trace_v2_loop_reflection_pass.json"
        trace = RunTrace.model_validate_json(trace_file.read_text(encoding="utf-8"))

        assert len(trace.steps) == 4
        assert len(trace.spans) == 4

        # Verify iteration indices
        assert [s.iteration_index for s in trace.spans] == [0, 0, 1, 1]

        # Verify span alignment handles reflection loop
        alignment = align_spans(trace.spans, trace.spans)
        assert alignment.matched_count == 4
        assert alignment.get_loop_iterations() == {0, 1}

        # Analyze with RuleEngine and Arbiter -> PASS (P5)
        res = RuleEngine().analyze(trace)
        bundle = Arbiter().run(run_id=trace.run_id, evidence=res.evidence)
        assert bundle.priority_level.value == "P5"
        assert bundle.primary_agent is None

    def test_v2_parallel_fork_trace(self, v2_traces_dir: Path) -> None:
        trace_file = v2_traces_dir / "trace_v2_parallel_fork_divergence.json"
        trace = RunTrace.model_validate_json(trace_file.read_text(encoding="utf-8"))

        assert len(trace.steps) == 4
        # Verify DAG structure: coordinator(1) -> market(2), financial(3) -> synthesizer(4)
        assert trace.steps[1].parent_step == "1"
        assert trace.steps[2].parent_step == "1"
        assert trace.steps[3].parent_step == "2,3"

        # Analyze with RuleEngine and Arbiter -> tool crash on financial_analyst (P2)
        res = RuleEngine().analyze(trace)
        rules_fired = [e.rule_match.rule_id for e in res.evidence if e.rule_match]
        assert "tool_failure_v1" in rules_fired

        bundle = Arbiter().run(run_id=trace.run_id, evidence=res.evidence)
        assert bundle.priority_level.value == "P2"
        assert bundle.primary_agent == "financial_analyst"

    def test_v2_entity_substitution_trace(self, v2_traces_dir: Path) -> None:
        trace_file = v2_traces_dir / "trace_v2_entity_substitution_swap.json"
        trace = RunTrace.model_validate_json(trace_file.read_text(encoding="utf-8"))

        assert len(trace.steps) == 3
        s_res = trace.steps[0]
        s_wr = trace.steps[1]

        # EntityTracker set difference check
        delta = EntityTracker.compare_steps(s_res, s_wr)
        assert delta.substitution_detected is True
        assert len(delta.invented_entities) == 3
        assert len(delta.dropped_entities) == 3

        # RuleEngine hallucination_v2 firing check
        res = RuleEngine().analyze(trace)
        rules_fired = [e.rule_match.rule_id for e in res.evidence if e.rule_match]
        assert "hallucination_v2" in rules_fired

        bundle = Arbiter().run(run_id=trace.run_id, evidence=res.evidence)
        assert bundle.priority_level.value == "P2"
        assert bundle.primary_agent == "writer"

    def test_v2_semantic_contradiction_trace(self, v2_traces_dir: Path) -> None:
        trace_file = v2_traces_dir / "trace_v2_semantic_contradiction.json"
        trace = RunTrace.model_validate_json(trace_file.read_text(encoding="utf-8"))

        assert len(trace.steps) == 3
        s_res = trace.steps[0]
        s_wr = trace.steps[1]

        # NLIEngine relation inversion check
        nli_engine = get_nli_engine()
        report = nli_engine.evaluate_steps(s_res, s_wr)
        assert report.has_contradiction is True

        # RuleEngine contradiction_v1 firing check
        res = RuleEngine().analyze(trace)
        rules_fired = [e.rule_match.rule_id for e in res.evidence if e.rule_match]
        assert "contradiction_v1" in rules_fired

        bundle = Arbiter().run(run_id=trace.run_id, evidence=res.evidence)
        assert bundle.priority_level.value == "P2"
        assert bundle.primary_agent == "writer"

    def test_v2_loop_deadlock_unresolved_trace(self, v2_traces_dir: Path) -> None:
        trace_file = v2_traces_dir / "trace_v2_loop_deadlock_unresolved.json"
        trace = RunTrace.model_validate_json(trace_file.read_text(encoding="utf-8"))

        assert len(trace.steps) == 6
        # RuleEngine researcher_quality_v1 check
        res = RuleEngine().analyze(trace)
        rules_fired = [e.rule_match.rule_id for e in res.evidence if e.rule_match]
        assert "researcher_quality_v1" in rules_fired

        bundle = Arbiter().run(run_id=trace.run_id, evidence=res.evidence)
        assert bundle.priority_level.value == "P2"
        assert bundle.primary_agent == "researcher"

    def test_phase_b_accuracy_evaluator_runner(self, v2_traces_dir: Path, tmp_path: Path) -> None:
        eval_json = tmp_path / "results_phase_b.json"
        eval_md = tmp_path / "EVALUATION_REPORT.md"

        results = run_phase_b_evaluation(
            v2_dir=v2_traces_dir, output_json=eval_json, output_md=eval_md
        )
        assert eval_json.exists()
        assert eval_md.exists()

        metrics = results["metrics"]
        assert metrics["total_runs"] == 5 if "total_runs" in metrics else results["total_runs"] == 5
        assert metrics["exact_attribution_accuracy"] >= 0.90
        assert metrics["binary_detection_accuracy"] >= 0.90
        assert metrics["passed"] is True
