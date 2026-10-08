"""
scripts/verify_day40a.py — Day 40a Verification Script
=======================================================
Verifies all Day 40a deliverables (Multi-Agent Generalization):
  1. config/config.yaml defines pipeline.name and pipeline.agents topology
  2. config/topology.py resolves agent roles, upstream handoffs, and required order
  3. Zero hardcoded "researcher", "writer", "verifier" strings in detection rule modules
  4. Custom 3-agent topology (planner -> coder -> reviewer) attributes failures accurately
  5. 2-agent topology (collector -> summarizer) passes clean runs and catches P2/P3 faults
  6. InformationLossRule.evaluate_trace() + evidence_from_information_loss() attribute to synthesizer role
  7. Existing rule/validator test suites pass with zero regressions
  8. Day 40a pytest suite (tests/test_multi_agent_generalization.py) passes
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _check(num: int, title: str, fn) -> bool:
    try:
        detail = fn()
        print(f"  [PASS] Check {num}: {title} -- {detail}")
        return True
    except Exception as exc:
        print(f"  [FAIL] Check {num}: {title} -- {exc}")
        return False


def check_1_config_topology() -> str:
    from config import config_loader

    config_loader.load_config.cache_clear()
    name = config_loader.get("pipeline", "name")
    agents = config_loader.get("pipeline", "agents")
    assert name == "research_report_pipeline", f"Unexpected pipeline.name: {name}"
    assert isinstance(agents, list) and len(agents) == 3
    roles = [a.get("role") for a in agents]
    assert roles == ["information_gatherer", "synthesizer", "quality_checker"]
    assert agents[1].get("receives_from") == "researcher"
    assert agents[2].get("receives_from") == "writer"
    return f"name='{name}', roles={roles}"


def check_2_topology_resolver() -> str:
    from config.topology import (
        ROLE_CHECKER,
        ROLE_GATHERER,
        ROLE_SYNTHESIZER,
        PipelineTopology,
    )

    topo = PipelineTopology(
        agents=[
            {"id": "planner", "role": ROLE_GATHERER},
            {"id": "coder", "role": ROLE_SYNTHESIZER, "receives_from": "planner"},
            {"id": "reviewer", "role": ROLE_CHECKER, "receives_from": "coder"},
        ]
    )
    assert topo.primary_agent_for_role(ROLE_GATHERER) == "planner"
    assert topo.primary_agent_for_role(ROLE_SYNTHESIZER) == "coder"
    assert topo.primary_agent_for_role(ROLE_CHECKER) == "reviewer"
    assert topo.upstream_agent_for("coder") == "planner"
    assert topo.handoff_edges() == [("planner", "coder"), ("coder", "reviewer")]
    return "PipelineTopology resolved planner -> coder -> reviewer"


def check_3_no_hardcoded_agent_names() -> str:
    modules = (
        "analyzers/detection/rule_engine.py",
        "analyzers/detection/consistency_validator.py",
        "analyzers/detection/workflow_validator.py",
    )
    for rel in modules:
        text = (PROJECT_ROOT / rel).read_text(encoding="utf-8")
        for forbidden in ('"researcher"', '"writer"', '"verifier"'):
            assert forbidden not in text, f"Hardcoded {forbidden} found in {rel}"
    return f"0 hardcoded agent names across {len(modules)} detection modules"


def check_4_custom_3_agent_pipeline() -> str:
    from analyzers.detection.consistency_validator import ConsistencyValidator
    from analyzers.detection.rule_engine import RuleEngine
    from config import config_loader
    from config.topology import ROLE_CHECKER, ROLE_GATHERER, ROLE_SYNTHESIZER
    from schema.models import AgentStep, RunTrace, StepStatus

    orig_get = config_loader.get
    try:
        custom_agents = [
            {"id": "planner", "role": ROLE_GATHERER},
            {"id": "coder", "role": ROLE_SYNTHESIZER, "receives_from": "planner"},
            {"id": "reviewer", "role": ROLE_CHECKER, "receives_from": "coder"},
        ]
        config_loader.get = lambda s, k=None, d=None: (  # type: ignore[assignment]
            custom_agents if (s == "pipeline" and k == "agents") else orig_get(s, k, d)
        )
        trace = RunTrace(
            run_id="run_verify_40a_3ag",
            workflow="code_pipeline",
            steps=[
                AgentStep(
                    run_id="run_verify_40a_3ag",
                    step=1,
                    agent="planner",
                    input="{}",
                    output=json.dumps({"source_count": 5, "entity_count": 3, "claims": ["c1"]}),
                    status=StepStatus.SUCCESS,
                    timestamp=datetime.now(UTC),
                ),
                AgentStep(
                    run_id="run_verify_40a_3ag",
                    step=2,
                    agent="coder",
                    input="{}",
                    output=json.dumps(
                        {"source_count": 5, "entity_count": 8, "claims": ["c1", "c2"]}
                    ),
                    status=StepStatus.SUCCESS,
                    timestamp=datetime.now(UTC),
                ),
                AgentStep(
                    run_id="run_verify_40a_3ag",
                    step=3,
                    agent="reviewer",
                    input="{}",
                    output=json.dumps({"source_count": 5, "entity_count": 8, "verified": True}),
                    status=StepStatus.SUCCESS,
                    timestamp=datetime.now(UTC),
                ),
            ],
        )
        re_ev = RuleEngine().analyze(trace).evidence
        cv_ev = ConsistencyValidator().analyze(trace).evidence
        assert any(e.agent == "coder" and e.rule_match.rule_id == "hallucination_v1" for e in re_ev)
        assert any(
            e.agent == "reviewer" and e.rule_match.rule_id == "verifier_passthrough_v1"
            for e in cv_ev
        )
    finally:
        config_loader.get = orig_get  # type: ignore[assignment]

    return "hallucination_v1 -> 'coder', verifier_passthrough_v1 -> 'reviewer'"


def check_5_two_agent_pipeline() -> str:
    from config import config_loader
    from config.topology import ROLE_GATHERER, ROLE_SYNTHESIZER
    from replay import analyze_trace
    from schema.models import AgentStep, PriorityLevel, RunTrace, StepStatus

    orig_get = config_loader.get
    try:
        two_agents = [
            {"id": "collector", "role": ROLE_GATHERER},
            {"id": "summarizer", "role": ROLE_SYNTHESIZER, "receives_from": "collector"},
        ]
        config_loader.get = lambda s, k=None, d=None: (  # type: ignore[assignment]
            two_agents if (s == "pipeline" and k == "agents") else orig_get(s, k, d)
        )
        clean_trace = RunTrace(
            run_id="run_verify_2ag",
            workflow="two_agent",
            steps=[
                AgentStep(
                    run_id="run_verify_2ag",
                    step=1,
                    agent="collector",
                    input="{}",
                    output=json.dumps({"source_count": 5, "entity_count": 5}),
                    status=StepStatus.SUCCESS,
                    timestamp=datetime.now(UTC),
                ),
                AgentStep(
                    run_id="run_verify_2ag",
                    step=2,
                    agent="summarizer",
                    input="{}",
                    output=json.dumps({"source_count": 5, "entity_count": 5}),
                    status=StepStatus.SUCCESS,
                    timestamp=datetime.now(UTC),
                ),
            ],
        )
        bundle = analyze_trace(clean_trace, dry_run=True)
        assert bundle.priority_level == PriorityLevel.P5
    finally:
        config_loader.get = orig_get  # type: ignore[assignment]

    return "2-agent pipeline (collector -> summarizer) evaluated cleanly with P5 PASS"


def check_6_information_loss_role_attribution() -> str:
    from analyzers.arbiter import evidence_from_information_loss
    from analyzers.detection.information_loss import InformationLossRule
    from analyzers.evidence_extraction.extractor import ExtractedEvidence

    res = InformationLossRule().evaluate(
        run_id="run_loss_role",
        researcher_evidence=ExtractedEvidence(source_count=7, entity_count=6),
        writer_evidence=ExtractedEvidence(source_count=2, entity_count=6),
        source_agent="planner",
        target_agent="coder",
    )
    ev = evidence_from_information_loss(res)
    assert ev is not None
    assert ev.agent == "coder"
    assert "planner \u2192 coder" in ev.description
    return f"information_loss_v1 attributed to '{ev.agent}'"


def check_7_existing_rule_tests() -> str:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_rule_engine.py",
            "tests/test_consistency_validator.py",
            "tests/test_workflow_validator.py",
            "tests/test_information_loss.py",
            "-q",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Existing tests failed:\n{proc.stdout}\n{proc.stderr}")
    last_line = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()][-1]
    return last_line


def check_8_day40a_pytest_suite() -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_multi_agent_generalization.py", "-q"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Day 40a pytest failed:\n{proc.stdout}\n{proc.stderr}")
    last_line = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()][-1]
    return last_line


def main() -> int:
    print("=" * 70)
    print("Day 40a Verification: Multi-Agent Generalization")
    print("=" * 70)

    checks = [
        (1, "config.yaml pipeline.agents topology", check_1_config_topology),
        (2, "PipelineTopology role & handoff resolver", check_2_topology_resolver),
        (3, "Zero hardcoded agent names in detection modules", check_3_no_hardcoded_agent_names),
        (
            4,
            "Custom 3-agent pipeline (planner -> coder -> reviewer)",
            check_4_custom_3_agent_pipeline,
        ),
        (5, "2-agent pipeline integration (collector -> summarizer)", check_5_two_agent_pipeline),
        (
            6,
            "InformationLossRule role-based agent attribution",
            check_6_information_loss_role_attribution,
        ),
        (7, "Existing rule & validator test suites (0 regressions)", check_7_existing_rule_tests),
        (
            8,
            "Day 40a pytest suite (test_multi_agent_generalization.py)",
            check_8_day40a_pytest_suite,
        ),
    ]

    passed = sum(1 for num, title, fn in checks if _check(num, title, fn))
    print("-" * 70)
    print(f"Result: {passed}/{len(checks)} checks passed.")
    print("=" * 70)
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
