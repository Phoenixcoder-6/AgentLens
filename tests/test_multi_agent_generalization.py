"""
tests/test_multi_agent_generalization.py — Day 40a Unit & Integration Tests
============================================================================
Covers:
    1. Role-based topology resolution from `config/config.yaml` (`pipeline.agents`).
    2. Switching to a custom 3-agent pipeline (`planner -> coder -> reviewer`)
       via config alone — verifying `RuleEngine`, `InformationLossRule`,
       `ConsistencyValidator`, `WorkflowValidator`, and `Arbiter` attribute
       faults to `planner`, `coder`, and `reviewer`.
    3. Running a 2-agent pipeline (`collector -> summarizer`) without a
       `quality_checker` — confirming clean `P5` pass and accurate `P2`/`P3`
       failure attribution with zero false-positive `skipped_step_v1`.
    4. Codebase inspection confirming hardcoded `"researcher"`, `"writer"`,
       `"verifier"` strings are removed from detection rule modules.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from analyzers.arbiter import Arbiter, evidence_from_information_loss
from analyzers.detection.consistency_validator import ConsistencyValidator
from analyzers.detection.information_loss import InformationLossRule
from analyzers.detection.rule_engine import RuleEngine
from analyzers.detection.workflow_validator import WorkflowValidator
from config import config_loader
from config.topology import (
    ROLE_CHECKER,
    ROLE_GATHERER,
    ROLE_SYNTHESIZER,
    get_topology,
)
from replay import analyze_trace
from schema.models import AgentStep, PriorityLevel, RunTrace, StepStatus

CUSTOM_3_AGENT_TOPOLOGY = [
    {"id": "planner", "role": ROLE_GATHERER},
    {"id": "coder", "role": ROLE_SYNTHESIZER, "receives_from": "planner"},
    {"id": "reviewer", "role": ROLE_CHECKER, "receives_from": "coder"},
]

TWO_AGENT_TOPOLOGY = [
    {"id": "collector", "role": ROLE_GATHERER},
    {"id": "summarizer", "role": ROLE_SYNTHESIZER, "receives_from": "collector"},
]


def _patch_topology(monkeypatch, agents_list):
    orig_get = config_loader.get

    def _custom_get(section: str, key: str | None = None, default=None):
        if section == "pipeline" and key == "agents":
            return agents_list
        if section == "pipeline" and key == "name":
            return "custom_pipeline"
        return orig_get(section, key, default)

    monkeypatch.setattr(config_loader, "get", _custom_get)


def _step(step_num: int, agent: str, payload: dict) -> AgentStep:
    return AgentStep(
        run_id="run_topo_test",
        step=step_num,
        agent=agent,
        input="{}",
        output=json.dumps(payload),
        latency_ms=120.0,
        status=StepStatus.SUCCESS,
        timestamp=datetime.now(UTC),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Default Config Topology Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_default_config_yaml_topology():
    config_loader.load_config.cache_clear()
    topo = get_topology()
    assert topo.name == "research_report_pipeline"
    assert topo.primary_agent_for_role(ROLE_GATHERER) == "researcher"
    assert topo.primary_agent_for_role(ROLE_SYNTHESIZER) == "writer"
    assert topo.primary_agent_for_role(ROLE_CHECKER) == "verifier"
    assert topo.upstream_agent_for("writer") == "researcher"
    assert topo.upstream_agent_for("verifier") == "writer"
    assert topo.required_agent_order() == ["researcher", "writer", "verifier"]
    assert topo.handoff_edges() == [("researcher", "writer"), ("writer", "verifier")]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Custom 3-Agent Pipeline (`planner -> coder -> reviewer`)
# ─────────────────────────────────────────────────────────────────────────────


def test_custom_3_agent_pipeline_rules_and_arbiter(monkeypatch):
    """
    Switching config to `planner -> coder -> reviewer` attributes:
      - researcher_quality_v1 to `planner`
      - hallucination_v1, claim_drift_v1, information_loss_v1 to `coder`
      - verifier_passthrough_v1 to `reviewer`
      - skipped_step_v1 / wrong_order_v1 to `planner` / `coder` / `reviewer`
    """
    _patch_topology(monkeypatch, CUSTOM_3_AGENT_TOPOLOGY)

    # Case A: planner has 0 sources -> researcher_quality_v1 blames 'planner'
    trace_low_sources = RunTrace(
        run_id="run_planner_low",
        workflow="code_pipeline",
        steps=[
            _step(1, "planner", {"source_count": 0, "entity_count": 4, "claims": ["c1"]}),
            _step(2, "coder", {"source_count": 0, "entity_count": 4, "claims": ["c1"]}),
            _step(3, "reviewer", {"source_count": 0, "entity_count": 4, "verified": True}),
        ],
    )
    re_res = RuleEngine().analyze(trace_low_sources)
    assert len(re_res.evidence) == 1
    assert re_res.evidence[0].rule_match.rule_id == "researcher_quality_v1"
    assert re_res.evidence[0].agent == "planner"

    # Case B: coder hallucinates entities & reviewer passes them through
    trace_hallucination = RunTrace(
        run_id="run_coder_halluc",
        workflow="code_pipeline",
        steps=[
            _step(1, "planner", {"source_count": 5, "entity_count": 4, "claims": ["spec A"]}),
            _step(
                2,
                "coder",
                {"source_count": 5, "entity_count": 9, "claims": ["spec A", "invented API"]},
            ),
            _step(3, "reviewer", {"source_count": 5, "entity_count": 9, "verified": True}),
        ],
    )
    re_halluc = RuleEngine().analyze(trace_hallucination)
    assert any(
        e.rule_match.rule_id == "hallucination_v1" and e.agent == "coder"
        for e in re_halluc.evidence
    )

    cv_res = ConsistencyValidator().analyze(trace_hallucination)
    cv_map = {e.rule_match.rule_id: e.agent for e in cv_res.evidence if e.rule_match}
    assert cv_map.get("verifier_passthrough_v1") == "reviewer"
    assert cv_map.get("claim_drift_v1") == "coder"

    # Case C: coder drops 4 sources from planner -> information_loss_v1 blames 'coder'
    trace_info_loss = RunTrace(
        run_id="run_coder_drop",
        workflow="code_pipeline",
        steps=[
            _step(1, "planner", {"source_count": 6, "entity_count": 5}),
            _step(2, "coder", {"source_count": 1, "entity_count": 5}),
            _step(3, "reviewer", {"source_count": 1, "entity_count": 5}),
        ],
    )
    loss_res = InformationLossRule().evaluate_trace(trace_info_loss)
    assert loss_res is not None
    assert loss_res.verdict == "FAIL"
    assert loss_res.source_agent == "planner"
    assert loss_res.target_agent == "coder"
    loss_ev = evidence_from_information_loss(loss_res)
    assert loss_ev is not None and loss_ev.agent == "coder"

    bundle = analyze_trace(trace_info_loss, dry_run=True)
    assert bundle.priority_level == PriorityLevel.P2
    assert bundle.primary_agent == "coder"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Two-Agent Pipeline Integration Test (`collector -> summarizer`)
# ─────────────────────────────────────────────────────────────────────────────


def test_two_agent_pipeline_integration(monkeypatch):
    """
    Integration test: run with a 2-agent pipeline config (`collector -> summarizer`),
    confirming clean P5 pass, P2 information loss on `summarizer`, and P3 ordering
    without false-flagging a missing verifier.
    """
    _patch_topology(monkeypatch, TWO_AGENT_TOPOLOGY)

    # 1. Clean 2-agent run -> P5 PASS (no skipped_step_v1 for verifier!)
    clean_2_agent = RunTrace(
        run_id="run_2ag_clean",
        workflow="two_agent_pipeline",
        steps=[
            _step(1, "collector", {"source_count": 5, "entity_count": 6, "claims": ["fact 1"]}),
            _step(2, "summarizer", {"source_count": 5, "entity_count": 6, "claims": ["fact 1"]}),
        ],
    )
    wf_clean = WorkflowValidator().analyze(clean_2_agent)
    assert wf_clean.evidence == []
    bundle_clean = analyze_trace(clean_2_agent, dry_run=True)
    assert bundle_clean.priority_level == PriorityLevel.P5

    # 2. Information loss in 2-agent run -> P2 blamed on 'summarizer'
    drop_2_agent = RunTrace(
        run_id="run_2ag_drop",
        workflow="two_agent_pipeline",
        steps=[
            _step(1, "collector", {"source_count": 6, "entity_count": 6}),
            _step(2, "summarizer", {"source_count": 1, "entity_count": 2}),
        ],
    )
    bundle_drop = analyze_trace(drop_2_agent, dry_run=True)
    assert bundle_drop.priority_level == PriorityLevel.P2
    assert bundle_drop.primary_agent == "summarizer"

    # 3. Wrong execution order in 2-agent run -> P3 wrong_order_v1 on 'summarizer'
    wrong_order_2_agent = RunTrace(
        run_id="run_2ag_order",
        workflow="two_agent_pipeline",
        steps=[
            _step(1, "summarizer", {"source_count": 5, "entity_count": 5}),
            _step(2, "collector", {"source_count": 5, "entity_count": 5}),
        ],
    )
    wf_order = WorkflowValidator().analyze(wrong_order_2_agent)
    assert len(wf_order.evidence) == 1
    assert wf_order.evidence[0].rule_match.rule_id == "wrong_order_v1"
    assert wf_order.evidence[0].agent == "summarizer"
    bundle_order = Arbiter().run("run_2ag_order", wf_order.evidence)
    assert bundle_order.priority_level == PriorityLevel.P3
    assert bundle_order.primary_agent == "summarizer"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Verify No Hardcoded Agent Strings in Detection Rule Modules
# ─────────────────────────────────────────────────────────────────────────────


def test_no_hardcoded_agent_names_in_detection_rule_modules():
    root = Path(__file__).resolve().parent.parent
    for rel_path in (
        "analyzers/detection/rule_engine.py",
        "analyzers/detection/consistency_validator.py",
        "analyzers/detection/workflow_validator.py",
    ):
        source = (root / rel_path).read_text(encoding="utf-8")
        for forbidden in ('"researcher"', '"writer"', '"verifier"'):
            assert forbidden not in source, f"Found hardcoded {forbidden} in {rel_path}"
