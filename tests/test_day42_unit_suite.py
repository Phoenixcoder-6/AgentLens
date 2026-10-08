"""
tests/test_day42_unit_suite.py — Day 42 Comprehensive Unit Test Suite
======================================================================
Covers all Day 42 requirements:
  1. Normalizer & all rules (including P3 wiring, rule versioning, stale detection)
  2. Diff alignment (including mismatched agent counts: 1 vs 3, 2 vs 4, disjoint)
  3. Storage (DB schema & Alembic migration upgrade/downgrade)
  4. Arbiter determinism: all 24 permutations of evidence -> identical output
  5. LLM cache: cache hit skips LLM call; cache miss triggers call and stores result
  6. Alert system: P1/P2 verdict triggers alert; PASS (P5) does not
"""

from __future__ import annotations

import itertools
from pathlib import Path
from unittest.mock import MagicMock, patch

from analyzers.alerter import Alerter
from analyzers.arbiter import Arbiter
from analyzers.detection.consistency_validator import ConsistencyValidator
from analyzers.detection.information_loss import InformationLossRule
from analyzers.detection.rule_engine import RuleEngine
from analyzers.detection.workflow_validator import WorkflowValidator
from analyzers.diff_engine import AlignmentStatus, DiffEngine, GraphAligner
from analyzers.evidence_extraction.extractor import EvidenceExtractor
from analyzers.rule_catalog import (
    RULE_CATALOG,
    get_current_rule_version,
    is_rule_version_stale,
)
from normalizer.normalizer import Normalizer
from schema.models import (
    SCHEMA_VERSION,
    AgentStep,
    AnalysisBundle,
    EvidenceRecord,
    EvidenceSource,
    FailureCategory,
    HandoffState,
    PriorityLevel,
    RuleMatch,
    RuleSeverity,
    RunTrace,
    StepStatus,
    TokenUsage,
)
from storage.db import DatabaseManager
from storage.llm_cache import LLMCache

# ─────────────────────────────────────────────────────────────────────────────
# 1. Normalizer, All Rules, P3 Wiring, Rule Versioning & Stale Detection
# ─────────────────────────────────────────────────────────────────────────────


class TestNormalizerAndRulesDay42:
    def test_normalizer_stamps_schema_version_and_computes_handoff_diff(self):
        trace = RunTrace(
            run_id="run_norm_42",
            workflow="research_summary",
            steps=[
                AgentStep(
                    run_id="run_norm_42",
                    step=1,
                    agent="researcher",
                    input="query",
                    output="sources",
                    latency_ms=110.0,
                    tokens=TokenUsage(prompt=20, completion=30, total=50),
                    handoff=HandoffState(
                        input_state={"query": "AI"},
                        output_state={"query": "AI", "sources": ["s1", "s2"]},
                    ),
                )
            ],
        )
        norm_run = Normalizer().normalize_run(trace)
        assert norm_run.schema_version == SCHEMA_VERSION
        assert norm_run.run_id == "run_norm_42"
        assert len(norm_run.steps) == 1
        assert norm_run.steps[0].input_state == {"query": "AI"}
        assert norm_run.steps[0].output_state == {"query": "AI", "sources": ["s1", "s2"]}

    def test_p3_workflow_validator_wiring_into_arbiter(self):
        """WorkflowValidator skipped_step_v1 / wrong_order_v1 wires into Arbiter as P3 WORKFLOW."""
        trace = RunTrace(
            run_id="run_p3_wire_42",
            workflow="research_summary",
            steps=[
                AgentStep(
                    run_id="run_p3_wire_42",
                    step=1,
                    agent="writer",
                    output="Draft without researcher",
                ),
                AgentStep(
                    run_id="run_p3_wire_42",
                    step=2,
                    agent="researcher",
                    output="Late research",
                ),
            ],
        )
        wf_res = WorkflowValidator().run(trace)
        assert not wf_res.skipped
        rule_ids = {e.rule_match.rule_id for e in wf_res.evidence if e.rule_match}
        assert "skipped_step_v1" in rule_ids
        assert "wrong_order_v1" in rule_ids

        bundle = Arbiter().run(run_id=trace.run_id, evidence=wf_res.evidence)
        assert bundle.priority_level == PriorityLevel.P3
        assert bundle.primary_cause == FailureCategory.WORKFLOW

    def test_all_rules_stamp_rule_version_and_stale_detection_works(self, tmp_path, monkeypatch):
        """Every fired RuleMatch carries a version, and stale rule versions are detected."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        trace = RunTrace(
            run_id="run_rules_version_42",
            workflow="research_summary",
            steps=[
                AgentStep(
                    run_id="run_rules_version_42",
                    step=1,
                    agent="researcher",
                    output="0 sources",
                    tool_calls=[
                        {"name": "search", "input": "q", "output": None, "error": "Error: timeout"}
                    ],
                    handoff=HandoffState(
                        output_state={
                            "extracted_evidence": {
                                "source_count": 0,
                                "entity_count": 2,
                                "claims": ["Claim A"],
                                "extraction_failed": False,
                            }
                        }
                    ),
                ),
                AgentStep(
                    run_id="run_rules_version_42",
                    step=2,
                    agent="writer",
                    output="Added entities and new claim",
                    handoff=HandoffState(
                        output_state={
                            "extracted_evidence": {
                                "source_count": 0,
                                "entity_count": 5,
                                "claims": ["Claim A", "Claim B"],
                                "extraction_failed": False,
                            }
                        }
                    ),
                ),
                AgentStep(
                    run_id="run_rules_version_42",
                    step=3,
                    agent="verifier",
                    output="Approved",
                    handoff=HandoffState(
                        output_state={
                            "extracted_evidence": {
                                "source_count": 0,
                                "entity_count": 5,
                                "claims": ["Claim A", "Claim B"],
                                "extraction_failed": False,
                            }
                        }
                    ),
                ),
            ],
        )

        re_ev = RuleEngine().run(trace).evidence
        cv_ev = ConsistencyValidator().run(trace).evidence
        il_ev = InformationLossRule().run(trace).evidence
        all_ev = re_ev + cv_ev + il_ev
        assert len(all_ev) >= 4

        for ev in all_ev:
            assert ev.rule_match is not None
            assert ev.rule_match.rule_version == "1.0.0"
            assert not is_rule_version_stale(ev.rule_match.rule_id, ev.rule_match.rule_version)

        # Persist into SQLite and test stale detection when catalog version is bumped
        db = DatabaseManager(str(tmp_path / "stale_rules.db"))
        db.initialize()
        db.insert_run(
            "run_rules_version_42",
            "research_summary",
            "2026-01-01T00:00:00Z",
            "FAILURE",
            100.0,
            50,
            "1.0",
        )
        db.insert_rule_matches(
            "run_rules_version_42", [e.rule_match for e in all_ev if e.rule_match]
        )

        assert db.get_stale_rule_matches() == []

        # Bump hallucination_v1 to 2.0.0 in a custom catalog snapshot
        bumped_catalog = {k: dict(v) for k, v in RULE_CATALOG.items()}
        bumped_catalog["hallucination_v1"]["version"] = "2.0.0"  # type: ignore[index]

        assert get_current_rule_version("hallucination_v1", catalog=bumped_catalog) == "2.0.0"  # type: ignore[arg-type]
        assert is_rule_version_stale("hallucination_v1", "1.0.0", catalog=bumped_catalog) is True  # type: ignore[arg-type]

        stale_rows = db.get_stale_rule_matches(catalog=bumped_catalog)
        assert len(stale_rows) == 1
        assert stale_rows[0]["rule_id"] == "hallucination_v1"
        assert stale_rows[0]["rule_version"] == "1.0.0"
        assert stale_rows[0]["current_version"] == "2.0.0"
        assert stale_rows[0]["is_stale"] is True


# ─────────────────────────────────────────────────────────────────────────────
# 2. Diff Alignment (Mismatched Agent Counts)
# ─────────────────────────────────────────────────────────────────────────────


class TestDiffAlignmentMismatchedCountsDay42:
    @staticmethod
    def _step(agent: str, idx: int, out: str = "ok") -> AgentStep:
        return AgentStep(
            run_id="r",
            step=idx,
            agent=agent,
            output=out,
            status=StepStatus.SUCCESS,
        )

    def test_align_1_agent_vs_4_agents(self):
        trace_a = RunTrace(
            run_id="run_1_agent",
            workflow="wf",
            steps=[self._step("researcher", 1)],
        )
        trace_b = RunTrace(
            run_id="run_4_agents",
            workflow="wf",
            steps=[
                self._step("planner", 1),
                self._step("researcher", 2),
                self._step("writer", 3),
                self._step("verifier", 4),
            ],
        )
        alignment = GraphAligner.align_traces(trace_a, trace_b)
        assert alignment.matched_count == 1
        assert alignment.missing_in_a_count == 3
        assert alignment.missing_in_b_count == 0
        assert alignment.is_fully_aligned is False

        diff_res = DiffEngine(baseline_trace=trace_a).run(trace_b)
        assert len(diff_res.evidence) >= 3
        assert all(e.source == EvidenceSource.DIFF_ENGINE for e in diff_res.evidence)

    def test_align_completely_disjoint_agent_pipelines(self):
        trace_a = RunTrace(
            run_id="run_disjoint_a",
            workflow="wf_a",
            steps=[self._step("researcher", 1), self._step("writer", 2)],
        )
        trace_b = RunTrace(
            run_id="run_disjoint_b",
            workflow="wf_b",
            steps=[self._step("planner", 1), self._step("coder", 2), self._step("reviewer", 3)],
        )
        alignment = GraphAligner.align_traces(trace_a, trace_b)
        assert alignment.matched_count == 0
        assert alignment.missing_in_b_count == 2
        assert alignment.missing_in_a_count == 3
        statuses = [p.status for p in alignment.pairs]
        assert statuses.count(AlignmentStatus.MISSING_IN_B) == 2
        assert statuses.count(AlignmentStatus.MISSING_IN_A) == 3


# ─────────────────────────────────────────────────────────────────────────────
# 3. Storage & Alembic Migration Up/Down
# ─────────────────────────────────────────────────────────────────────────────


class TestStorageAndAlembicDay42:
    def test_db_schema_and_alembic_up_down(self, tmp_path):
        from alembic.config import Config

        from alembic import command

        db_file = tmp_path / "day42_alembic.db"
        db = DatabaseManager(str(db_file))
        db.initialize()

        assert set(db.table_counts().keys()) == {
            "runs",
            "steps",
            "analysis",
            "metrics",
            "llm_cache",
        }

        root_dir = Path(__file__).resolve().parent.parent
        cfg = Config(str(root_dir / "alembic.ini"))
        cfg.set_main_option("script_location", str(root_dir / "alembic"))
        cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_file.as_posix()}")

        command.downgrade(cfg, "base")
        command.upgrade(cfg, "head")
        with db.connection() as conn:
            idx_names = {
                str(r[0])
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'index'"
                ).fetchall()
            }
        assert "ix_runs_timestamp" in idx_names
        assert "ix_steps_run_id" in idx_names

        command.downgrade(cfg, "base")
        with db.connection() as conn:
            idx_after = {
                str(r[0])
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'index'"
                ).fetchall()
            }
        assert "ix_runs_timestamp" not in idx_after
        assert "ix_steps_run_id" not in idx_after


# ─────────────────────────────────────────────────────────────────────────────
# 4. Arbiter Determinism Across All 24 Permutations
# ─────────────────────────────────────────────────────────────────────────────


class TestArbiterExhaustiveDeterminismDay42:
    def test_all_24_permutations_yield_identical_bundle(self):
        ev_p2_a = EvidenceRecord(
            source=EvidenceSource.RULE_ENGINE,
            description="Rule B",
            value="FAIL",
            rule_match=RuleMatch(
                rule_id="rule_b_v1",
                category=FailureCategory.REASONING,
                description="Rule B",
                severity=RuleSeverity.HIGH,
                agent="writer",
            ),
            agent="writer",
            confidence=0.95,
        )
        ev_p2_b = EvidenceRecord(
            source=EvidenceSource.RULE_ENGINE,
            description="Rule A",
            value="FAIL",
            rule_match=RuleMatch(
                rule_id="rule_a_v1",
                category=FailureCategory.EXECUTION,
                description="Rule A",
                severity=RuleSeverity.HIGH,
                agent="researcher",
            ),
            agent="researcher",
            confidence=0.80,
        )
        ev_p3 = EvidenceRecord(
            source=EvidenceSource.WORKFLOW_VALIDATOR,
            description="Skipped step",
            value="FAIL",
            rule_match=RuleMatch(
                rule_id="skipped_step_v1",
                category=FailureCategory.WORKFLOW,
                description="Skipped step",
                severity=RuleSeverity.HIGH,
                agent="verifier",
            ),
            agent="verifier",
            confidence=1.0,
        )
        ev_p4 = EvidenceRecord(
            source=EvidenceSource.STATISTICAL_ANOMALY,
            description="Latency outlier",
            value=9500.0,
            agent="writer",
            confidence=0.75,
        )

        items = [ev_p2_a, ev_p2_b, ev_p3, ev_p4]
        arbiter = Arbiter()
        baseline: AnalysisBundle | None = None

        permutations = list(itertools.permutations(items))
        assert len(permutations) == 24

        for perm in permutations:
            bundle = arbiter.run("run_perm_42", list(perm))
            if baseline is None:
                baseline = bundle
            else:
                assert bundle.primary_cause == baseline.primary_cause
                assert bundle.priority_level == baseline.priority_level
                assert bundle.primary_agent == baseline.primary_agent
                assert bundle.grounded == baseline.grounded
                assert bundle.summary == baseline.summary

        assert baseline is not None
        # Tie-break ascending: rule_a_v1 < rule_b_v1 -> researcher EXECUTION wins
        assert baseline.priority_level == PriorityLevel.P2
        assert baseline.primary_cause == FailureCategory.EXECUTION
        assert baseline.primary_agent == "researcher"


# ─────────────────────────────────────────────────────────────────────────────
# 5. LLM Cache: Hit Skips Call, Miss Triggers Call & Stores Result
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMCacheHitMissDay42:
    def test_cache_miss_calls_llm_and_cache_hit_skips_llm(self, tmp_path):
        db = DatabaseManager(str(tmp_path / "cache_day42.db"))
        db.initialize()
        cache = LLMCache(db=db)

        with patch.object(EvidenceExtractor, "_build_llm", return_value=None):
            extractor = EvidenceExtractor()
        extractor._cache = cache

        call_counter = {"count": 0}

        class FakeLLM:
            def invoke(self, messages):
                call_counter["count"] += 1
                resp = MagicMock()
                resp.content = (
                    '{"source_count": 3, "entity_count": 4, "tool_calls": [], '
                    '"claims": ["c1"], "references": [], "numbers": [], "dates": []}'
                )
                resp.response_metadata = {"token_usage": {"total_tokens": 42}}
                return resp

        extractor._llm = FakeLLM()  # type: ignore[assignment]

        # 1. Cache miss -> triggers FakeLLM.invoke() and stores result in SQLite
        ev1 = extractor.extract("Sample agent output with 3 sources", agent="researcher")
        assert not ev1.extraction_failed
        assert ev1.source_count == 3
        assert ev1.entity_count == 4
        assert call_counter["count"] == 1
        assert db.table_counts()["llm_cache"] == 1

        # 2. Cache hit -> skips FakeLLM.invoke() (call_counter stays 1)
        ev2 = extractor.extract("Sample agent output with 3 sources", agent="researcher")
        assert not ev2.extraction_failed
        assert ev2.source_count == 3
        assert ev2.entity_count == 4
        assert call_counter["count"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 6. Alert System: P1/P2 Verdict Triggers Alert; PASS (P5) Does Not
# ─────────────────────────────────────────────────────────────────────────────


class TestAlertSystemDay42:
    def test_p1_and_p2_trigger_alert_while_pass_p5_does_not(self, tmp_path):
        Alerter.reset_cooldowns()
        alerter = Alerter.__new__(Alerter)
        alerter._enabled = True
        alerter._on_verdict = ["P1", "P2"]
        alerter._channel = "log"
        alerter._webhook_url = ""
        alerter._cooldown_minutes = 60
        alerter._log_dir = str(tmp_path)

        p1_bundle = AnalysisBundle(
            run_id="run_alert_p1",
            primary_cause=FailureCategory.REASONING,
            priority_level=PriorityLevel.P1,
            grounded=True,
            primary_agent="writer",
            summary="Ground truth mismatch",
        )
        p2_bundle = AnalysisBundle(
            run_id="run_alert_p2",
            primary_cause=FailureCategory.EXECUTION,
            priority_level=PriorityLevel.P2,
            grounded=False,
            primary_agent="researcher",
            summary="Tool failure",
        )
        pass_bundle = AnalysisBundle(
            run_id="run_alert_pass",
            primary_cause=FailureCategory.UNKNOWN,
            priority_level=PriorityLevel.P5,
            grounded=False,
            primary_agent=None,
            summary="PASS — no failures detected",
        )

        assert alerter.fire("run_alert_p1", p1_bundle) is True
        assert alerter.fire("run_alert_p2", p2_bundle) is True
        assert alerter.fire("run_alert_pass", pass_bundle) is False
