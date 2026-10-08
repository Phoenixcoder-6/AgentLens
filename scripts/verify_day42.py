"""
scripts/verify_day42.py -- Day 42 Verification Suite (8 Checks)
================================================================
Verifies Day 42: Unit Tests & 75% Coverage Gate:
  1. Normalizer & all rules (including P3 wiring, rule versioning, stale detection)
  2. Diff alignment (including mismatched agent counts: 1 vs 4, disjoint pipelines)
  3. Storage (DB schema & Alembic migration upgrade/downgrade)
  4. Arbiter determinism (all 24 permutations of mixed evidence -> identical output)
  5. LLM cache (cache hit skips LLM call; cache miss triggers call and stores result)
  6. Alert system (P1/P2 verdict triggers alert; PASS / P5 does not)
  7. Dedicated Day 42 unit test suite (tests/test_day42_unit_suite.py) passes
  8. Full test suite coverage gate: pytest --cov --cov-fail-under=75 passes
"""

from __future__ import annotations

import itertools
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alembic.config import Config  # noqa: E402

from alembic import command  # noqa: E402
from analyzers.alerter import Alerter  # noqa: E402
from analyzers.arbiter import Arbiter  # noqa: E402
from analyzers.detection.workflow_validator import WorkflowValidator  # noqa: E402
from analyzers.diff_engine import AlignmentStatus, GraphAligner  # noqa: E402
from analyzers.evidence_extraction.extractor import EvidenceExtractor  # noqa: E402
from analyzers.rule_catalog import (  # noqa: E402
    RULE_CATALOG,
    is_rule_version_stale,
)
from normalizer.normalizer import Normalizer  # noqa: E402
from schema.models import (  # noqa: E402
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
)
from storage.db import DatabaseManager  # noqa: E402
from storage.llm_cache import LLMCache  # noqa: E402


def main() -> int:
    passed = 0
    total = 8
    print("=" * 72)
    print("Day 42 Verification Suite: Unit Tests & 75% Coverage Gate")
    print("=" * 72)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Check 1: Normalizer, P3 wiring, rule versioning & stale detection
        try:
            trace = RunTrace(
                run_id="run_v42_1",
                workflow="research_summary",
                steps=[
                    AgentStep(
                        run_id="run_v42_1",
                        step=1,
                        agent="writer",
                        output="Draft without researcher",
                        handoff=HandoffState(
                            input_state={"a": 1},
                            output_state={"a": 1, "b": 2},
                        ),
                    )
                ],
            )
            norm = Normalizer().normalize_run(trace)
            assert norm.schema_version == SCHEMA_VERSION
            assert norm.steps[0].input_state == {"a": 1}
            assert norm.steps[0].output_state == {"a": 1, "b": 2}

            wf_res = WorkflowValidator().run(trace)
            assert any(
                e.rule_match and e.rule_match.rule_id == "skipped_step_v1" for e in wf_res.evidence
            )
            bundle = Arbiter().run("run_v42_1", wf_res.evidence)
            assert bundle.priority_level == PriorityLevel.P3
            assert bundle.primary_cause == FailureCategory.WORKFLOW

            db = DatabaseManager(str(tmp_path / "stale42.db"))
            db.initialize()
            db.insert_run(
                "run_v42_1", "research_summary", "2026-01-01T00:00:00Z", "FAILURE", 10.0, 5, "1.0"
            )
            db.insert_rule_matches(
                "run_v42_1", [e.rule_match for e in wf_res.evidence if e.rule_match]
            )

            assert db.get_stale_rule_matches() == []
            bumped = {k: dict(v) for k, v in RULE_CATALOG.items()}
            bumped["skipped_step_v1"]["version"] = "2.0.0"  # type: ignore[index]
            assert is_rule_version_stale("skipped_step_v1", "1.0.0", catalog=bumped)  # type: ignore[arg-type]
            stale = db.get_stale_rule_matches(catalog=bumped)
            assert len(stale) >= 1 and stale[0]["current_version"] == "2.0.0"
            print(
                "[PASS] Check 1: Normalizer, P3 wiring, rule versioning, and stale detection verified"
            )
            passed += 1
        except Exception as exc:
            print(f"[FAIL] Check 1: {exc}")

        # Check 2: Diff alignment with mismatched agent counts
        try:
            t_a = RunTrace(
                run_id="ra",
                workflow="wf",
                steps=[AgentStep(run_id="ra", step=1, agent="researcher", output="o1")],
            )
            t_b = RunTrace(
                run_id="rb",
                workflow="wf",
                steps=[
                    AgentStep(run_id="rb", step=1, agent="planner", output="o0"),
                    AgentStep(run_id="rb", step=2, agent="researcher", output="o1"),
                    AgentStep(run_id="rb", step=3, agent="writer", output="o2"),
                    AgentStep(run_id="rb", step=4, agent="verifier", output="o3"),
                ],
            )
            al = GraphAligner.align_traces(t_a, t_b)
            assert al.matched_count == 1
            assert al.missing_in_a_count == 3
            assert any(p.status == AlignmentStatus.MISSING_IN_A for p in al.pairs)
            print("[PASS] Check 2: Diff alignment handles mismatched agent counts (1 vs 4)")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] Check 2: {exc}")

        # Check 3: Storage schema & Alembic migration upgrade/downgrade
        try:
            mig_db_file = tmp_path / "alembic_v42.db"
            mig_db = DatabaseManager(str(mig_db_file))
            mig_db.initialize()
            assert set(mig_db.table_counts().keys()) == {
                "runs",
                "steps",
                "analysis",
                "metrics",
                "llm_cache",
            }
            cfg = Config(str(ROOT / "alembic.ini"))
            cfg.set_main_option("script_location", str(ROOT / "alembic"))
            cfg.set_main_option("sqlalchemy.url", f"sqlite:///{mig_db_file.as_posix()}")

            command.downgrade(cfg, "base")
            command.upgrade(cfg, "head")
            with mig_db.connection() as conn:
                idxs = {
                    str(r[0])
                    for r in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'index'"
                    ).fetchall()
                }
            assert "ix_runs_timestamp" in idxs and "ix_steps_run_id" in idxs

            command.downgrade(cfg, "base")
            with mig_db.connection() as conn:
                idxs_down = {
                    str(r[0])
                    for r in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'index'"
                    ).fetchall()
                }
            assert "ix_runs_timestamp" not in idxs_down
            print(
                "[PASS] Check 3: Storage schema and Alembic upgrade(head) / downgrade(base) verified"
            )
            passed += 1
        except Exception as exc:
            print(f"[FAIL] Check 3: {exc}")

        # Check 4: Arbiter determinism across all 24 permutations
        try:
            evs = [
                EvidenceRecord(
                    source=EvidenceSource.RULE_ENGINE,
                    description="R-B",
                    value="FAIL",
                    rule_match=RuleMatch(
                        rule_id="rule_b",
                        category=FailureCategory.REASONING,
                        description="R-B",
                        severity=RuleSeverity.HIGH,
                        agent="writer",
                    ),
                    agent="writer",
                ),
                EvidenceRecord(
                    source=EvidenceSource.RULE_ENGINE,
                    description="R-A",
                    value="FAIL",
                    rule_match=RuleMatch(
                        rule_id="rule_a",
                        category=FailureCategory.EXECUTION,
                        description="R-A",
                        severity=RuleSeverity.HIGH,
                        agent="researcher",
                    ),
                    agent="researcher",
                ),
                EvidenceRecord(
                    source=EvidenceSource.WORKFLOW_VALIDATOR,
                    description="WF",
                    value="FAIL",
                    rule_match=RuleMatch(
                        rule_id="skipped_step_v1",
                        category=FailureCategory.WORKFLOW,
                        description="WF",
                        severity=RuleSeverity.HIGH,
                        agent="verifier",
                    ),
                    agent="verifier",
                ),
                EvidenceRecord(
                    source=EvidenceSource.STATISTICAL_ANOMALY,
                    description="STAT",
                    value=9999.0,
                    agent="writer",
                    confidence=0.8,
                ),
            ]
            arb = Arbiter()
            outputs = [arb.run("run_det_42", list(p)) for p in itertools.permutations(evs)]
            assert len(outputs) == 24
            first = outputs[0]
            assert all(
                o.primary_cause == first.primary_cause
                and o.priority_level == first.priority_level
                and o.primary_agent == first.primary_agent
                for o in outputs
            )
            print(
                "[PASS] Check 4: Arbiter determinism verified across all 24 evidence permutations"
            )
            passed += 1
        except Exception as exc:
            print(f"[FAIL] Check 4: {exc}")

        # Check 5: LLM Cache hit skips LLM call; miss triggers call and stores result
        try:
            cache_db = DatabaseManager(str(tmp_path / "llm_cache42.db"))
            cache_db.initialize()
            cache = LLMCache(db=cache_db)
            with patch.object(EvidenceExtractor, "_build_llm", return_value=None):
                ext = EvidenceExtractor()
            ext._cache = cache
            calls = {"n": 0}

            class _FakeLLM:
                def invoke(self, _msgs):
                    calls["n"] += 1
                    r = MagicMock()
                    r.content = '{"source_count": 2, "entity_count": 3, "tool_calls": [], "claims": [], "references": [], "numbers": [], "dates": []}'
                    r.response_metadata = {"token_usage": {"total_tokens": 25}}
                    return r

            ext._llm = _FakeLLM()  # type: ignore[assignment]
            r1 = ext.extract("Output for cache test", agent="researcher")
            r2 = ext.extract("Output for cache test", agent="researcher")
            assert r1.source_count == 2 and r2.source_count == 2
            assert calls["n"] == 1
            print(
                "[PASS] Check 5: LLM cache miss calls LLM & stores result; cache hit skips LLM call"
            )
            passed += 1
        except Exception as exc:
            print(f"[FAIL] Check 5: {exc}")

        # Check 6: Alert system (P1/P2 fires; PASS / P5 does not)
        try:
            Alerter.reset_cooldowns()
            alerter = Alerter.__new__(Alerter)
            alerter._enabled = True
            alerter._on_verdict = ["P1", "P2"]
            alerter._channel = "log"
            alerter._webhook_url = ""
            alerter._cooldown_minutes = 60
            alerter._log_dir = str(tmp_path)

            b_p1 = AnalysisBundle(
                run_id="r_p1",
                primary_cause=FailureCategory.REASONING,
                priority_level=PriorityLevel.P1,
                grounded=True,
                primary_agent="writer",
                summary="P1",
            )
            b_p2 = AnalysisBundle(
                run_id="r_p2",
                primary_cause=FailureCategory.EXECUTION,
                priority_level=PriorityLevel.P2,
                grounded=False,
                primary_agent="researcher",
                summary="P2",
            )
            b_pass = AnalysisBundle(
                run_id="r_pass",
                primary_cause=FailureCategory.UNKNOWN,
                priority_level=PriorityLevel.P5,
                grounded=False,
                primary_agent=None,
                summary="PASS",
            )
            assert alerter.fire("r_p1", b_p1) is True
            assert alerter.fire("r_p2", b_p2) is True
            assert alerter.fire("r_pass", b_pass) is False
            print("[PASS] Check 6: Alert system fires on P1/P2 and suppresses on PASS (P5)")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] Check 6: {exc}")

        # Check 7: pytest tests/test_day42_unit_suite.py
        try:
            proc_u = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/test_day42_unit_suite.py", "-q"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=90,
            )
            assert proc_u.returncode == 0, f"{proc_u.stdout}\n{proc_u.stderr}"
            print("[PASS] Check 7: pytest tests/test_day42_unit_suite.py passed")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] Check 7: {exc}")

        # Check 8: Full coverage gate (pytest --cov --cov-fail-under=75)
        try:
            proc_cov = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "--cov",
                    "--cov-fail-under=75",
                    "-q",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=300,
            )
            assert proc_cov.returncode == 0, f"{proc_cov.stdout}\n{proc_cov.stderr}"
            m = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+(?:\.\d+)?)%", proc_cov.stdout)
            cov_str = f"{m.group(1)}%" if m else ">=75%"
            print(
                f"[PASS] Check 8: Coverage gate passed (total coverage = {cov_str}, threshold = 75%)"
            )
            passed += 1
        except Exception as exc:
            print(f"[FAIL] Check 8: {exc}")

    print("-" * 72)
    print(f"Summary: {passed}/{total} checks passed")
    print("=" * 72)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
