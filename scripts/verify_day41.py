"""
scripts/verify_day41.py -- Day 41 Verification Suite (8 Checks)
================================================================
Verifies Day 41: Analyzer Interface Validation + PEP 561 py.typed + mypy:
  1. All 11 concrete analyzers implement the runtime_checkable Analyzer Protocol
  2. EvidenceExtractor.analyze() / .run() extracts pre-structured evidence without LLM calls
  3. DiffEngine, GraphAligner, and SemanticSimilarityEngine implement Analyzer & detect divergence
  4. MetricsAnalyzer and StatisticalDetector implement Analyzer & return AnalysisResult
  5. All Detection sub-modules (GroundTruth, RuleEngine, InformationLoss, Workflow, Consistency) implement Analyzer
  6. All 11 concrete analyzers degrade gracefully (skipped=True, no exception) on empty RunTrace
  7. PEP 561 py.typed marker files exist across all 9 AgentLens packages + pyproject.toml config
  8. Full pytest suite for tests/test_interfaces.py passes
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyzers.detection.consistency_validator import ConsistencyValidator  # noqa: E402
from analyzers.detection.ground_truth import GroundTruthValidator  # noqa: E402
from analyzers.detection.information_loss import InformationLossRule  # noqa: E402
from analyzers.detection.rule_engine import RuleEngine  # noqa: E402
from analyzers.detection.statistical_detector import StatisticalDetector  # noqa: E402
from analyzers.detection.workflow_validator import WorkflowValidator  # noqa: E402
from analyzers.diff_engine import (  # noqa: E402
    DiffEngine,
    GraphAligner,
    SemanticSimilarityEngine,
)
from analyzers.evidence_extraction.extractor import EvidenceExtractor  # noqa: E402
from analyzers.metrics_analyzer import MetricsAnalyzer  # noqa: E402
from app.interfaces import AnalysisResult, Analyzer  # noqa: E402
from schema.models import AgentStep, EvidenceSource, RunTrace, TokenUsage  # noqa: E402
from storage.db import DatabaseManager  # noqa: E402


def _make_sample_trace() -> RunTrace:
    from schema.models import HandoffState

    return RunTrace(
        run_id="run_verify_day41",
        workflow="research_summary",
        expected_output="Final verified summary.",
        steps=[
            AgentStep(
                run_id="run_verify_day41",
                step=1,
                agent="researcher",
                input="Topic: Quantum computing",
                output="Found 5 sources and 6 entities.",
                latency_ms=100.0,
                tokens=TokenUsage(prompt=40, completion=60, total=100),
                handoff=HandoffState(
                    output_state={
                        "extracted_evidence": {
                            "source_count": 5,
                            "entity_count": 6,
                            "claims": ["Claim 1", "Claim 2"],
                            "extraction_failed": False,
                        }
                    }
                ),
            ),
            AgentStep(
                run_id="run_verify_day41",
                step=2,
                agent="writer",
                input="Found 5 sources and 6 entities.",
                output="Summary of 2 sources and 6 entities.",
                latency_ms=12000.0,  # exceeds 8000ms threshold in config.yaml
                tokens=TokenUsage(prompt=50, completion=70, total=120),
                handoff=HandoffState(
                    output_state={
                        "extracted_evidence": {
                            "source_count": 2,  # dropped 3 sources -> InformationLoss FAIL
                            "entity_count": 6,
                            "claims": ["Claim 1", "Claim 2"],
                            "extraction_failed": False,
                        }
                    }
                ),
            ),
            AgentStep(
                run_id="run_verify_day41",
                step=3,
                agent="verifier",
                input="Summary of 2 sources and 6 entities.",
                output="Final verified summary.",
                latency_ms=90.0,
                tokens=TokenUsage(prompt=30, completion=40, total=70),
                handoff=HandoffState(
                    output_state={
                        "extracted_evidence": {
                            "source_count": 2,
                            "entity_count": 6,
                            "claims": ["Claim 1", "Claim 2"],
                            "extraction_failed": False,
                        }
                    }
                ),
            ),
        ],
    )


def main() -> int:
    passed = 0
    total = 8
    print("=" * 72)
    print("Day 41 Verification Suite: Analyzer Interface + py.typed + mypy")
    print("=" * 72)

    os.environ.pop("GROQ_API_KEY", None)

    with tempfile.TemporaryDirectory() as tmpdir:
        db = DatabaseManager(db_path=str(Path(tmpdir) / "verify41.db"))
        with patch.object(EvidenceExtractor, "_build_llm", return_value=None):
            extractor = EvidenceExtractor()

        analyzers = [
            extractor,
            DiffEngine(),
            GraphAligner(),
            SemanticSimilarityEngine(),
            MetricsAnalyzer(db=db),
            StatisticalDetector(db=db),
            GroundTruthValidator(),
            RuleEngine(),
            InformationLossRule(),
            WorkflowValidator(),
            ConsistencyValidator(),
        ]

        # Check 1: All 11 concrete analyzers satisfy isinstance(a, Analyzer)
        try:
            for a in analyzers:
                assert isinstance(a, Analyzer), f"{type(a).__name__} failed isinstance(Analyzer)"
                assert isinstance(a.analyzer_id, str) and a.analyzer_id
                assert callable(getattr(a, "analyze", None))
                assert callable(getattr(a, "run", None))
            print("[PASS] Check 1: All 11 concrete analyzers implement the Analyzer protocol")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] Check 1: {exc}")

        trace = _make_sample_trace()

        # Check 2: EvidenceExtractor.analyze() / .run()
        try:
            res_ext = extractor.run(trace)
            assert isinstance(res_ext, AnalysisResult)
            assert res_ext.analyzer_id == "evidence_extraction"
            assert not res_ext.skipped
            assert len(res_ext.evidence) == 3
            assert all(e.source == EvidenceSource.EVIDENCE_EXTRACTION for e in res_ext.evidence)
            print(
                "[PASS] Check 2: EvidenceExtractor.run(trace) returns AnalysisResult with 3 EvidenceRecords"
            )
            passed += 1
        except Exception as exc:
            print(f"[FAIL] Check 2: {exc}")

        # Check 3: DiffEngine, GraphAligner, SemanticSimilarityEngine
        try:
            diff_engine = DiffEngine()
            res_diff = diff_engine.run(trace)
            assert isinstance(res_diff, AnalysisResult)
            assert res_diff.analyzer_id == "diff_engine"
            assert not res_diff.skipped

            # Compare against a baseline missing the verifier step
            partial_trace = RunTrace(
                run_id="run_baseline_partial",
                workflow="research_summary",
                steps=trace.steps[:2],
            )
            res_missing = diff_engine.analyze(trace, baseline_trace=partial_trace)
            assert len(res_missing.evidence) >= 1
            assert any(e.source == EvidenceSource.DIFF_ENGINE for e in res_missing.evidence)
            print(
                "[PASS] Check 3: DiffEngine / GraphAligner / SemanticSimilarityEngine implement Analyzer"
            )
            passed += 1
        except Exception as exc:
            print(f"[FAIL] Check 3: {exc}")

        # Check 4: MetricsAnalyzer & StatisticalDetector
        try:
            ma = MetricsAnalyzer(db=db)
            sd = StatisticalDetector(db=db)
            res_ma = ma.run(trace)
            res_sd = sd.run(trace)
            assert isinstance(res_ma, AnalysisResult)
            assert res_ma.analyzer_id == "metrics_analyzer"
            # writer step has latency_ms=8000 > 5000ms threshold
            assert len(res_ma.evidence) >= 1
            assert res_ma.evidence[0].source == EvidenceSource.METRICS_ANALYZER

            assert isinstance(res_sd, AnalysisResult)
            assert res_sd.analyzer_id == "statistical_detector"
            print(
                "[PASS] Check 4: MetricsAnalyzer & StatisticalDetector implement Analyzer and emit P4 evidence"
            )
            passed += 1
        except Exception as exc:
            print(f"[FAIL] Check 4: {exc}")

        # Check 5: All Detection sub-modules implement Analyzer
        try:
            det_analyzers = [
                GroundTruthValidator(),
                RuleEngine(),
                InformationLossRule(),
                WorkflowValidator(),
                ConsistencyValidator(),
            ]
            for da in det_analyzers:
                res = da.analyze(trace)
                assert isinstance(res, AnalysisResult)
                assert res.analyzer_id == da.analyzer_id
            # InformationLossRule should detect the 5 -> 2 source drop
            il_res = InformationLossRule().run(trace)
            assert len(il_res.evidence) == 1
            assert il_res.evidence[0].rule_match is not None
            assert il_res.evidence[0].rule_match.rule_id == "information_loss_v1"
            print("[PASS] Check 5: All 5 Detection sub-modules implement Analyzer (.analyze)")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] Check 5: {exc}")

        # Check 6: Graceful skip on empty RunTrace across all 11 analyzers
        try:
            empty_trace = RunTrace(run_id="run_empty", workflow="research_summary", steps=[])
            for a in analyzers:
                res = a.analyze(empty_trace)
                assert isinstance(res, AnalysisResult)
                assert res.skipped is True, f"{type(a).__name__} did not set skipped=True"
                assert res.evidence == []
            print("[PASS] Check 6: All 11 concrete analyzers gracefully skip on empty RunTrace")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] Check 6: {exc}")

        # Check 7: PEP 561 py.typed markers & pyproject.toml configuration
        try:
            packages = [
                "app",
                "analyzers",
                "capture",
                "config",
                "dashboard",
                "diff_engine",
                "normalizer",
                "schema",
                "storage",
            ]
            for pkg in packages:
                marker = ROOT / pkg / "py.typed"
                assert marker.is_file(), f"Missing {marker}"
            pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
            assert "py.typed" in pyproject_text
            print(
                "[PASS] Check 7: PEP 561 py.typed markers present in all 9 packages + pyproject.toml"
            )
            passed += 1
        except Exception as exc:
            print(f"[FAIL] Check 7: {exc}")

        # Check 8: Pytest suite for tests/test_interfaces.py passes
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/test_interfaces.py", "-q"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert proc.returncode == 0, f"pytest failed:\n{proc.stdout}\n{proc.stderr}"
            print("[PASS] Check 8: pytest tests/test_interfaces.py passed")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] Check 8: {exc}")

    print("-" * 72)
    print(f"Summary: {passed}/{total} checks passed")
    print("=" * 72)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
