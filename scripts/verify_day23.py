"""
scripts/verify_day23.py — Day 23 Verification Script
=====================================================
Verifies 12 critical Day 23 requirements:
  1. DDL: _CREATE_LLM_CACHE in DatabaseManager
  2. Table: llm_cache initialized in SQLite
  3. Key: LLMCache.compute_key deterministic SHA256
  4. Cache HIT: set() and get() store/retrieve responses
  5. Cache EXPIRATION: expired records return None and purge
  6. Extractor Cache: second call hits cache without LLM invocation
  7. Extractor Fallback Model: primary error triggers fallback model
  8. Extractor Degradation: complete LLM failure returns extraction_failed=True
  9. RuleEngine Degradation: skips extraction rules on extraction failure
 10. ConsistencyValidator Degradation: skips extraction rules on extraction failure
 11. Explainer Fallback Model: primary error triggers fallback model
 12. Explainer Degradation: complete API failure returns rule-based summary
"""

from __future__ import annotations

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

# Force UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import UTC

from analyzers.detection import ConsistencyValidator, RuleEngine
from analyzers.evidence_extraction.extractor import EvidenceExtractor, ExtractedEvidence
from analyzers.explainer import LLMExplainer
from schema.models import (
    AgentStep,
    AnalysisBundle,
    FailureCategory,
    PriorityLevel,
    RunTrace,
    StepStatus,
)
from storage.db import DatabaseManager
from storage.llm_cache import LLMCache

PASS = "[PASS]"
FAIL = "[FAIL]"


def main() -> int:
    print("=== Day 23 Verification: Graceful Degradation & LLM Caching ===\n")
    passed = 0
    total = 0

    # ── Test 1: DDL contains llm_cache ──────────────────────────────────────
    total += 1
    try:
        from storage.db import _CREATE_LLM_CACHE

        if "llm_cache" in _CREATE_LLM_CACHE:
            print(f"  {PASS} [1] _CREATE_LLM_CACHE table DDL present")
            passed += 1
        else:
            print(f"  {FAIL} [1] DDL missing llm_cache")
    except Exception as e:
        print(f"  {FAIL} [1] DDL check error: {e}")

    # ── Test 2: DatabaseManager initializes llm_cache table ──────────────────
    total += 1
    try:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            db_path = tf.name
        db = DatabaseManager(db_path)
        db.initialize()
        counts = db.table_counts()
        if "llm_cache" in counts:
            print(f"  {PASS} [2] DatabaseManager initialized llm_cache table")
            passed += 1
        else:
            print(f"  {FAIL} [2] table_counts missing llm_cache")
    except Exception as e:
        print(f"  {FAIL} [2] DB init error: {e}")

    # ── Test 3: LLMCache compute_key deterministic ───────────────────────────
    total += 1
    try:
        cache = LLMCache(db)
        k1 = cache.compute_key("sys", "user", "modelA", 0.0)
        k2 = cache.compute_key("sys", "user", "modelA", 0.0)
        if k1 == k2 and len(k1) == 64:
            print(f"  {PASS} [3] LLMCache compute_key is deterministic SHA-256")
            passed += 1
        else:
            print(f"  {FAIL} [3] Hash key mismatch or invalid length")
    except Exception as e:
        print(f"  {FAIL} [3] Key compute error: {e}")

    # ── Test 4: LLMCache set and get hit ─────────────────────────────────────
    total += 1
    try:
        cache.set("sys", "user_test4", "modelA", "cached_text_4", token_cost=88)
        retrieved = cache.get("sys", "user_test4", "modelA")
        if retrieved and retrieved[0] == "cached_text_4" and retrieved[1] == 88:
            print(f"  {PASS} [4] LLMCache set and get hit verified")
            passed += 1
        else:
            print(f"  {FAIL} [4] Cache get returned incorrect data: {retrieved}")
    except Exception as e:
        print(f"  {FAIL} [4] Cache set/get error: {e}")

    # ── Test 5: Cache expiration and purge ──────────────────────────────────
    total += 1
    try:
        cache.set("sys", "user_exp", "modelA", "expired_text", token_cost=10, ttl_hours=-1)
        miss = cache.get("sys", "user_exp", "modelA")
        from datetime import datetime

        purged = db.purge_expired_llm_cache(datetime.now(UTC).isoformat())
        if miss is None and purged >= 1:
            print(f"  {PASS} [5] Cache expiration and purge verified (purged={purged})")
            passed += 1
        else:
            print(f"  {FAIL} [5] Expiration failed: miss={miss}, purged={purged}")
    except Exception as e:
        print(f"  {FAIL} [5] Expiration error: {e}")

    # ── Test 6: EvidenceExtractor hits cache on repeat call ──────────────────
    total += 1
    try:
        extractor = EvidenceExtractor()
        mock_cache = MagicMock()
        mock_cache.get.return_value = ('{"source_count": 4, "entity_count": 2}', 120)
        extractor._cache = mock_cache

        res = extractor.extract("test input", agent="researcher")
        if res.source_count == 4 and not res.extraction_failed:
            print(f"  {PASS} [6] EvidenceExtractor hits cache on repeat call")
            passed += 1
        else:
            print(f"  {FAIL} [6] EvidenceExtractor cache hit failed: {res}")
    except Exception as e:
        print(f"  {FAIL} [6] Extractor cache error: {e}")

    # ── Test 7: EvidenceExtractor fallback model invocation ──────────────────
    total += 1
    try:
        extractor = EvidenceExtractor()
        extractor._cache = MagicMock()
        extractor._cache.get.return_value = None

        mock_primary = MagicMock()
        mock_primary.invoke.side_effect = RuntimeError("Primary 500 error")

        mock_fallback = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = '{"source_count": 1, "entity_count": 1}'
        mock_resp.response_metadata = {"token_usage": {"total_tokens": 50}}
        mock_fallback.invoke.return_value = mock_resp

        extractor._llm = mock_primary
        extractor._fallback_llm = mock_fallback
        extractor._fallback_model_name = "llama-3.3-70b-versatile"

        raw, tokens = extractor._call_llm("sys", "user")
        if raw == '{"source_count": 1, "entity_count": 1}' and mock_fallback.invoke.called:
            print(f"  {PASS} [7] EvidenceExtractor invoked fallback model when primary failed")
            passed += 1
        else:
            print(f"  {FAIL} [7] Extractor fallback model call failed")
    except Exception as e:
        print(f"  {FAIL} [7] Extractor fallback error: {e}")

    # ── Test 8: EvidenceExtractor degradation on total failure ──────────────
    total += 1
    try:
        extractor = EvidenceExtractor()
        with patch.object(extractor, "_call_llm", side_effect=RuntimeError("API total collapse")):
            res = extractor.extract("some input", agent="writer")

        if res.extraction_failed and "RuntimeError" in (res.error_message or ""):
            print(f"  {PASS} [8] EvidenceExtractor degrades gracefully on total failure")
            passed += 1
        else:
            print(f"  {FAIL} [8] Extractor total failure did not set extraction_failed=True")
    except Exception as e:
        print(f"  {FAIL} [8] Extractor total failure error: {e}")

    # ── Test 9: RuleEngine skips extraction rules on failure ─────────────────
    total += 1
    try:
        step = AgentStep(
            run_id="r1", step=1, agent="researcher", output="out", status=StepStatus.SUCCESS
        )
        trace = RunTrace(run_id="r1", workflow="wf", steps=[step])

        failed_ev = ExtractedEvidence(extraction_failed=True, error_message="Failed")
        with patch("analyzers.detection.rule_engine.EvidenceExtractor") as MockExt:
            m = MagicMock()
            m.extract.return_value = failed_ev
            MockExt.return_value = m

            with patch("os.getenv", return_value="key"):
                engine = RuleEngine()
                analysis = engine.analyze(trace)

        rule_ids = [e.rule_match.rule_id for e in analysis.evidence if e.rule_match]
        if "researcher_quality_v1" not in rule_ids:
            print(f"  {PASS} [9] RuleEngine skipped extraction rules when extraction_failed=True")
            passed += 1
        else:
            print(f"  {FAIL} [9] RuleEngine fired extraction rules despite failure: {rule_ids}")
    except Exception as e:
        print(f"  {FAIL} [9] RuleEngine skip error: {e}")

    # ── Test 10: ConsistencyValidator skips extraction rules on failure ──────
    total += 1
    try:
        step = AgentStep(
            run_id="r1", step=1, agent="verifier", output="out", status=StepStatus.SUCCESS
        )
        trace = RunTrace(run_id="r1", workflow="wf", steps=[step])

        failed_ev = ExtractedEvidence(extraction_failed=True, error_message="Failed")
        with patch("analyzers.detection.consistency_validator.EvidenceExtractor") as MockExt:
            m = MagicMock()
            m.extract.return_value = failed_ev
            MockExt.return_value = m

            with patch("os.getenv", return_value="key"):
                validator = ConsistencyValidator()
                analysis = validator.analyze(trace)

        rule_ids = [e.rule_match.rule_id for e in analysis.evidence if e.rule_match]
        if "verifier_passthrough_v1" not in rule_ids and "claim_drift_v1" not in rule_ids:
            print(
                f"  {PASS} [10] ConsistencyValidator skipped extraction rules when extraction_failed=True"
            )
            passed += 1
        else:
            print(f"  {FAIL} [10] ConsistencyValidator fired rules despite failure: {rule_ids}")
    except Exception as e:
        print(f"  {FAIL} [10] ConsistencyValidator skip error: {e}")

    # ── Test 11: LLMExplainer fallback model invocation ──────────────────────
    total += 1
    try:
        bundle = AnalysisBundle(
            run_id="r1",
            primary_cause=FailureCategory.REASONING,
            priority_level=PriorityLevel.P2,
            primary_agent="writer",
            grounded=False,
        )
        with patch("os.getenv", return_value="fake_key"):
            explainer = LLMExplainer()
            explainer._cache = MagicMock()
            explainer._cache.get.return_value = None

            mock_primary = MagicMock()
            mock_primary.with_structured_output.side_effect = RuntimeError("Primary explainer down")

            mock_fallback = MagicMock()
            mock_struct = MagicMock()
            mock_resp = MagicMock(summary="Fallback summary", suggested_fix="Fallback fix")
            mock_struct.invoke.return_value = mock_resp
            mock_fallback.with_structured_output.return_value = mock_struct

            explainer._llm = mock_primary
            explainer._fallback_llm = mock_fallback
            explainer._fallback_model_name = "llama-3.3-70b-versatile"

            res_b = explainer.explain(bundle)
            if res_b.summary == "Fallback summary" and mock_fallback.with_structured_output.called:
                print(f"  {PASS} [11] LLMExplainer invoked fallback model when primary failed")
                passed += 1
            else:
                print(
                    f"  {FAIL} [11] LLMExplainer fallback model call failed: summary={res_b.summary}"
                )
    except Exception as e:
        print(f"  {FAIL} [11] Explainer fallback model error: {e}")

    # ── Test 12: LLMExplainer degradation on complete API failure ────────────
    total += 1
    try:
        bundle = AnalysisBundle(
            run_id="r1",
            primary_cause=FailureCategory.REASONING,
            priority_level=PriorityLevel.P2,
            primary_agent="writer",
            grounded=False,
        )
        with patch("os.getenv", return_value="fake_key"):
            explainer = LLMExplainer()
            explainer._cache = MagicMock()
            explainer._cache.get.return_value = None

            explainer._llm = MagicMock()
            explainer._llm.with_structured_output.side_effect = RuntimeError(
                "All LLM services down"
            )
            explainer._fallback_llm = None

            res_b = explainer.explain(bundle)
            if res_b.summary and res_b.suggested_fix:
                print(
                    f"  {PASS} [12] LLMExplainer degrades gracefully to rule-based explanation on total failure"
                )
                passed += 1
            else:
                print(f"  {FAIL} [12] Explainer total failure did not generate fallback summary")
    except Exception as e:
        print(f"  {FAIL} [12] Explainer total failure error: {e}")

    print(f"\n{passed}/{total} verifications passed.")
    if passed == total:
        print("✅ Day 23 Verification Complete — All 12 checks passed!")
        return 0
    else:
        print(f"❌ {total - passed} verifications failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
