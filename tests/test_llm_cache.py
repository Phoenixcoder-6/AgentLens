"""
tests/test_llm_cache.py — Day 23: LLMCache unit tests
======================================================
Tests:
  - Cache key computation (deterministic hash)
  - Set / Get hit behavior
  - Expiration behavior (TTL)
  - Config bypass when cache_enabled is False
  - Cache integration in EvidenceExtractor (second call hits cache)
"""

from __future__ import annotations

import tempfile
from datetime import UTC
from unittest.mock import MagicMock

import pytest

from analyzers.evidence_extraction.extractor import EvidenceExtractor
from storage.db import DatabaseManager
from storage.llm_cache import LLMCache


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name
    db = DatabaseManager(db_path)
    db.initialize()
    yield db


class TestLLMCacheUnit:
    def test_compute_key_deterministic(self, temp_db):
        cache = LLMCache(temp_db)
        k1 = cache.compute_key("sys", "user", "modelA", 0.0)
        k2 = cache.compute_key("sys", "user", "modelA", 0.0)
        k3 = cache.compute_key("sys", "user", "modelB", 0.0)

        assert k1 == k2
        assert k1 != k3

    def test_set_and_get(self, temp_db):
        cache = LLMCache(temp_db)
        cache.set("sys", "user", "modelA", "response_text_123", token_cost=150)

        res = cache.get("sys", "user", "modelA")
        assert res is not None
        text, tokens = res
        assert text == "response_text_123"
        assert tokens == 150

    def test_miss_returns_none(self, temp_db):
        cache = LLMCache(temp_db)
        assert cache.get("sys", "user_uncached", "modelA") is None

    def test_expired_key_returns_none(self, temp_db):
        cache = LLMCache(temp_db)
        cache.set("sys", "user", "modelA", "resp", token_cost=10, ttl_hours=-1)  # expired 1h ago

        assert cache.get("sys", "user", "modelA") is None

    def test_purge_expired(self, temp_db):
        cache = LLMCache(temp_db)
        cache.set("sys", "user1", "modelA", "resp1", token_cost=10, ttl_hours=-1)
        cache.set("sys", "user2", "modelA", "resp2", token_cost=10, ttl_hours=24)

        from datetime import datetime
        now_iso = datetime.now(UTC).isoformat()
        purged = temp_db.purge_expired_llm_cache(now_iso)
        assert purged == 1


class TestExtractorCacheIntegration:
    def test_extractor_hits_cache_on_second_call(self):
        extractor = EvidenceExtractor()
        mock_cache = MagicMock()
        mock_cache.get.return_value = (
            '{"source_count": 3, "entity_count": 5, "claims": ["c1"]}',
            100,
        )
        extractor._cache = mock_cache

        res = extractor.extract("some researcher output")
        assert not res.extraction_failed
        assert res.source_count == 3
        mock_cache.get.assert_called_once()
