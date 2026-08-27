"""
storage/llm_cache.py — Day 23: SQLite LLM Response Cache Manager
===================================================================
Provides deterministic prompt-response caching to eliminate duplicate LLM calls,
reduce token costs, and prevent Groq 429 rate limit hits.

Cache key formula:
  sha256(f"{model}:{temperature}:{system_prompt}:{user_prompt}")

Features:
  - Configurable TTL (default 24h from config.yaml)
  - Auto-initialization of DB schema
  - Graceful bypass if disabled or on DB error
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

from config import config_loader
from storage.db import DatabaseManager


class LLMCache:
    """
    Manages caching of LLM responses in SQLite (`llm_cache` table).

    Usage:
        cache = LLMCache()
        cached = cache.get(system_prompt, user_prompt, model="openai/gpt-oss-120b")
        if cached:
            raw_text, tokens = cached
        else:
            raw_text, tokens = call_llm(...)
            cache.set(system_prompt, user_prompt, model="openai/gpt-oss-120b", response_text=raw_text, token_cost=tokens)
    """

    def __init__(self, db: DatabaseManager | None = None) -> None:
        self.db = db or DatabaseManager()
        self.db.initialize()

    def _is_enabled(self) -> bool:
        return bool(config_loader.get("llm", "cache_enabled", True))

    def _get_ttl_hours(self) -> int:
        return int(config_loader.get("llm", "cache_ttl_hours", 24))

    def compute_key(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.0,
    ) -> str:
        """Compute SHA256 hex digest for prompt inputs."""
        payload = f"{model}:{temperature:.2f}:{system_prompt}:{user_prompt}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.0,
    ) -> tuple[str, int] | None:
        """
        Retrieve cached response if enabled and unexpired.
        Returns (response_text, token_cost) or None.
        """
        if not self._is_enabled():
            return None

        key = self.compute_key(system_prompt, user_prompt, model, temperature)
        now_iso = datetime.now(UTC).isoformat()
        try:
            return self.db.get_cached_llm_response(key, now_iso)
        except Exception:
            return None

    def set(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        response_text: str,
        token_cost: int,
        temperature: float = 0.0,
        ttl_hours: int | None = None,
    ) -> None:
        """Store response text in cache with TTL."""
        if not self._is_enabled():
            return

        ttl = ttl_hours if ttl_hours is not None else self._get_ttl_hours()
        key = self.compute_key(system_prompt, user_prompt, model, temperature)
        now = datetime.now(UTC)
        created_at = now.isoformat()
        expires_at = (now + timedelta(hours=ttl)).isoformat()
        combined_prompt = f"SYSTEM: {system_prompt}\nUSER: {user_prompt}"

        try:
            self.db.set_cached_llm_response(
                cache_key=key,
                prompt=combined_prompt[:2000],
                model=model,
                response_text=response_text,
                token_cost=token_cost,
                created_at=created_at,
                expires_at=expires_at,
            )
        except Exception:
            pass
