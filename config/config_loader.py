"""
config/config_loader.py
Loads and exposes the master config.yaml as a typed namespace.
All modules import from here — never open config.yaml directly.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


CONFIG_PATH = Path(__file__).parent / "config.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Load and cache config.yaml. Returns the full config as a dict."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.yaml not found at {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get(section: str, key: str | None = None, default: Any = None) -> Any:
    """
    Fetch a value from config.yaml.

    Usage:
        get("llm", "model")          → "llama-3.3-70b-versatile"
        get("storage")               → full storage section dict
        get("metrics", "latency_threshold_ms")  → 5000
    """
    cfg = load_config()
    section_data = cfg.get(section, {})
    if key is None:
        return section_data
    return section_data.get(key, default)
