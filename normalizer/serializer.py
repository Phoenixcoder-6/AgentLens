"""
normalizer/serializer.py — Safe Type Serializer
================================================
Day 7: Handles serialization of every type that can appear inside a
raw captured AgentStep event, converting it to a JSON-safe Python native.

Supported types:
    datetime / date / time  → ISO 8601 string
    numpy scalars           → Python int / float
    numpy ndarray           → nested list
    pandas DataFrame        → list of row dicts
    pandas Series           → list of values
    Pydantic BaseModel      → dict via .model_dump()
    Enum                    → .value
    bytes                   → base64 string
    str that looks like JSON → parsed dict/list (deserialize direction)
    Everything else          → str(value) fallback

Two directions are supported:
    to_serializable(value)  → convert TO JSON-safe Python native (for storage)
    safe_loads(text)        → parse a raw JSON string safely (for reading back)
"""

from __future__ import annotations

import base64
import json
from datetime import date, datetime, time, timezone
from enum import Enum
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Optional imports — these are all soft dependencies.
# AgentLens works without numpy/pandas; the serializer gracefully skips them.
# ─────────────────────────────────────────────────────────────────────────────

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

try:
    from pydantic import BaseModel as PydanticBaseModel
    _PYDANTIC_AVAILABLE = True
except ImportError:
    _PYDANTIC_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Direction 1: to_serializable — convert any value to JSON-safe native
# ─────────────────────────────────────────────────────────────────────────────

def to_serializable(value: Any) -> Any:
    """
    Recursively convert a value to a JSON-serializable Python native.

    Rules (applied in order):
        None, bool, int, float, str  → returned as-is
        datetime                     → ISO 8601 string (UTC-aware)
        date / time                  → ISO 8601 string
        Enum                         → .value
        bytes                        → base64-encoded string
        Pydantic BaseModel           → dict via .model_dump()
        numpy scalar                 → int or float
        numpy ndarray                → nested list
        pandas DataFrame             → list of row dicts
        pandas Series                → list of values
        dict                         → recursively processed
        list / tuple / set           → recursively processed list
        anything else                → str(value) fallback
    """
    # ── Primitives — returned immediately ────────────────────────────────────
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    # ── datetime (must come before date — datetime is a subclass of date) ───
    if isinstance(value, datetime):
        if value.tzinfo is None:
            # Treat naive datetimes as UTC
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()

    # ── date / time ──────────────────────────────────────────────────────────
    if isinstance(value, (date, time)):
        return value.isoformat()

    # ── Enum ─────────────────────────────────────────────────────────────────
    if isinstance(value, Enum):
        return value.value

    # ── bytes ────────────────────────────────────────────────────────────────
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("utf-8")

    # ── Pydantic BaseModel ────────────────────────────────────────────────────
    if _PYDANTIC_AVAILABLE and isinstance(value, PydanticBaseModel):
        return to_serializable(value.model_dump())

    # ── numpy ─────────────────────────────────────────────────────────────────
    if _NUMPY_AVAILABLE:
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.ndarray):
            return to_serializable(value.tolist())

    # ── pandas ────────────────────────────────────────────────────────────────
    if _PANDAS_AVAILABLE:
        if isinstance(value, pd.DataFrame):
            return to_serializable(value.to_dict(orient="records"))
        if isinstance(value, pd.Series):
            return to_serializable(value.tolist())
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, pd.NA.__class__):
            return None

    # ── Containers — recurse ─────────────────────────────────────────────────
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_serializable(v) for v in value]

    # ── Final fallback ────────────────────────────────────────────────────────
    return str(value)


# ─────────────────────────────────────────────────────────────────────────────
# Direction 2: safe_loads — parse a raw JSON string safely
# ─────────────────────────────────────────────────────────────────────────────

def safe_loads(text: Any) -> Any:
    """
    Safely parse a value that might be a JSON string, a plain string, or
    already a dict/list.

    Cases:
        dict / list        → returned as-is (already parsed)
        valid JSON string  → parsed to dict / list
        non-JSON string    → returned as plain string
        anything else      → returned as-is
    """
    if isinstance(text, (dict, list)):
        return text

    if isinstance(text, str):
        stripped = text.strip()
        if stripped.startswith(("{", "[")):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass
        return text

    return text


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: safe_dumps — serialize any value to a JSON string
# ─────────────────────────────────────────────────────────────────────────────

def safe_dumps(value: Any, indent: int = None) -> str:
    """
    Serialize any value to a JSON string using to_serializable as the encoder.
    Never raises — falls back to str(value) on total failure.
    """
    try:
        return json.dumps(to_serializable(value), indent=indent, ensure_ascii=False)
    except Exception:
        return str(value)
