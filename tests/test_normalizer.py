"""
tests/test_normalizer.py — Day 7: Normalizer + Serializer tests
================================================================
Verifies:
    - to_serializable handles datetime, Enum, bytes, dict, list, Pydantic
    - to_serializable handles numpy and pandas when available
    - safe_loads correctly parses JSON strings and passes dicts through
    - Normalizer.normalize_step produces correct NormalizedStep
    - schema_version is stamped on every normalized record
    - timestamp is always UTC-aware datetime after normalization
    - status is always StepStatus enum after normalization
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, date, time
from enum import Enum

import pytest

from normalizer.serializer import to_serializable, safe_loads, safe_dumps
from normalizer.normalizer import Normalizer, NormalizedStep, NormalizedRun
from schema.models import (
    AgentStep, RunTrace, HandoffState, StepStatus,
    TokenUsage, SCHEMA_VERSION
)


# ─────────────────────────────────────────────────────────────────────────────
# Serializer — to_serializable
# ─────────────────────────────────────────────────────────────────────────────

class TestToSerializable:

    def test_none_passthrough(self):
        assert to_serializable(None) is None

    def test_bool_passthrough(self):
        assert to_serializable(True) is True
        assert to_serializable(False) is False

    def test_int_passthrough(self):
        assert to_serializable(42) == 42

    def test_float_passthrough(self):
        assert to_serializable(3.14) == 3.14

    def test_str_passthrough(self):
        assert to_serializable("hello") == "hello"

    def test_datetime_with_tz_to_iso(self):
        dt = datetime(2026, 7, 17, 12, 0, 0, tzinfo=timezone.utc)
        result = to_serializable(dt)
        assert isinstance(result, str)
        assert "2026-07-17" in result
        assert "12:00:00" in result

    def test_naive_datetime_treated_as_utc(self):
        dt = datetime(2026, 7, 17, 12, 0, 0)  # no tzinfo
        result = to_serializable(dt)
        assert isinstance(result, str)
        assert "+00:00" in result or "Z" in result.upper() or "UTC" in result or "00:00" in result

    def test_date_to_iso(self):
        result = to_serializable(date(2026, 7, 17))
        assert result == "2026-07-17"

    def test_time_to_iso(self):
        result = to_serializable(time(14, 30, 0))
        assert result == "14:30:00"

    def test_enum_to_value(self):
        class Color(Enum):
            RED = "red"
            BLUE = "blue"
        assert to_serializable(Color.RED) == "red"

    def test_bytes_to_base64(self):
        result = to_serializable(b"hello")
        import base64
        assert result == base64.b64encode(b"hello").decode()

    def test_dict_recursive(self):
        dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
        result = to_serializable({"key": dt, "nested": {"val": 42}})
        assert isinstance(result["key"], str)
        assert result["nested"]["val"] == 42

    def test_list_recursive(self):
        dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
        result = to_serializable([dt, 42, "hello"])
        assert isinstance(result[0], str)
        assert result[1] == 42

    def test_tuple_to_list(self):
        result = to_serializable((1, 2, 3))
        assert result == [1, 2, 3]

    def test_set_to_list(self):
        result = to_serializable({42})
        assert result == [42]

    def test_pydantic_model_to_dict(self):
        tokens = TokenUsage(prompt=10, completion=20, total=30)
        result = to_serializable(tokens)
        assert isinstance(result, dict)
        assert result["prompt"] == 10
        assert result["completion"] == 20

    def test_step_status_enum(self):
        result = to_serializable(StepStatus.SUCCESS)
        assert result == "SUCCESS"

    def test_unknown_type_falls_back_to_str(self):
        class WeirdObj:
            def __str__(self):
                return "weird"
        result = to_serializable(WeirdObj())
        assert result == "weird"

    def test_result_is_json_serializable(self):
        """Final output must always be serializable by stdlib json."""
        dt = datetime(2026, 7, 17, tzinfo=timezone.utc)
        tokens = TokenUsage(prompt=5, completion=10, total=15)
        value = {
            "dt": dt,
            "tokens": tokens,
            "items": [1, 2, None],
            "nested": {"flag": True},
        }
        result = to_serializable(value)
        # Should not raise
        json_str = json.dumps(result)
        assert json_str is not None


class TestToSerializableNumpy:
    """Numpy tests — skipped if numpy is not installed."""

    @pytest.fixture(autouse=True)
    def skip_if_no_numpy(self):
        try:
            import numpy as np
            self.np = np
        except ImportError:
            pytest.skip("numpy not installed")

    def test_numpy_int64(self):
        result = to_serializable(self.np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_numpy_float32(self):
        result = to_serializable(self.np.float32(3.14))
        assert isinstance(result, float)

    def test_numpy_bool(self):
        result = to_serializable(self.np.bool_(True))
        assert result is True

    def test_numpy_array_to_list(self):
        arr = self.np.array([1, 2, 3])
        result = to_serializable(arr)
        assert result == [1, 2, 3]

    def test_nested_numpy_in_dict(self):
        d = {"count": self.np.int64(10), "score": self.np.float32(0.95)}
        result = to_serializable(d)
        assert isinstance(result["count"], int)
        assert isinstance(result["score"], float)


class TestToSerializablePandas:
    """Pandas tests — skipped if pandas is not installed."""

    @pytest.fixture(autouse=True)
    def skip_if_no_pandas(self):
        try:
            import pandas as pd
            self.pd = pd
        except ImportError:
            pytest.skip("pandas not installed")

    def test_dataframe_to_list_of_dicts(self):
        df = self.pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = to_serializable(df)
        assert isinstance(result, list)
        assert result[0] == {"a": 1, "b": 3}

    def test_series_to_list(self):
        s = self.pd.Series([10, 20, 30])
        result = to_serializable(s)
        assert result == [10, 20, 30]

    def test_timestamp_to_iso(self):
        ts = self.pd.Timestamp("2026-07-17 12:00:00")
        result = to_serializable(ts)
        assert isinstance(result, str)
        assert "2026-07-17" in result


# ─────────────────────────────────────────────────────────────────────────────
# Serializer — safe_loads
# ─────────────────────────────────────────────────────────────────────────────

class TestSafeLoads:

    def test_dict_passthrough(self):
        d = {"key": "value"}
        assert safe_loads(d) == d

    def test_list_passthrough(self):
        lst = [1, 2, 3]
        assert safe_loads(lst) == lst

    def test_valid_json_object_string(self):
        result = safe_loads('{"key": "value", "count": 5}')
        assert result == {"key": "value", "count": 5}

    def test_valid_json_array_string(self):
        result = safe_loads('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_plain_string_returned_as_is(self):
        result = safe_loads("hello world")
        assert result == "hello world"

    def test_invalid_json_string_returned_as_is(self):
        result = safe_loads("{not valid json}")
        assert result == "{not valid json}"

    def test_empty_string_returned_as_is(self):
        result = safe_loads("")
        assert result == ""

    def test_none_returned_as_is(self):
        result = safe_loads(None)
        assert result is None

    def test_integer_returned_as_is(self):
        result = safe_loads(42)
        assert result == 42


# ─────────────────────────────────────────────────────────────────────────────
# Normalizer — normalize_step
# ─────────────────────────────────────────────────────────────────────────────

def _make_agent_step(**overrides) -> AgentStep:
    """Helper: create a minimal AgentStep with sensible defaults."""
    defaults = dict(
        run_id="run_test01",
        step=1,
        agent="researcher",
        input='{"topic": "AI", "research_findings": "", "source_count": 0}',
        output='{"research_findings": "SOURCES:\\n- Book A", "source_count": 1}',
        latency_ms=1234.5,
        status=StepStatus.SUCCESS,
        prompt="added=['research_findings', 'source_count']",
        timestamp=datetime(2026, 7, 17, 12, 0, 0, tzinfo=timezone.utc),
        handoff=HandoffState(
            input_state={"topic": "AI", "research_findings": "", "source_count": 0},
            filtered_state={"research_findings": "SOURCES:\\n- Book A", "source_count": 1},
            output_state={"topic": "AI", "research_findings": "SOURCES:\\n- Book A", "source_count": 1},
        ),
    )
    defaults.update(overrides)
    return AgentStep(**defaults)


class TestNormalizerStep:

    def setup_method(self):
        self.normalizer = Normalizer()

    def test_schema_version_stamped(self):
        step = _make_agent_step()
        result = self.normalizer.normalize_step(step)
        assert result.schema_version == SCHEMA_VERSION

    def test_run_id_preserved(self):
        step = _make_agent_step(run_id="run_abc123")
        result = self.normalizer.normalize_step(step)
        assert result.run_id == "run_abc123"

    def test_agent_name_preserved(self):
        step = _make_agent_step(agent="writer")
        result = self.normalizer.normalize_step(step)
        assert result.agent == "writer"

    def test_input_state_is_dict(self):
        step = _make_agent_step()
        result = self.normalizer.normalize_step(step)
        assert isinstance(result.input_state, dict)
        assert result.input_state["topic"] == "AI"

    def test_output_state_is_dict(self):
        step = _make_agent_step()
        result = self.normalizer.normalize_step(step)
        assert isinstance(result.output_state, dict)
        assert result.output_state["source_count"] == 1

    def test_timestamp_is_utc_aware_datetime(self):
        step = _make_agent_step()
        result = self.normalizer.normalize_step(step)
        assert isinstance(result.timestamp, datetime)
        assert result.timestamp.tzinfo is not None

    def test_naive_timestamp_converted_to_utc(self):
        step = _make_agent_step(timestamp=datetime(2026, 7, 17, 12, 0, 0))  # no tzinfo
        result = self.normalizer.normalize_step(step)
        assert result.timestamp.tzinfo is not None

    def test_timestamp_string_parsed(self):
        step = _make_agent_step(timestamp="2026-07-17T12:00:00+00:00")
        result = self.normalizer.normalize_step(step)
        assert isinstance(result.timestamp, datetime)

    def test_status_is_step_status_enum(self):
        step = _make_agent_step(status=StepStatus.SUCCESS)
        result = self.normalizer.normalize_step(step)
        assert isinstance(result.status, StepStatus)
        assert result.status == StepStatus.SUCCESS

    def test_latency_is_float(self):
        step = _make_agent_step(latency_ms=999)
        result = self.normalizer.normalize_step(step)
        assert isinstance(result.latency_ms, float)
        assert result.latency_ms == 999.0

    def test_diff_summary_preserved(self):
        step = _make_agent_step(prompt="added=['source_count']")
        result = self.normalizer.normalize_step(step)
        assert result.diff_summary == "added=['source_count']"

    def test_error_preserved(self):
        step = _make_agent_step(
            status=StepStatus.ERROR,
            error="ValueError: Something broke"
        )
        result = self.normalizer.normalize_step(step)
        assert result.error == "ValueError: Something broke"

    def test_raw_strings_preserved(self):
        step = _make_agent_step()
        result = self.normalizer.normalize_step(step)
        assert isinstance(result.raw_input, str)
        assert isinstance(result.raw_output, str)


# ─────────────────────────────────────────────────────────────────────────────
# Normalizer — normalize_run
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizerRun:

    def setup_method(self):
        self.normalizer = Normalizer()

    def _make_run(self) -> RunTrace:
        steps = [
            _make_agent_step(step=1, agent="researcher"),
            _make_agent_step(step=2, agent="writer"),
            _make_agent_step(step=3, agent="verifier"),
        ]
        return RunTrace(
            run_id="run_fulltest",
            workflow="research_report_pipeline",
            timestamp=datetime(2026, 7, 17, 12, 0, 0, tzinfo=timezone.utc),
            steps=steps,
            total_latency_ms=3703.5,
            total_tokens=0,
            status=StepStatus.SUCCESS,
        )

    def test_schema_version_stamped_on_run(self):
        run = self._make_run()
        result = self.normalizer.normalize_run(run)
        assert result.schema_version == SCHEMA_VERSION

    def test_all_steps_normalized(self):
        run = self._make_run()
        result = self.normalizer.normalize_run(run)
        assert len(result.steps) == 3

    def test_each_step_has_schema_version(self):
        run = self._make_run()
        result = self.normalizer.normalize_run(run)
        for step in result.steps:
            assert step.schema_version == SCHEMA_VERSION

    def test_agent_names_in_order(self):
        run = self._make_run()
        result = self.normalizer.normalize_run(run)
        assert [s.agent for s in result.steps] == ["researcher", "writer", "verifier"]

    def test_total_latency_is_float(self):
        run = self._make_run()
        result = self.normalizer.normalize_run(run)
        assert isinstance(result.total_latency_ms, float)

    def test_run_timestamp_is_utc_datetime(self):
        run = self._make_run()
        result = self.normalizer.normalize_run(run)
        assert isinstance(result.timestamp, datetime)
        assert result.timestamp.tzinfo is not None
