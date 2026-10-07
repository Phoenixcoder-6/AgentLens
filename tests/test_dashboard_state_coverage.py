"""
Additional unit tests for dashboard.state.

Focus:
- DB/data-layer helpers
- filtering/sorting
- aggregation
- metrics/timeline helpers
- rule statistics
- verdict/cost helpers

Streamlit UI is intentionally not tested here.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _level(value: str):
    return SimpleNamespace(value=value)


def _bundle(
    run_id: str = "r1",
    priority: str = "P2",
    primary_agent: str | None = "writer",
    cause: str = "information_loss",
):
    return SimpleNamespace(
        run_id=run_id,
        priority_level=_level(priority),
        primary_agent=primary_agent,
        primary_cause=_level(cause),
    )


def _state(
    bundle=None,
    loss_result=None,
):
    return SimpleNamespace(
        bundle=bundle,
        loss_result=loss_result,
    )


def _db():
    db = MagicMock()
    db.initialize.return_value = None
    return db


# ---------------------------------------------------------------------------
# _extract_topic
# ---------------------------------------------------------------------------


class TestExtractTopicCoverage:
    def test_empty_string(self):
        from dashboard.state import _extract_topic

        assert _extract_topic("") == ""

    def test_topic_from_input_state(self):
        from dashboard.state import _extract_topic

        trace = {
            "steps": [
                {
                    "handoff": {
                        "input_state": {
                            "topic": "Artificial Intelligence",
                        }
                    }
                }
            ]
        }

        assert _extract_topic(json.dumps(trace)) == "Artificial Intelligence"

    def test_topic_from_output_state(self):
        from dashboard.state import _extract_topic

        trace = {
            "steps": [
                {
                    "handoff": {
                        "output_state": {
                            "topic": "Machine Learning",
                        }
                    }
                }
            ]
        }

        assert _extract_topic(json.dumps(trace)) == "Machine Learning"

    def test_topic_from_filtered_state(self):
        from dashboard.state import _extract_topic

        trace = {
            "steps": [
                {
                    "handoff": {
                        "filtered_state": {
                            "topic": "RAG systems",
                        }
                    }
                }
            ]
        }

        assert _extract_topic(json.dumps(trace)) == "RAG systems"

    def test_nested_json_handoff_and_state(self):
        from dashboard.state import _extract_topic

        trace = {
            "steps": [
                {
                    "handoff": json.dumps(
                        {
                            "input_state": json.dumps(
                                {"topic": "Agentic AI"}
                            )
                        }
                    )
                }
            ]
        }

        assert _extract_topic(json.dumps(trace)) == "Agentic AI"

    def test_top_level_initial_state(self):
        from dashboard.state import _extract_topic

        trace = {
            "steps": [],
            "initial_state": {
                "topic": "Generative AI",
            },
        }

        assert _extract_topic(json.dumps(trace)) == "Generative AI"

    def test_string_initial_state(self):
        from dashboard.state import _extract_topic

        trace = {
            "steps": [],
            "initial_state": json.dumps(
                {"topic": "Large Language Models"}
            ),
        }

        assert _extract_topic(json.dumps(trace)) == "Large Language Models"

    def test_invalid_json_returns_empty(self):
        from dashboard.state import _extract_topic

        assert _extract_topic("{not-valid-json") == ""

    def test_invalid_nested_handoff_continues(self):
        from dashboard.state import _extract_topic

        trace = {
            "steps": [
                {
                    "handoff": "{bad-json",
                },
                {
                    "handoff": {
                        "input_state": {
                            "topic": "Fallback Topic",
                        }
                    },
                },
            ]
        }

        assert _extract_topic(json.dumps(trace)) == "Fallback Topic"

    def test_topic_is_truncated_to_60_chars(self):
        from dashboard.state import _extract_topic

        topic = "A" * 100

        trace = {
            "steps": [
                {
                    "handoff": {
                        "input_state": {
                            "topic": topic,
                        }
                    }
                }
            ]
        }

        result = _extract_topic(json.dumps(trace))

        assert len(result) == 60


# ---------------------------------------------------------------------------
# get_db
# ---------------------------------------------------------------------------


class TestGetDb:
    def test_get_db_initializes_database_once(self):
        import dashboard.state as state

        fake_db = MagicMock()

        with patch.object(state, "_db", None), patch(
            "dashboard.state.DatabaseManager",
            return_value=fake_db,
        ):
            result = state.get_db()

        assert result is fake_db
        fake_db.initialize.assert_called_once()


# ---------------------------------------------------------------------------
# get_steps
# ---------------------------------------------------------------------------


class TestGetStepsCoverage:
    def test_get_steps_maps_defaults(self):
        import dashboard.state as state

        db = _db()
        db.get_steps_for_run.return_value = [
            {
                "step": 1,
                "agent": "researcher",
                "status": "SUCCESS",
                "latency_ms": 120,
                "tokens_prompt": 20,
                "tokens_completion": 30,
                "tokens_total": 50,
            },
            {
                "step": 2,
                "agent": "writer",
            },
        ]

        with patch.object(state, "get_db", return_value=db):
            result = state.get_steps("run-1")

        assert len(result) == 2
        assert result[0].agent == "researcher"
        assert result[0].tokens_total == 50

        assert result[1].status == "unknown"
        assert result[1].latency_ms == 0
        assert result[1].tokens_prompt == 0
        assert result[1].tokens_completion == 0
        assert result[1].tokens_total == 0


# ---------------------------------------------------------------------------
# get_trace_steps / _parse_handoff
# ---------------------------------------------------------------------------


class TestTraceHelpers:
    def test_get_trace_steps_returns_empty_for_missing_run(self):
        import dashboard.state as state

        db = _db()
        db.get_run.return_value = None

        with patch.object(state, "get_db", return_value=db):
            assert state.get_trace_steps("missing") == []

    def test_get_trace_steps_returns_empty_without_trace(self):
        import dashboard.state as state

        db = _db()
        db.get_run.return_value = {"run_id": "r1", "trace_json": ""}

        with patch.object(state, "get_db", return_value=db):
            assert state.get_trace_steps("r1") == []

    def test_get_trace_steps_reads_steps(self):
        import dashboard.state as state

        steps = [{"agent": "researcher", "step": 1}]
        db = _db()
        db.get_run.return_value = {
            "trace_json": json.dumps({"steps": steps})
        }

        with patch.object(state, "get_db", return_value=db):
            assert state.get_trace_steps("r1") == steps

    def test_parse_handoff_none(self):
        from dashboard.state import _parse_handoff

        assert _parse_handoff(None) == {}

    def test_parse_handoff_valid_json(self):
        from dashboard.state import _parse_handoff

        assert _parse_handoff('{"input_state": {"x": 1}}') == {
            "input_state": {"x": 1}
        }

    def test_parse_handoff_invalid_json(self):
        from dashboard.state import _parse_handoff

        assert _parse_handoff("{bad}") == {}

    def test_parse_handoff_non_dict(self):
        from dashboard.state import _parse_handoff

        assert _parse_handoff(["not", "a", "dict"]) == {}


# ---------------------------------------------------------------------------
# list_runs
# ---------------------------------------------------------------------------


class TestListRunsCoverage:
    def _make_db(self):
        db = _db()

        db.list_runs.return_value = [
            {
                "run_id": "r1",
                "workflow": "research_workflow",
                "timestamp": "2026-10-01T10:00:00",
                "status": "SUCCESS",
            },
            {
                "run_id": "r2",
                "workflow": "writer_workflow",
                "timestamp": "2026-10-02T11:00:00",
                "status": "FAILED",
            },
        ]

        db.get_steps_for_run.side_effect = lambda run_id: {
            "r1": [
                {
                    "agent": "researcher",
                    "latency_ms": 100,
                    "tokens_total": 50,
                },
                {
                    "agent": "writer",
                    "latency_ms": 200,
                    "tokens_total": 100,
                },
            ],
            "r2": [
                {
                    "agent": "writer",
                    "latency_ms": 500,
                    "tokens_total": 200,
                }
            ],
        }[run_id]

        db.get_run.side_effect = lambda run_id: {
            "r1": {
                "trace_json": json.dumps(
                    {
                        "steps": [
                            {
                                "handoff": {
                                    "input_state": {
                                        "topic": "AI research"
                                    }
                                }
                            }
                        ]
                    }
                )
            },
            "r2": {"trace_json": ""},
        }[run_id]

        return db

    def test_list_runs_basic(self):
        import dashboard.state as state

        db = self._make_db()

        with patch.object(state, "get_db", return_value=db):
            result = state.list_runs()

        assert len(result) == 2
        assert result[0].run_id == "r1"
        assert result[0].topic == "AI research"
        assert result[0].latency_ms == 300
        assert result[0].tokens_total == 150
        assert result[0].step_count == 2
        assert result[1].topic == "writer_workflow"

    def test_agent_filter(self):
        import dashboard.state as state

        db = self._make_db()

        with patch.object(state, "get_db", return_value=db):
            result = state.list_runs(agent_filter="researcher")

        assert [r.run_id for r in result] == ["r1"]

    def test_agent_filter_excludes_all(self):
        import dashboard.state as state

        db = self._make_db()

        with patch.object(state, "get_db", return_value=db):
            result = state.list_runs(agent_filter="verifier")

        assert result == []

    def test_date_filters(self):
        import dashboard.state as state

        db = self._make_db()

        with patch.object(state, "get_db", return_value=db):
            result = state.list_runs(
                date_from="2026-10-02",
                date_to="2026-10-02",
            )

        assert [r.run_id for r in result] == ["r2"]

    def test_priority_sorting_uses_cached_verdict(self):
        import dashboard.state as state

        db = self._make_db()

        state1 = _state(_bundle("r1", "P4"))
        state2 = _state(_bundle("r2", "P1"))

        with (
            patch.object(state, "get_db", return_value=db),
            patch.object(
                state,
                "_analysis_cache",
                {"r1": state1, "r2": state2},
            ),
        ):
            result = state.list_runs(sort_by="priority")

        assert [r.run_id for r in result] == ["r2", "r1"]

    def test_latency_sorting(self):
        import dashboard.state as state

        db = self._make_db()

        with patch.object(state, "get_db", return_value=db):
            result = state.list_runs(sort_by="latency")

        assert [r.run_id for r in result] == ["r2", "r1"]

    def test_verdict_filter(self):
        import dashboard.state as state

        db = self._make_db()

        with patch.object(state, "get_db", return_value=db), patch.object(
            state,
            "_analysis_cache",
            {
                "r1": _state(_bundle("r1", "P2")),
                "r2": _state(_bundle("r2", "P4")),
            },
        ):
            result = state.list_runs(verdict_filter="P2")

        assert [r.run_id for r in result] == ["r1"]


# ---------------------------------------------------------------------------
# get_unique_agents
# ---------------------------------------------------------------------------


class TestUniqueAgents:
    def test_returns_sorted_unique_agents(self):
        import dashboard.state as state

        db = _db()
        db.list_runs.return_value = [
            {"run_id": "r1"},
            {"run_id": "r2"},
        ]
        db.get_steps_for_run.side_effect = [
            [
                {"agent": "writer"},
                {"agent": "researcher"},
                {"agent": ""},
            ],
            [
                {"agent": "researcher"},
                {"agent": "verifier"},
            ],
        ]

        with patch.object(state, "get_db", return_value=db):
            result = state.get_unique_agents()

        assert result == ["researcher", "verifier", "writer"]


# ---------------------------------------------------------------------------
# get_aggregate_stats
# ---------------------------------------------------------------------------


class TestAggregateStats:
    def test_empty_runs(self):
        from dashboard.state import get_aggregate_stats

        assert get_aggregate_stats([]) == {
            "total": 0,
            "analyzed": 0,
            "p1_p2_count": 0,
            "avg_latency": 0.0,
            "total_tokens": 0,
            "top_failing_agent": None,
        }

    def test_calculates_stats_and_top_agent(self):
        import dashboard.state as state

        runs = [
            SimpleNamespace(
                run_id="r1",
                verdict_level="P1",
                latency_ms=100,
                tokens_total=50,
            ),
            SimpleNamespace(
                run_id="r2",
                verdict_level="P3",
                latency_ms=200,
                tokens_total=70,
            ),
            SimpleNamespace(
                run_id="r3",
                verdict_level="UNANALYZED",
                latency_ms=300,
                tokens_total=30,
            ),
        ]

        with patch.object(
            state,
            "_analysis_cache",
            {
                "r1": _state(_bundle(primary_agent="writer")),
                "r2": _state(_bundle(primary_agent="writer")),
                "r3": _state(None),
            },
        ):
            result = state.get_aggregate_stats(runs)

        assert result["total"] == 3
        assert result["analyzed"] == 2
        assert result["p1_p2_count"] == 1
        assert result["avg_latency"] == 200
        assert result["total_tokens"] == 150
        assert result["top_failing_agent"] == "writer"


# ---------------------------------------------------------------------------
# get_rule_stats
# ---------------------------------------------------------------------------


class TestRuleStats:
    def test_catalog_rules_without_db_stats(self):
        import dashboard.state as state

        fake_catalog = {
            "RULE-A": {
                "name": "Rule A",
                "category": "latency",
                "version": "v1",
                "description": "Latency rule",
            },
            "RULE-B": {
                "name": "Rule B",
                "category": "quality",
                "version": "v2",
                "description": "Quality rule",
            },
        }

        db = _db()
        db.get_rule_stats.return_value = []

        with (
            patch(
                "analyzers.rule_catalog.RULE_CATALOG",
                fake_catalog,
            ),
            patch.object(state, "get_db", return_value=db),
        ):
            result = state.get_rule_stats()

        assert len(result) == 2
        assert result[0]["times_fired"] == 0
        assert result[0]["version"] in ("v1", "v2")

    def test_catalog_and_uncatalogued_rules(self):
        import dashboard.state as state

        fake_catalog = {
            "RULE-A": {
                "name": "Rule A",
                "category": "latency",
                "version": "v1",
                "description": "Latency rule",
            }
        }

        db = _db()
        db.get_rule_stats.return_value = [
            {
                "rule_id": "RULE-A",
                "rule_version": "v3",
                "category": "latency",
                "times_fired": 2,
                "last_triggered": "2026-10-01",
                "example_run": "r1",
            },
            {
                "rule_id": "UNKNOWN-RULE",
                "rule_version": "v1",
                "category": "custom",
                "times_fired": 5,
                "last_triggered": "2026-10-02",
                "example_run": "r2",
            },
        ]

        with (
            patch(
                "analyzers.rule_catalog.RULE_CATALOG",
                fake_catalog,
            ),
            patch.object(state, "get_db", return_value=db),
        ):
            result = state.get_rule_stats()

        assert result[0]["rule_id"] == "UNKNOWN-RULE"
        assert result[0]["times_fired"] == 5

        known = next(r for r in result if r["rule_id"] == "RULE-A")
        assert known["version"] == "v3"
        assert known["times_fired"] == 2

    def test_category_filter(self):
        import dashboard.state as state

        fake_catalog = {
            "RULE-A": {
                "name": "Rule A",
                "category": "latency",
                "version": "v1",
                "description": "Latency",
            },
            "RULE-B": {
                "name": "Rule B",
                "category": "quality",
                "version": "v1",
                "description": "Quality",
            },
        }

        db = _db()
        db.get_rule_stats.return_value = []

        with (
            patch(
                "analyzers.rule_catalog.RULE_CATALOG",
                fake_catalog,
            ),
            patch.object(state, "get_db", return_value=db),
        ):
            result = state.get_rule_stats(category="quality")

        assert len(result) == 1
        assert result[0]["rule_id"] == "RULE-B"

    def test_db_failure_returns_catalog(self):
        import dashboard.state as state

        fake_catalog = {
            "RULE-A": {
                "name": "Rule A",
                "category": "latency",
                "version": "v1",
                "description": "Latency",
            }
        }

        db = _db()
        db.get_rule_stats.side_effect = RuntimeError("DB unavailable")

        with (
            patch(
                "analyzers.rule_catalog.RULE_CATALOG",
                fake_catalog,
            ),
            patch.object(state, "get_db", return_value=db),
        ):
            result = state.get_rule_stats()

        assert len(result) == 1
        assert result[0]["times_fired"] == 0


# ---------------------------------------------------------------------------
# cache helper
# ---------------------------------------------------------------------------


class TestCacheHelpers:
    def test_get_cached(self):
        import dashboard.state as state

        cached = _state(_bundle())

        with patch.object(
            state,
            "_analysis_cache",
            {"r1": cached},
        ):
            assert state.get_cached("r1") is cached
            assert state.get_cached("missing") is None


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


class TestMetricsData:
    def test_metrics_are_aggregated_per_agent(self):
        import dashboard.state as state

        db = _db()
        db.list_runs.return_value = [
            {"run_id": "r1"},
            {"run_id": "r2"},
        ]
        db.get_steps_for_run.side_effect = [
            [
                {
                    "agent": "researcher",
                    "latency_ms": 100,
                    "tokens_total": 20,
                },
                {
                    "agent": "writer",
                    "latency_ms": 200,
                    "tokens_total": 40,
                },
            ],
            [
                {
                    "agent": "researcher",
                    "latency_ms": 300,
                    "tokens_total": 60,
                }
            ],
        ]

        with patch.object(state, "get_db", return_value=db):
            result = state.get_metrics_data()

        assert result["researcher"]["avg_latency_ms"] == 200
        assert result["researcher"]["max_latency_ms"] == 300
        assert result["researcher"]["avg_tokens"] == 40
        assert result["researcher"]["total_tokens"] == 80
        assert result["researcher"]["run_count"] == 2

        assert result["writer"]["avg_latency_ms"] == 200
        assert result["writer"]["total_tokens"] == 40


# ---------------------------------------------------------------------------
# failure timeline / cause breakdown
# ---------------------------------------------------------------------------


class TestFailureAndCauseHelpers:
    def test_failure_timeline_counts_p1_and_p2(self):
        import dashboard.state as state

        db = _db()
        db.list_runs.return_value = [
            {
                "run_id": "r1",
                "timestamp": "2026-10-01T10:00:00",
            },
            {
                "run_id": "r2",
                "timestamp": "2026-10-01T11:00:00",
            },
            {
                "run_id": "r3",
                "timestamp": "2026-10-02T12:00:00",
            },
            {
                "run_id": "r4",
                "timestamp": "",
            },
        ]

        cache = {
            "r1": _state(_bundle(priority="P1")),
            "r2": _state(_bundle(priority="P2")),
            "r3": _state(_bundle(priority="P5")),
            "r4": _state(None),
        }

        with (
            patch.object(state, "get_db", return_value=db),
            patch.object(state, "_analysis_cache", cache),
        ):
            result = state.get_failure_timeline(days=14)

        assert result == [
            {
                "date": "2026-10-01",
                "p1": 1,
                "p2": 1,
                "total": 2,
            }
        ]

    def test_failure_timeline_respects_days(self):
        import dashboard.state as state

        db = _db()
        db.list_runs.return_value = [
            {"run_id": "r1", "timestamp": "2026-10-01T10:00:00"},
            {"run_id": "r2", "timestamp": "2026-10-02T10:00:00"},
            {"run_id": "r3", "timestamp": "2026-10-03T10:00:00"},
        ]

        cache = {
            "r1": _state(_bundle(priority="P1")),
            "r2": _state(_bundle(priority="P2")),
            "r3": _state(_bundle(priority="P1")),
        }

        with (
            patch.object(state, "get_db", return_value=db),
            patch.object(state, "_analysis_cache", cache),
        ):
            result = state.get_failure_timeline(days=2)

        assert [r["date"] for r in result] == [
            "2026-10-02",
            "2026-10-03",
        ]

    def test_cause_breakdown(self):
        import dashboard.state as state

        cache = {
            "r1": _state(
                _bundle(cause="information_loss")
            ),
            "r2": _state(
                _bundle(cause="information_loss")
            ),
            "r3": _state(
                _bundle(cause="hallucination")
            ),
            "r4": _state(None),
        }

        with patch.object(
            state,
            "_analysis_cache",
            cache,
        ):
            result = state.get_cause_breakdown()

        assert result[0] == {
            "cause": "information_loss",
            "count": 2,
        }
        assert result[1] == {
            "cause": "hallucination",
            "count": 1,
        }


# ---------------------------------------------------------------------------
# total cost
# ---------------------------------------------------------------------------


class TestCostEstimate:
    def test_total_cost_estimate(self):
        import dashboard.state as state

        db = _db()
        db.list_runs.return_value = [
            {"run_id": "r1"},
            {"run_id": "r2"},
        ]

        with (
            patch.object(state, "get_db", return_value=db),
            patch.object(
                state,
                "_total_tokens",
                side_effect=[1000, 2000],
            ),
        ):
            result = state.total_cost_estimate()

        assert result == 3000 * state.COST_PER_TOKEN


# ---------------------------------------------------------------------------
# verdict_for_bundle
# ---------------------------------------------------------------------------


class TestVerdictForBundle:
    def test_none_bundle_is_unknown(self):
        from dashboard.state import verdict_for_bundle

        assert verdict_for_bundle(None) == "UNKNOWN"

    def test_p5_is_pass(self):
        from dashboard.state import verdict_for_bundle

        assert verdict_for_bundle(
            _bundle(priority="P5")
        ) == "PASS"

    def test_cached_loss_verdict_is_returned(self):
        import dashboard.state as state

        bundle = _bundle(priority="P2")
        loss_result = SimpleNamespace(verdict="CRITICAL")

        with patch.object(
            state,
            "_analysis_cache",
            {
                "r1": _state(
                    bundle=bundle,
                    loss_result=loss_result,
                )
            },
        ):
            assert state.verdict_for_bundle(bundle) == "CRITICAL"

    def test_non_p5_without_loss_result_is_warning(self):
        import dashboard.state as state

        bundle = _bundle(priority="P2")

        with patch.object(
            state,
            "_analysis_cache",
            {"r1": _state(bundle=bundle, loss_result=None)},
        ):
            assert state.verdict_for_bundle(bundle) == "WARNING"

    def test_uncached_non_p5_is_warning(self):
        import dashboard.state as state

        bundle = _bundle(priority="P1")

        with patch.object(state, "_analysis_cache", {}):
            assert state.verdict_for_bundle(bundle) == "WARNING"

class TestComputeDiffCoverage:
    def test_compute_diff_missing_trace(self):
        import dashboard.state as state

        with patch.object(state, "_load_run_trace", return_value=None):
            result = state.compute_diff("run-a", "run-b")

        assert result.run_a == "run-a"
        assert result.run_b == "run-b"
        assert result.steps == []
        assert result.rows == []
        assert result.first_divergence == "(trace not found)"
        assert result.overall_similarity == 0.0

    def test_compute_diff_full_path(self):
        import dashboard.state as state

        trace_a = object()
        trace_b = object()

        step_a = SimpleNamespace(agent="researcher")
        step_b = SimpleNamespace(agent="researcher")

        pair = SimpleNamespace(
            agent="researcher",
            status=SimpleNamespace(value="MATCHED"),
            step_a=step_a,
            step_b=step_b,
        )

        alignment = SimpleNamespace(
            pairs=[pair],
            matched_count=1,
            missing_in_a_count=0,
            missing_in_b_count=0,
        )

        score = SimpleNamespace(
            agent="researcher",
            similarity=0.92,
            diverged=False,
            method="semantic",
        )

        sim_report = SimpleNamespace(
            scores=[score],
            first_divergence_agent=None,
            average_similarity=0.92,
        )

        db = MagicMock()
        db.get_steps_for_run.side_effect = lambda run_id: [
            {
                "agent": "researcher",
                "latency_ms": 100,
                "tokens_total": 50,
            }
        ]

        with (
            patch.object(state, "_load_run_trace", side_effect=[trace_a, trace_b]),
            patch(
                "diff_engine.align_traces",
                return_value=alignment,
            ),
            patch(
                "diff_engine.score_similarity",
                return_value=sim_report,
            ),
            patch.object(state, "get_db", return_value=db),
        ):
            result = state.compute_diff("run-a", "run-b")

        assert result.run_a == "run-a"
        assert result.run_b == "run-b"
        assert result.overall_similarity == 0.92
        assert result.first_divergence == "(none)"
        assert result.matched_count == 1
        assert result.missing_in_a_count == 0
        assert result.missing_in_b_count == 0

        assert len(result.rows) == 1
        row = result.rows[0]

        assert row.agent == "researcher"
        assert row.match_status == "MATCHED"
        assert row.lat_a == 100.0
        assert row.lat_b == 100.0
        assert row.lat_delta == 0.0
        assert row.tok_a == 50
        assert row.tok_b == 50
        assert row.tok_delta == 0
        assert row.sim == 0.92
        assert row.diverged is False
        assert row.method == "semantic"

        assert len(result.steps) == 1
        assert result.steps[0]["agent"] == "researcher"

    def test_compute_diff_missing_in_a_and_missing_in_b(self):
        import dashboard.state as state

        trace_a = object()
        trace_b = object()

        pair_a = SimpleNamespace(
            agent="researcher",
            status=SimpleNamespace(value="MISSING_IN_B"),
            step_a=SimpleNamespace(agent="researcher"),
            step_b=None,
        )

        pair_b = SimpleNamespace(
            agent="writer",
            status=SimpleNamespace(value="MISSING_IN_A"),
            step_a=None,
            step_b=SimpleNamespace(agent="writer"),
        )

        alignment = SimpleNamespace(
            pairs=[pair_a, pair_b],
            matched_count=0,
            missing_in_a_count=1,
            missing_in_b_count=1,
        )

        sim_report = SimpleNamespace(
            scores=[],
            first_divergence_agent="researcher",
            average_similarity=0.0,
        )

        db = MagicMock()
        db.get_steps_for_run.side_effect = lambda run_id: {
            "run-a": [
                {
                    "agent": "researcher",
                    "latency_ms": 150,
                    "tokens_total": 80,
                }
            ],
            "run-b": [
                {
                    "agent": "writer",
                    "latency_ms": 300,
                    "tokens_total": 120,
                }
            ],
        }[run_id]

        with (
            patch.object(state, "_load_run_trace", side_effect=[trace_a, trace_b]),
            patch(
                "diff_engine.align_traces",
                return_value=alignment,
            ),
            patch(
                "diff_engine.score_similarity",
                return_value=sim_report,
            ),
            patch.object(state, "get_db", return_value=db),
        ):
            result = state.compute_diff("run-a", "run-b")

        assert len(result.rows) == 2

        researcher = result.rows[0]
        assert researcher.match_status == "MISSING_IN_B"
        assert researcher.lat_a == 150.0
        assert researcher.lat_b == 0.0
        assert researcher.lat_delta == -150.0
        assert researcher.tok_a == 80
        assert researcher.tok_b == 0
        assert researcher.diverged is True
        assert researcher.method == "n/a"

        writer = result.rows[1]
        assert writer.match_status == "MISSING_IN_A"
        assert writer.lat_a == 0.0
        assert writer.lat_b == 300.0
        assert writer.lat_delta == 300.0
        assert writer.tok_a == 0
        assert writer.tok_b == 120
        assert writer.diverged is True
        assert writer.method == "n/a"

        assert result.missing_in_a_count == 1
        assert result.missing_in_b_count == 1
        assert result.first_divergence == "researcher"
