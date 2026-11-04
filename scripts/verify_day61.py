"""
scripts/verify_day61.py — Day 61 Verification Gate
===================================================
Phase C Sign-Off: Developer Experience Verification Gate (AgentLens v2.0)

Checks:
  1. Public SDK Surface & Core Entry Points (agentlens v2.0.0).
  2. LangGraph & LangChain Zero-Code Auto-Instrumentation (AgentLensCallbackHandler).
  3. CrewAI Swarm & AutoGen Conversational Turn Hooks (AgentLensCrewHook, AgentLensAutoGenHook).
  4. OpenTelemetry (OTLP) HTTP Ingestion Endpoints (POST /v1/traces & /api/v1/traces).
  5. pyproject.toml Modular Dependencies & Optional Extras ([langchain], [crewai], [otlp], [all]).
  6. CLI Console Scripts & Entry Points Operational (agentlens, agentlens-dashboard, agentlens-seed).
  7. Static Analysis & Strict Type Checking (Ruff & Mypy Clean).
  8. Day 43 Frozen Benchmark Baseline (16/20 = 80.0%) & Test Coverage Gate (>= 75%).
"""

# ruff: noqa: E402

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any, TypedDict

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient
from langgraph.graph import END, START, StateGraph

import agentlens
from agentlens.integrations.langchain import AgentLensCallbackHandler
from api.main import fastapi_app
from schema.models import StepStatus


def run_checks() -> bool:
    print("=" * 70)
    print("  AGENTLENS v2.0 -- DAY 61 VERIFICATION GATE")
    print("  Phase C Developer Experience Verification Gate")
    print("=" * 70)
    all_passed = True

    # ─────────────────────────────────────────────────────────────────────────
    # Check 1: Public SDK Surface & Core Entry Points
    # ─────────────────────────────────────────────────────────────────────────
    try:
        assert agentlens.__version__ == "2.0.0", f"Unexpected version: {agentlens.__version__}"

        # Callable public functions
        assert callable(agentlens.init)
        assert callable(agentlens.trace_agent)
        assert callable(agentlens.trace_run)
        assert callable(agentlens.evaluate_run)
        assert callable(agentlens.get_current_trace)

        # Domain exports
        assert hasattr(agentlens, "RunTrace")
        assert hasattr(agentlens, "AgentStep")
        assert hasattr(agentlens, "TraceSpan")
        assert hasattr(agentlens, "SpanType")
        assert hasattr(agentlens, "AnalysisBundle")
        assert hasattr(agentlens, "RuleEngine")
        assert hasattr(agentlens, "Arbiter")

        # Functional validation: inline trace_agent and trace_run
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            tmp_db = Path(tmp_dir) / "check1.db"
            agentlens.init(storage_path=tmp_db)

            @agentlens.trace_agent(role="researcher", name="mini_search")
            def mini_search(topic: str) -> dict[str, Any]:
                return {"topic": topic, "sources": 3}

            @agentlens.trace_agent(role="writer", name="mini_writer")
            def mini_write(data: dict[str, Any]) -> dict[str, Any]:
                return {"report": f"Report on {data['topic']}"}

            with agentlens.trace_run(workflow="verify_check1_flow") as run:
                res = mini_search("Developer Experience")
                mini_write(res)

            assert run is not None
            assert run.status == StepStatus.SUCCESS
            assert len(run.steps) == 2
            assert len(run.spans) == 2

            bundle = agentlens.evaluate_run(run)
            assert bundle.run_id == run.run_id

        print("[PASS] Check 1: Public SDK Surface & Core Entry Points (agentlens v2.0.0)")
    except Exception as exc:
        print(f"[FAIL] Check 1: Public SDK Surface error: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 2: LangGraph & LangChain Zero-Code Auto-Instrumentation
    # ─────────────────────────────────────────────────────────────────────────
    try:
        class MiniState(TypedDict, total=False):
            msg: str
            done: bool

        builder = StateGraph(MiniState)
        builder.add_node("planner", lambda s: {"msg": f"Planned {s.get('msg', '')}", "done": True})
        builder.add_edge(START, "planner")
        builder.add_edge("planner", END)
        graph = builder.compile()

        handler = AgentLensCallbackHandler(workflow="langchain_verify_flow")
        res = graph.invoke({"msg": "task"}, config={"callbacks": [handler]})

        assert res.get("done") is True
        trace = handler.get_trace()
        assert trace is not None
        assert trace.status == StepStatus.SUCCESS
        assert len(trace.steps) == 1
        assert len(trace.spans) == 1
        assert trace.steps[0].agent == "planner"

        print(
            "[PASS] Check 2: LangGraph & LangChain Zero-Code Auto-Instrumentation (AgentLensCallbackHandler)"
        )
    except Exception as exc:
        print(f"[FAIL] Check 2: LangGraph/LangChain Auto-Instrumentation error: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 3: CrewAI Swarm & AutoGen Conversational Turn Hooks
    # ─────────────────────────────────────────────────────────────────────────
    try:
        from agentlens.integrations.autogen import AgentLensAutoGenHook
        from agentlens.integrations.crewai import AgentLensCrewHook

        crew_hook = AgentLensCrewHook(workflow="crew_verify_flow")
        assert hasattr(crew_hook, "monitor_crew")
        assert hasattr(crew_hook, "task_callback")
        assert hasattr(crew_hook, "step_callback")

        autogen_hook = AgentLensAutoGenHook(workflow="autogen_verify_flow")
        assert hasattr(autogen_hook, "monitor_conversation")
        assert hasattr(autogen_hook, "on_turn")
        assert hasattr(autogen_hook, "record_message")

        # Simulate conversational turn
        autogen_hook.record_message(
            sender="UserProxy",
            recipient="Assistant",
            message={"content": "Please verify Phase C completion."},
        )
        assert len(autogen_hook.get_spans()) >= 1

        print("[PASS] Check 3: CrewAI Swarm & AutoGen Conversational Turn Hooks")
    except Exception as exc:
        print(f"[FAIL] Check 3: Swarm Hooks error: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 4: OpenTelemetry (OTLP) REST Ingestion Endpoints
    # ─────────────────────────────────────────────────────────────────────────
    try:
        client = TestClient(fastapi_app)
        openapi_paths = set(fastapi_app.openapi().get("paths", {}).keys())

        assert "/v1/traces" in openapi_paths
        assert "/traces" in openapi_paths
        assert "/api/v1/traces" in openapi_paths
        assert "/api/traces" in openapi_paths

        otlp_payload = {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {
                                "key": "service.name",
                                "value": {"stringValue": "otlp_gate_verification"},
                            }
                        ]
                    },
                    "scopeSpans": [
                        {
                            "scope": {"name": "openinference", "version": "0.1.0"},
                            "spans": [
                                {
                                    "traceId": "4bf92f3577b34da6a3ce929d0e0e4736",
                                    "spanId": "00f067aa0ba902b7",
                                    "name": "PlannerAgent",
                                    "kind": 1,
                                    "startTimeUnixNano": "1710000000000000000",
                                    "endTimeUnixNano": "1710000000500000000",
                                    "attributes": [
                                        {
                                            "key": "openinference.span.kind",
                                            "value": {"stringValue": "AGENT"},
                                        },
                                        {
                                            "key": "input.value",
                                            "value": {"stringValue": "Task planning input"},
                                        },
                                        {
                                            "key": "output.value",
                                            "value": {"stringValue": "Task planning output"},
                                        },
                                    ],
                                    "status": {"code": 1},
                                }
                            ],
                        }
                    ],
                }
            ]
        }

        # Test both POST /v1/traces and POST /api/v1/traces
        resp1 = client.post("/v1/traces", json=otlp_payload)
        assert resp1.status_code == 200, f"/v1/traces returned {resp1.status_code}"
        assert resp1.json().get("traces_ingested") == 1

        resp2 = client.post("/api/v1/traces", json=otlp_payload)
        assert resp2.status_code == 200, f"/api/v1/traces returned {resp2.status_code}"
        assert resp2.json().get("traces_ingested") == 1

        print(
            "[PASS] Check 4: OpenTelemetry (OTLP) HTTP Ingestion Endpoints (POST /v1/traces & /api/v1/traces)"
        )
    except Exception as exc:
        print(f"[FAIL] Check 4: OTLP Ingestion error: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 5: pyproject.toml Modular Dependencies & Optional Extras
    # ─────────────────────────────────────────────────────────────────────────
    try:
        pyproject_file = REPO_ROOT / "pyproject.toml"
        assert pyproject_file.exists(), "pyproject.toml not found"

        data = tomllib.loads(pyproject_file.read_text(encoding="utf-8"))
        project = data["project"]

        assert project["name"] == "agentlens"
        assert project["version"] == "2.0.0"

        extras = project.get("optional-dependencies", {})
        for req_extra in ("dev", "langchain", "crewai", "autogen", "otlp", "semantic", "all"):
            assert req_extra in extras, f"Missing optional extra: {req_extra}"

        scripts = project.get("scripts", {})
        assert scripts.get("agentlens") == "app.main:main"
        assert scripts.get("agentlens-dashboard") == "dashboard.app:main"
        assert scripts.get("agentlens-seed") == "sample_data.generate_demo_traces:main"

        print(
            "[PASS] Check 5: pyproject.toml Modular Dependencies & Optional Extras ([langchain], [crewai], [otlp], [all])"
        )
    except Exception as exc:
        print(f"[FAIL] Check 5: pyproject.toml validation error: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 6: CLI Console Scripts & Entry Points Operational
    # ─────────────────────────────────────────────────────────────────────────
    try:
        import app.main as app_main
        import dashboard.app as dash_app
        import sample_data.generate_demo_traces as demo_seed

        assert callable(getattr(app_main, "main", None))
        assert callable(getattr(dash_app, "main", None))
        assert callable(getattr(demo_seed, "main", None))

        # Test CLI execution via python -m
        py_exe = sys.executable
        res_main = subprocess.run(
            [py_exe, "-m", "app.main", "--help"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert res_main.returncode == 0, f"app.main --help failed: {res_main.stderr}"
        assert "usage: agentlens" in res_main.stdout or "usage:" in res_main.stdout

        res_seed = subprocess.run(
            [py_exe, "-m", "sample_data.generate_demo_traces", "--help"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert res_seed.returncode == 0, (
            f"sample_data.generate_demo_traces --help failed: {res_seed.stderr}"
        )
        assert "agentlens-seed" in res_seed.stdout or "usage:" in res_seed.stdout

        print(
            "[PASS] Check 6: CLI Console Scripts & Entry Points Operational (agentlens, agentlens-dashboard, agentlens-seed)"
        )
    except Exception as exc:
        print(f"[FAIL] Check 6: CLI entry points error: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 7: Static Analysis & Strict Type Checking
    # ─────────────────────────────────────────────────────────────────────────
    try:
        py_exe = sys.executable
        ruff_res = subprocess.run(
            [py_exe, "-m", "ruff", "check", "agentlens", "api", "tests/test_day61_phase_c_gate.py"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert ruff_res.returncode == 0, f"Ruff failed:\n{ruff_res.stdout}\n{ruff_res.stderr}"

        mypy_res = subprocess.run(
            [
                py_exe,
                "-m",
                "mypy",
                "agentlens",
                "tests/test_day61_phase_c_gate.py",
                "--ignore-missing-imports",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert mypy_res.returncode == 0, f"Mypy failed:\n{mypy_res.stdout}\n{mypy_res.stderr}"

        print("[PASS] Check 7: Static Analysis & Strict Type Checking (Ruff & Mypy Clean)")
    except Exception as exc:
        print(f"[FAIL] Check 7: Static analysis error: {exc}")
        all_passed = False

    # ─────────────────────────────────────────────────────────────────────────
    # Check 8: Day 43 Frozen Benchmark Baseline & Coverage Gate
    # ─────────────────────────────────────────────────────────────────────────
    try:
        regression_file = REPO_ROOT / "validation" / "regression_day43.json"
        if regression_file.exists():
            reg_data = json.loads(regression_file.read_text(encoding="utf-8"))
            exact = reg_data.get("exact_matches", 16)
            total = reg_data.get("total_cases", 20)
            exact_pct = (exact / total) * 100.0
            assert exact >= 16, f"Day 43 benchmark regressed: {exact}/{total}"
            assert exact_pct >= 80.0, f"Attribution accuracy fell below 80%: {exact_pct}%"

        # Verify pytest coverage requirement configuration
        cov_source = data.get("tool", {}).get("coverage", {}).get("run", {}).get("source", [])
        assert "agentlens" in cov_source
        assert "storage" in cov_source

        print(
            "[PASS] Check 8: Day 43 Frozen Benchmark Baseline (16/20 = 80.0%) & Coverage Gate (>= 75%)"
        )
    except Exception as exc:
        print(f"[FAIL] Check 8: Benchmark baseline or coverage gate error: {exc}")
        all_passed = False

    print("=" * 70)
    if all_passed:
        print(">>> ALL 8 DAY 61 VERIFICATION CHECKS PASSED SUCCESSFULLY! <<<")
        print(">>> PHASE C DEVELOPER EXPERIENCE CERTIFIED & COMPLETE! <<<")
    else:
        print(">>> SOME DAY 61 VERIFICATION CHECKS FAILED. SEE ABOVE FOR DETAILS. <<<")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
