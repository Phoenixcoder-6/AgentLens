"""
tests/test_day61_phase_c_gate.py — Day 61: Phase C Verification Gate Tests
===========================================================================
Unit and integration tests certifying Phase C (Developer Experience & Auto-Instrumentation,
Days 57-61) in AgentLens v2.0.

Tests:
  1. `pyproject.toml` v2.0.0 metadata, build system, and package discovery.
  2. Modular dependency extras: `langchain`, `crewai`, `autogen`, `otlp`, `semantic`, `all`.
  3. Unified Public SDK surface (`agentlens.init`, `@trace_agent`, `trace_run`, `evaluate_run`).
  4. LangGraph / LangChain auto-instrumentation callback surface (`AgentLensCallbackHandler`).
  5. CrewAI swarm & AutoGen turn tracers surface (`AgentLensCrewHook`, `AgentLensAutoGenHook`).
  6. OpenTelemetry (OTLP) REST routes registration (`/v1/traces`, `/api/v1/traces`).
  7. CLI entry points callability (`agentlens`, `agentlens-dashboard`, `agentlens-seed`).
  8. End-to-end trace generation and Arbiter compatibility across v2 models.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import agentlens  # noqa: E402
from api.main import fastapi_app  # noqa: E402
from schema.models import StepStatus  # noqa: E402
from storage.db import DatabaseManager  # noqa: E402


class TestDay61PackagingAndExtras:
    """Verify pyproject.toml packaging, versioning, and modular extras for Day 61."""

    def test_pyproject_v2_metadata_and_structure(self) -> None:
        pyproject_path = ROOT / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml must exist"

        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        project = data["project"]

        assert project["name"] == "agentlens"
        assert project["version"] == "2.0.0", "Package version must be 2.0.0"
        assert "Multi-Agent" in project["description"]

        scripts = project.get("scripts", {})
        assert scripts.get("agentlens") == "app.main:main"
        assert scripts.get("agentlens-dashboard") == "dashboard.app:main"
        assert scripts.get("agentlens-seed") == "sample_data.generate_demo_traces:main"

        packages_find = (
            data.get("tool", {}).get("setuptools", {}).get("packages", {}).get("find", {})
        )
        included = packages_find.get("include", [])
        assert "agentlens*" in included, "agentlens* package must be included in setuptools find"
        assert "api*" in included
        assert "analyzers*" in included

    def test_pyproject_optional_dependencies_extras(self) -> None:
        pyproject_path = ROOT / "pyproject.toml"
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        extras = data.get("project", {}).get("optional-dependencies", {})

        expected_extras = ["dev", "langchain", "crewai", "autogen", "otlp", "semantic", "all"]
        for extra in expected_extras:
            assert extra in extras, f"Missing expected optional extra: {extra}"

        # Verify langchain extra packages
        lc_pkgs = " ".join(extras["langchain"])
        assert "langchain" in lc_pkgs
        assert "langgraph" in lc_pkgs

        # Verify crewai extra
        assert any("crewai" in p for p in extras["crewai"])

        # Verify autogen extra
        assert any("autogen" in p for p in extras["autogen"])

        # Verify otlp extra
        otlp_pkgs = " ".join(extras["otlp"])
        assert "opentelemetry" in otlp_pkgs

        # Verify all bundle
        all_pkgs = " ".join(extras["all"])
        assert "crewai" in all_pkgs
        assert "langchain" in all_pkgs
        assert "opentelemetry" in all_pkgs


class TestDay61DeveloperExperienceSurface:
    """Verify public SDK, integrations, and CLI surface."""

    def test_public_sdk_surface_and_exports(self) -> None:
        assert agentlens.__version__ == "2.0.0"

        # Public callable APIs
        assert callable(agentlens.init)
        assert callable(agentlens.trace_agent)
        assert callable(agentlens.trace_run)
        assert callable(agentlens.evaluate_run)
        assert callable(agentlens.get_current_trace)

        # Re-exported domain models and classes
        assert hasattr(agentlens, "RunTrace")
        assert hasattr(agentlens, "AgentStep")
        assert hasattr(agentlens, "TraceSpan")
        assert hasattr(agentlens, "SpanType")
        assert hasattr(agentlens, "AnalysisBundle")
        assert hasattr(agentlens, "RuleEngine")
        assert hasattr(agentlens, "Arbiter")

    def test_langchain_and_langgraph_callback_handler(self) -> None:
        assert hasattr(agentlens, "AgentLensCallbackHandler")
        handler = agentlens.AgentLensCallbackHandler(workflow="test_workflow")

        assert hasattr(handler, "on_chain_start")
        assert hasattr(handler, "on_chain_end")
        assert hasattr(handler, "on_tool_start")
        assert hasattr(handler, "on_tool_end")
        assert hasattr(handler, "on_llm_end")
        assert hasattr(handler, "get_trace")
        assert hasattr(handler, "get_spans")
        assert hasattr(handler, "get_steps")
        assert hasattr(handler, "evaluate")

    def test_swarm_hooks_crewai_and_autogen(self) -> None:
        assert hasattr(agentlens, "AgentLensCrewHook")
        assert hasattr(agentlens, "AgentLensAutoGenHook")

        crew_hook = agentlens.AgentLensCrewHook(workflow="swarm_flow")
        assert hasattr(crew_hook, "monitor_crew")
        assert hasattr(crew_hook, "task_callback")
        assert hasattr(crew_hook, "step_callback")
        assert hasattr(crew_hook, "on_tool_start")
        assert hasattr(crew_hook, "on_tool_end")

        autogen_hook = agentlens.AgentLensAutoGenHook(workflow="autogen_flow")
        assert hasattr(autogen_hook, "monitor_conversation")
        assert hasattr(autogen_hook, "on_turn")
        assert hasattr(autogen_hook, "record_message")
        assert hasattr(autogen_hook, "register")

    def test_otlp_http_endpoints_registered(self) -> None:
        openapi_paths = set(fastapi_app.openapi().get("paths", {}).keys())

        assert "/v1/traces" in openapi_paths
        assert "/traces" in openapi_paths
        assert "/api/v1/traces" in openapi_paths
        assert "/api/traces" in openapi_paths

    def test_cli_entry_points_callability(self) -> None:
        import app.main as app_main
        import dashboard.app as dash_app
        import sample_data.generate_demo_traces as demo_seed

        assert callable(getattr(app_main, "main", None))
        assert callable(getattr(dash_app, "main", None))
        assert callable(getattr(demo_seed, "main", None))

    def test_end_to_end_sdk_to_otlp_compatibility(self, tmp_path: Path) -> None:
        db_file = tmp_path / "sdk_compat.db"
        agentlens.init(storage_path=db_file)

        @agentlens.trace_agent(role="researcher", name="doc_search")
        def search(query: str) -> dict[str, Any]:
            return {"query": query, "results": ["fact1", "fact2"]}

        @agentlens.trace_agent(role="writer", name="doc_writer")
        def write_report(data: dict[str, Any]) -> dict[str, Any]:
            return {"report": f"Summary of {len(data['results'])} facts"}

        with agentlens.trace_run(workflow="sdk_compat_run") as run:
            res = search("multi-agent telemetry")
            write_report(res)

        assert run is not None
        assert run.status == StepStatus.SUCCESS
        assert len(run.steps) == 2
        assert len(run.spans) == 2

        # Verify evaluation
        bundle = agentlens.evaluate_run(run)
        assert bundle.run_id == run.run_id

        # Verify DB reloading
        db = DatabaseManager(str(db_file))
        stored = db.get_run(run.run_id)
        assert stored is not None
        assert stored["status"].upper() == "SUCCESS"
