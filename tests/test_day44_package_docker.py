"""
tests/test_day44_package_docker.py -- Day 44 Package + Sample Data + Docker Tests
==================================================================================
Tests for Day 44 deliverables:
  1. `pyproject.toml` editable package configuration & CLI entry points.
  2. Zero-setup sample trace generator (`sample_data/generate_demo_traces.py`)
     covering normal (P5), grounded (P1), heuristic (P2/P3), and diff pairs.
  3. `GET /health` endpoint returning `{"status": "ok", "db": "connected", "llm": "reachable", "uptime_seconds": ...}`.
  4. `Dockerfile` and `docker-compose.yml` structure, environment injection, and healthcheck.
  5. `DB_PATH` environment variable support in `DatabaseManager`.
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.main import fastapi_app  # noqa: E402
from diff_engine import DiffEngine  # noqa: E402
from sample_data.generate_demo_traces import (  # noqa: E402
    DEMO_TRACES_DIR,
    generate_demo_traces,
)
from schema.models import RunTrace  # noqa: E402
from storage.db import DatabaseManager  # noqa: E402


class TestDay44PackagingAndEntryPoints:
    """Verify `pyproject.toml` packaging and entry points for `pip install -e .`."""

    def test_pyproject_metadata_and_entry_points(self) -> None:
        pyproject_path = ROOT / "pyproject.toml"
        assert pyproject_path.exists()
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

        project = data["project"]
        assert project["name"] == "agentlens"
        assert project["version"] == "1.0.0"

        scripts = project.get("scripts", {})
        assert scripts.get("agentlens") == "app.main:main"
        assert scripts.get("agentlens-dashboard") == "dashboard.app:main"
        assert scripts.get("agentlens-seed") == "sample_data.generate_demo_traces:main"

        packages_find = (
            data.get("tool", {}).get("setuptools", {}).get("packages", {}).get("find", {})
        )
        included = packages_find.get("include", [])
        for pkg in ("api*", "app*", "analyzers*", "capture*", "dashboard*", "storage*"):
            assert pkg in included


class TestDay44SampleDataGenerator:
    """Verify zero-setup sample traces (normal, grounded, heuristic, diff pairs)."""

    def test_generate_demo_traces_writes_files_and_seeds_sqlite(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "demo_traces"
        db_file = tmp_path / "demo.db"

        manifest = generate_demo_traces(output_dir=out_dir, seed_db=True, db_path=db_file)

        assert manifest["total_traces"] == 6
        assert (out_dir / "manifest.json").exists()

        cats = manifest["categories"]
        assert len(cats["normal"]) >= 1
        assert len(cats["grounded"]) >= 1
        assert len(cats["heuristic"]) >= 2
        assert len(cats["diff_pair"]) == 2

        by_id = {t["run_id"]: t for t in manifest["traces"]}
        assert by_id["demo_normal_pass_01"]["verdict"] == "PASS"
        assert by_id["demo_normal_pass_01"]["priority_level"] == "P5"

        assert by_id["demo_grounded_p1_01"]["verdict"] == "FAIL"
        assert by_id["demo_grounded_p1_01"]["priority_level"] == "P1"
        assert by_id["demo_grounded_p1_01"]["grounded"] is True

        assert by_id["demo_heuristic_p2_hallucination"]["verdict"] == "FAIL"
        assert by_id["demo_heuristic_p2_hallucination"]["priority_level"] == "P2"
        assert by_id["demo_heuristic_p2_hallucination"]["grounded"] is False

        assert by_id["demo_heuristic_p3_workflow"]["verdict"] == "FAIL"
        assert by_id["demo_heuristic_p3_workflow"]["priority_level"] == "P3"

        # Verify SQLite DB was seeded
        db = DatabaseManager(db_path=db_file)
        runs = db.list_runs(limit=20)
        assert len(runs) == 6

    def test_diff_pair_traces_diverge_at_writer(self) -> None:
        manifest = generate_demo_traces(seed_db=False)
        assert manifest["total_traces"] == 6

        raw_a = json.loads(
            (DEMO_TRACES_DIR / "demo_diff_pair_baseline_a.json").read_text(encoding="utf-8")
        )
        raw_b = json.loads(
            (DEMO_TRACES_DIR / "demo_diff_pair_diverged_b.json").read_text(encoding="utf-8")
        )
        raw_a.pop("demo_metadata", None)
        raw_b.pop("demo_metadata", None)

        trace_a = RunTrace(**raw_a)
        trace_b = RunTrace(**raw_b)

        diff_report = DiffEngine().compare(trace_a, trace_b)
        assert diff_report.first_divergence_agent == "writer"


class TestDay44HealthEndpoint:
    """Verify `GET /health` returns the Day 44 health payload."""

    def test_health_endpoint_day44_contract(self) -> None:
        with TestClient(fastapi_app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            body = resp.json()

            assert body["status"] == "ok"
            assert body["db"] == "connected"
            assert body["llm"] == "reachable"
            assert isinstance(body["uptime_seconds"], (int, float))
            assert body["uptime_seconds"] >= 0.0

    def test_health_endpoint_degraded_when_llm_unreachable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AGENTLENS_FORCE_LLM_UNREACHABLE", "1")
        with TestClient(fastapi_app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            body = resp.json()

            assert body["status"] == "degraded"
            assert body["db"] == "connected"
            assert body["llm"] == "unreachable"


class TestDay44DockerAndCompose:
    """Verify `Dockerfile`, `docker-compose.yml`, and `DB_PATH` env var support."""

    def test_dockerfile_contains_required_directives(self) -> None:
        dockerfile = ROOT / "Dockerfile"
        assert dockerfile.exists()
        content = dockerfile.read_text(encoding="utf-8")

        assert "FROM python:3.11-slim" in content
        assert "WORKDIR /app" in content
        assert "COPY . ." in content
        assert "RUN pip install -e ." in content
        assert "EXPOSE 8080" in content
        assert (
            "HEALTHCHECK --interval=30s CMD curl -f http://localhost:8080/health || exit 1"
            in content
        )
        assert 'CMD ["python", "-m", "dashboard.app"]' in content

    def test_docker_compose_configures_env_and_volumes(self) -> None:
        compose_path = ROOT / "docker-compose.yml"
        assert compose_path.exists()
        parsed: dict[str, Any] = yaml.safe_load(compose_path.read_text(encoding="utf-8"))

        services = parsed.get("services", {})
        assert "agentlens" in services
        svc = services["agentlens"]

        env_list = svc.get("environment", [])
        env_joined = " ".join(str(x) for x in env_list)
        assert "GROQ_API_KEY" in env_joined
        assert "DB_PATH" in env_joined
        assert "SLACK_WEBHOOK_URL" in env_joined

        volumes = svc.get("volumes", [])
        assert any("/app/data" in str(v) for v in volumes)

    def test_database_manager_honors_db_path_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        custom_db = tmp_path / "env_configured.db"
        monkeypatch.setenv("DB_PATH", str(custom_db))

        db = DatabaseManager()
        db.initialize()
        assert Path(db.db_path) == custom_db
        assert custom_db.exists()
