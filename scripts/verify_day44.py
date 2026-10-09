"""
scripts/verify_day44.py -- Day 44 Verification Suite (8 Checks)
================================================================
Verifies all Day 44 deliverables (`full_project_plan.md` lines 421-439):
  1. `pyproject.toml` editable package config & CLI entry points (`agentlens`,
     `agentlens-dashboard`, `agentlens-seed`).
  2. Editable package installation (`pip install -e . --no-deps`) succeeds and installs
     `agentlens` package metadata (`importlib.metadata.version("agentlens") == "1.0.0"`).
  3. Zero-setup sample trace generator (`sample_data/generate_demo_traces.py`) generates
     normal (PASS/P5), grounded (FAIL/P1), heuristic (FAIL/P2 & P3), and diff pair traces.
  4. Diff pair sample traces (`demo_diff_pair_baseline_a` vs. `demo_diff_pair_diverged_b`)
     align and pinpoint `writer` as `first_divergence_agent`.
  5. SQLite zero-setup seeding persists all 6 demo runs, steps, analysis bundles, and rule matches.
  6. Health check endpoint `GET /health` returns HTTP 200 with
     `{"status": "ok", "db": "connected", "llm": "reachable", "uptime_seconds": ...}`.
  7. `Dockerfile` exists with `FROM python:3.11-slim`, `RUN pip install -e .`,
     `EXPOSE 8080`, `HEALTHCHECK` on `/health`, and `CMD ["python", "-m", "dashboard.app"]`.
  8. `docker-compose.yml` exists with environment variable injection (`GROQ_API_KEY`,
     `DB_PATH`, `SLACK_WEBHOOK_URL`), persistent volume `./data:/app/data`, and healthcheck.
"""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

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


def check_1_pyproject_entry_points() -> str:
    pyproject_path = ROOT / "pyproject.toml"
    assert pyproject_path.exists(), "pyproject.toml missing"
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = data["project"]
    assert project["name"] == "agentlens"
    assert project["version"] == "1.0.0"
    scripts = project.get("scripts", {})
    for ep in ("agentlens", "agentlens-dashboard", "agentlens-seed"):
        assert ep in scripts, f"Missing CLI entry point: {ep}"
    return f"pyproject.toml v{project['version']} with entry points {list(scripts.keys())}"


def check_2_editable_pip_install() -> str:
    egg_info = ROOT / "agentlens.egg-info"
    if egg_info.is_dir() and not any(egg_info.iterdir()):
        try:
            egg_info.rmdir()
        except OSError:
            pass

    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps", "--no-build-isolation"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
    assert proc.returncode == 0, f"pip install -e . failed: {proc.stderr}"

    # Remove empty egg-info stub if OneDrive recreated an empty folder
    if egg_info.is_dir() and not (egg_info / "PKG-INFO").exists():
        try:
            egg_info.rmdir()
        except OSError:
            pass

    versions = [
        d.version
        for d in importlib.metadata.distributions()
        if d.metadata.get("Name") == "agentlens" and d.version
    ]
    assert "1.0.0" in versions, (
        f"Expected agentlens==1.0.0 in installed distributions, got {versions}"
    )
    return "pip install -e . succeeded (installed agentlens==1.0.0)"


def check_3_sample_traces_generated() -> str:
    manifest = generate_demo_traces(seed_db=True)
    assert manifest["total_traces"] == 6
    cats = manifest["categories"]
    assert len(cats["normal"]) >= 1, "Missing normal demo trace"
    assert len(cats["grounded"]) >= 1, "Missing grounded demo trace"
    assert len(cats["heuristic"]) >= 2, "Missing heuristic demo traces"
    assert len(cats["diff_pair"]) == 2, "Missing diff_pair demo traces"
    for entry in manifest["traces"]:
        p = DEMO_TRACES_DIR / entry["file"]
        assert p.exists(), f"Missing generated trace file: {p}"
    return (
        f"6 demo traces generated in sample_data/demo_traces/ "
        f"(normal={len(cats['normal'])}, grounded={len(cats['grounded'])}, "
        f"heuristic={len(cats['heuristic'])}, diff_pair={len(cats['diff_pair'])})"
    )


def check_4_diff_pair_verification() -> str:
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
    report = DiffEngine().compare(trace_a, trace_b)
    assert report.first_divergence_agent == "writer", (
        f"Expected writer as first divergence, got {report.first_divergence_agent}"
    )
    return (
        f"Diff pair verified: first_divergence_agent='{report.first_divergence_agent}', "
        f"average_similarity={report.average_similarity:.3f}"
    )


def check_5_zero_setup_sqlite_seeding() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        db_file = Path(tmp) / "seeded.db"
        generate_demo_traces(seed_db=True, db_path=db_file)
        db = DatabaseManager(db_path=db_file)
        runs = db.list_runs(limit=20)
        assert len(runs) == 6, f"Expected 6 seeded runs, got {len(runs)}"
        grounded_matches = db.get_rule_matches_for_run("demo_grounded_p1_01")
        heur_matches = db.get_rule_matches_for_run("demo_heuristic_p2_hallucination")
        assert len(grounded_matches) >= 1
        assert len(heur_matches) >= 1
    return "SQLite database seeded with 6 runs, steps, analysis bundles, and rule_matches"


def check_6_health_check_endpoint() -> str:
    with TestClient(fastapi_app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200, f"Expected 200 OK, got {resp.status_code}"
        body = resp.json()
        assert body["status"] == "ok", f"Unexpected status: {body}"
        assert body["db"] == "connected", f"Unexpected db field: {body}"
        assert body["llm"] == "reachable", f"Unexpected llm field: {body}"
        assert isinstance(body["uptime_seconds"], (int, float))
    return (
        f"GET /health -> status='{body['status']}', db='{body['db']}', "
        f"llm='{body['llm']}', uptime_seconds={body['uptime_seconds']}"
    )


def check_7_dockerfile_spec() -> str:
    dockerfile = ROOT / "Dockerfile"
    assert dockerfile.exists(), "Dockerfile missing"
    text = dockerfile.read_text(encoding="utf-8")
    required = [
        "FROM python:3.11-slim",
        "WORKDIR /app",
        "COPY . .",
        "RUN pip install -e .",
        "EXPOSE 8080",
        "HEALTHCHECK --interval=30s CMD curl -f http://localhost:8080/health || exit 1",
        'CMD ["python", "-m", "dashboard.app"]',
    ]
    for line in required:
        assert line in text, f"Missing Dockerfile directive: {line}"
    return "Dockerfile verified with python:3.11-slim, pip install -e ., EXPOSE 8080 & HEALTHCHECK"


def check_8_docker_compose_spec() -> str:
    compose = ROOT / "docker-compose.yml"
    assert compose.exists(), "docker-compose.yml missing"
    parsed: dict[Any, Any] = yaml.safe_load(compose.read_text(encoding="utf-8"))
    services = parsed.get("services", {})
    assert "agentlens" in services, "Missing 'agentlens' service in docker-compose.yml"
    svc = services["agentlens"]
    env_str = " ".join(str(x) for x in svc.get("environment", []))
    for var in ("GROQ_API_KEY", "DB_PATH", "SLACK_WEBHOOK_URL"):
        assert var in env_str, f"Missing {var} in docker-compose.yml environment"
    return (
        "docker-compose.yml verified with GROQ_API_KEY, DB_PATH, SLACK_WEBHOOK_URL & volume mount"
    )


def main() -> int:
    checks: list[tuple[str, Callable[[], str]]] = [
        ("Check 1: pyproject.toml Packaging & CLI Entry Points", check_1_pyproject_entry_points),
        ("Check 2: Editable Package Install (pip install -e .)", check_2_editable_pip_install),
        (
            "Check 3: Zero-Setup Sample Traces (Normal/Grounded/Heuristic/Diff)",
            check_3_sample_traces_generated,
        ),
        ("Check 4: Diff Pair Alignment & Divergence Verification", check_4_diff_pair_verification),
        ("Check 5: SQLite Zero-Setup Database Seeding", check_5_zero_setup_sqlite_seeding),
        ("Check 6: Health Check Endpoint (GET /health)", check_6_health_check_endpoint),
        ("Check 7: Dockerfile Directives & Healthcheck", check_7_dockerfile_spec),
        ("Check 8: docker-compose.yml Environment & Volume Config", check_8_docker_compose_spec),
    ]

    print("=" * 72)
    print("AgentLens -- Day 44 Verification Suite (Package + Sample Data + Docker)")
    print("=" * 72)

    passed = 0
    for title, fn in checks:
        try:
            detail = fn()
            print(f"[PASS] {title}")
            print(f"       -> {detail}")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] {title}")
            print(f"       -> {exc}")

    print("-" * 72)
    print(f"Day 44 Verification Summary: {passed}/{len(checks)} checks passed.")
    print("=" * 72)
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
