"""
scripts/verify_day27.py — Day 27 Verification Script
=====================================================
Validates:
  1. EvidenceSource.STATISTICAL_ANOMALY exists in schema
  2. STATISTICAL_ANOMALY maps to P4 in Arbiter
  3. StatisticalDetector can be imported from analyzers.detection
  4. StatisticalDetector._build_baselines returns empty dict below min threshold
  5. StatisticalDetector builds correct baseline from sufficient runs
  6. Latency outlier is flagged as EvidenceRecord with correct source
  7. Token outlier is flagged as EvidenceRecord with correct source
  8. Normal step produces no anomaly
  9. Confidence is clamped to [0.5, 0.99] range for outliers
 10. AgentBaseline.active property works correctly
 11. backup_db.backup_database creates a valid DB file
 12. Alembic versions directory has at least one migration file
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

# ─── ensure project root on path ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS = "✅"
FAIL = "❌"
results: list[tuple[str, bool, str]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    results.append((label, ok, detail))
    status = PASS if ok else FAIL
    print(f"  {status} {label}" + (f"  [{detail}]" if detail else ""))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

TS = "2026-09-06T12:00:00+00:00"
SCHEMA_VER = "1.0"


def _seed_run(db, run_id: str, latencies: list, tokens: list, agents: list | None = None):
    _agents = agents or ["researcher", "writer", "verifier"]
    db.insert_run(run_id, "pipeline", TS, "SUCCESS", sum(latencies), sum(tokens), SCHEMA_VER)
    for i, (lat, tok) in enumerate(zip(latencies, tokens, strict=False)):
        agent = _agents[i % len(_agents)]
        db.insert_step(run_id, i+1, agent, "SUCCESS", lat, tok//3, tok-tok//3*2, tok, "", TS, SCHEMA_VER)


# ─────────────────────────────────────────────────────────────────────────────
# Check 1: EvidenceSource.STATISTICAL_ANOMALY exists
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] schema.models — EvidenceSource.STATISTICAL_ANOMALY")
try:
    from schema.models import EvidenceSource  # type: ignore[import]
    has_source = hasattr(EvidenceSource, "STATISTICAL_ANOMALY")
    check("EvidenceSource.STATISTICAL_ANOMALY exists", has_source,
          str(EvidenceSource.STATISTICAL_ANOMALY) if has_source else "MISSING")
except Exception as e:
    check("EvidenceSource.STATISTICAL_ANOMALY exists", False, str(e))

# ─────────────────────────────────────────────────────────────────────────────
# Check 2: Arbiter maps STATISTICAL_ANOMALY → P4
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] analyzers.arbiter — STATISTICAL_ANOMALY → P4")
try:
    from analyzers.arbiter import SOURCE_TO_PRIORITY  # type: ignore[import]
    from schema.models import PriorityLevel  # type: ignore[import]
    src = EvidenceSource.STATISTICAL_ANOMALY
    mapped = SOURCE_TO_PRIORITY.get(src)
    check("STATISTICAL_ANOMALY mapped to P4",
          mapped == PriorityLevel.P4, f"got {mapped}")
except Exception as e:
    check("STATISTICAL_ANOMALY mapped to P4", False, str(e))

# ─────────────────────────────────────────────────────────────────────────────
# Check 3: StatisticalDetector importable from analyzers.detection
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] analyzers.detection — StatisticalDetector importable")
try:
    from analyzers.detection import StatisticalDetector  # type: ignore[import]
    check("StatisticalDetector importable", True, "ok")
except Exception as e:
    check("StatisticalDetector importable", False, str(e))
    sys.exit(1)  # rest of checks depend on this

# ─────────────────────────────────────────────────────────────────────────────
# Check 4: Empty baselines when insufficient runs
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] StatisticalDetector — no baselines below min threshold")
try:
    from storage.db import DatabaseManager  # type: ignore[import]
    with tempfile.TemporaryDirectory() as td:
        db = DatabaseManager(db_path=os.path.join(td, "test.db"))
        db.initialize()
        # Seed only 3 runs (default min is 5)
        for i in range(3):
            _seed_run(db, f"r{i}", [100.0], [500], ["researcher"])
        detector = StatisticalDetector(db)
        baselines = detector.get_baselines()
        check("No baselines with 3 runs (min=5)", len(baselines) == 0,
              f"got {len(baselines)} baselines")
except Exception as e:
    check("No baselines with 3 runs (min=5)", False, str(e))

# ─────────────────────────────────────────────────────────────────────────────
# Check 5: Correct baseline with sufficient runs
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] StatisticalDetector — correct baseline with 6 runs")
try:
    with tempfile.TemporaryDirectory() as td:
        db = DatabaseManager(db_path=os.path.join(td, "test.db"))
        db.initialize()
        for i in range(6):
            _seed_run(db, f"r{i}", [200.0], [1000], ["researcher"])
        detector = StatisticalDetector(db)
        baselines = detector.get_baselines()
        bl = baselines.get("researcher")
        ok = (bl is not None
              and abs(bl.latency_mean - 200.0) < 1.0
              and abs(bl.token_mean - 1000.0) < 1.0)
        check("Baseline mean=200ms tokens=1000 with 6 runs", ok,
              f"mean={bl.latency_mean:.1f}" if bl else "no baseline")
except Exception as e:
    check("Correct baseline with 6 runs", False, str(e))

# ─────────────────────────────────────────────────────────────────────────────
# Check 6: Latency outlier flagged
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] StatisticalDetector — latency outlier flagged")
try:
    with tempfile.TemporaryDirectory() as td:
        db = DatabaseManager(db_path=os.path.join(td, "test.db"))
        db.initialize()
        # Seed stable runs with slight variation so stddev > 0
        for i, lat in enumerate([100.0, 105.0, 98.0, 102.0, 99.0, 101.0]):
            _seed_run(db, f"r{i}", [lat], [500], ["researcher"])
        _seed_run(db, "run_spike", [9000.0], [500], ["researcher"])
        detector = StatisticalDetector(db)
        report = detector.analyze_run("run_spike")
        lat_anomalies = [e for e in report.anomalies if "Latency" in e.description]
        ok = (
            report.has_anomalies
            and len(lat_anomalies) >= 1
            and lat_anomalies[0].source == EvidenceSource.STATISTICAL_ANOMALY
        )
        check("Latency outlier detected with correct source", ok,
              f"{len(lat_anomalies)} lat anomaly(s)")
except Exception as e:
    check("Latency outlier flagged", False, str(e))

# ─────────────────────────────────────────────────────────────────────────────
# Check 7: Token outlier flagged
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] StatisticalDetector — token outlier flagged")
try:
    with tempfile.TemporaryDirectory() as td:
        db = DatabaseManager(db_path=os.path.join(td, "test.db"))
        db.initialize()
        # Seed stable runs with slight variation so stddev > 0
        for i, tok in enumerate([500, 510, 490, 505, 495, 502]):
            _seed_run(db, f"r{i}", [100.0], [tok], ["writer"])
        _seed_run(db, "run_tok", [100.0], [50000], ["writer"])
        detector = StatisticalDetector(db)
        report = detector.analyze_run("run_tok")
        tok_anomalies = [e for e in report.anomalies if "Token" in e.description]
        ok = len(tok_anomalies) >= 1 and tok_anomalies[0].source == EvidenceSource.STATISTICAL_ANOMALY
        check("Token outlier detected with correct source", ok,
              f"{len(tok_anomalies)} tok anomaly(s)")
except Exception as e:
    check("Token outlier flagged", False, str(e))

# ─────────────────────────────────────────────────────────────────────────────
# Check 8: Normal step → no anomaly
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8] StatisticalDetector — normal step produces no anomaly")
try:
    with tempfile.TemporaryDirectory() as td:
        db = DatabaseManager(db_path=os.path.join(td, "test.db"))
        db.initialize()
        # Seed with slight variation so stddev > 0 for latency
        # Keep tokens identical so no zero-stddev token false-positive
        for i, lat in enumerate([100.0, 105.0, 98.0, 102.0, 99.0, 101.0]):
            _seed_run(db, f"r{i}", [lat], [500], ["researcher"])
        # Normal step: latency within 1 sigma of mean (~101ms), same tokens
        _seed_run(db, "run_normal", [103.0], [500], ["researcher"])
        detector = StatisticalDetector(db)
        report = detector.analyze_run("run_normal")
        check("Normal step produces no anomaly", not report.has_anomalies,
              f"{len(report.anomalies)} anomalies")
except Exception as e:
    check("Normal step produces no anomaly", False, str(e))

# ─────────────────────────────────────────────────────────────────────────────
# Check 9: Confidence in [0.5, 0.99]
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9] StatisticalDetector — confidence clamped to [0.5, 0.99]")
try:
    from analyzers.detection.statistical_detector import _confidence_from_z  # type: ignore[import]
    c_at_thresh = _confidence_from_z(2.5, 2.5)
    c_very_high = _confidence_from_z(100.0, 2.5)
    ok = (abs(c_at_thresh - 0.5) < 0.01
          and c_very_high <= 0.99)
    check("_confidence_from_z boundary values correct", ok,
          f"at-threshold={c_at_thresh:.3f}, extreme={c_very_high:.3f}")
except Exception as e:
    check("Confidence boundary check", False, str(e))

# ─────────────────────────────────────────────────────────────────────────────
# Check 10: AgentBaseline.active property
# ─────────────────────────────────────────────────────────────────────────────
print("\n[10] AgentBaseline.active property")
try:
    from analyzers.detection.statistical_detector import AgentBaseline  # type: ignore[import]
    bl = AgentBaseline(
        agent="test", n_runs=5, latency_mean=100.0, latency_std=10.0,
        token_mean=500.0, token_std=50.0
    )
    check("AgentBaseline.active=True when n_runs>=1", bl.active is True, f"n_runs={bl.n_runs}")
except Exception as e:
    check("AgentBaseline.active property", False, str(e))

# ─────────────────────────────────────────────────────────────────────────────
# Check 11: backup_db creates valid DB file
# ─────────────────────────────────────────────────────────────────────────────
print("\n[11] scripts.backup_db — creates valid DB backup")
try:
    from scripts.backup_db import backup_database  # type: ignore[import]
    with tempfile.TemporaryDirectory() as td:
        # Create a minimal source DB
        src_path = os.path.join(td, "source.db")
        conn = sqlite3.connect(src_path)
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO test VALUES (1)")
        conn.commit()
        conn.close()

        backup_dir = os.path.join(td, "backups")
        dest = backup_database(db_path=src_path, backup_dir=backup_dir, quiet=True)

        # Verify backup is a valid SQLite DB with same content
        bconn = sqlite3.connect(str(dest))
        rows = bconn.execute("SELECT id FROM test").fetchall()
        bconn.close()
        ok = dest.exists() and rows == [(1,)]
        check("backup_database creates valid consistent backup", ok,
              f"file={dest.name}, rows={rows}")
except Exception as e:
    check("backup_db creates valid DB file", False, str(e))

# ─────────────────────────────────────────────────────────────────────────────
# Check 12: Alembic versions directory has migration file
# ─────────────────────────────────────────────────────────────────────────────
print("\n[12] Alembic — versions directory contains migration file")
try:
    versions_dir = ROOT / "alembic" / "versions"
    migration_files = [
        f for f in versions_dir.iterdir()
        if f.suffix == ".py" and not f.name.startswith("__")
    ] if versions_dir.exists() else []
    ok = len(migration_files) >= 1
    check("At least one Alembic migration file exists", ok,
          f"found: {[f.name for f in migration_files]}")
except Exception as e:
    check("Alembic versions directory check", False, str(e))

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
pct = 100 * passed // total if total else 0
print(f"Day 27 verification: {passed}/{total} checks passed ({pct}%)")
if passed == total:
    print("🎉 All checks passed — Day 27 complete!")
else:
    failed = [(label, detail) for label, ok, detail in results if not ok]
    print("Failed checks:")
    for label, detail in failed:
        print(f"  ❌ {label}: {detail}")
    sys.exit(1)
