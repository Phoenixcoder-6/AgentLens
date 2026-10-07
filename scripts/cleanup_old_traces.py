"""
scripts/cleanup_old_traces.py — Trace Retention Cleanup Utility
================================================================
Day 39: Deletes traces older than `capture.retention_days` (default: 90 days)
from both `data/traces/*.json` and `data/agentlens.db`.

Usage:
    python scripts/cleanup_old_traces.py
    python scripts/cleanup_old_traces.py --days 30 --dry-run
    python scripts/cleanup_old_traces.py --days 90 --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config_loader import get  # noqa: E402
from storage.db import DatabaseManager  # noqa: E402


def _parse_iso_timestamp(ts_str: str) -> datetime | None:
    """Parse an ISO-8601 timestamp string into a UTC-aware datetime."""
    if not ts_str or not isinstance(ts_str, str):
        return None
    try:
        cleaned = ts_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except Exception:
        return None


def _get_trace_file_timestamp(file_path: Path) -> datetime:
    """
    Determine the timestamp of a JSON trace file.
    Prefers the internal `timestamp` JSON field; falls back to file mtime.
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "timestamp" in payload:
            parsed = _parse_iso_timestamp(str(payload["timestamp"]))
            if parsed is not None:
                return parsed
    except Exception:
        pass

    mtime = os.path.getmtime(file_path)
    return datetime.fromtimestamp(mtime, tz=UTC)


def cleanup_old_traces(
    retention_days: int | None = None,
    db_path: str | None = None,
    traces_dir: str | None = None,
    dry_run: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    """
    Purge JSON trace files and SQLite database runs older than `retention_days`.

    Returns a summary dictionary with counts and lists of removed files/runs.
    """
    if retention_days is None:
        retention_days = int(get("capture", "retention_days", 90) or 90)
    if db_path is None:
        db_path = str(get("storage", "db_path", "data/agentlens.db") or "data/agentlens.db")
    if traces_dir is None:
        traces_dir = str(get("storage", "traces_dir", "data/traces") or "data/traces")

    current_time = now.astimezone(UTC) if now is not None else datetime.now(UTC)
    cutoff_dt = current_time - timedelta(days=retention_days)
    cutoff_iso = cutoff_dt.isoformat()

    deleted_files: list[str] = []
    kept_files: int = 0

    traces_path = Path(traces_dir)
    if traces_path.exists() and traces_path.is_dir():
        for json_file in sorted(traces_path.glob("*.json")):
            file_dt = _get_trace_file_timestamp(json_file)
            if file_dt < cutoff_dt:
                deleted_files.append(str(json_file))
                if not dry_run:
                    try:
                        json_file.unlink()
                    except OSError:
                        pass
            else:
                kept_files += 1

    deleted_db_runs: list[str] = []
    if db_path and os.path.exists(db_path):
        db = DatabaseManager(db_path=db_path)
        db.initialize()
        deleted_db_runs = db.delete_runs_older_than(cutoff_iso=cutoff_iso, dry_run=dry_run)

    return {
        "retention_days": retention_days,
        "cutoff_iso": cutoff_iso,
        "dry_run": dry_run,
        "deleted_files_count": len(deleted_files),
        "deleted_files": deleted_files,
        "kept_files_count": kept_files,
        "deleted_db_runs_count": len(deleted_db_runs),
        "deleted_db_runs": deleted_db_runs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Delete AgentLens traces older than capture.retention_days."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Override capture.retention_days from config/config.yaml",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Override SQLite database path",
    )
    parser.add_argument(
        "--traces-dir",
        type=str,
        default=None,
        help="Override trace JSON directory path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview files and DB runs that would be deleted without removing them",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print summary as JSON",
    )
    args = parser.parse_args(argv)

    summary = cleanup_old_traces(
        retention_days=args.days,
        db_path=args.db_path,
        traces_dir=args.traces_dir,
        dry_run=args.dry_run,
    )

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        mode = "DRY-RUN" if summary["dry_run"] else "LIVE"
        print(
            f"[Trace Cleanup ({mode})] retention_days={summary['retention_days']} "
            f"(cutoff={summary['cutoff_iso']})"
        )
        print(
            f"  JSON trace files removed : {summary['deleted_files_count']} "
            f"(kept: {summary['kept_files_count']})"
        )
        print(f"  SQLite DB runs removed   : {summary['deleted_db_runs_count']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
