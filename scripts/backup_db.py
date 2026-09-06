"""
scripts/backup_db.py — AgentLens Database Backup Utility
=========================================================
Day 27: Creates a timestamped copy of the AgentLens SQLite database in
the backups/ directory.

Usage:
    python scripts/backup_db.py                       # uses default DB path
    python scripts/backup_db.py --db-path custom.db   # custom DB path
    python scripts/backup_db.py --keep 10             # keep only last N backups

Output:
    backups/agentlens_2026-09-06T21-05-33.db

The script uses SQLite's built-in backup API (sqlite3.Connection.backup) which
produces a clean, consistent snapshot even if the DB is actively being written.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path


def _timestamp_str() -> str:
    """Return a filesystem-safe ISO-8601 timestamp (colons replaced with dashes)."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")


def backup_database(
    db_path: str = "data/agentlens.db",
    backup_dir: str = "backups",
    keep: int | None = None,
    quiet: bool = False,
) -> Path:
    """
    Create a timestamped backup copy of the AgentLens SQLite database.

    Args:
        db_path:    Path to the source database.
        backup_dir: Directory to write backups into (created if missing).
        keep:       If set, delete oldest backups keeping only this many.
        quiet:      Suppress console output.

    Returns:
        Path to the created backup file.

    Raises:
        FileNotFoundError: If the source database does not exist.
        RuntimeError:      If the backup fails for any reason.
    """
    src = Path(db_path)
    if not src.exists():
        raise FileNotFoundError(f"Source database not found: {src}")

    out_dir = Path(backup_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = src.stem  # e.g. "agentlens"
    ts = _timestamp_str()
    dest = out_dir / f"{stem}_{ts}.db"

    # Use SQLite's native backup API for a consistent snapshot
    try:
        src_conn = sqlite3.connect(str(src))
        dst_conn = sqlite3.connect(str(dest))
        with dst_conn:
            src_conn.backup(dst_conn)
        src_conn.close()
        dst_conn.close()
    except Exception as exc:
        raise RuntimeError(f"Backup failed: {exc}") from exc

    size_kb = dest.stat().st_size / 1024
    if not quiet:
        print(f"[backup_db] Backup created: {dest}  ({size_kb:.1f} KB)")

    # Prune old backups if requested
    if keep is not None and keep > 0:
        _prune_old_backups(out_dir, stem, keep, quiet=quiet)

    return dest


def _prune_old_backups(out_dir: Path, stem: str, keep: int, quiet: bool = False) -> None:
    """Delete oldest backups, keeping only the most recent `keep` files."""
    pattern = re.compile(rf"^{re.escape(stem)}_\d{{4}}-\d{{2}}-\d{{2}}T\d{{2}}-\d{{2}}-\d{{2}}\.db$")
    backups = sorted(
        [f for f in out_dir.iterdir() if f.is_file() and pattern.match(f.name)],
        key=lambda f: f.name,
    )
    while len(backups) > keep:
        oldest = backups.pop(0)
        oldest.unlink()
        if not quiet:
            print(f"[backup_db] Pruned old backup: {oldest.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a timestamped backup of the AgentLens SQLite database."
    )
    parser.add_argument(
        "--db-path",
        default="data/agentlens.db",
        help="Path to the source database (default: data/agentlens.db)",
    )
    parser.add_argument(
        "--backup-dir",
        default="backups",
        help="Directory to write backups into (default: backups/)",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=None,
        help="Keep only the N most recent backups (optional)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output",
    )
    args = parser.parse_args()

    try:
        dest = backup_database(
            db_path=args.db_path,
            backup_dir=args.backup_dir,
            keep=args.keep,
            quiet=args.quiet,
        )
        if not args.quiet:
            print(f"[backup_db] Done: {dest}")
    except FileNotFoundError as e:
        print(f"[backup_db] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"[backup_db] ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
