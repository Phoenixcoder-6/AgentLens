"""add_performance_indexes

Revision ID: ec1ab42def9b
Revises:
Create Date: 2026-09-06 21:07:33.860651

Day 27 — DB Hardening: Add performance indexes on high-frequency query columns.

Indexes added:
    ix_runs_timestamp     — runs.timestamp    (newest-first list queries)
    ix_analysis_run_id    — analysis.run_id   (FK join from analysis table)
    ix_llm_cache_key      — llm_cache.cache_key   (cache lookup hot path)
    ix_llm_cache_expires  — llm_cache.expires_at  (purge expired cache)
    ix_steps_run_id       — steps.run_id      (FK join in get_steps_for_run)

Note on SQLite:
    CREATE INDEX IF NOT EXISTS is used so this migration is idempotent
    and safe to apply even if some indexes already exist.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "ec1ab42def9b"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add performance indexes to high-frequency query columns."""
    # Use op.execute() — the correct Alembic pattern for raw DDL.
    # op.execute() uses the current migration connection automatically.
    op.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_runs_timestamp ON runs (timestamp DESC)"))
    op.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_analysis_run_id ON analysis (run_id)"))
    op.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_llm_cache_key ON llm_cache (cache_key)"))
    op.execute(
        sa.text("CREATE INDEX IF NOT EXISTS ix_llm_cache_expires_at ON llm_cache (expires_at)")
    )
    op.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_steps_run_id ON steps (run_id)"))


def downgrade() -> None:
    """Drop the performance indexes added in this migration."""
    op.execute(sa.text("DROP INDEX IF EXISTS ix_runs_timestamp"))
    op.execute(sa.text("DROP INDEX IF EXISTS ix_analysis_run_id"))
    op.execute(sa.text("DROP INDEX IF EXISTS ix_llm_cache_key"))
    op.execute(sa.text("DROP INDEX IF EXISTS ix_llm_cache_expires_at"))
    op.execute(sa.text("DROP INDEX IF EXISTS ix_steps_run_id"))
