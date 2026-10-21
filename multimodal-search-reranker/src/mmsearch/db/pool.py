"""asyncpg connection pool with the pgvector codec registered on every connection.

``register_vector`` teaches asyncpg to send/receive numpy arrays as the ``vector``
type in binary form, so query embeddings travel to Postgres without string formatting.
"""

from __future__ import annotations

from pathlib import Path

import asyncpg
from pgvector.asyncpg import register_vector

SQL_DIR = Path(__file__).resolve().parents[3] / "sql"


def migration_files(sql_dir: Path = SQL_DIR) -> list[Path]:
    """001_init.sql, 002_images.sql, ... in order. 000_* runs only at cluster creation."""
    return sorted(p for p in sql_dir.glob("[0-9][0-9][0-9]_*.sql") if not p.name.startswith("000_"))


async def _init_connection(conn: asyncpg.Connection, ef_search: int) -> None:
    await register_vector(conn)
    # Session default; individual requests can still override it per transaction.
    await conn.execute(f"SET hnsw.ef_search = {int(ef_search)}")


async def create_pool(dsn: str, min_size: int, max_size: int, ef_search: int) -> asyncpg.Pool:
    async def init(conn: asyncpg.Connection) -> None:
        await _init_connection(conn, ef_search)

    return await asyncpg.create_pool(
        dsn, min_size=min_size, max_size=max_size, init=init, command_timeout=30
    )


MIGRATION_LOCK_ID = 7_340_001  # arbitrary, constant across processes


async def init_db(dsn: str, sql_dir: Path = SQL_DIR) -> None:
    """Apply all migrations (each is idempotent). Must run before ``create_pool``.

    The API and the CLI both call this at start-up, often at the same moment
    (``make demo``). Two concurrent ``CREATE TABLE IF NOT EXISTS`` can still collide in
    the system catalogs, so the whole migration runs under a transaction-scoped
    advisory lock: the second caller waits, then finds everything already in place.
    """
    conn = await asyncpg.connect(dsn)
    try:
        async with conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock($1)", MIGRATION_LOCK_ID)
            for f in migration_files(sql_dir):
                await conn.execute(f.read_text())
    finally:
        await conn.close()
