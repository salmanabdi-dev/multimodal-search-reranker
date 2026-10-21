"""HNSW index management.

HNSW builds a multi-layer proximity graph. ``m`` is the number of links per node and
``ef_construction`` is the beam width used while inserting. Building once after a
bulk load is far faster than maintaining the graph during thousands of inserts.
"""

from __future__ import annotations

import logging
import re
import time

import asyncpg

log = logging.getLogger(__name__)
INDEX_NAME = "chunks_embedding_hnsw"


async def drop_hnsw(conn: asyncpg.Connection) -> None:
    await conn.execute(f"DROP INDEX IF EXISTS {INDEX_NAME}")


async def build_hnsw(
    conn: asyncpg.Connection, m: int, ef_construction: int, maintenance_work_mem: str
) -> float:
    if not re.fullmatch(r"\d+(kB|MB|GB)", maintenance_work_mem):
        raise ValueError(f"bad maintenance_work_mem: {maintenance_work_mem!r}")
    t0 = time.perf_counter()
    await conn.execute(f"SET maintenance_work_mem = '{maintenance_work_mem}'")
    await conn.execute(
        f"CREATE INDEX IF NOT EXISTS {INDEX_NAME} ON chunks "
        f"USING hnsw (embedding vector_cosine_ops) "
        f"WITH (m = {int(m)}, ef_construction = {int(ef_construction)})"
    )
    await conn.execute("ANALYZE chunks")
    elapsed = time.perf_counter() - t0
    log.info("built HNSW index (m=%d, ef_construction=%d) in %.1fs", m, ef_construction, elapsed)
    return elapsed
