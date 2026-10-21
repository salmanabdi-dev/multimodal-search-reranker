"""Every SQL statement in the project lives here.

Distances: chunks are stored L2-normalised and ``<=>`` is pgvector's cosine
distance, so ``similarity = 1 - distance`` is the cosine similarity.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import asyncpg
import numpy as np

from mmsearch.db.models import ChunkInsert, ChunkRow, ImageRow

# The ANN scan sits in a CTE so the join to documents cannot change the plan:
# Postgres runs ORDER BY ... LIMIT against the HNSW index, then joins only K rows.
_ANN_SQL = """
WITH ann AS (
    SELECT id, document_id, content, embedding <=> $1 AS distance
    FROM chunks
    ORDER BY embedding <=> $1
    LIMIT $2
)
SELECT ann.id AS chunk_id, ann.document_id, ann.content, ann.distance,
       d.external_id, d.title
FROM ann JOIN documents d ON d.id = ann.document_id
ORDER BY ann.distance
"""


_IMAGE_ANN_SQL = """
WITH ann AS (
    SELECT document_id, embedding <=> $1 AS distance
    FROM image_embeddings
    ORDER BY embedding <=> $1
    LIMIT $2
)
SELECT ann.document_id, ann.distance, d.external_id, i.caption, i.file_path,
       i.mime_type, i.width, i.height
FROM ann
JOIN documents d ON d.id = ann.document_id
JOIN image_embeddings i ON i.document_id = ann.document_id
ORDER BY ann.distance
"""


def _image_rows(records: Sequence[Any]) -> list[ImageRow]:
    return [
        ImageRow(document_id=r["document_id"], external_id=r["external_id"],
                 caption=r["caption"], file_path=r["file_path"], mime_type=r["mime_type"],
                 width=r["width"], height=r["height"], distance=float(r["distance"]))
        for r in records
    ]


def _rows(records: Sequence[Any]) -> list[ChunkRow]:
    return [
        ChunkRow(
            chunk_id=r["chunk_id"],
            document_id=r["document_id"],
            external_id=r["external_id"],
            title=r["title"],
            content=r["content"],
            distance=float(r["distance"]),
        )
        for r in records
    ]


class Repository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self.pool = pool

    # ---------------- writes ----------------

    async def upsert_document_with_chunks(
        self,
        conn: asyncpg.Connection,
        *,
        external_id: str,
        title: str | None,
        source: str | None,
        metadata: dict[str, Any],
        chunks: Sequence[ChunkInsert],
        embedding_model: str,
    ) -> int:
        """Insert or replace one document and all of its chunks (call inside a transaction)."""
        doc_id = await conn.fetchval(
            """
            INSERT INTO documents (external_id, title, source, metadata)
            VALUES ($1, $2, $3, $4::jsonb)
            ON CONFLICT (external_id) DO UPDATE
                SET title = EXCLUDED.title, source = EXCLUDED.source,
                    metadata = EXCLUDED.metadata
            RETURNING id
            """,
            external_id, title, source, json.dumps(metadata),
        )
        await conn.execute("DELETE FROM chunks WHERE document_id = $1", doc_id)
        await conn.executemany(
            """
            INSERT INTO chunks (document_id, chunk_index, content, token_count,
                                embedding, embedding_model)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            [(doc_id, c.chunk_index, c.content, c.token_count, c.embedding, embedding_model)
             for c in chunks],
        )
        return doc_id

    # ---------------- reads ----------------

    async def ann_search(
        self, query_vec: np.ndarray, k: int, ef_search: int | None = None
    ) -> list[ChunkRow]:
        """Approximate nearest neighbours via the HNSW index (one round trip by default)."""
        async with self.pool.acquire() as conn:
            if ef_search is None:
                return _rows(await conn.fetch(_ANN_SQL, query_vec, k))
            async with conn.transaction():
                # set_config(..., is_local=true) == SET LOCAL, but accepts a bind parameter.
                await conn.execute(
                    "SELECT set_config('hnsw.ef_search', $1, true)", str(int(ef_search))
                )
                return _rows(await conn.fetch(_ANN_SQL, query_vec, k))

    async def exact_search(self, query_vec: np.ndarray, k: int) -> list[ChunkRow]:
        """Brute-force search with the index disabled: ground truth for ANN recall."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SET LOCAL enable_indexscan = off")
                return _rows(await conn.fetch(_ANN_SQL, query_vec, k))

    async def stats(self) -> dict[str, Any]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT count(*) AS total, "
                "count(*) FILTER (WHERE modality = 'text') AS text, "
                "count(*) FILTER (WHERE modality = 'image') AS image FROM documents"
            )
            chunks = await conn.fetchval("SELECT count(*) FROM chunks")
            models = [r["embedding_model"] for r in
                      await conn.fetch("SELECT DISTINCT embedding_model FROM chunks")]
            image_models = [r["embedding_model"] for r in
                            await conn.fetch("SELECT DISTINCT embedding_model FROM image_embeddings")]
            indexes = {r["indexname"] for r in await conn.fetch(
                "SELECT indexname FROM pg_indexes WHERE indexname IN "
                "('chunks_embedding_hnsw', 'image_embeddings_hnsw')")}
            pgvector = await conn.fetchval(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            )
        return {"documents": row["total"], "text_documents": row["text"], "chunks": chunks,
                "embedding_models": models, "hnsw_index": "chunks_embedding_hnsw" in indexes,
                "images": row["image"], "image_embedding_models": image_models,
                "image_hnsw_index": "image_embeddings_hnsw" in indexes,
                "pgvector_version": pgvector}

    # ---------------- images ----------------

    async def find_image_by_sha(self, conn: asyncpg.Connection, sha256: str) -> dict | None:
        r = await conn.fetchrow(
            "SELECT d.id, d.external_id, i.width, i.height FROM image_embeddings i "
            "JOIN documents d ON d.id = i.document_id WHERE i.sha256 = $1", sha256)
        return dict(r) if r else None

    async def set_image_caption(self, conn: asyncpg.Connection, document_id: int,
                                caption: str) -> None:
        await conn.execute("UPDATE image_embeddings SET caption = $2 WHERE document_id = $1",
                           document_id, caption)
        await conn.execute("UPDATE documents SET title = $2 WHERE id = $1", document_id, caption)

    async def upsert_image(
        self, conn: asyncpg.Connection, *, external_id: str, caption: str | None,
        source: str | None, metadata: dict[str, Any], sha256: str, file_path: str,
        mime_type: str, width: int, height: int, nbytes: int, embedding: np.ndarray,
        embedding_model: str,
    ) -> int:
        """Insert or replace one image (call inside a transaction)."""
        doc_id = await conn.fetchval(
            """
            INSERT INTO documents (external_id, title, source, modality, metadata)
            VALUES ($1, $2, $3, 'image', $4::jsonb)
            ON CONFLICT (external_id) DO UPDATE
                SET title = EXCLUDED.title, source = EXCLUDED.source,
                    metadata = EXCLUDED.metadata
                WHERE documents.modality = 'image'
            RETURNING id
            """,
            external_id, caption, source, json.dumps(metadata),
        )
        if doc_id is None:  # the conflict was with a text document: refuse, don't convert
            raise ValueError(f"external_id {external_id!r} already belongs to a text document")
        await conn.execute(
            """
            INSERT INTO image_embeddings (document_id, sha256, file_path, mime_type, width,
                                          height, bytes, caption, embedding, embedding_model)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (document_id) DO UPDATE
                SET sha256 = EXCLUDED.sha256, file_path = EXCLUDED.file_path,
                    mime_type = EXCLUDED.mime_type, width = EXCLUDED.width,
                    height = EXCLUDED.height, bytes = EXCLUDED.bytes,
                    caption = EXCLUDED.caption, embedding = EXCLUDED.embedding,
                    embedding_model = EXCLUDED.embedding_model
            """,
            doc_id, sha256, file_path, mime_type, width, height, nbytes, caption,
            embedding, embedding_model,
        )
        return doc_id

    async def image_ann_search(self, query_vec: np.ndarray, k: int,
                               ef_search: int | None = None) -> list[ImageRow]:
        async with self.pool.acquire() as conn:
            if ef_search is None:
                return _image_rows(await conn.fetch(_IMAGE_ANN_SQL, query_vec, k))
            async with conn.transaction():
                await conn.execute(
                    "SELECT set_config('hnsw.ef_search', $1, true)", str(int(ef_search)))
                return _image_rows(await conn.fetch(_IMAGE_ANN_SQL, query_vec, k))

    async def image_exact_search(self, query_vec: np.ndarray, k: int) -> list[ImageRow]:
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SET LOCAL enable_indexscan = off")
                return _image_rows(await conn.fetch(_IMAGE_ANN_SQL, query_vec, k))

    async def get_image(self, external_id: str) -> ImageRow | None:
        async with self.pool.acquire() as conn:
            r = await conn.fetchrow(
                "SELECT d.id AS document_id, d.external_id, i.caption, i.file_path, "
                "i.mime_type, i.width, i.height, 0.0::float8 AS distance "
                "FROM documents d JOIN image_embeddings i ON i.document_id = d.id "
                "WHERE d.external_id = $1", external_id)
        return _image_rows([r])[0] if r else None

    async def image_count(self) -> int:
        async with self.pool.acquire() as conn:
            return await conn.fetchval("SELECT count(*) FROM image_embeddings")

    async def ping(self) -> bool:
        async with self.pool.acquire() as conn:
            return await conn.fetchval("SELECT 1") == 1
