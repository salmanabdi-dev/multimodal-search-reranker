"""Ingestion: load -> clean -> chunk -> embed (batched) -> store -> build HNSW."""

from __future__ import annotations

import itertools
import logging
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass

from mmsearch.config import Settings
from mmsearch.db.index import build_hnsw, drop_hnsw
from mmsearch.db.models import ChunkInsert
from mmsearch.db.repository import Repository
from mmsearch.embeddings.base import Encoder
from mmsearch.ingestion.chunker import TokenChunker
from mmsearch.ingestion.loaders import RawDocument
from mmsearch.ingestion.text import clean_text, passage_text

log = logging.getLogger(__name__)


@dataclass
class IngestStats:
    documents: int = 0
    chunks: int = 0
    skipped: int = 0
    embed_seconds: float = 0.0
    db_seconds: float = 0.0
    index_seconds: float = 0.0
    total_seconds: float = 0.0

    def as_dict(self) -> dict:
        return {k: round(v, 2) if isinstance(v, float) else v for k, v in asdict(self).items()}


def _batched(items: Iterable[RawDocument], n: int) -> Iterator[list[RawDocument]]:
    it = iter(items)
    while batch := list(itertools.islice(it, n)):
        yield batch


class IngestionPipeline:
    def __init__(self, repo: Repository, encoder: Encoder, chunker: TokenChunker,
                 settings: Settings) -> None:
        if encoder.dim != settings.embedding_dim:
            raise ValueError(
                f"encoder dim {encoder.dim} != MMS_EMBEDDING_DIM {settings.embedding_dim} "
                "(and the vector(N) column in sql/001_init.sql)"
            )
        self.repo = repo
        self.encoder = encoder
        self.chunker = chunker
        self.settings = settings

    async def run(self, docs: Iterable[RawDocument], *, rebuild_index: bool = True,
                  batch_docs: int = 128) -> IngestStats:
        stats = IngestStats()
        t0 = time.perf_counter()
        if rebuild_index:
            async with self.repo.pool.acquire() as conn:
                await drop_hnsw(conn)

        for batch in _batched(docs, batch_docs):
            await self._ingest_batch(batch, stats)
            log.info("ingested %d documents / %d chunks", stats.documents, stats.chunks)

        if rebuild_index:
            async with self.repo.pool.acquire() as conn:
                stats.index_seconds = await build_hnsw(
                    conn, self.settings.hnsw_m, self.settings.hnsw_ef_construction,
                    self.settings.index_maintenance_work_mem,
                )
        stats.total_seconds = time.perf_counter() - t0
        return stats

    async def _ingest_batch(self, batch: list[RawDocument], stats: IngestStats) -> None:
        prepared = []
        passages: list[str] = []
        for doc in batch:
            title = clean_text(doc.title) if doc.title else None
            chunks = self.chunker.chunk(clean_text(doc.text), title)
            if not chunks:
                stats.skipped += 1
                continue
            prepared.append((doc, title, chunks))
            passages.extend(passage_text(title, c.content) for c in chunks)
        if not prepared:
            return

        t = time.perf_counter()
        vectors = self.encoder.encode(passages, batch_size=self.settings.embed_batch_size)
        stats.embed_seconds += time.perf_counter() - t

        t = time.perf_counter()
        i = 0
        async with self.repo.pool.acquire() as conn, conn.transaction():
            for doc, title, chunks in prepared:
                rows = [ChunkInsert(c.index, c.content, c.token_count, vectors[i + j])
                        for j, c in enumerate(chunks)]
                i += len(chunks)
                await self.repo.upsert_document_with_chunks(
                    conn, external_id=doc.external_id, title=title, source=doc.source,
                    metadata=doc.metadata, chunks=rows, embedding_model=self.encoder.name,
                )
                stats.documents += 1
                stats.chunks += len(chunks)
        stats.db_seconds += time.perf_counter() - t
