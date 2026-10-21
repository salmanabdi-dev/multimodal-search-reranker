"""Stage 1: query vector -> HNSW -> candidate documents.

We search over *chunks* (that is what is embedded) but rank *documents* (that is what
users want and what relevance labels describe). Rows arrive sorted by distance, so
the first chunk we see for a document is its best one; later chunks of the same
document are dropped. We fetch more chunks (K) than documents needed (N) so enough
distinct documents survive the collapse.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from mmsearch.db.models import ChunkRow

if TYPE_CHECKING:
    from mmsearch.config import Settings


@dataclass(frozen=True)
class Candidate:
    document_id: int
    external_id: str
    title: str | None
    chunk_id: int
    content: str
    vector_score: float           # cosine similarity of the best chunk
    vector_rank: int              # 1-based rank after stage 1
    rerank_score: float | None = None


def collapse_to_documents(rows: Sequence[ChunkRow], n_docs: int) -> list[Candidate]:
    seen: set[int] = set()
    out: list[Candidate] = []
    for row in rows:
        if row.document_id in seen:
            continue
        seen.add(row.document_id)
        out.append(Candidate(
            document_id=row.document_id, external_id=row.external_id, title=row.title,
            chunk_id=row.chunk_id, content=row.content, vector_score=row.similarity,
            vector_rank=len(out) + 1,
        ))
        if len(out) == n_docs:
            break
    return out


class VectorRetriever:
    def __init__(self, repo, settings: Settings) -> None:  # noqa: ANN001 - Repository
        self.repo = repo
        self.settings = settings

    async def retrieve(self, query_vec: np.ndarray, n_docs: int,
                       ef_search: int | None = None) -> list[Candidate]:
        k = self.settings.k_chunks_for(n_docs)
        ef = max(ef_search if ef_search is not None else self.settings.ef_search, k)
        # The pool sets ef_search=settings.ef_search per session; only override (one extra
        # statement) when a different value is needed.
        override = None if ef == self.settings.ef_search else ef
        rows = await self.repo.ann_search(query_vec, k, override)
        return collapse_to_documents(rows, n_docs)
