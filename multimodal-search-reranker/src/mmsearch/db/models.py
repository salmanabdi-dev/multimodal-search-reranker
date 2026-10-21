"""Plain row types shared by the repository, retrieval and tests (no DB driver import)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChunkRow:
    chunk_id: int
    document_id: int
    external_id: str
    title: str | None
    content: str
    distance: float           # cosine distance from the query, in [0, 2]

    @property
    def similarity(self) -> float:
        return 1.0 - self.distance


@dataclass(frozen=True)
class ChunkInsert:
    chunk_index: int
    content: str
    token_count: int
    embedding: np.ndarray


@dataclass(frozen=True)
class ImageRow:
    document_id: int
    external_id: str
    caption: str | None
    file_path: str
    mime_type: str
    width: int
    height: int
    distance: float = 0.0     # cosine distance from the query (0 for direct lookups)

    @property
    def similarity(self) -> float:
        return 1.0 - self.distance
