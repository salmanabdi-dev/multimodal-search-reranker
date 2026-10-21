"""Cross-modal stage 1: CLIP text vector -> image HNSW -> image candidates.

One image is one vector (no chunking), so there is no collapse step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ImageCandidate:
    document_id: int
    external_id: str
    caption: str | None
    file_path: str
    mime_type: str
    width: int
    height: int
    clip_score: float                 # cosine(query text, image) in CLIP space
    vector_rank: int
    caption_score: float | None = None   # cross-encoder logit on (query, caption)
    relevance: float | None = None       # calibrated, comparable across modalities


class ImageRetriever:
    def __init__(self, repo, settings) -> None:  # noqa: ANN001
        self.repo = repo
        self.settings = settings

    async def retrieve(self, query_vec: np.ndarray, n: int,
                       ef_search: int | None = None) -> list[ImageCandidate]:
        ef = max(ef_search if ef_search is not None else self.settings.ef_search, n)
        override = None if ef == self.settings.ef_search else ef
        rows = await self.repo.image_ann_search(query_vec, n, override)
        return [
            ImageCandidate(document_id=r.document_id, external_id=r.external_id,
                           caption=r.caption, file_path=r.file_path, mime_type=r.mime_type,
                           width=r.width, height=r.height, clip_score=r.similarity,
                           vector_rank=i)
            for i, r in enumerate(rows, 1)
        ]
