"""Merging per-modality results into one ranked list.

Each modality is ranked by its own best signal (text: cross-encoder; images: caption
fusion over CLIP), and each final score is calibrated to a relevance probability
(see calibration.py). Only then are the lists merged, by that probability. Merging
raw scores would let whichever modality happens to produce bigger numbers win.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from mmsearch.retrieval.image_search import ImageCandidate
from mmsearch.retrieval.vector_search import Candidate
from mmsearch.search.calibration import Platt, sigmoid

MODALITIES = ("text", "image")


@dataclass(frozen=True)
class UnifiedHit:
    modality: str                       # "text" | "image"
    relevance: float                    # calibrated probability, comparable across modalities
    text: Candidate | None = None
    image: ImageCandidate | None = None


def text_relevance(c: Candidate, vector_calibration: Platt) -> float:
    if c.rerank_score is not None:
        return sigmoid(c.rerank_score)   # ms-marco logits are ~log-odds already
    return vector_calibration(c.vector_score)


def merge_hits(text: Sequence[Candidate], images: Sequence[ImageCandidate], top_k: int,
               vector_calibration: Platt) -> list[UnifiedHit]:
    hits = [UnifiedHit("text", text_relevance(c, vector_calibration), text=c) for c in text]
    hits += [UnifiedHit("image", c.relevance if c.relevance is not None else 0.0, image=c)
             for c in images]
    # Stable sort: within a modality the per-modality order is preserved.
    return sorted(hits, key=lambda h: h.relevance, reverse=True)[:top_k]
