"""Stage 2 for images: caption fusion.

Why not a visual cross-encoder? Models such as BLIP's image-text matching head *are*
the image analogue of our text cross-encoder (cross-attention between query tokens
and image patches), but the vision tower must run per (query, image) pair at query
time: hundreds of milliseconds per query on CPU, which breaks the latency budget.

What we do instead, per image candidate:
* p_clip     = calibrated CLIP similarity (always available)
* p_caption  = sigmoid(cross-encoder logit on (query, caption)), when a caption exists
* relevance  = w * p_caption + (1 - w) * p_clip, or p_clip alone without a caption

The caption scores come from the *same* text cross-encoder call as the text
passages (one batched forward pass), so image reranking adds a few short pairs to an
existing batch rather than a new model call. Uncaptioned images keep CLIP order.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

from mmsearch.retrieval.image_search import ImageCandidate
from mmsearch.search.calibration import Platt, sigmoid


class CaptionFusionReranker:
    def __init__(self, clip_calibration: Platt, caption_weight: float) -> None:
        self.calib = clip_calibration
        self.w = caption_weight

    def relevance(self, clip_score: float, caption_logit: float | None) -> float:
        p_clip = self.calib(clip_score)
        if caption_logit is None:
            return p_clip
        return self.w * sigmoid(caption_logit) + (1 - self.w) * p_clip

    def fuse(self, candidates: Sequence[ImageCandidate],
             caption_logits: Mapping[int, float]) -> list[ImageCandidate]:
        """caption_logits: candidate index -> cross-encoder logit (captioned ones only)."""
        out = [
            dataclasses.replace(c, caption_score=caption_logits.get(i),
                                relevance=self.relevance(c.clip_score, caption_logits.get(i)))
            for i, c in enumerate(candidates)
        ]
        return sorted(out, key=lambda c: c.relevance, reverse=True)

    def calibrate_only(self, candidates: Sequence[ImageCandidate]) -> list[ImageCandidate]:
        """No reranking: CLIP order, but with comparable relevance attached for merging."""
        return [dataclasses.replace(c, relevance=self.calib(c.clip_score)) for c in candidates]
