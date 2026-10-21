"""The two-stage search path. API, evaluation and benchmarks all call this, so what we
measure is exactly what we serve.

Model forward passes are CPU-bound and would block the event loop, so they run on a
small dedicated thread pool (PyTorch releases the GIL inside its kernels). One
inference thread by default: on CPU, two concurrent forward passes just fight over
the same cores.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

from mmsearch.ingestion.text import passage_text
from mmsearch.observability.timing import StageTimer
from mmsearch.retrieval.image_search import ImageCandidate
from mmsearch.retrieval.vector_search import Candidate, VectorRetriever
from mmsearch.search.calibration import Platt
from mmsearch.search.unified import MODALITIES, UnifiedHit, merge_hits

if TYPE_CHECKING:
    from mmsearch.config import Settings

log = logging.getLogger(__name__)

_WARMUP_QUERIES = [
    "test",
    "what are the health effects of vitamin d",
    "how does a cross encoder rerank search results compared with a bi encoder retriever "
    "when queries are long and contain several distinct clauses",
]


@dataclass
class SearchOutcome:
    query: str
    results: list[Candidate]
    timings_ms: dict[str, float]
    reranked: bool
    candidates_considered: int


@dataclass
class UnifiedOutcome:
    query: str
    hits: list[UnifiedHit]
    timings_ms: dict[str, float]
    reranked: bool
    modalities: list[str]               # modalities actually searched
    candidates_considered: dict[str, int]


class SearchService:
    def __init__(self, encoder: Any, retriever: VectorRetriever, reranker: Any,
                 settings: Settings, executor_initializer: Callable[[], None] | None = None,
                 *, clip: Any = None, image_retriever: Any = None, image_reranker: Any = None,
                 text_vec_calibration: Platt | None = None, image_count: int = 0,
                 ) -> None:
        self.encoder = encoder
        self.retriever = retriever
        self.reranker = reranker
        self.settings = settings
        # Image side is optional: without it the service is exactly the text system.
        self.clip = clip
        self.image_retriever = image_retriever
        self.image_reranker = image_reranker
        self.text_vec_calibration = text_vec_calibration or Platt(
            getattr(settings, "text_vec_calib_a", 12.0), getattr(settings, "text_vec_calib_b", -6.0))
        self.image_count = image_count  # cached; skip the image branch on an empty index
        self._executor = ThreadPoolExecutor(
            max_workers=settings.inference_workers, thread_name_prefix="inference",
            initializer=executor_initializer,
        )

    @property
    def images_enabled(self) -> bool:
        return self.clip is not None and self.image_retriever is not None

    async def _run(self, fn: Callable, *args: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, partial(fn, *args))

    run = _run  # public: the image upload path runs CLIP on the same executor

    # -- building blocks (also used by evaluation) --

    async def embed_query(self, query: str):  # noqa: ANN201
        return await self._run(self.encoder.encode_query, query)

    async def retrieve(self, query: str, n_docs: int, ef_search: int | None = None
                       ) -> list[Candidate]:
        vec = await self.embed_query(query)
        return await self.retriever.retrieve(vec, n_docs, ef_search)

    async def rerank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        if not candidates:
            return []
        passages = [passage_text(c.title, c.content) for c in candidates]
        scores = await self._run(self.reranker.score, query, passages)
        scored = [dataclasses.replace(c, rerank_score=float(s))
                  for c, s in zip(candidates, scores, strict=True)]
        # Stable sort: equal scores keep stage-1 order.
        return sorted(scored, key=lambda c: c.rerank_score, reverse=True)

    async def retrieve_images(self, query: str, n: int, ef_search: int | None = None
                              ) -> list[ImageCandidate]:
        vec = await self._run(self.clip.encode_query, query)
        return await self.image_retriever.retrieve(vec, n, ef_search)

    async def rerank_images(self, query: str, candidates: list[ImageCandidate]
                            ) -> list[ImageCandidate]:
        _, images = await self._rerank_both(query, [], candidates)
        return images

    async def _rerank_both(self, query: str, text: list[Candidate],
                           images: list[ImageCandidate]
                           ) -> tuple[list[Candidate], list[ImageCandidate]]:
        """One cross-encoder pass over text passages AND image captions."""
        captioned = [i for i, c in enumerate(images) if c.caption]
        pairs = [passage_text(c.title, c.content) for c in text]
        pairs += [images[i].caption for i in captioned]
        scores = await self._run(self.reranker.score, query, pairs) if pairs else []
        text_scored = sorted(
            (dataclasses.replace(c, rerank_score=float(s))
             for c, s in zip(text, scores[:len(text)], strict=True)),
            key=lambda c: c.rerank_score, reverse=True)
        caption_logits = {i: float(s) for i, s in zip(captioned, scores[len(text):], strict=True)}
        return text_scored, (self.image_reranker.fuse(images, caption_logits) if images else [])

    # -- the request paths --

    async def search_unified(self, query: str, top_k: int | None = None, *,
                             modalities: tuple[str, ...] | list[str] = MODALITIES,
                             rerank: bool = True, n_text: int | None = None,
                             n_image: int | None = None) -> UnifiedOutcome:
        """n_text / n_image override the configured candidate counts (used by the
        optimisation sweep; the API always uses the configured values)."""
        top_k = top_k or self.settings.default_top_k
        do_text = "text" in modalities
        do_image = "image" in modalities and self.images_enabled and self.image_count > 0
        timer = StageTimer()
        tasks: dict[str, asyncio.Task] = {}
        try:
            # Encodings run on the inference thread one after another (they share the
            # cores); each ANN query is launched as soon as its vector exists, so the
            # text DB round trip overlaps with CLIP encoding.
            if do_text:
                n = max(n_text or self.settings.rerank_candidates, top_k) if rerank else top_k
                with timer.stage("embed_text"):
                    tvec = await self._run(self.encoder.encode_query, query)
                tasks["text"] = asyncio.create_task(self.retriever.retrieve(tvec, n))
            if do_image:
                m = max(n_image or self.settings.image_candidates, top_k) if rerank else top_k
                with timer.stage("embed_clip"):
                    cvec = await self._run(self.clip.encode_query, query)
                tasks["image"] = asyncio.create_task(self.image_retriever.retrieve(cvec, m))
            with timer.stage("retrieve"):
                text = await tasks["text"] if "text" in tasks else []
                images = await tasks["image"] if "image" in tasks else []
        finally:
            for t in tasks.values():
                if not t.done():
                    t.cancel()

        considered = {"text": len(text), "image": len(images)}
        if rerank and (text or images):
            with timer.stage("rerank"):
                text, images = await self._rerank_both(query, text, images)
        elif images:
            images = self.image_reranker.calibrate_only(images)
        with timer.stage("merge"):
            hits = merge_hits(text, images, top_k, self.text_vec_calibration)

        searched = [m for m, on in (("text", do_text), ("image", do_image)) if on]
        return UnifiedOutcome(query=query, hits=hits, timings_ms=timer.as_dict(),
                              reranked=rerank, modalities=searched,
                              candidates_considered=considered)

    async def search(self, query: str, top_k: int | None = None, *, rerank: bool = True,
                     n_candidates: int | None = None) -> SearchOutcome:
        top_k = top_k or self.settings.default_top_k
        n = max(n_candidates or self.settings.rerank_candidates, top_k) if rerank else top_k
        timer = StageTimer()

        with timer.stage("embed"):
            vec = await self.embed_query(query)
        with timer.stage("retrieve"):
            candidates = await self.retriever.retrieve(vec, n)
        if rerank and candidates:
            with timer.stage("rerank"):
                candidates = await self.rerank(query, candidates)

        return SearchOutcome(query=query, results=candidates[:top_k],
                             timings_ms=timer.as_dict(), reranked=rerank,
                             candidates_considered=len(candidates))

    async def warmup(self, rounds: int = 5, with_db: bool = True) -> None:
        """First forward passes are slow (allocations, kernel selection, lazy init).
        Pay that cost at start-up, not on a user's first request."""
        n = self.settings.rerank_candidates
        dummy = ["warm up passage " * k for k in range(1, n + 1)]
        for _ in range(rounds):
            for q in _WARMUP_QUERIES:
                await self._run(self.encoder.encode_query, q)
                await self._run(self.reranker.score, q, dummy)
                if self.images_enabled:
                    await self._run(self.clip.encode_query, q)
                if with_db:
                    await self.search_unified(q)
        log.info("warm-up complete (%d rounds, db=%s, images=%s)", rounds, with_db,
                 self.images_enabled and self.image_count > 0)

    def close(self) -> None:
        self._executor.shutdown(wait=False)
