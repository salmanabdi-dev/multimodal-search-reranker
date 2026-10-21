"""Wiring: load models, configure torch threads, assemble the SearchService."""

from __future__ import annotations

import logging

import torch

from mmsearch.config import Settings
from mmsearch.db.repository import Repository
from mmsearch.embeddings.bi_encoder import BiEncoder
from mmsearch.reranking.cross_encoder import CrossEncoderReranker
from mmsearch.retrieval.vector_search import VectorRetriever
from mmsearch.search.calibration import Platt
from mmsearch.search.service import SearchService

log = logging.getLogger(__name__)


def configure_torch(threads: int) -> None:
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:  # may only be set once per process, before any parallel work
        pass


def load_bi_encoder(s: Settings) -> BiEncoder:
    return BiEncoder(s.bi_encoder_model, s.bi_encoder_revision, s.device,
                     s.query_max_length, s.doc_max_length, s.query_prefix)


def load_cross_encoder(s: Settings) -> CrossEncoderReranker:
    return CrossEncoderReranker(s.cross_encoder_model, s.cross_encoder_revision, s.device,
                                s.rerank_max_length, s.rerank_quantize)


def load_clip(s: Settings):  # noqa: ANN201
    from mmsearch.embeddings.clip_encoder import ClipEncoder
    return ClipEncoder(s.clip_model, s.clip_revision, s.device, s.clip_text_max_length)


def build_service(settings: Settings, repo: Repository, image_count: int = 0) -> SearchService:
    configure_torch(settings.torch_threads)
    encoder = load_bi_encoder(settings)
    reranker = load_cross_encoder(settings)
    image_kwargs = {}
    if settings.enable_images:
        from mmsearch.reranking.image_reranker import CaptionFusionReranker
        from mmsearch.retrieval.image_search import ImageRetriever
        image_kwargs = dict(
            clip=load_clip(settings),
            image_retriever=ImageRetriever(repo, settings),
            image_reranker=CaptionFusionReranker(
                Platt(settings.clip_calib_a, settings.clip_calib_b), settings.caption_weight),
            image_count=image_count,
        )
    return SearchService(
        encoder, VectorRetriever(repo, settings), reranker, settings,
        # Thread counts are per-thread state in some OpenMP builds: set them in the worker too.
        executor_initializer=lambda: torch.set_num_threads(settings.torch_threads),
        text_vec_calibration=Platt(settings.text_vec_calib_a, settings.text_vec_calib_b),
        **image_kwargs,
    )
