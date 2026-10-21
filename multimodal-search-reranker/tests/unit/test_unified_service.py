import asyncio

import pytest

pytest.importorskip("PIL")

from mmsearch.ingestion.images import ImageIngestionPipeline, RawImage  # noqa: E402
from mmsearch.media.storage import MediaStore  # noqa: E402
from mmsearch.reranking.image_reranker import CaptionFusionReranker  # noqa: E402
from mmsearch.retrieval.image_search import ImageRetriever  # noqa: E402
from mmsearch.retrieval.vector_search import VectorRetriever  # noqa: E402
from mmsearch.search.calibration import Platt  # noqa: E402
from mmsearch.search.service import SearchService  # noqa: E402
from tests.unit.fakes import (  # noqa: E402
    FakeClip,
    FakeEncoder,
    FakeImageRepo,
    FakeRepo,
    RecordingReranker,
    chunk_row,
    make_mm_settings,
    png_bytes,
)

COLORS = {"red": (220, 30, 30), "blue": (30, 60, 220), "green": (30, 170, 60)}


def build(tmp_path, captions=None):
    s = make_mm_settings(rerank_candidates=4, image_candidates=6, default_top_k=5)
    captions = captions or {}
    text_rows = [chunk_row(1, 1, 0.10, "red apples are sweet"),
                 chunk_row(2, 2, 0.20, "the sky is blue"),
                 chunk_row(3, 3, 0.30, "nothing relevant")]
    img_repo, clip, reranker = FakeImageRepo(), FakeClip(), RecordingReranker()
    pipeline = ImageIngestionPipeline(img_repo, clip, MediaStore(tmp_path), s)
    raws = [RawImage(png_bytes(rgb, seed=k), external_id=f"{name}{k}",
                     caption=captions.get(f"{name}{k}"))
            for name, rgb in COLORS.items() for k in range(2)]
    stats = asyncio.run(pipeline.ingest_many(raws))
    svc = SearchService(
        FakeEncoder(), VectorRetriever(FakeRepo(text_rows), s), reranker, s,
        clip=clip, image_retriever=ImageRetriever(img_repo, s),
        image_reranker=CaptionFusionReranker(Platt(10.0, -5.0), 0.5),
        image_count=len(img_repo.images))
    return svc, img_repo, reranker, stats, pipeline


def test_cross_modal_retrieval_finds_matching_images(tmp_path):
    svc, *_ = build(tmp_path)
    out = asyncio.run(svc.search_unified("something red", top_k=2, modalities=["image"]))
    assert [h.image.external_id[:3] for h in out.hits] == ["red", "red"]
    assert out.modalities == ["image"]
    assert "embed_text" not in out.timings_ms and "embed_clip" in out.timings_ms
    svc.close()


def test_unified_search_returns_both_modalities_with_one_rerank_call(tmp_path):
    svc, _, reranker, _, _ = build(tmp_path, captions={"red0": "a red ball", "blue0": "blue"})
    out = asyncio.run(svc.search_unified("red", top_k=20))
    assert {h.modality for h in out.hits} == {"text", "image"}
    assert out.modalities == ["text", "image"]
    assert len(reranker.calls) == 1                       # passages + captions, one batch
    batch = reranker.calls[0]
    assert "a red ball" in batch and "blue" in batch
    assert len(batch) == out.candidates_considered["text"] + 2
    rel = [h.relevance for h in out.hits]
    assert rel == sorted(rel, reverse=True)               # merged by calibrated relevance
    red_ball = next(h for h in out.hits if h.image and h.image.external_id == "red0")
    assert red_ball.image.caption_score is not None
    assert {"embed_text", "embed_clip", "retrieve", "rerank", "merge"} <= out.timings_ms.keys()
    svc.close()


def test_no_rerank_still_merges_with_calibrated_scores(tmp_path):
    svc, _, reranker, _, _ = build(tmp_path)
    out = asyncio.run(svc.search_unified("blue", top_k=5, rerank=False))
    assert reranker.calls == []
    assert all(h.relevance is not None for h in out.hits)
    assert "rerank" not in out.timings_ms
    svc.close()


def test_image_branch_skipped_when_index_empty(tmp_path):
    svc, img_repo, *_ = build(tmp_path)
    svc.image_count = 0
    out = asyncio.run(svc.search_unified("red", modalities=["text", "image"]))
    assert out.modalities == ["text"]
    assert all(h.modality == "text" for h in out.hits)
    svc.close()


def test_ingestion_dedupes_by_content_and_batches_clip_calls(tmp_path):
    svc, img_repo, _, stats, pipeline = build(tmp_path)
    assert stats.images == 6 and stats.duplicates == 0
    calls_before = pipeline.clip.image_calls
    again = asyncio.run(pipeline.ingest_one(
        RawImage(png_bytes(COLORS["red"], seed=0), external_id="other-id", caption="new cap")))
    assert again.created is False and again.external_id == "red0"
    assert img_repo.images["red0"]["caption"] == "new cap"
    assert pipeline.clip.image_calls == calls_before      # no re-embedding
    assert len(list(tmp_path.rglob("*.png"))) == 6
    # duplicates inside one batch are embedded once
    dup = png_bytes((10, 200, 200))
    st = asyncio.run(pipeline.ingest_many([RawImage(dup, external_id="x1"),
                                           RawImage(dup, external_id="x2")]))
    assert (st.images, st.duplicates) == (1, 1)
    svc.close()


def test_ingestion_rejects_bad_files_but_keeps_going(tmp_path):
    svc, img_repo, _, _, pipeline = build(tmp_path)
    st = asyncio.run(pipeline.ingest_many([RawImage(b"junk", external_id="bad"),
                                           RawImage(png_bytes((1, 2, 3)), external_id="ok")]))
    assert (st.images, st.rejected) == (1, 1)
    assert "bad" in st.errors[0]
    svc.close()


def test_image_external_id_cannot_overwrite_text_document(tmp_path):
    svc, img_repo, _, _, pipeline = build(tmp_path)
    img_repo.text_ids.add("doc-1")
    with pytest.raises(ValueError):
        asyncio.run(pipeline.ingest_one(RawImage(png_bytes((9, 9, 9)), external_id="doc-1")))
    svc.close()
