from mmsearch.reranking.image_reranker import CaptionFusionReranker
from mmsearch.retrieval.image_search import ImageCandidate
from mmsearch.retrieval.vector_search import Candidate
from mmsearch.search.calibration import Platt, sigmoid
from mmsearch.search.unified import merge_hits


def img(ext_id: str, clip: float, rank: int, caption: str | None = None) -> ImageCandidate:
    return ImageCandidate(rank, ext_id, caption, f"{ext_id}.png", "image/png", 10, 10,
                          clip, rank)


def txt(ext_id: str, rank: int, ce: float | None, vec: float = 0.5) -> Candidate:
    return Candidate(rank, ext_id, None, rank, "", vec, rank, ce)


CALIB = Platt(40.0, -10.0)


def test_uncaptioned_images_keep_clip_order():
    f = CaptionFusionReranker(CALIB, 0.5)
    out = f.fuse([img("a", 0.30, 1), img("b", 0.28, 2), img("c", 0.20, 3)], {})
    assert [c.external_id for c in out] == ["a", "b", "c"]
    assert out[0].relevance == CALIB(0.30)


def test_caption_evidence_can_reorder():
    f = CaptionFusionReranker(CALIB, 0.5)
    cands = [img("a", 0.27, 1, "a dog"), img("b", 0.26, 2, "red circle")]
    out = f.fuse(cands, {0: -8.0, 1: 6.0})
    assert [c.external_id for c in out] == ["b", "a"]
    assert out[0].caption_score == 6.0
    assert out[0].relevance == 0.5 * sigmoid(6.0) + 0.5 * CALIB(0.26)


def test_weight_zero_is_pure_clip():
    f = CaptionFusionReranker(CALIB, 0.0)
    out = f.fuse([img("a", 0.30, 1, "x"), img("b", 0.20, 2, "y")], {0: -10.0, 1: 10.0})
    assert [c.external_id for c in out] == ["a", "b"]


def test_merge_interleaves_by_calibrated_relevance():
    text = [txt("t1", 1, 4.0), txt("t2", 2, -3.0)]                  # p = 0.98, 0.05
    images = CaptionFusionReranker(CALIB, 0.5).calibrate_only(
        [img("i1", 0.30, 1), img("i2", 0.22, 2)])                    # p = 0.88, 0.23
    hits = merge_hits(text, images, top_k=4, vector_calibration=Platt(12, -6))
    assert [(h.modality, (h.text or h.image).external_id) for h in hits] == [
        ("text", "t1"), ("image", "i1"), ("image", "i2"), ("text", "t2")]
    assert merge_hits(text, images, 2, Platt(12, -6))[1].modality == "image"


def test_merge_without_rerank_uses_vector_calibration():
    hits = merge_hits([txt("t", 1, None, vec=0.6)], [], 5, Platt(12, -6))
    assert hits[0].relevance == Platt(12, -6)(0.6)
