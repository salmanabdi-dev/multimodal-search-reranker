import hashlib

import pytest

pytest.importorskip("PIL")

from mmsearch.evaluation.datasets import load_dataset  # noqa: E402
from mmsearch.evaluation.image_runner import evaluate_images, rank_clip, rank_fused  # noqa: E402
from mmsearch.ingestion.images import load_image_directory  # noqa: E402
from mmsearch.media.synthetic import generate  # noqa: E402
from mmsearch.retrieval.image_search import ImageCandidate  # noqa: E402
from mmsearch.search.calibration import Platt  # noqa: E402


def digest(root):
    h = hashlib.sha256()
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(f.relative_to(root).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def test_generator_is_byte_for_byte_deterministic(tmp_path):
    a, b = generate(tmp_path / "a", seed=3, instances=2), generate(tmp_path / "b", seed=3,
                                                                   instances=2)
    assert digest(a) == digest(b)
    assert digest(a) != digest(generate(tmp_path / "c", seed=4, instances=2))


def test_generated_labels_and_captions(tmp_path):
    root = generate(tmp_path / "d", instances=2)
    ds = load_dataset(root)
    assert len(ds.queries) == 24 + 6 + 4
    assert ds.relevant("red-circle", 2) == {"red_circle_0", "red_circle_1"}
    assert len(ds.relevant("red-any")) == 8            # 4 shapes x 2 instances
    raws = list(load_image_directory(root / "images"))
    assert len(raws) == 48
    with_caption = [r for r in raws if r.caption]
    assert 0 < len(with_caption) < len(raws)          # deliberately partial captions
    assert {r.external_id for r in raws} >= set(ds.qrels["any-star"])


def cand(ext, clip, rank, cap_score=None):
    return ImageCandidate(rank, ext, "c" if cap_score is not None else None, "", "", 1, 1,
                          clip, rank, cap_score)


def test_image_ranking_and_evaluation():
    from mmsearch.evaluation.datasets import EvalDataset
    cands = [cand("x", 0.31, 1, -9.0), cand("y", 0.30, 2), cand("a", 0.29, 3, 8.0),
             cand("b", 0.28, 4, 7.0), cand("z", 0.27, 5), cand("c", 0.26, 6, 9.0)]
    ds = EvalDataset("t", {"q": "q"}, {"q": {"a": 2, "b": 2, "c": 1}})
    assert rank_clip(list(reversed(cands)))[:2] == ["x", "y"]
    calib = Platt(40.0, -10.0)
    assert rank_fused(cands, 0.0, calib) == rank_clip(cands)
    # a, b: good caption + decent CLIP -> top. y (no caption, strong CLIP, p=0.88) stays
    # above c (perfect caption, weak CLIP, fused 0.80): captions add evidence, they don't
    # override the image. x (contradicting caption) falls to last.
    assert rank_fused(cands, 0.5, calib) == ["a", "b", "y", "c", "z", "x"]
    r = evaluate_images({"q": cands}, ds, ["q"], n=6, weight=0.5, calibration=calib,
                        bootstrap=False)
    assert r.a["p@5"] == pytest.approx(0.4) and r.b["p@5"] == pytest.approx(0.6)
    assert r.extra == {"caption_weight": 0.5}
