"""Cross-modal evaluation: text queries -> images.

    A = CLIP order (cross-modal bi-encoder alone)
    B = caption fusion (CLIP + cross-encoder on whatever caption exists)

As with text, each query's candidate pool is retrieved and caption-scored once;
every caption weight w is then derived from the stored scores, so a weight sweep
costs no extra model calls and A and B always see the same candidates.

Calibration (Platt scaling of CLIP cosines) is fit on dev queries only, against all
(query, image) pairs from an exact scan, then checked on test by log loss.
"""

from __future__ import annotations

import logging

from mmsearch.evaluation.datasets import EvalDataset
from mmsearch.evaluation.runner import ConfigResult, compare_rankings
from mmsearch.reranking.image_reranker import CaptionFusionReranker
from mmsearch.retrieval.image_search import ImageCandidate
from mmsearch.search.calibration import Platt, fit_platt, log_loss

log = logging.getLogger(__name__)


async def collect_image_runs(service, dataset: EvalDataset, qids: list[str], n: int,  # noqa: ANN001
                             ) -> dict[str, list[ImageCandidate]]:
    runs = {}
    for qid in qids:
        query = dataset.queries[qid]
        runs[qid] = await service.rerank_images(query, await service.retrieve_images(query, n))
    return runs


def rank_clip(cands: list[ImageCandidate]) -> list[str]:
    return [c.external_id for c in sorted(cands, key=lambda c: c.vector_rank)]


def rank_fused(cands: list[ImageCandidate], weight: float, calibration: Platt) -> list[str]:
    fusion = CaptionFusionReranker(calibration, weight)
    return [c.external_id for c in sorted(
        cands, key=lambda c: fusion.relevance(c.clip_score, c.caption_score), reverse=True)]


def evaluate_images(runs: dict[str, list[ImageCandidate]], dataset: EvalDataset,
                    qids: list[str], *, n: int, weight: float, calibration: Platt,
                    bootstrap: bool = True) -> ConfigResult:
    return compare_rankings(
        dataset, qids, {q: rank_clip(runs[q]) for q in qids},
        {q: rank_fused(runs[q], weight, calibration) for q in qids},
        n=n, bootstrap=bootstrap, extra={"caption_weight": weight})


async def clip_score_pairs(service, dataset: EvalDataset, qids: list[str]  # noqa: ANN001
                           ) -> tuple[list[float], list[int]]:
    """Every (query, image) cosine with its 0/1 label, via exact search over all images."""
    scores, labels = [], []
    k = max(service.image_count, 1)
    for qid in qids:
        vec = await service.run(service.clip.encode_query, dataset.queries[qid])
        rel = dataset.relevant(qid)
        for row in await service.image_retriever.repo.image_exact_search(vec, k):
            scores.append(row.similarity)
            labels.append(int(row.external_id in rel))
    return scores, labels


async def fit_clip_calibration(service, dataset: EvalDataset, dev: list[str],  # noqa: ANN001
                               test: list[str], current: Platt) -> dict:
    s_dev, y_dev = await clip_score_pairs(service, dataset, dev)
    fitted = fit_platt(s_dev, y_dev)
    s_test, y_test = await clip_score_pairs(service, dataset, test)
    return {
        "fitted": fitted,
        "pairs_dev": len(s_dev), "pairs_test": len(s_test),
        "test_log_loss_current": log_loss([current(x) for x in s_test], y_test),
        "test_log_loss_fitted": log_loss([fitted(x) for x in s_test], y_test),
        "base_rate_test": sum(y_test) / max(len(y_test), 1),
    }
