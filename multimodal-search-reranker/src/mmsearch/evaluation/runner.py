"""System A (vector only) vs System B (vector + cross-encoder) on the same candidates.

For each query we retrieve the largest candidate pool once, score every candidate
with the cross-encoder once, and then derive every configuration from that:

    A(k)    = top-k by vector similarity
    B(N, k) = the first N candidates reordered by cross-encoder score, then top-k

Cross-encoder scores are pointwise (each pair scored independently), so reranking
the first N of a pool gives exactly the same scores as reranking a pool of size N.
That makes an N sweep cheap and guarantees A and B differ only by reranking.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import Any

from mmsearch.evaluation.datasets import EvalDataset
from mmsearch.evaluation.metrics import (
    mrr_at_k,
    ndcg_at_k,
    paired_bootstrap,
    precision_at_k,
    recall_at_k,
    relative_improvement,
)
from mmsearch.retrieval.vector_search import Candidate

log = logging.getLogger(__name__)
TARGET_GAIN = 0.28


def rank_vector(cands: list[Candidate]) -> list[str]:
    return [c.external_id for c in sorted(cands, key=lambda c: c.vector_rank)]


def rank_reranked(cands: list[Candidate], n: int) -> list[str]:
    """Rerank the head (first n by vector rank); the tail keeps vector order."""
    by_vec = sorted(cands, key=lambda c: c.vector_rank)
    head = sorted(by_vec[:n], key=lambda c: c.rerank_score, reverse=True)
    return [c.external_id for c in head + by_vec[n:]]


async def collect_runs(service, dataset: EvalDataset, qids: list[str], max_n: int,  # noqa: ANN001
                       ef_search: int | None = None) -> dict[str, list[Candidate]]:
    runs: dict[str, list[Candidate]] = {}
    for i, qid in enumerate(qids, 1):
        query = dataset.queries[qid]
        cands = await service.retrieve(query, max_n, ef_search)
        runs[qid] = await service.rerank(query, cands)
        if i % 50 == 0:
            log.info("scored %d/%d queries", i, len(qids))
    return runs


@dataclass
class ConfigResult:
    n: int
    ef_search: int | None
    queries: int
    a: dict[str, float]
    b: dict[str, float]
    pool_recall: float                       # Recall@N of the candidate pool: B's ceiling
    rel_gain_p5: float
    bootstrap: dict[str, float] = field(default_factory=dict)
    extra: dict[str, float] = field(default_factory=dict)   # e.g. caption_weight for images
    per_query_p5: dict[str, tuple[float, float]] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "per_query_p5"}


def compare_rankings(dataset: EvalDataset, qids: list[str], ranked_a: dict[str, list[str]],
                     ranked_b: dict[str, list[str]], *, n: int, k: int = 5, min_level: int = 1,
                     ef_search: int | None = None, bootstrap: bool = True,
                     extra: dict[str, float] | None = None) -> ConfigResult:
    """Modality-agnostic: two rankings of document ids per query -> A vs B metrics.
    Pool recall is Recall@n of ranking A (the candidate pool B reorders)."""
    metrics_a: dict[str, list[float]] = {"p@5": [], "p@5_strict": [], "ndcg@10": [], "mrr@10": []}
    metrics_b: dict[str, list[float]] = {key: [] for key in metrics_a}
    pool_recall: list[float] = []
    per_query: dict[str, tuple[float, float]] = {}

    for qid in qids:
        rel = dataset.relevant(qid, min_level)
        strict = dataset.relevant(qid, max(min_level, 2))
        grades = dataset.qrels[qid]
        ra, rb = ranked_a[qid], ranked_b[qid]
        for out, ranked in ((metrics_a, ra), (metrics_b, rb)):
            out["p@5"].append(precision_at_k(ranked, rel, k))
            out["p@5_strict"].append(precision_at_k(ranked, strict, k) if strict else 0.0)
            out["ndcg@10"].append(ndcg_at_k(ranked, grades, 10))
            out["mrr@10"].append(mrr_at_k(ranked, rel, 10))
        pool_recall.append(recall_at_k(ra, rel, n))
        per_query[qid] = (metrics_a["p@5"][-1], metrics_b["p@5"][-1])

    mean_a = {m: statistics.fmean(v) for m, v in metrics_a.items()}
    mean_b = {m: statistics.fmean(v) for m, v in metrics_b.items()}
    result = ConfigResult(
        n=n, ef_search=ef_search, queries=len(qids), a=mean_a, b=mean_b,
        pool_recall=statistics.fmean(pool_recall),
        rel_gain_p5=relative_improvement(mean_a["p@5"], mean_b["p@5"]),
        per_query_p5=per_query, extra=extra or {},
    )
    if bootstrap and len(qids) > 1:
        result.bootstrap = paired_bootstrap(metrics_a["p@5"], metrics_b["p@5"])
    return result


def evaluate(runs: dict[str, list[Candidate]], dataset: EvalDataset, qids: list[str], n: int,
             *, k: int = 5, min_level: int = 1, ef_search: int | None = None,
             bootstrap: bool = True) -> ConfigResult:
    return compare_rankings(
        dataset, qids, {q: rank_vector(runs[q]) for q in qids},
        {q: rank_reranked(runs[q], n) for q in qids}, n=n, k=k, min_level=min_level,
        ef_search=ef_search, bootstrap=bootstrap)


def pick_best(results: list[ConfigResult], tolerance: float = 0.002) -> ConfigResult:
    """Highest dev P@5 for B; within `tolerance` of the best, prefer the smaller N (faster)."""
    best = max(r.b["p@5"] for r in results)
    near = [r for r in results if r.b["p@5"] >= best - tolerance]
    return min(near, key=lambda r: (r.n, r.ef_search or 0))


def format_result(label: str, r: ConfigResult, a_name: str = "A vector",
                  b_name: str = "B rerank", target: float | None = TARGET_GAIN) -> str:
    extra = "".join(f", {k}={v}" for k, v in r.extra.items())
    lines = [
        f"\n== {label}: {r.queries} queries, N={r.n}, "
        f"ef_search={r.ef_search or 'default'}{extra} ==",
        f"{'metric':<12}{a_name:>10}{b_name:>10}{'rel. gain':>11}",
    ]
    for m in ("p@5", "p@5_strict", "ndcg@10", "mrr@10"):
        gain = relative_improvement(r.a[m], r.b[m])
        lines.append(f"{m:<12}{r.a[m]:>10.4f}{r.b[m]:>10.4f}{gain:>+10.1%}")
    lines.append(f"candidate pool Recall@{r.n}: {r.pool_recall:.4f}  (upper bound for reranking)")
    if r.bootstrap:
        bs = r.bootstrap
        lines.append(f"P@5 relative gain 95% CI: [{bs['rel_ci_low']:+.1%}, {bs['rel_ci_high']:+.1%}]"
                     f"  p(B<=A)={bs['p_value_b_not_better']:.4f}")
    if target is None:
        lines.append(f"Top-5 precision gain: {r.rel_gain_p5:+.1%}")
    else:
        met = "MET" if r.rel_gain_p5 >= target else "NOT MET"
        lines.append(f"Top-5 precision gain: {r.rel_gain_p5:+.1%}  (target +{target:.0%}: {met})")
    return "\n".join(lines)
