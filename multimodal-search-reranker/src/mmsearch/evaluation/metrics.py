"""Ranking metrics. ``ranked`` is a list of document ids, best first.

Precision@k = (# relevant in the top k) / k. The denominator is always k, even if
fewer results were returned: an empty slot is a slot the user didn't get a good
answer in.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return sum(1 for d in ranked[:k] if d in relevant) / k


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for d in ranked[:k] if d in relevant) / len(relevant)


def mrr_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    for i, d in enumerate(ranked[:k], 1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], grades: Mapping[str, int], k: int) -> float:
    """Graded: a grade-2 document at rank 1 is worth more than a grade-1 one."""
    def dcg(gs: Sequence[int]) -> float:
        return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gs))
    ideal = dcg(sorted((g for g in grades.values() if g > 0), reverse=True)[:k])
    if ideal == 0:
        return 0.0
    return dcg([max(grades.get(d, 0), 0) for d in ranked[:k]]) / ideal


def relative_improvement(baseline: float, improved: float) -> float:
    return (improved - baseline) / baseline if baseline > 0 else float("inf")


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_resamples: int = 10_000,
                     seed: int = 0, alpha: float = 0.05) -> dict[str, float]:
    """Resample queries with replacement; CI on mean(b) - mean(a) and on the relative gain.

    Pairing matters: both systems answered the same queries, so we resample query
    indices once and score both systems on the same resample.
    """
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a_arr) != len(b_arr) or len(a_arr) == 0:
        raise ValueError("need two equal-length, non-empty score lists")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a_arr), size=(n_resamples, len(a_arr)))
    ma, mb = a_arr[idx].mean(axis=1), b_arr[idx].mean(axis=1)
    diff = mb - ma
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(ma > 0, diff / ma, np.nan)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "mean_diff": float(b_arr.mean() - a_arr.mean()),
        "diff_ci_low": float(np.percentile(diff, lo)),
        "diff_ci_high": float(np.percentile(diff, hi)),
        "rel_ci_low": float(np.nanpercentile(rel, lo)),
        "rel_ci_high": float(np.nanpercentile(rel, hi)),
        "p_value_b_not_better": float((diff <= 0).mean()),
    }
