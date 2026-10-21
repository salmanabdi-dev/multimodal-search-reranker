import math

import pytest

from mmsearch.evaluation.metrics import (
    mrr_at_k,
    ndcg_at_k,
    paired_bootstrap,
    precision_at_k,
    recall_at_k,
    relative_improvement,
)


def test_precision_denominator_is_always_k():
    assert precision_at_k(["a", "b", "x"], {"a", "b"}, 5) == pytest.approx(0.4)
    assert precision_at_k(["x", "y", "a", "b", "c"], {"a", "b", "c"}, 5) == pytest.approx(0.6)


def test_recall_and_mrr():
    assert recall_at_k(["x", "a"], {"a", "b"}, 2) == 0.5
    assert recall_at_k(["x"], set(), 1) == 0.0
    assert mrr_at_k(["x", "y", "a"], {"a"}, 10) == pytest.approx(1 / 3)
    assert mrr_at_k(["x"], {"a"}, 10) == 0.0


def test_ndcg_perfect_and_graded():
    grades = {"a": 2, "b": 1}
    assert ndcg_at_k(["a", "b"], grades, 10) == pytest.approx(1.0)
    swapped = ndcg_at_k(["b", "a"], grades, 10)
    expected = (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))
    assert swapped == pytest.approx(expected)
    assert ndcg_at_k(["x"], {}, 10) == 0.0


def test_relative_improvement():
    assert relative_improvement(0.5, 0.64) == pytest.approx(0.28)


def test_paired_bootstrap_detects_consistent_gain():
    a = [0.2, 0.4, 0.2, 0.6, 0.4] * 20
    b = [x + 0.2 for x in a]
    bs = paired_bootstrap(a, b, n_resamples=2000)
    assert bs["mean_diff"] == pytest.approx(0.2)
    assert bs["diff_ci_low"] > 0
    assert bs["p_value_b_not_better"] == 0.0


def test_paired_bootstrap_rejects_mismatch():
    with pytest.raises(ValueError):
        paired_bootstrap([1.0], [1.0, 2.0])
