"""Turning each modality's raw score into a comparable relevance probability.

A cross-encoder logit of 3.1 and a CLIP cosine of 0.29 cannot be sorted against each
other: they live on unrelated scales. Platt scaling fits p = sigmoid(a*s + b) on
labelled (score, relevant?) pairs, so each modality's score becomes "estimated
probability this result is relevant", and results from different modalities can be
merged by that probability.

The ms-marco cross-encoder was trained with a binary cross-entropy objective, so its
logit is already roughly log-odds: sigmoid(logit) needs no fitting.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


@dataclass(frozen=True)
class Platt:
    a: float
    b: float

    def __call__(self, score: float) -> float:
        return sigmoid(self.a * score + self.b)


def fit_platt(scores: Sequence[float], labels: Sequence[int], iterations: int = 50,
              l2: float = 1e-4) -> Platt:
    """1-D logistic regression by Newton's method (tiny problem, converges in a few steps)."""
    x = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    if x.size == 0 or y.min() == y.max():
        raise ValueError("need both relevant and non-relevant examples")
    X = np.stack([x, np.ones_like(x)], axis=1)
    w = np.zeros(2)
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -35, 35)))
        grad = X.T @ (p - y) + l2 * w
        hess = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return Platt(a=float(w[0]), b=float(w[1]))


def log_loss(probs: Sequence[float], labels: Sequence[int]) -> float:
    p = np.clip(np.asarray(probs, dtype=float), 1e-7, 1 - 1e-7)
    y = np.asarray(labels, dtype=float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
