"""Measured optimisation: choose the most precise configuration that meets the latency target.

The cross-encoder dominates query latency, and its cost is set by three knobs that
also affect precision: N (candidates scored), the token limit per pair, and int8
quantisation. Torch thread count affects latency only. This module measures, on the
machine it runs on, both sides of that trade:

Stage 1 (threads): p95 of the default configuration at each thread count. Pick the
fastest; precision is unaffected by threads.

Stage 2 (joint grid): for each reranker variant (max_length x quantised), dev-split
P@5 on BEIR for every N, and p95 latency of text-only and unified search for every N.
Cross-encoder scores are pointwise, so each variant scores the largest candidate pool
once per query and every smaller N is derived from those scores.

Selection: highest dev P@5 among points whose p95 is under the target (unified p95
when images are indexed, since that is the API default; text p95 otherwise). The
selected point is then scored once on the held-out test split. Nothing is tuned on
test.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from mmsearch.benchmarks.latency import summarize

log = logging.getLogger(__name__)


@dataclass
class GridPoint:
    n: int
    max_length: int
    quantize: bool
    dev_p5: float
    dev_p5_vector: float
    text_p95: float
    text_p50: float
    unified_p95: float | None = None
    unified_p50: float | None = None

    def constraint_p95(self) -> float:
        return self.unified_p95 if self.unified_p95 is not None else self.text_p95


@dataclass
class Selection:
    point: GridPoint | None
    reason: str
    candidates_meeting_target: int = 0
    fallback: bool = False


def select_point(points: Sequence[GridPoint], target_ms: float,
                 tolerance: float = 0.002) -> Selection:
    """Best dev P@5 under the target; ties (within `tolerance`) go to the faster point.
    If nothing meets the target, return the fastest point and say so."""
    ok = [p for p in points if p.constraint_p95() < target_ms]
    if not ok:
        fastest = min(points, key=lambda p: p.constraint_p95(), default=None)
        return Selection(fastest, f"no configuration met p95 < {target_ms:.0f} ms; "
                                  "selected the fastest", 0, fallback=True)
    best = max(p.dev_p5 for p in ok)
    near = [p for p in ok if p.dev_p5 >= best - tolerance]
    chosen = min(near, key=lambda p: p.constraint_p95())
    return Selection(chosen, f"best dev P@5 with p95 < {target_ms:.0f} ms", len(ok))


def thread_grid(cpus: int | None = None) -> list[int]:
    cpus = cpus or os.cpu_count() or 1
    grid = sorted({t for t in (1, 2, 4, 6, 8, 12, 16) if t <= cpus} | {min(cpus, 16)})
    return grid


def env_lines(threads: int, point: GridPoint) -> list[str]:
    return [f"MMS_TORCH_THREADS={threads}",
            f"MMS_RERANK_CANDIDATES={point.n}",
            f"MMS_RERANK_MAX_LENGTH={point.max_length}",
            f"MMS_RERANK_QUANTIZE={'true' if point.quantize else 'false'}"]


async def _latency(service, queries: list[str], *, iterations: int, warmup: int,  # noqa: ANN001
                   modalities: tuple[str, ...], n_text: int | None = None) -> dict[str, float]:
    for i in range(warmup):
        await service.search_unified(queries[i % len(queries)], modalities=modalities,
                                     n_text=n_text)
    samples = []
    for i in range(iterations):
        t0 = time.perf_counter_ns()
        await service.search_unified(queries[i % len(queries)], modalities=modalities,
                                     n_text=n_text)
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    return summarize(samples)


@dataclass
class OptimizeResult:
    threads: list[dict[str, Any]] = field(default_factory=list)
    best_threads: int = 0
    grid: list[GridPoint] = field(default_factory=list)
    selection: Selection | None = None
    test: dict[str, Any] | None = None
    env: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        sel = self.selection
        return {
            "threads": self.threads, "best_threads": self.best_threads,
            "grid": [asdict(p) for p in self.grid],
            "selection": None if sel is None else {
                "point": None if sel.point is None else asdict(sel.point),
                "reason": sel.reason, "fallback": sel.fallback,
                "candidates_meeting_target": sel.candidates_meeting_target},
            "test": self.test, "env": self.env,
        }


async def optimize(service, settings, dataset, dev: list[str], test: list[str],  # noqa: ANN001
                   queries: list[str], *, n_grid: Sequence[int],
                   max_lengths: Sequence[int], quantize_options: Sequence[bool],
                   iterations: int, warmup: int, rerankers: dict[bool, Any],
                   ) -> OptimizeResult:
    """rerankers: {False: unquantised CrossEncoderReranker, True: quantised one}. Passed
    explicitly so a quantised default in .env cannot leak into the 'unquantised' points."""
    import torch

    from mmsearch.evaluation.runner import collect_runs, evaluate

    res = OptimizeResult()
    with_images = service.images_enabled and service.image_count > 0
    unified = ("text", "image")

    # ---- stage 1: threads
    for t in thread_grid():
        await service.run(torch.set_num_threads, t)
        lat = await _latency(service, queries, iterations=iterations, warmup=warmup,
                             modalities=unified if with_images else ("text",))
        res.threads.append({"threads": t, **lat})
        log.info("threads=%d p95=%.2f ms", t, lat["p95"])
    res.best_threads = min(res.threads, key=lambda r: r["p95"])["threads"]
    await service.run(torch.set_num_threads, res.best_threads)

    # ---- stage 2: reranker variants x N
    original = service.reranker
    max_n = max(n_grid)
    for quant in quantize_options:
        reranker = rerankers.get(quant)
        if reranker is None:
            continue
        for max_len in max_lengths:
            reranker.max_length = max_len
            service.reranker = reranker
            runs = await collect_runs(service, dataset, dev, max_n)
            for n in n_grid:
                q = evaluate(runs, dataset, dev, n, bootstrap=False)
                text_lat = await _latency(service, queries, iterations=iterations,
                                          warmup=warmup, modalities=("text",), n_text=n)
                point = GridPoint(n=n, max_length=max_len, quantize=quant,
                                  dev_p5=q.b["p@5"], dev_p5_vector=q.a["p@5"],
                                  text_p95=text_lat["p95"], text_p50=text_lat["p50"])
                if with_images:
                    u = await _latency(service, queries, iterations=iterations,
                                       warmup=warmup, modalities=unified, n_text=n)
                    point.unified_p95, point.unified_p50 = u["p95"], u["p50"]
                res.grid.append(point)
                log.info("quant=%s len=%d N=%d dev P@5=%.4f text p95=%.2f unified p95=%s",
                         quant, max_len, n, point.dev_p5, point.text_p95, point.unified_p95)

    res.selection = select_point(res.grid, settings.latency_target_ms)
    chosen = res.selection.point
    if chosen is not None:
        reranker = rerankers[chosen.quantize]
        reranker.max_length = chosen.max_length
        service.reranker = reranker
        runs = await collect_runs(service, dataset, test, max(chosen.n, 10))
        final = evaluate(runs, dataset, test, chosen.n)
        res.test = final.summary()
        res.env = env_lines(res.best_threads, chosen)
    # leave the service as configured
    for r in rerankers.values():
        r.max_length = settings.rerank_max_length
    service.reranker = original
    await service.run(torch.set_num_threads, settings.torch_threads)
    return res
