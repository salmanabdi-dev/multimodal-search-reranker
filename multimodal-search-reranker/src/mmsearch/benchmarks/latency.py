"""Search latency benchmark (mean / p50 / p95 / p99).

Reproducibility rules:
* fixed query file, visited in a fixed order (i mod len)
* models already warmed by the service; the benchmark adds its own warm-up requests
* concurrency 1, no result caching anywhere in the path
* every report records hardware, library versions, model names and all search settings

Configurations (each run separately, same queries): text only, images only, and the
unified text+image path, reranked; plus unified without reranking for comparison.

Two modes:
* in-process: calls SearchService.search directly; measures the pipeline itself
* http: hits a running API with keep-alive; adds the HTTP + JSON layer. Reports both
  the client-observed time and the server's own total (from Server-Timing).
"""

from __future__ import annotations

import os
import platform
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_QUERIES = Path(__file__).resolve().parents[3] / "data" / "benchmark" / "queries.txt"


def load_queries(path: Path = DEFAULT_QUERIES) -> list[str]:
    qs = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    qs = [q for q in qs if q and not q.startswith("#")]
    if not qs:
        raise ValueError(f"no queries in {path}")
    return qs


def summarize(samples_ms: Sequence[float]) -> dict[str, float]:
    a = np.asarray(samples_ms, dtype=float)
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 3),
        "std": round(float(a.std(ddof=1)) if a.size > 1 else 0.0, 3),
        "min": round(float(a.min()), 3),
        "p50": round(float(np.percentile(a, 50)), 3),
        "p95": round(float(np.percentile(a, 95)), 3),
        "p99": round(float(np.percentile(a, 99)), 3),
        "max": round(float(a.max()), 3),
    }


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith(("model name", "hardware", "cpu model")):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _cpu_model(),
        "logical_cpus": os.cpu_count(),
    }
    try:
        import torch
        import transformers
        info |= {"torch": torch.__version__, "transformers": transformers.__version__,
                 "torch_threads": torch.get_num_threads(),
                 "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    except ImportError:
        pass
    return info


CONFIGS = {  # label -> (modalities, rerank)
    "text": (("text",), True),
    "image": (("image",), True),
    "text+image": (("text", "image"), True),
    "text+image_vector_only": (("text", "image"), False),
}


def configs_for(has_text: bool, has_images: bool) -> dict[str, tuple[tuple[str, ...], bool]]:
    return {k: v for k, v in CONFIGS.items()
            if all((m == "text" and has_text) or (m == "image" and has_images) for m in v[0])}


async def bench_inprocess(service, queries: list[str], *, warmup: int, iterations: int,  # noqa: ANN001
                          modalities: tuple[str, ...] = ("text",), rerank: bool = True
                          ) -> dict[str, Any]:
    async def call(q: str):  # noqa: ANN202
        return await service.search_unified(q, modalities=modalities, rerank=rerank)

    for i in range(warmup):
        await call(queries[i % len(queries)])

    totals: list[float] = []
    stages: dict[str, list[float]] = {}
    for i in range(iterations):
        t0 = time.perf_counter_ns()
        outcome = await call(queries[i % len(queries)])
        totals.append((time.perf_counter_ns() - t0) / 1e6)
        for name, v in outcome.timings_ms.items():
            if name != "total":
                stages.setdefault(name, []).append(v)

    return {"total": summarize(totals),
            "stages": {k: summarize(v) for k, v in stages.items()},
            "stage_mean_share": {k: round(statistics.fmean(v) / statistics.fmean(totals), 3)
                                 for k, v in stages.items()}}


def _server_total(header: str | None) -> float | None:
    for part in (header or "").split(","):
        name, _, dur = part.strip().partition(";dur=")
        if name == "total" and dur:
            return float(dur)
    return None


def bench_http(url: str, queries: list[str], *, warmup: int, iterations: int,
               modalities: tuple[str, ...] = ("text",), rerank: bool = True) -> dict[str, Any]:
    import httpx

    client_ms: list[float] = []
    server_ms: list[float] = []
    with httpx.Client(base_url=url, timeout=10.0) as client:
        def call(q: str) -> httpx.Response:
            r = client.post("/search", json={"query": q, "top_k": 5, "rerank": rerank,
                                             "modalities": list(modalities)})
            r.raise_for_status()
            return r

        for i in range(warmup):
            call(queries[i % len(queries)])
        for i in range(iterations):
            t0 = time.perf_counter_ns()
            r = call(queries[i % len(queries)])
            client_ms.append((time.perf_counter_ns() - t0) / 1e6)
            st = _server_total(r.headers.get("server-timing"))
            if st is not None:
                server_ms.append(st)

    out: dict[str, Any] = {"client_total": summarize(client_ms)}
    if server_ms:
        out["server_total"] = summarize(server_ms)
    return out
