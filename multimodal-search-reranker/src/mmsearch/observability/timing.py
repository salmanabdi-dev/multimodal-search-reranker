"""Per-stage timing used by the API (Server-Timing header) and the benchmark."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager


class StageTimer:
    """Collects wall-clock durations (ms) for named stages of one request."""

    def __init__(self) -> None:
        self._start = time.perf_counter_ns()
        self.stages: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + (time.perf_counter_ns() - t0) / 1e6

    def total_ms(self) -> float:
        return (time.perf_counter_ns() - self._start) / 1e6

    def as_dict(self) -> dict[str, float]:
        out = {k: round(v, 3) for k, v in self.stages.items()}
        out["total"] = round(self.total_ms(), 3)
        return out

    def server_timing_header(self) -> str:
        return ", ".join(f"{k};dur={v:.2f}" for k, v in self.as_dict().items())
