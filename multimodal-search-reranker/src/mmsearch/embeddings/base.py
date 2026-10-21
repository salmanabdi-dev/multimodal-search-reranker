"""Encoder interface.

Ingestion, retrieval and evaluation depend on this protocol rather than on a concrete
model, which is what lets a different encoder (e.g. a CLIP image/text encoder for a
multimodal extension) slot in without touching the pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class Encoder(Protocol):
    name: str

    @property
    def dim(self) -> int: ...

    def encode(self, texts: Sequence[str], *, is_query: bool = False,
               batch_size: int = 64) -> np.ndarray:
        """Return float32 array of shape (len(texts), dim), rows L2-normalised."""
        ...

    def encode_query(self, query: str) -> np.ndarray: ...
