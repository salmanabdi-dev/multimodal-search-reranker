"""Turning a transformer's per-token outputs into one sentence vector.

A transformer returns one hidden vector per token: shape (batch, seq_len, hidden).
all-MiniLM-L6-v2 was trained with *mean pooling*: average the token vectors, but only
over real tokens. Padding positions (attention_mask == 0) must be excluded or short
texts in a padded batch would be dragged toward the padding representation.

After pooling we L2-normalise, so every vector has length 1. Then
cosine(a, b) == dot(a, b), and pgvector's cosine distance is simply 1 - dot.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)   # (B, T, 1)
    summed = (last_hidden_state * mask).sum(dim=1)                    # (B, H)
    counts = mask.sum(dim=1).clamp(min=1e-9)                          # (B, 1)
    return summed / counts


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=1)
