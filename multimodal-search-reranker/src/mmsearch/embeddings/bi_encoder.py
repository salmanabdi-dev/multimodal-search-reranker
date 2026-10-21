"""Bi-encoder: embeds queries and documents *independently* into one vector space.

Independence is the point. Document vectors are computed once at ingestion and
indexed, so a search only needs one forward pass (for the query) plus an index
lookup. The price is that the model never sees query and document together, which
is why a second, more precise stage (the cross-encoder) follows.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from mmsearch.embeddings.pooling import l2_normalize, mean_pool

log = logging.getLogger(__name__)


class BiEncoder:
    def __init__(
        self,
        model_name: str,
        revision: str = "main",
        device: str = "cpu",
        query_max_length: int = 64,
        doc_max_length: int = 256,
        query_prefix: str = "",
    ) -> None:
        self.name = model_name
        self.device = torch.device(device)
        self.query_max_length = query_max_length
        self.doc_max_length = doc_max_length
        self.query_prefix = query_prefix
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        model = AutoModel.from_pretrained(model_name, revision=revision)
        if self.device.type == "cuda":
            model = model.half()
        self.model = model.to(self.device).eval()
        log.info("loaded bi-encoder %s (dim=%d) on %s", model_name, self.dim, self.device)

    @property
    def dim(self) -> int:
        return int(self.model.config.hidden_size)

    def encode(self, texts: Sequence[str], *, is_query: bool = False,
               batch_size: int = 64) -> np.ndarray:
        texts = list(texts)
        if is_query and self.query_prefix:
            texts = [self.query_prefix + t for t in texts]
        max_len = self.query_max_length if is_query else self.doc_max_length
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        if not texts:
            return out

        # Batch texts of similar length together: less padding = less wasted compute.
        order = np.argsort([-len(t) for t in texts], kind="stable")
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                idx = order[start:start + batch_size]
                enc = self.tokenizer(
                    [texts[i] for i in idx], padding=True, truncation=True,
                    max_length=max_len, return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**enc).last_hidden_state
                vecs = l2_normalize(mean_pool(hidden, enc["attention_mask"]))
                out[idx] = vecs.float().cpu().numpy()
        return out

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([query], is_query=True)[0]


class HFTokenizerAdapter:
    """Exposes just what the chunker needs from a Hugging Face fast tokenizer."""

    def __init__(self, tokenizer) -> None:  # noqa: ANN001 - any HF fast tokenizer
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("chunking needs a fast tokenizer (offset mappings)")
        self._tok = tokenizer
        self.special_tokens = tokenizer.num_special_tokens_to_add(pair=False)

    def offsets(self, text: str) -> list[tuple[int, int]]:
        enc = self._tok(text, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=False, verbose=False)
        return [tuple(o) for o in enc["offset_mapping"]]
