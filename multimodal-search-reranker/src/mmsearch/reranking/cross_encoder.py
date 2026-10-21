"""Cross-encoder: scores a (query, passage) pair by reading both at once.

Input is ``[CLS] query [SEP] passage [SEP]``. Every query token can attend to every
passage token, so the model sees exact term matches, negation, and which entity a
phrase is about: things a single pooled vector blurs. The output is one logit per
pair, a relevance score. Nothing can be precomputed, so cost grows linearly with the
number of candidates. That is why it only runs on stage-1's short list.

All N pairs go through one batched forward pass; on CPU that is much faster than N
separate calls.
"""

from __future__ import annotations

import logging
import platform
from collections.abc import Sequence

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

log = logging.getLogger(__name__)


def _quantize_dynamic(model: torch.nn.Module) -> torch.nn.Module:
    """int8 weights for Linear layers; activations quantised on the fly. CPU only."""
    engines = torch.backends.quantized.supported_engines
    preferred = "qnnpack" if platform.machine().lower() in {"arm64", "aarch64"} else "fbgemm"
    if preferred in engines:
        torch.backends.quantized.engine = preferred
    return torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        revision: str = "main",
        device: str = "cpu",
        max_length: int = 256,
        quantize: bool = False,
    ) -> None:
        self.name = model_name
        self.device = torch.device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, revision=revision)
        model.eval()
        if quantize and self.device.type == "cpu":
            model = _quantize_dynamic(model)
            log.info("cross-encoder quantised to int8 (engine=%s)",
                     torch.backends.quantized.engine)
        elif self.device.type == "cuda":
            model = model.half()
        self.model = model.to(self.device)
        log.info("loaded cross-encoder %s on %s", model_name, self.device)

    def score(self, query: str, passages: Sequence[str], batch_size: int = 64) -> np.ndarray:
        """Relevance logit for each (query, passage). Higher = more relevant."""
        if not passages:
            return np.empty(0, dtype=np.float32)
        scores: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(passages), batch_size):
                batch = list(passages[start:start + batch_size])
                enc = self.tokenizer(
                    [query] * len(batch), batch, padding=True, truncation="longest_first",
                    max_length=self.max_length, return_tensors="pt",
                ).to(self.device)
                logits = self.model(**enc).logits
                # ms-marco cross-encoders have a single output; 2-class heads use the margin.
                s = logits[:, 0] if logits.shape[-1] == 1 else logits[:, 1] - logits[:, 0]
                scores.append(s.float().cpu().numpy())
        return np.concatenate(scores)
