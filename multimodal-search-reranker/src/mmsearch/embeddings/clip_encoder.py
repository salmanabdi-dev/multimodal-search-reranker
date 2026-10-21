"""CLIP: one embedding space shared by images and text.

CLIP has two towers, a vision transformer and a text transformer, trained jointly on
~400M (image, caption) pairs with a contrastive loss: the true pair's vectors are
pulled together, every other pairing in the batch is pushed apart. After training, a
caption and an image of the same thing land near each other, which is what makes
*cross-modal* retrieval possible: embed the user's text with the text tower, then
search vectors produced by the image tower.

Like the MiniLM bi-encoder, the two sides are encoded independently, so image vectors
are computed once at ingestion and indexed; a query costs one small text forward pass.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPModel

from mmsearch.embeddings.pooling import l2_normalize

log = logging.getLogger(__name__)


def _features(out) -> torch.Tensor:  # noqa: ANN001
    """get_*_features returns a tensor; guard against output objects in other versions."""
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("text_embeds", "image_embeds", "pooler_output"):
        if getattr(out, attr, None) is not None:
            return getattr(out, attr)
    raise TypeError(f"unexpected CLIP output type {type(out)!r}")


class ClipEncoder:
    def __init__(self, model_name: str, revision: str = "main", device: str = "cpu",
                 text_max_length: int = 77) -> None:
        self.name = model_name
        self.device = torch.device(device)
        self.text_max_length = text_max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name, revision=revision)
        model = CLIPModel.from_pretrained(model_name, revision=revision)
        if self.device.type == "cuda":
            model = model.half()
        self.model = model.to(self.device).eval()
        log.info("loaded CLIP %s (dim=%d) on %s", model_name, self.dim, self.device)

    @property
    def dim(self) -> int:
        return int(self.model.config.projection_dim)

    def encode_text(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start:start + batch_size])
                enc = self.tokenizer(batch, padding=True, truncation=True,
                                     max_length=self.text_max_length,
                                     return_tensors="pt").to(self.device)
                vecs = l2_normalize(_features(self.model.get_text_features(**enc)))
                out[start:start + len(batch)] = vecs.float().cpu().numpy()
        return out

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode_text([query])[0]

    def encode_images(self, images: Sequence[Image.Image], batch_size: int = 16) -> np.ndarray:
        """Images must already be RGB (see media.storage.decode_image)."""
        out = np.empty((len(images), self.dim), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(images), batch_size):
                batch = list(images[start:start + batch_size])
                pixels = self.image_processor(images=batch, return_tensors="pt")["pixel_values"]
                pixels = pixels.to(self.device, dtype=self.model.dtype)
                vecs = l2_normalize(_features(self.model.get_image_features(pixel_values=pixels)))
                out[start:start + len(batch)] = vecs.float().cpu().numpy()
        return out
