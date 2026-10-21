"""Lightweight stand-ins so pipeline logic can be tested without models or a database."""

from __future__ import annotations

import re
from types import SimpleNamespace

import numpy as np

from mmsearch.db.models import ChunkRow


class WhitespaceTokenizer:
    """Every whitespace-separated word is one token; [CLS]/[SEP] cost 2."""
    special_tokens = 2

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [m.span() for m in re.finditer(r"\S+", text)]


def make_settings(**overrides):
    base = dict(inference_workers=1, default_top_k=5, rerank_candidates=10,
                candidate_chunks=20, ef_search=40)
    base.update(overrides)
    s = SimpleNamespace(**base)
    s.k_chunks_for = lambda n: max(s.candidate_chunks, 2 * n)
    s.ef_for = lambda k: max(s.ef_search, k)
    return s


def chunk_row(chunk_id: int, doc_id: int, distance: float, content: str = "x") -> ChunkRow:
    return ChunkRow(chunk_id=chunk_id, document_id=doc_id, external_id=f"d{doc_id}",
                    title=f"title {doc_id}", content=content, distance=distance)


class FakeEncoder:
    name = "fake"
    dim = 4

    def encode_query(self, q: str) -> np.ndarray:
        return np.ones(4, dtype=np.float32) / 2


class FakeRepo:
    def __init__(self, rows: list[ChunkRow]) -> None:
        self.rows = rows
        self.calls: list[tuple[int, int | None]] = []

    async def ann_search(self, vec, k, ef_search=None):  # noqa: ANN001
        self.calls.append((k, ef_search))
        return sorted(self.rows, key=lambda r: r.distance)[:k]


class KeywordReranker:
    """Scores a passage by how many query words it contains."""

    def score(self, query: str, passages):  # noqa: ANN001
        words = set(query.lower().split())
        return np.array([sum(w in p.lower() for w in words) for p in passages], dtype=np.float32)


# ---------------------------------------------------------------- multimodal fakes

import contextlib  # noqa: E402

from mmsearch.db.models import ImageRow  # noqa: E402

COLOR_WORDS = {"red": (1, 0, 0), "green": (0, 1, 0), "blue": (0, 0, 1),
               "yellow": (1, 1, 0), "purple": (0.5, 0, 0.6), "orange": (1, 0.5, 0)}


def _unit(v) -> np.ndarray:  # noqa: ANN001
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) or 1.0)


class FakeClip:
    """Images -> normalised mean colour; queries -> the colour words they mention.
    Crude, but it makes 'red' retrieve red images through the real code path."""
    name = "fake-clip"
    dim = 4

    def __init__(self) -> None:
        self.image_calls = 0

    def encode_images(self, images, batch_size=16):  # noqa: ANN001
        self.image_calls += 1
        out = []
        for img in images:
            r, g, b = np.asarray(img.convert("RGB"), dtype=np.float32).reshape(-1, 3).mean(0) / 255
            # subtract the light background so the shape's colour dominates
            out.append(_unit([1 - g - b + r, 1 - r - b + g, 1 - r - g + b, 0.05]))
        return np.stack(out)

    def encode_query(self, q: str) -> np.ndarray:
        for word, rgb in COLOR_WORDS.items():
            if word in q.lower():
                return _unit([*rgb, 0.05])
        return _unit([1, 1, 1, 1])


class FakeConn:
    @contextlib.asynccontextmanager
    async def transaction(self):
        yield


class FakePool:
    @contextlib.asynccontextmanager
    async def acquire(self):
        yield FakeConn()


class FakeImageRepo:
    """In-memory stand-in for the image half of Repository."""

    def __init__(self) -> None:
        self.pool = FakePool()
        self.images: dict[str, dict] = {}   # external_id -> row
        self.text_ids: set[str] = set()
        self.ann_calls: list[tuple[int, int | None]] = []

    async def find_image_by_sha(self, conn, sha):  # noqa: ANN001
        for r in self.images.values():
            if r["sha256"] == sha:
                return {"id": r["id"], "external_id": r["external_id"],
                        "width": r["width"], "height": r["height"]}
        return None

    async def set_image_caption(self, conn, document_id, caption):  # noqa: ANN001
        for r in self.images.values():
            if r["id"] == document_id:
                r["caption"] = caption

    async def upsert_image(self, conn, **kw):  # noqa: ANN001
        if kw["external_id"] in self.text_ids:
            raise ValueError("already belongs to a text document")
        row = self.images.get(kw["external_id"]) or {"id": len(self.images) + 1}
        row.update(kw)
        self.images[kw["external_id"]] = row
        return row["id"]

    async def image_ann_search(self, vec, k, ef_search=None):  # noqa: ANN001
        self.ann_calls.append((k, ef_search))
        rows = [ImageRow(document_id=r["id"], external_id=r["external_id"],
                         caption=r["caption"], file_path=r["file_path"],
                         mime_type=r["mime_type"], width=r["width"], height=r["height"],
                         distance=float(1 - np.dot(vec, r["embedding"])))
                for r in self.images.values()]
        return sorted(rows, key=lambda r: r.distance)[:k]

    async def image_exact_search(self, vec, k):  # noqa: ANN001
        return await self.image_ann_search(vec, k)


class RecordingReranker(KeywordReranker):
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def score(self, query, passages):  # noqa: ANN001
        self.calls.append(list(passages))
        return super().score(query, passages)


def make_mm_settings(**overrides):
    s = make_settings(**overrides)
    defaults = dict(clip_dim=4, max_image_bytes=5_000_000, max_image_pixels=4_000_000,
                    image_embed_batch_size=8, image_candidates=10, caption_weight=0.5,
                    clip_calib_a=10.0, clip_calib_b=-5.0, text_vec_calib_a=12.0,
                    text_vec_calib_b=-6.0)
    for key, value in defaults.items():
        if not hasattr(s, key):
            setattr(s, key, value)
    return s


def png_bytes(color=(255, 0, 0), size=(64, 64), fmt="PNG", seed=0) -> bytes:  # noqa: ANN001
    import io

    from PIL import Image, ImageDraw
    img = Image.new("RGB", size, (250, 250, 250))
    ImageDraw.Draw(img).ellipse([8 + seed % 5, 8, size[0] - 8, size[1] - 8], fill=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()
