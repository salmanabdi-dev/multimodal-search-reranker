"""Image ingestion: validate -> content-address -> store file -> CLIP embed -> DB row.

Used two ways: one image at a time from the upload endpoint, and in batches from
the CLI (``ingest-images``). Batching matters for CLIP: one forward pass over 16
images is far cheaper than 16 passes.

Deduplication is by SHA-256 of the bytes. Uploading the same image twice returns
the existing document (optionally updating its caption) without re-embedding.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mmsearch.ingestion.text import clean_text
from mmsearch.media.storage import DecodedImage, ImageValidationError, MediaStore, decode_image

log = logging.getLogger(__name__)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".gif")


@dataclass
class RawImage:
    data: bytes
    external_id: str | None = None
    caption: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageIngestResult:
    external_id: str
    document_id: int
    sha256: str
    created: bool
    width: int
    height: int


@dataclass
class ImageIngestStats:
    images: int = 0
    duplicates: int = 0
    rejected: int = 0
    embed_seconds: float = 0.0
    total_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


async def _inline(fn: Callable, *args: Any) -> Any:
    return fn(*args)


def load_image_directory(path: str | Path, captions_file: str | Path | None = None
                         ) -> Iterator[RawImage]:
    """Every image under ``path``; id = file stem. Optional captions JSONL:
    ``{"_id": "<stem>", "caption": "..."}`` (a dataset's ``captions.jsonl`` is used
    automatically if present next to the images folder)."""
    root = Path(path)
    captions: dict[str, str] = {}
    cfile = Path(captions_file) if captions_file else root.parent / "captions.jsonl"
    if cfile.exists():
        with cfile.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    obj = json.loads(line)
                    if obj.get("caption"):
                        captions[str(obj["_id"])] = obj["caption"]
    for f in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS):
        yield RawImage(data=f.read_bytes(), external_id=f.stem, caption=captions.get(f.stem),
                       source=root.parent.name or root.name)


class ImageIngestionPipeline:
    def __init__(self, repo, clip, store: MediaStore, settings,  # noqa: ANN001
                 run: Callable[..., Awaitable[Any]] = _inline) -> None:
        if clip.dim != settings.clip_dim:
            raise ValueError(f"CLIP dim {clip.dim} != MMS_CLIP_DIM {settings.clip_dim}")
        self.repo, self.clip, self.store, self.settings = repo, clip, store, settings
        # `run` executes model calls; the API passes the service's inference executor so
        # uploads never block the event loop and never run concurrently with a search.
        self.run = run

    def _decode(self, raw: RawImage) -> DecodedImage:
        return decode_image(raw.data, max_bytes=self.settings.max_image_bytes,
                            max_pixels=self.settings.max_image_pixels)

    async def ingest_one(self, raw: RawImage) -> ImageIngestResult:
        """Raises ImageValidationError for bad input."""
        decoded = self._decode(raw)
        try:
            return (await self._ingest_decoded([(raw, decoded)]))[0]
        except Exception as exc:
            # Two concurrent uploads of the same new image: the loser hits the sha256
            # unique constraint. Retrying finds the winner's row and returns it.
            if type(exc).__name__ != "UniqueViolationError":
                raise
            return (await self._ingest_decoded([(raw, decoded)]))[0]

    async def ingest_many(self, raws: Iterable[RawImage], batch_size: int | None = None
                          ) -> ImageIngestStats:
        stats = ImageIngestStats()
        t0 = time.perf_counter()
        batch: list[tuple[RawImage, DecodedImage]] = []
        size = batch_size or self.settings.image_embed_batch_size

        async def flush() -> None:
            t = time.perf_counter()
            for res in await self._ingest_decoded(batch):
                if res.created:
                    stats.images += 1
                else:
                    stats.duplicates += 1
            stats.embed_seconds += time.perf_counter() - t
            batch.clear()

        for raw in raws:
            try:
                batch.append((raw, self._decode(raw)))
            except ImageValidationError as exc:
                stats.rejected += 1
                stats.errors.append(f"{raw.external_id}: {exc}")
                continue
            if len(batch) >= size:
                await flush()
                log.info("ingested %d images (%d duplicates)", stats.images, stats.duplicates)
        if batch:
            await flush()
        stats.total_seconds = time.perf_counter() - t0
        return stats

    async def _ingest_decoded(self, items: list[tuple[RawImage, DecodedImage]]
                              ) -> list[ImageIngestResult]:
        results: list[ImageIngestResult | None] = [None] * len(items)
        todo: list[int] = []
        seen: dict[str, int] = {}  # sha -> index of first occurrence within this batch
        async with self.repo.pool.acquire() as conn:
            for i, (raw, dec) in enumerate(items):
                existing = await self.repo.find_image_by_sha(conn, dec.sha256)
                if existing:
                    if raw.caption:
                        await self.repo.set_image_caption(conn, existing["id"],
                                                          clean_text(raw.caption))
                    results[i] = ImageIngestResult(existing["external_id"], existing["id"],
                                                   dec.sha256, False, dec.width, dec.height)
                elif dec.sha256 in seen:
                    results[i] = -1  # resolved after the first copy is inserted
                else:
                    seen[dec.sha256] = i
                    todo.append(i)

        if todo:
            vectors = await self.run(self.clip.encode_images, [items[i][1].image for i in todo],
                                     self.settings.image_embed_batch_size)
            async with self.repo.pool.acquire() as conn, conn.transaction():
                for vec, i in zip(vectors, todo, strict=True):
                    raw, dec = items[i]
                    rel = self.store.save(raw.data, dec)
                    ext_id = raw.external_id or f"img-{dec.sha256[:16]}"
                    caption = clean_text(raw.caption) if raw.caption else None
                    doc_id = await self.repo.upsert_image(
                        conn, external_id=ext_id, caption=caption, source=raw.source,
                        metadata=raw.metadata, sha256=dec.sha256, file_path=rel,
                        mime_type=dec.mime_type, width=dec.width, height=dec.height,
                        nbytes=dec.nbytes, embedding=vec, embedding_model=self.clip.name,
                    )
                    results[i] = ImageIngestResult(ext_id, doc_id, dec.sha256, True,
                                                   dec.width, dec.height)

        for i, r in enumerate(results):
            if r == -1:
                first = results[seen[items[i][1].sha256]]
                results[i] = ImageIngestResult(first.external_id, first.document_id,
                                               first.sha256, False, first.width, first.height)
        return results  # type: ignore[return-value]
