"""FastAPI application.

Start-up (lifespan) does all expensive work before the first request is accepted:
DB pool, model loading, a model/index consistency check, and warm-up.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from mmsearch.api.routes import router
from mmsearch.config import Settings, get_settings

log = logging.getLogger("mmsearch")
STATIC_DIR = Path(__file__).parent / "static"


def _check_index(s: Settings, stats: dict) -> None:
    """Refuse to serve vectors from a different model than the one that will embed queries."""
    if stats["chunks"] and stats["embedding_models"] != [s.bi_encoder_model]:
        raise RuntimeError(
            f"text index was built with {stats['embedding_models']} but the API is configured "
            f"for {s.bi_encoder_model!r}; re-ingest or fix MMS_BI_ENCODER_MODEL")
    if s.enable_images and stats["images"] and stats["image_embedding_models"] != [s.clip_model]:
        raise RuntimeError(
            f"image index was built with {stats['image_embedding_models']} but the API is "
            f"configured for {s.clip_model!r}; re-ingest images or fix MMS_CLIP_MODEL")
    if stats["chunks"] and not stats["hnsw_index"]:
        log.warning("no text HNSW index: searches will fall back to a sequential scan")
    if not stats["chunks"] and not stats["images"]:
        log.warning("index is empty: run `make ingest-sample` and/or `make ingest-images`")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    if getattr(app.state, "service", None) is not None:  # injected (tests)
        yield
        return

    from mmsearch.db.pool import create_pool, init_db
    from mmsearch.db.repository import Repository
    from mmsearch.search.factory import build_service

    s: Settings = app.state.settings
    await init_db(s.database_url)   # idempotent migrations: upgrades existing volumes
    pool = await create_pool(s.database_url, s.db_pool_min, s.db_pool_max, s.ef_search)
    repo = Repository(pool)
    stats = await repo.stats()
    _check_index(s, stats)

    service = build_service(s, repo, image_count=stats["images"])
    if service.images_enabled:
        from mmsearch.ingestion.images import ImageIngestionPipeline
        from mmsearch.media.storage import MediaStore

        app.state.media_store = MediaStore(s.media_dir)
        app.state.image_pipeline = ImageIngestionPipeline(
            repo, service.clip, app.state.media_store, s, run=service.run)
    await service.warmup(with_db=stats["chunks"] > 0 or stats["images"] > 0)
    app.state.repo, app.state.service = repo, service
    log.info("ready: %s", stats)
    try:
        yield
    finally:
        service.close()
        await pool.close()


def create_app(settings: Settings | None = None, service=None, repo=None) -> FastAPI:  # noqa: ANN001
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = FastAPI(title="Multimodal Search & Reranker", version="0.1.0", lifespan=_lifespan)
    app.state.settings, app.state.service, app.state.repo = settings, service, repo
    app.state.image_pipeline = app.state.media_store = None
    app.include_router(router)
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
    return app


app = create_app()
