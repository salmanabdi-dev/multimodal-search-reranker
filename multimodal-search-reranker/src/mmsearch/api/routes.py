from __future__ import annotations

from urllib.parse import quote

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import ValidationError

from mmsearch.api.schemas import ImageUploadResponse, SearchRequest, SearchResponse, SearchResult
from mmsearch.search.service import SearchService, UnifiedOutcome

router = APIRouter()
SNIPPET_CHARS = 320


def image_url(external_id: str) -> str:
    return f"/images/{quote(external_id, safe='')}"


def _service(request: Request) -> SearchService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(503, "models are still loading")
    return service


def _result(rank: int, hit) -> SearchResult:  # noqa: ANN001
    if hit.modality == "text":
        c = hit.text
        return SearchResult(
            rank=rank, modality="text", document_id=c.external_id, title=c.title,
            snippet=c.content[:SNIPPET_CHARS], relevance=round(hit.relevance, 5),
            vector_score=round(c.vector_score, 5), vector_rank=c.vector_rank,
            rerank_score=None if c.rerank_score is None else round(c.rerank_score, 5))
    c = hit.image
    return SearchResult(
        rank=rank, modality="image", document_id=c.external_id, title=c.caption,
        image_url=image_url(c.external_id), width=c.width, height=c.height,
        relevance=round(hit.relevance, 5), vector_score=round(c.clip_score, 5),
        vector_rank=c.vector_rank,
        rerank_score=None if c.caption_score is None else round(c.caption_score, 5))


def _to_response(outcome: UnifiedOutcome) -> SearchResponse:
    return SearchResponse(
        query=outcome.query, reranked=outcome.reranked, modalities=outcome.modalities,
        candidates_considered=outcome.candidates_considered, timings_ms=outcome.timings_ms,
        results=[_result(i, h) for i, h in enumerate(outcome.hits, 1)],
    )


async def _search(request: Request, response: Response, body: SearchRequest) -> SearchResponse:
    outcome = await _service(request).search_unified(
        body.query, body.top_k, modalities=body.modalities, rerank=body.rerank)
    response.headers["Server-Timing"] = ", ".join(
        f"{k};dur={v:.2f}" for k, v in outcome.timings_ms.items())
    return _to_response(outcome)


@router.post("/search", response_model=SearchResponse)
async def search_post(body: SearchRequest, request: Request, response: Response):
    return await _search(request, response, body)


@router.get("/search", response_model=SearchResponse)
async def search_get(request: Request, response: Response,
                     q: str = Query(..., min_length=1, max_length=1000),
                     top_k: int = Query(5, ge=1, le=20), rerank: bool = True,
                     modalities: str = Query("text,image", description="comma-separated")):
    try:
        body = SearchRequest(query=q, top_k=top_k, rerank=rerank,
                             modalities=[m.strip() for m in modalities.split(",") if m.strip()])
    except ValidationError as exc:
        raise HTTPException(422, exc.errors(include_url=False, include_context=False)) from exc
    return await _search(request, response, body)


@router.post("/images", response_model=ImageUploadResponse, status_code=201)
async def upload_image(request: Request, response: Response,
                       file: UploadFile = File(..., description="JPEG, PNG, WEBP or GIF"),
                       caption: str | None = Form(None, max_length=500),
                       external_id: str | None = Form(None, max_length=200,
                                                      pattern=r"^[A-Za-z0-9._:-]+$")):
    from mmsearch.media.storage import ImageValidationError
    from mmsearch.ingestion.images import RawImage

    pipeline = getattr(request.app.state, "image_pipeline", None)
    if pipeline is None:
        raise HTTPException(503, "image ingestion is disabled or still loading")
    limit = request.app.state.settings.max_image_bytes
    data = await file.read(limit + 1)  # read at most one byte past the limit
    if len(data) > limit:
        raise HTTPException(413, f"image larger than {limit} bytes")
    try:
        res = await pipeline.ingest_one(RawImage(
            data=data, external_id=external_id, caption=caption or None, source="upload",
            metadata={"filename": file.filename} if file.filename else {}))
    except ImageValidationError as exc:
        raise HTTPException(415, str(exc)) from exc
    except ValueError as exc:  # external_id collides with a text document
        raise HTTPException(409, str(exc)) from exc
    if res.created:
        _service(request).image_count += 1
    else:
        response.status_code = 200
    return ImageUploadResponse(document_id=res.external_id, created=res.created,
                               sha256=res.sha256, width=res.width, height=res.height,
                               image_url=image_url(res.external_id))


@router.get("/images/{external_id}")
async def get_image(external_id: str, request: Request):
    repo = getattr(request.app.state, "repo", None)
    store = getattr(request.app.state, "media_store", None)
    if repo is None or store is None:
        raise HTTPException(503, "not ready")
    row = await repo.get_image(external_id)
    if row is None:
        raise HTTPException(404, "no such image")
    path = store.resolve(row.file_path)
    if not path.exists():
        raise HTTPException(410, "image file is missing from the media store")
    # Content-addressed files never change, so they can be cached forever.
    return FileResponse(path, media_type=row.mime_type,
                        headers={"Cache-Control": "public, max-age=31536000, immutable"})


@router.get("/health")
async def health():
    """Liveness: the process is up."""
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request):
    """Readiness: models loaded, warm-up done, database reachable."""
    repo = getattr(request.app.state, "repo", None)
    if getattr(request.app.state, "service", None) is None or repo is None:
        raise HTTPException(503, "not ready")
    try:
        await repo.ping()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"database unavailable: {exc}") from exc
    return {"status": "ready"}


@router.get("/stats")
async def stats(request: Request):
    repo = getattr(request.app.state, "repo", None)
    if repo is None:
        raise HTTPException(503, "not ready")
    s = request.app.state.settings
    return {
        "index": await repo.stats(),
        "config": {
            "bi_encoder": s.bi_encoder_model, "cross_encoder": s.cross_encoder_model,
            "clip": s.clip_model if s.enable_images else None,
            "candidate_chunks": s.candidate_chunks, "rerank_candidates": s.rerank_candidates,
            "image_candidates": s.image_candidates, "caption_weight": s.caption_weight,
            "ef_search": s.ef_search, "device": s.device, "torch_threads": s.torch_threads,
        },
    }
