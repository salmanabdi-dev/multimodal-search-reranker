import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
try:  # python-multipart renamed its import package in 0.0.13
    import python_multipart  # noqa: F401
except ImportError:
    pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402

from mmsearch.api.main import create_app  # noqa: E402
from mmsearch.config import Settings  # noqa: E402
from mmsearch.db.models import ImageRow  # noqa: E402
from mmsearch.ingestion.images import ImageIngestResult  # noqa: E402
from mmsearch.media.storage import MediaStore, decode_image  # noqa: E402
from mmsearch.retrieval.image_search import ImageCandidate  # noqa: E402
from mmsearch.retrieval.vector_search import Candidate  # noqa: E402
from mmsearch.search.service import UnifiedOutcome  # noqa: E402
from mmsearch.search.unified import UnifiedHit  # noqa: E402
from tests.unit.fakes import png_bytes  # noqa: E402


class StubService:
    image_count = 0

    def __init__(self):
        self.calls = []

    async def search_unified(self, query, top_k=None, *, modalities=("text", "image"),
                             rerank=True):
        self.calls.append((query, top_k, list(modalities), rerank))
        hits = []
        if "text" in modalities:
            c = Candidate(1, "doc-1", "Title", 7, "some content", 0.81, 4,
                          2.5 if rerank else None)
            hits.append(UnifiedHit("text", 0.92, text=c))
        if "image" in modalities:
            i = ImageCandidate(2, "red circle", "a red circle", "ab/ab.png", "image/png",
                               64, 48, 0.31, 1, 3.0 if rerank else None, 0.88)
            hits.append(UnifiedHit("image", 0.88, image=i))
        return UnifiedOutcome(query, hits, {"embed_text": 5.0, "retrieve": 2.0, "total": 9.0},
                              rerank, list(modalities), {"text": 25, "image": 20})


class StubPipeline:
    def __init__(self):
        self.seen = {}

    async def ingest_one(self, raw):
        if raw.external_id == "doc-1":
            raise ValueError("external_id 'doc-1' already belongs to a text document")
        d = decode_image(raw.data, max_bytes=10**7, max_pixels=10**8)  # may raise
        created = d.sha256 not in self.seen
        ext = self.seen.setdefault(d.sha256, raw.external_id or f"img-{d.sha256[:16]}")
        return ImageIngestResult(ext, 1, d.sha256, created, d.width, d.height)


class StubRepo:
    def __init__(self, rel):
        self.rel = rel

    async def ping(self):
        return True

    async def stats(self):
        return {"documents": 1, "chunks": 1, "images": 1}

    async def get_image(self, external_id):
        if external_id != "red circle":
            return None
        return ImageRow(2, "red circle", "a red circle", self.rel, "image/png", 64, 64)


@pytest.fixture
def client(tmp_path):
    store = MediaStore(tmp_path)
    data = png_bytes()
    rel = store.save(data, decode_image(data, max_bytes=10**7, max_pixels=10**8))
    service = StubService()
    app = create_app(Settings(_env_file=None, max_image_bytes=200_000),
                     service=service, repo=StubRepo(rel))
    app.state.media_store, app.state.image_pipeline = store, StubPipeline()
    with TestClient(app) as c:
        c.service, c.image_bytes = service, data
        yield c


def test_unified_search_returns_both_modalities(client):
    r = client.post("/search", json={"query": "  red  ", "top_k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "red" and body["modalities"] == ["text", "image"]
    text, image = body["results"]
    assert text == {
        "rank": 1, "modality": "text", "document_id": "doc-1", "title": "Title",
        "snippet": "some content", "image_url": None, "width": None, "height": None,
        "relevance": 0.92, "vector_score": 0.81, "vector_rank": 4, "rerank_score": 2.5}
    assert image["modality"] == "image" and image["rank"] == 2
    assert image["image_url"] == "/images/red%20circle"          # ids are URL-quoted
    assert image["title"] == "a red circle" and image["snippet"] is None
    assert (image["width"], image["height"]) == (64, 48)
    assert body["candidates_considered"] == {"text": 25, "image": 20}
    assert "total;dur=9.00" in r.headers["server-timing"]


def test_modality_filter_and_get_variant(client):
    r = client.get("/search", params={"q": "red", "modalities": "image", "rerank": "false"})
    assert r.status_code == 200
    assert [x["modality"] for x in r.json()["results"]] == ["image"]
    assert client.service.calls[-1] == ("red", 5, ["image"], False)
    client.post("/search", json={"query": "x", "modalities": ["text", "text"]})
    assert client.service.calls[-1][2] == ["text"]                # de-duplicated


def test_validation(client):
    assert client.post("/search", json={"query": "   "}).status_code == 422
    assert client.post("/search", json={"query": "x", "top_k": 50}).status_code == 422
    assert client.post("/search", json={"query": "x", "modalities": []}).status_code == 422
    assert client.post("/search", json={"query": "x", "modalities": ["audio"]}).status_code == 422
    assert client.get("/search", params={"q": "   "}).status_code == 422
    assert client.get("/search", params={"q": "x", "modalities": "video"}).status_code == 422


def test_upload_creates_then_dedupes(client):
    files = {"file": ("x.png", png_bytes((0, 0, 255)), "image/png")}
    r = client.post("/images", files=files, data={"caption": "a blue dot"})
    assert r.status_code == 201
    body = r.json()
    assert body["created"] is True and body["image_url"].startswith("/images/img-")
    assert client.service.image_count == 1
    r2 = client.post("/images", files={"file": ("y.png", png_bytes((0, 0, 255)), "image/png")})
    assert r2.status_code == 200 and r2.json()["created"] is False
    assert r2.json()["document_id"] == body["document_id"]
    assert client.service.image_count == 1


def test_upload_rejections(client):
    bad = client.post("/images", files={"file": ("x.png", b"not an image", "image/png")})
    assert bad.status_code == 415
    big = client.post("/images", files={"file": ("x.png", b"0" * 300_000, "image/png")})
    assert big.status_code == 413
    clash = client.post("/images", files={"file": ("x.png", png_bytes(), "image/png")},
                        data={"external_id": "doc-1"})
    assert clash.status_code == 409
    weird = client.post("/images", files={"file": ("x.png", png_bytes(), "image/png")},
                        data={"external_id": "../etc"})
    assert weird.status_code == 422


def test_image_file_is_served(client):
    r = client.get("/images/red%20circle")
    assert r.status_code == 200
    assert r.content == client.image_bytes
    assert r.headers["content-type"] == "image/png"
    assert "immutable" in r.headers["cache-control"]
    assert client.get("/images/nope").status_code == 404


def test_health_ready_ui(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"status": "ready"}
    page = client.get("/").text
    assert "Search the index" in page and "Add image" in page


def test_not_ready_without_service():
    app = create_app(Settings(_env_file=None))
    c = TestClient(app)  # no lifespan: nothing loaded
    assert c.get("/ready").status_code == 503
    assert c.get("/search", params={"q": "x"}).status_code == 503
    assert c.post("/images", files={"file": ("x.png", b"x", "image/png")}).status_code == 503
