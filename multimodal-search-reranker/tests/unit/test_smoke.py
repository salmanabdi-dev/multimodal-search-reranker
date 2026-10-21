import json

import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("PIL")

from mmsearch.smoke import run_smoke  # noqa: E402


def file_part(req: httpx.Request) -> bytes:
    """Extract the bytes of the multipart field named 'file'."""
    boundary = req.headers["content-type"].split("boundary=")[1].encode()
    for part in req.content.split(b"--" + boundary):
        head, _, body = part.partition(b"\r\n\r\n")
        if b'name="file"' in head:
            return body[:-2] if body.endswith(b"\r\n") else body
    raise AssertionError("no file part")


def fake_api(broken: str | None = None):
    images = {"count": 96, "uploaded": {}}

    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path == "/ready":
            return httpx.Response(200, json={"status": "ready"})
        if path == "/stats":
            return httpx.Response(200, json={"index": {
                "chunks": 5000, "images": images["count"], "text_documents": 3633,
                "hnsw_index": broken != "hnsw", "image_hnsw_index": True,
                "pgvector_version": "0.8.0"}})
        if path == "/search":
            body = json.loads(req.content)
            mods = body["modalities"]
            if mods == ["image"]:
                ids = ["blue_star_0", "blue_circle_1", "blue_square_0", "blue_star_2",
                       "red_circle_0" if broken != "clip" else "red_star_0"]
                if broken == "clip":
                    ids[:3] = ["green_star_0", "red_square_1", "yellow_star_0"]
                res = [{"modality": "image", "document_id": i, "rank": k + 1, "vector_rank": k + 1,
                        "relevance": 0.9 - k / 10, "rerank_score": None, "title": None}
                       for k, i in enumerate(ids)]
            elif mods == ["text"]:
                res = [{"modality": "text", "document_id": f"d{k}", "rank": k + 1,
                        "vector_rank": 5 - k, "relevance": 0.9 - k / 10, "rerank_score": 1.0,
                        "title": "Vitamin D"} for k in range(5)]
            else:
                visual = "circle" in body["query"]
                first, second = ("image", "text") if visual else ("text", "image")
                res = [{"modality": m, "document_id": f"x{k}", "rank": k + 1, "vector_rank": 1,
                        "relevance": 0.9 - k / 20, "rerank_score": None, "title": "t"}
                       for k, m in enumerate([first, second] * 3)]
            return httpx.Response(200, json={"results": res, "modalities": mods},
                                  headers={"server-timing": "embed_text;dur=5, rerank;dur=30"})
        if path == "/images" and req.method == "POST":
            data = file_part(req)
            if not data.startswith(b"\x89PNG"):
                return httpx.Response(415, json={"detail": "not an image"})
            if data in images["uploaded"]:           # content-addressed, like the real API
                return httpx.Response(200, json={"created": False, **images["uploaded"][data]})
            images["png"], images["count"] = data, images["count"] + 1
            images["uploaded"][data] = {"document_id": "img-1", "image_url": "/images/img-1"}
            return httpx.Response(201, json={"created": True, **images["uploaded"][data]})
        if path == "/images/img-1":
            return httpx.Response(200, content=images["png"])
        return httpx.Response(404)

    return httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))


def test_all_checks_pass_against_a_healthy_api():
    checks = run_smoke("http://test", client=fake_api())
    assert [c["ok"] for c in checks] == [True] * 8, [c for c in checks if not c["ok"]]
    assert checks[4]["detail"] == "4/5 results are blue images"


@pytest.mark.parametrize("broken, failing", [("hnsw", "HNSW indexes"),
                                             ("clip", "CLIP cross-modal retrieval")])
def test_failures_are_reported_not_raised(broken, failing):
    checks = {c["name"]: c for c in run_smoke("http://test", client=fake_api(broken))}
    assert checks[failing]["ok"] is False
    assert sum(not c["ok"] for c in checks.values()) == 1
