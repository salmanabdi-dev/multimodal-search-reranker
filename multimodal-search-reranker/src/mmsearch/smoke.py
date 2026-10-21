"""Functional checks against a running API, one per capability.

Run after the index holds BEIR NFCorpus plus the synthetic images. Each check records
pass/fail and a short detail; `mmsearch smoke` writes them to reports/smoke_*.json and
exits non-zero if any fail. The integration tests prove the same things against a
throwaway database; these prove them against the deployed container.
"""

from __future__ import annotations

import io
import time
from collections.abc import Callable
from typing import Any


def _png(seed: int) -> bytes:
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (96, 96), (250, 250, 250))
    ImageDraw.Draw(img).polygon([(10, 80), (48, 8 + seed % 7), (86, 80)], fill=(20, 140, 140))
    img.putpixel((0, 0), (seed % 256, (seed >> 8) % 256, 7))  # unique bytes per run
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def run_smoke(url: str, client: Any = None) -> list[dict[str, Any]]:
    import httpx

    checks: list[dict[str, Any]] = []
    owns_client = client is None
    client = client or httpx.Client(base_url=url, timeout=30.0)

    def check(name: str, fn: Callable[[], str]) -> None:
        t0 = time.perf_counter()
        try:
            detail, ok = fn(), True
        except AssertionError as exc:
            detail, ok = str(exc) or "assertion failed", False
        except Exception as exc:  # noqa: BLE001
            detail, ok = f"{type(exc).__name__}: {exc}", False
        checks.append({"name": name, "ok": ok, "detail": detail,
                       "ms": round((time.perf_counter() - t0) * 1000, 1)})

    def search(q: str, modalities: list[str], top_k: int = 5, rerank: bool = True) -> dict:
        r = client.post("/search", json={"query": q, "top_k": top_k, "rerank": rerank,
                                         "modalities": modalities})
        assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:200]}"
        body = r.json()
        body["_timing"] = r.headers.get("server-timing", "")
        return body

    stats: dict = {}

    def ready() -> str:
        r = client.get("/ready")
        assert r.status_code == 200, f"/ready returned {r.status_code}"
        return "API ready, database reachable"

    def ingestion() -> str:
        stats.update(client.get("/stats").json()["index"])
        assert stats["chunks"] > 0, "no text chunks indexed"
        assert stats["images"] > 0, "no images indexed"
        return (f"{stats['text_documents']} documents / {stats['chunks']} chunks, "
                f"{stats['images']} images")

    def hnsw() -> str:
        assert stats.get("hnsw_index"), "text HNSW index missing"
        assert stats.get("image_hnsw_index"), "image HNSW index missing"
        return f"text and image HNSW indexes present (pgvector {stats.get('pgvector_version')})"

    def text_rerank() -> str:
        b = search("does vitamin d supplementation reduce cancer risk", ["text"])
        res = b["results"]
        assert len(res) == 5 and all(x["modality"] == "text" for x in res), "expected 5 docs"
        assert all(x["rerank_score"] is not None for x in res), "missing cross-encoder scores"
        moved = sum(x["vector_rank"] != x["rank"] for x in res)
        assert "rerank" in b["_timing"], "no rerank stage in Server-Timing"
        return f"cross-encoder reordered {moved}/5; top: {res[0]['title']!r}"

    def clip_retrieval() -> str:
        b = search("something blue", ["image"])
        ids = [x["document_id"] for x in b["results"]]
        hits = sum(i.startswith("blue_") for i in ids)
        assert hits >= 4, f"only {hits}/5 blue images: {ids}"
        return f"{hits}/5 results are blue images"

    def unified_visual() -> str:
        b = search("a red circle", ["text", "image"], top_k=10)
        kinds = [x["modality"] for x in b["results"]]
        assert b["modalities"] == ["text", "image"], f"searched {b['modalities']}"
        assert kinds[0] == "image", f"expected an image first, got {kinds}"
        rel = [x["relevance"] for x in b["results"]]
        assert rel == sorted(rel, reverse=True), "merged list not sorted by relevance"
        return f"image first; modalities in top 10: {sorted(set(kinds))}"

    def unified_textual() -> str:
        b = search("is coffee consumption linked to high blood pressure", ["text", "image"])
        assert b["results"][0]["modality"] == "text", "expected a document first"
        return f"document first: {b['results'][0]['title']!r}"

    def upload() -> str:
        before = client.get("/stats").json()["index"]["images"]
        data = _png(int(time.time()))
        r = client.post("/images", files={"file": ("smoke.png", data, "image/png")},
                        data={"caption": "smoke test teal triangle"})
        assert r.status_code == 201, f"upload returned {r.status_code}: {r.text[:200]}"
        body = r.json()
        again = client.post("/images", files={"file": ("again.png", data, "image/png")})
        assert again.status_code == 200 and again.json()["created"] is False, "no dedupe"
        served = client.get(body["image_url"])
        assert served.status_code == 200 and served.content == data, "served bytes differ"
        after = client.get("/stats").json()["index"]["images"]
        assert after == before + 1, f"image count {before} -> {after}"
        bad = client.post("/images", files={"file": ("x.png", b"not an image", "image/png")})
        assert bad.status_code == 415, f"invalid upload returned {bad.status_code}"
        return f"uploaded {body['document_id']}, deduplicated, served, invalid file rejected"

    for name, fn in [("api ready", ready), ("document + image ingestion", ingestion),
                     ("HNSW indexes", hnsw), ("text search + cross-encoder", text_rerank),
                     ("CLIP cross-modal retrieval", clip_retrieval),
                     ("unified search, visual query", unified_visual),
                     ("unified search, textual query", unified_textual),
                     ("image upload", upload)]:
        check(name, fn)
    if owns_client:
        client.close()
    return checks
