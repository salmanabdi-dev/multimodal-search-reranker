"""Multimodal end-to-end: real Postgres+pgvector, real MiniLM, CLIP and cross-encoder.

Setup ingests the text sample corpus plus the synthetic shapes dataset (48 images).
Images arrive through BOTH paths: most through the bulk pipeline (the CLI path),
and every red image through HTTP uploads, so search tests also prove uploaded
images are searchable immediately.

    make test-integration
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics

import pytest

pytestmark = pytest.mark.integration

DSN = os.environ.get("MMS_TEST_DATABASE_URL")
if not DSN:
    pytest.skip("MMS_TEST_DATABASE_URL not set", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from mmsearch.api.main import create_app  # noqa: E402
from mmsearch.config import Settings  # noqa: E402
from mmsearch.db.pool import create_pool, init_db  # noqa: E402
from mmsearch.db.repository import Repository  # noqa: E402
from mmsearch.embeddings.bi_encoder import HFTokenizerAdapter  # noqa: E402
from mmsearch.evaluation.datasets import SAMPLE_DIR, load_dataset  # noqa: E402
from mmsearch.evaluation.metrics import precision_at_k  # noqa: E402
from mmsearch.ingestion.chunker import TokenChunker  # noqa: E402
from mmsearch.ingestion.images import ImageIngestionPipeline, load_image_directory  # noqa: E402
from mmsearch.ingestion.loaders import load_jsonl  # noqa: E402
from mmsearch.ingestion.pipeline import IngestionPipeline  # noqa: E402
from mmsearch.media.storage import MediaStore  # noqa: E402
from mmsearch.media.synthetic import COLORS, generate  # noqa: E402
from mmsearch.search.factory import configure_torch, load_bi_encoder, load_clip  # noqa: E402

UPLOADED_PREFIX = "red_"


async def _prepare(s: Settings, dataset) -> None:  # noqa: ANN001
    await init_db(s.database_url)
    pool = await create_pool(s.database_url, 1, 2, s.ef_search)
    try:
        async with pool.acquire() as conn:
            await conn.execute("TRUNCATE documents, chunks, image_embeddings "
                               "RESTART IDENTITY CASCADE")
        repo = Repository(pool)
        configure_torch(s.torch_threads)
        enc = load_bi_encoder(s)
        await IngestionPipeline(repo, enc, TokenChunker(HFTokenizerAdapter(enc.tokenizer)), s
                                ).run(load_jsonl(SAMPLE_DIR / "corpus.jsonl"))
        bulk = [r for r in load_image_directory(dataset / "images")
                if not r.external_id.startswith(UPLOADED_PREFIX)]
        stats = await ImageIngestionPipeline(repo, load_clip(s), MediaStore(s.media_dir), s
                                             ).ingest_many(bulk)
        assert stats.images == 40 and stats.rejected == 0
    finally:
        await pool.close()


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    return generate(tmp_path_factory.mktemp("synth"), seed=7, instances=2)


@pytest.fixture(scope="module")
def settings(tmp_path_factory):
    return Settings(_env_file=None, database_url=DSN,
                    media_dir=str(tmp_path_factory.mktemp("media")))


def _upload(client, path, caption=None, external_id=None):  # noqa: ANN001
    data = {k: v for k, v in (("caption", caption), ("external_id", external_id)) if v}
    return client.post("/images", files={"file": (path.name, path.read_bytes(), "image/png")},
                       data=data)


@pytest.fixture(scope="module")
def client(dataset, settings):
    asyncio.run(_prepare(settings, dataset))
    captions = {}
    for line in (dataset / "captions.jsonl").read_text().splitlines():
        obj = json.loads(line)
        captions[obj["_id"]] = obj["caption"]
    with TestClient(create_app(settings)) as c:
        # The red images go in over HTTP, so every search test below covers both paths.
        c.red = sorted((dataset / "images").glob(f"{UPLOADED_PREFIX}*.png"))
        c.uploads = [_upload(c, p, caption=captions.get(p.stem), external_id=p.stem)
                     for p in c.red]
        yield c


def _search(client, query, modalities=("image",), top_k=5, rerank=True):  # noqa: ANN001
    r = client.post("/search", json={"query": query, "top_k": top_k, "rerank": rerank,
                                     "modalities": list(modalities)})
    assert r.status_code == 200, r.text
    return r.json()


# ------------------------------------------------------------------ ingestion

def test_bulk_ingestion_built_image_index(client, settings):
    idx = client.get("/stats").json()["index"]
    assert idx["images"] == 40
    assert idx["text_documents"] == 46
    assert idx["image_hnsw_index"] is True
    assert idx["image_embedding_models"] == [settings.clip_model]


def test_upload_ingests_dedupes_and_serves(client):
    red = client.red
    assert len(red) == 8
    for r in client.uploads:
        assert r.status_code == 201, r.text
        assert r.json()["created"] is True
    again = _upload(client, red[0], external_id="a-different-id")
    assert again.status_code == 200 and again.json()["created"] is False
    assert again.json()["document_id"] == red[0].stem       # same bytes, same document
    assert client.get("/stats").json()["index"]["images"] == 48
    served = client.get(again.json()["image_url"])
    assert served.status_code == 200 and served.content == red[0].read_bytes()


def test_upload_rejects_non_images_and_id_clashes(client):
    r = client.post("/images", files={"file": ("x.png", b"<?php echo 1; ?>", "image/png")})
    assert r.status_code == 415
    from PIL import Image
    import io
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (1, 2, 3)).save(buf, "PNG")   # new bytes, so not a duplicate
    clash = client.post("/images", data={"external_id": "s01"},   # s01 is a text document
                        files={"file": ("x.png", buf.getvalue(), "image/png")})
    assert clash.status_code == 409, clash.text


# ------------------------------------------------------------------ cross-modal retrieval

@pytest.mark.parametrize("color", list(COLORS))
def test_text_query_retrieves_images_of_that_colour(client, color):
    res = _search(client, f"something {color}")
    ids = [x["document_id"] for x in res["results"]]
    assert len(ids) == 5
    assert sum(i.startswith(f"{color}_") for i in ids) >= 4, ids


def test_colour_and_shape_query(client):
    res = _search(client, "a red circle", top_k=5)
    ids = [x["document_id"] for x in res["results"]]
    assert ids[0].startswith("red_"), ids
    assert any(i.startswith("red_circle") for i in ids), ids


def test_image_hnsw_agrees_with_exact_search(client, settings):
    vec = client.app.state.service.clip.encode_query("a blue star")

    async def compare():
        pool = await create_pool(settings.database_url, 1, 1, settings.ef_search)
        try:
            repo = Repository(pool)
            approx = [r.external_id for r in await repo.image_ann_search(vec, 10, 100)]
            exact = [r.external_id for r in await repo.image_exact_search(vec, 10)]
        finally:
            await pool.close()
        return approx, exact

    approx, exact = asyncio.run(compare())
    assert set(approx) == set(exact)


# ------------------------------------------------------------------ combined search

def test_combined_search_returns_both_modalities_merged(client):
    res = _search(client, "a red circle", modalities=("text", "image"), top_k=20)
    assert res["modalities"] == ["text", "image"]
    kinds = {x["modality"] for x in res["results"]}
    assert kinds == {"text", "image"}
    rel = [x["relevance"] for x in res["results"]]
    assert rel == sorted(rel, reverse=True)
    assert res["results"][0]["modality"] == "image"         # visual query -> image first
    assert all(x["image_url"] for x in res["results"] if x["modality"] == "image")
    assert {"embed_text", "embed_clip", "retrieve", "rerank"} <= res["timings_ms"].keys()


def test_combined_search_puts_text_first_for_a_text_question(client):
    res = _search(client, "what are the symptoms of iron deficiency",
                  modalities=("text", "image"), top_k=5)
    top = res["results"][0]
    assert top["modality"] == "text"
    assert top["document_id"] in load_dataset(SAMPLE_DIR).relevant("q4")


def test_text_only_search_is_unchanged(client):
    res = _search(client, "why is the sky blue", modalities=("text",))
    assert res["modalities"] == ["text"]
    assert all(x["modality"] == "text" for x in res["results"])
    assert res["results"][0]["document_id"] in load_dataset(SAMPLE_DIR).relevant("q2")


def test_cross_modal_quality_smoke(client, dataset):
    """Colour queries over the full 48-image set: P@5 for CLIP alone and with fusion."""
    ds = load_dataset(dataset)
    qids = [q for q in ds.queries if q.endswith("-any")]
    p = {True: [], False: []}
    for rerank in (False, True):
        for q in qids:
            ids = [x["document_id"] for x in _search(client, ds.queries[q], rerank=rerank)["results"]]
            p[rerank].append(precision_at_k(ids, ds.relevant(q), 5))
    clip_only, fused = statistics.fmean(p[False]), statistics.fmean(p[True])
    print(f"\ncolour-query P@5: CLIP {clip_only:.3f}, fusion {fused:.3f}")
    assert clip_only >= 0.8
    assert fused >= clip_only - 0.1
