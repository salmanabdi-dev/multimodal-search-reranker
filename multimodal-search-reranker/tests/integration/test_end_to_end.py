"""End-to-end against real Postgres+pgvector and real models.

Runs against MMS_TEST_DATABASE_URL (a separate `mms_test` database created by
sql/000_test_db.sql), never the main database. Run with:

    make test-integration        # == pytest -m integration inside the api container
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pytest

pytestmark = pytest.mark.integration

DSN = os.environ.get("MMS_TEST_DATABASE_URL")
if not DSN:
    pytest.skip("MMS_TEST_DATABASE_URL not set", allow_module_level=True)

from mmsearch.config import Settings  # noqa: E402
from mmsearch.db.pool import create_pool, init_db  # noqa: E402
from mmsearch.db.repository import Repository  # noqa: E402
from mmsearch.embeddings.bi_encoder import HFTokenizerAdapter  # noqa: E402
from mmsearch.evaluation.datasets import SAMPLE_DIR, load_dataset  # noqa: E402
from mmsearch.evaluation.runner import collect_runs, evaluate  # noqa: E402
from mmsearch.ingestion.chunker import TokenChunker  # noqa: E402
from mmsearch.ingestion.loaders import load_jsonl  # noqa: E402
from mmsearch.ingestion.pipeline import IngestionPipeline  # noqa: E402
from mmsearch.retrieval.vector_search import VectorRetriever  # noqa: E402
from mmsearch.search.factory import configure_torch, load_bi_encoder, load_cross_encoder  # noqa: E402
from mmsearch.search.service import SearchService  # noqa: E402


@lru_cache
def settings() -> Settings:
    return Settings(_env_file=None, database_url=DSN)


@lru_cache
def models():
    configure_torch(settings().torch_threads)
    return load_bi_encoder(settings()), load_cross_encoder(settings())


@pytest.fixture
async def repo():
    await init_db(DSN)
    pool = await create_pool(DSN, 1, 2, settings().ef_search)
    async with pool.acquire() as conn:
        await conn.execute("TRUNCATE documents, chunks RESTART IDENTITY CASCADE")
    r = Repository(pool)
    enc, _ = models()
    chunker = TokenChunker(HFTokenizerAdapter(enc.tokenizer), 256, 32)
    await IngestionPipeline(r, enc, chunker, settings()).run(
        load_jsonl(SAMPLE_DIR / "corpus.jsonl"), rebuild_index=True)
    yield r
    await pool.close()


@pytest.fixture
async def service(repo):
    enc, ce = models()
    svc = SearchService(enc, VectorRetriever(repo, settings()), ce, settings())
    yield svc
    svc.close()


async def test_ingestion_populates_index(repo):
    st = await repo.stats()
    assert st["documents"] == 46
    assert st["chunks"] >= 46
    assert st["hnsw_index"] is True
    assert st["embedding_models"] == [settings().bi_encoder_model]


async def test_reingest_is_idempotent(repo):
    enc, _ = models()
    chunker = TokenChunker(HFTokenizerAdapter(enc.tokenizer), 256, 32)
    before = await repo.stats()
    await IngestionPipeline(repo, enc, chunker, settings()).run(
        load_jsonl(SAMPLE_DIR / "corpus.jsonl"), rebuild_index=False)
    assert (await repo.stats())["chunks"] == before["chunks"]


async def test_hnsw_matches_exact_on_small_corpus(repo):
    enc, _ = models()
    v = enc.encode_query("why is the sky blue")
    approx = [r.chunk_id for r in await repo.ann_search(v, 10, 100)]
    exact = [r.chunk_id for r in await repo.exact_search(v, 10)]
    assert set(approx) == set(exact)


async def test_search_returns_relevant_top_result(service):
    out = await service.search("what are the symptoms of iron deficiency", top_k=5)
    ds = load_dataset(SAMPLE_DIR)
    assert out.results[0].external_id in ds.relevant("q4")
    assert len(out.results) == 5
    assert out.results[0].rerank_score is not None


async def test_reranking_does_not_hurt_on_sample(service):
    ds = load_dataset(SAMPLE_DIR)
    qids = ds.judged_query_ids()
    runs = await collect_runs(service, ds, qids, 25)
    r = evaluate(runs, ds, qids, 25, bootstrap=False)
    assert r.b["p@5"] >= r.a["p@5"]


def test_embeddings_match_sentence_transformers():
    st = pytest.importorskip("sentence_transformers")
    enc, _ = models()
    ref = st.SentenceTransformer(settings().bi_encoder_model, device="cpu")
    texts = ["short", "a somewhat longer sentence about vector search and reranking"]
    ours = enc.encode(texts)
    theirs = ref.encode(texts, normalize_embeddings=True)
    assert np.allclose(ours, theirs, atol=1e-4)
