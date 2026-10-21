import asyncio

from mmsearch.retrieval.vector_search import VectorRetriever
from mmsearch.search.service import SearchService
from tests.unit.fakes import FakeEncoder, FakeRepo, KeywordReranker, chunk_row, make_settings


def build():
    rows = [
        chunk_row(1, 1, 0.10, "cast iron skillet seasoning"),
        chunk_row(2, 2, 0.15, "ironman triathlon training"),
        chunk_row(3, 3, 0.20, "iron deficiency symptoms tiredness"),
        chunk_row(4, 4, 0.25, "symptoms of low iron pale skin"),
        chunk_row(5, 5, 0.30, "ferritin blood test"),
        chunk_row(6, 6, 0.35, "unrelated"),
    ]
    s = make_settings(rerank_candidates=6, default_top_k=3)
    return SearchService(FakeEncoder(), VectorRetriever(FakeRepo(rows), s), KeywordReranker(), s)


def test_rerank_reorders_and_reports_timings():
    svc = build()
    out = asyncio.run(svc.search("iron deficiency symptoms", top_k=3))
    assert [c.document_id for c in out.results] == [3, 4, 1]
    assert out.results[0].vector_rank == 3
    assert out.results[0].rerank_score == 3.0
    assert {"embed", "retrieve", "rerank", "total"} <= out.timings_ms.keys()
    assert out.candidates_considered == 6
    svc.close()


def test_vector_only_skips_reranker():
    svc = build()
    out = asyncio.run(svc.search("iron deficiency symptoms", top_k=3, rerank=False))
    assert [c.document_id for c in out.results] == [1, 2, 3]
    assert all(c.rerank_score is None for c in out.results)
    assert "rerank" not in out.timings_ms
    svc.close()


def test_warmup_without_db():
    svc = build()
    asyncio.run(svc.warmup(rounds=1, with_db=False))
    svc.close()
