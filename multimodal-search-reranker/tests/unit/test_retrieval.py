import asyncio

from mmsearch.retrieval.vector_search import VectorRetriever, collapse_to_documents
from tests.unit.fakes import FakeRepo, chunk_row, make_settings


def test_collapse_keeps_best_chunk_per_document():
    rows = [chunk_row(1, 10, 0.1), chunk_row(2, 10, 0.2), chunk_row(3, 11, 0.3),
            chunk_row(4, 12, 0.4), chunk_row(5, 11, 0.5)]
    out = collapse_to_documents(rows, n_docs=10)
    assert [c.document_id for c in out] == [10, 11, 12]
    assert [c.chunk_id for c in out] == [1, 3, 4]
    assert [c.vector_rank for c in out] == [1, 2, 3]
    assert out[0].vector_score == 0.9


def test_collapse_stops_at_n():
    rows = [chunk_row(i, i, i / 100) for i in range(20)]
    assert len(collapse_to_documents(rows, 5)) == 5


def test_retriever_fetches_extra_chunks_and_forces_ef_at_least_k():
    repo = FakeRepo([chunk_row(i, i // 2, i / 100) for i in range(200)])
    r = VectorRetriever(repo, make_settings(candidate_chunks=20, ef_search=40))
    out = asyncio.run(r.retrieve(None, n_docs=30))
    k, ef = repo.calls[-1]
    assert k == 60                 # 2 * n_docs
    assert ef == 60                # ef_search raised to k, so an override is sent
    assert len(out) == 30


def test_retriever_uses_session_default_when_possible():
    repo = FakeRepo([chunk_row(i, i, i / 100) for i in range(50)])
    r = VectorRetriever(repo, make_settings(candidate_chunks=20, ef_search=40))
    asyncio.run(r.retrieve(None, n_docs=5))
    assert repo.calls[-1] == (20, None)
