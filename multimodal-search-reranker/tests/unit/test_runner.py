from mmsearch.evaluation.datasets import EvalDataset
from mmsearch.evaluation.runner import evaluate, pick_best, rank_reranked, rank_vector
from mmsearch.retrieval.vector_search import Candidate


def cand(doc: str, vrank: int, score: float) -> Candidate:
    return Candidate(document_id=vrank, external_id=doc, title=None, chunk_id=vrank, content="",
                     vector_score=1 - vrank / 100, vector_rank=vrank, rerank_score=score)


# vector order: x1 x2 a b x3 c d x4 ; reranker loves the relevant ones
CANDS = [cand("x1", 1, 0.1), cand("x2", 2, 0.2), cand("a", 3, 0.9), cand("b", 4, 0.8),
         cand("x3", 5, 0.0), cand("c", 6, 0.7), cand("d", 7, 0.6), cand("x4", 8, 0.05)]
DS = EvalDataset("t", {"q": "query"}, {"q": {"a": 2, "b": 1, "c": 1, "d": 1}})


def test_rank_functions():
    assert rank_vector(list(reversed(CANDS)))[:3] == ["x1", "x2", "a"]
    assert rank_reranked(CANDS, 4) == ["a", "b", "x2", "x1", "x3", "c", "d", "x4"]
    assert rank_reranked(CANDS, 8)[:4] == ["a", "b", "c", "d"]


def test_evaluate_a_vs_b():
    r = evaluate({"q": CANDS}, DS, ["q"], n=8, bootstrap=False)
    assert r.a["p@5"] == 0.4          # a, b in vector top 5
    assert r.b["p@5"] == 0.8          # a, b, c, d in reranked top 5
    assert r.rel_gain_p5 == 1.0
    assert r.pool_recall == 1.0
    small = evaluate({"q": CANDS}, DS, ["q"], n=4, bootstrap=False)
    assert small.b["p@5"] == 0.4      # c, d were outside the reranked head
    assert small.pool_recall == 0.5


def test_pick_best_prefers_smaller_n_on_ties():
    rs = [evaluate({"q": CANDS}, DS, ["q"], n=n, bootstrap=False) for n in (8, 7, 4)]
    assert pick_best(rs).n == 7
