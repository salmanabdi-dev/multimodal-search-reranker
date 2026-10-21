from mmsearch.evaluation.datasets import SAMPLE_DIR, load_dataset, split_query_ids
from mmsearch.ingestion.loaders import load_jsonl


def test_sample_dataset_loads():
    ds = load_dataset(SAMPLE_DIR, "test")
    qids = ds.judged_query_ids()
    assert len(qids) == 8
    for q in qids:
        assert len(ds.relevant(q)) >= 4      # P@5 is meaningful only with several positives
    assert ds.relevant("q4", min_level=2) == {"s19", "s20"}


def test_sample_corpus_ids_match_qrels():
    ids = {d.external_id for d in load_jsonl(SAMPLE_DIR / "corpus.jsonl")}
    ds = load_dataset(SAMPLE_DIR, "test")
    assert all(d in ids for rels in ds.qrels.values() for d in rels)


def test_split_is_deterministic_and_disjoint():
    qids = [f"q{i}" for i in range(100)]
    d1, t1 = split_query_ids(qids, 0.4, 13)
    d2, t2 = split_query_ids(list(reversed(qids)), 0.4, 13)
    assert (d1, t1) == (d2, t2)
    assert len(d1) == 40 and len(t1) == 60
    assert not set(d1) & set(t1)
