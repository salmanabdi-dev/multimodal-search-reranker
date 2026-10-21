from mmsearch.benchmarks.latency import DEFAULT_QUERIES, _server_total, load_queries, summarize


def test_summarize():
    s = summarize(list(range(1, 101)))
    assert s["n"] == 100
    assert s["p50"] == 50.5
    assert 95 <= s["p95"] <= 96
    assert s["min"] == 1 and s["max"] == 100


def test_server_timing_parse():
    assert _server_total("embed;dur=5.10, retrieve;dur=2.00, total;dur=40.25") == 40.25
    assert _server_total(None) is None


def test_benchmark_queries_file():
    qs = load_queries(DEFAULT_QUERIES)
    assert len(qs) >= 30
    assert not any(q.startswith("#") for q in qs)
