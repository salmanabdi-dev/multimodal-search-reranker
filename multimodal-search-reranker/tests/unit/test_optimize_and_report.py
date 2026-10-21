import json

from mmsearch.benchmarks.optimize import GridPoint, env_lines, select_point, thread_grid
from mmsearch.reporting import build_markdown, latest, parse_junit


def gp(n, dev, text, unified=None, length=256, quant=False):
    return GridPoint(n=n, max_length=length, quantize=quant, dev_p5=dev, dev_p5_vector=0.3,
                     text_p95=text, text_p50=text * 0.8, unified_p95=unified,
                     unified_p50=None if unified is None else unified * 0.8)


def test_select_best_precision_under_target():
    pts = [gp(25, 0.40, 80), gp(20, 0.39, 60), gp(15, 0.37, 50), gp(10, 0.33, 40)]
    sel = select_point(pts, 65)
    assert sel.point.n == 20 and not sel.fallback and sel.candidates_meeting_target == 3


def test_select_uses_unified_latency_when_images_present():
    pts = [gp(20, 0.39, 55, unified=70), gp(15, 0.37, 45, unified=60)]
    assert select_point(pts, 65).point.n == 15


def test_select_breaks_near_ties_toward_faster():
    pts = [gp(20, 0.3900, 60), gp(15, 0.3885, 50)]
    assert select_point(pts, 65).point.n == 15


def test_select_falls_back_to_fastest_and_says_so():
    sel = select_point([gp(20, 0.39, 90), gp(10, 0.33, 70)], 65)
    assert sel.fallback and sel.point.n == 10 and "no configuration" in sel.reason


def test_thread_grid_and_env():
    assert thread_grid(1) == [1]
    assert thread_grid(6) == [1, 2, 4, 6]
    assert thread_grid(32)[-1] == 16
    assert env_lines(4, gp(15, 0.4, 50, length=192, quant=True)) == [
        "MMS_TORCH_THREADS=4", "MMS_RERANK_CANDIDATES=15",
        "MMS_RERANK_MAX_LENGTH=192", "MMS_RERANK_QUANTIZE=true"]


JUNIT = """<?xml version="1.0"?><testsuites><testsuite name="pytest" tests="5" failures="1"
errors="0" skipped="1" time="12.5"><testcase classname="tests.unit.test_a" name="test_ok"/>
<testcase classname="tests.unit.test_a" name="test_bad"><failure message="x"/></testcase>
<testcase classname="t" name="s"><skipped/></testcase></testsuite></testsuites>"""


def _stats(v):
    return {"n": 10, "mean": v, "std": 1, "min": v, "p50": v, "p95": v + 10, "p99": v + 20,
            "max": v + 30}


def _result(n, a, b):
    return {"n": n, "ef_search": None, "queries": 194, "a": {"p@5": a, "p@5_strict": a / 2,
            "ndcg@10": a, "mrr@10": a}, "b": {"p@5": b, "p@5_strict": b / 2, "ndcg@10": b,
            "mrr@10": b}, "pool_recall": 0.6, "rel_gain_p5": (b - a) / a,
            "bootstrap": {"rel_ci_low": 0.05, "rel_ci_high": 0.2, "p_value_b_not_better": 0.01},
            "extra": {}}


def test_parse_junit(tmp_path):
    p = tmp_path / "junit_unit.xml"
    p.write_text(JUNIT)
    j = parse_junit(p)
    assert (j.tests, j.passed, j.failures, j.skipped) == (5, 3, 1, 1)
    assert j.failed == ["tests.unit.test_a::test_bad"] and not j.ok


def test_latest_ignores_tagged_variants(tmp_path):
    for name in ("eval_20260101T000000Z", "eval_20260102T000000Z",
                 "eval_images_20260103T000000Z", "eval_final_20260104T000000Z"):
        (tmp_path / f"{name}.json").write_text(json.dumps({"name": name}))
    assert latest(tmp_path, "eval")[1]["name"] == "eval_20260102T000000Z"
    assert latest(tmp_path, "eval_final")[1]["name"] == "eval_final_20260104T000000Z"
    assert latest(tmp_path, "smoke") is None


def test_build_markdown_reads_every_number_from_reports(tmp_path):
    (tmp_path / "junit_unit.xml").write_text(JUNIT)
    w = lambda name, obj: (tmp_path / f"{name}_20260101T000000Z.json").write_text(json.dumps(obj))  # noqa: E731
    w("eval_final", {"test": _result(15, 0.3000, 0.3500)})
    img = _result(20, 0.7, 0.8)
    img["extra"] = {"caption_weight": 0.5}
    w("eval_images", {"test": img, "calibration": {"fitted": {"a": 41.2, "b": -10.5},
      "test_log_loss_current": 0.3, "test_log_loss_fitted": 0.2}})
    bench = {"target_p95_ms": 65, "environment": {"cpu": "TestCPU", "logical_cpus": 8,
             "torch": "2.5.1"}, "index": {"pgvector_version": "0.8.0"}, "results": {
             "text": {"total": _stats(30), "stages": {"rerank": _stats(20)},
                      "stage_mean_share": {"rerank": 0.66}},
             "text+image": {"total": _stats(60), "stages": {}, "stage_mean_share": {}}}}
    w("bench_inprocess_final", bench)
    w("smoke", {"checks": [{"name": "api ready", "ok": True, "detail": "fine", "ms": 1}]})
    md = build_markdown(tmp_path)
    assert "TestCPU" in md and "pgvector 0.8.0" in md
    assert "| unit | 3 | 1 | 0 | 1 | 12 s |" in md and "test_bad" in md
    assert "+16.7%" in md and "**not met**" in md               # 0.30 -> 0.35 is +16.7%
    assert "| text | 30.00 | 30.00 | 40.00 | 50.00 | yes |" in md
    assert "| text+image | 60.00 | 60.00 | 70.00 | 80.00 | **no** |" in md
    assert "a = 41.200" in md and "(1/1 passed)" in md
    assert "integration | not run" in md
