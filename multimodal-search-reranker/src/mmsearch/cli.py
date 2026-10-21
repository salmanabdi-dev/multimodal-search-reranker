"""Command-line entry point: ``python -m mmsearch.cli <command>``.

    init-db            create extension + tables
    download-models    fetch both models into $HF_HOME, write manifest.json with commit hashes
    download-dataset   fetch a BEIR dataset (default: nfcorpus)
    ingest             load documents, chunk, embed, store, (re)build HNSW
    build-index        (re)build the HNSW index only
    eval               Precision@5: vector-only (A) vs vector + reranker (B)
    ann-recall         HNSW recall vs exact search, across ef_search values
    make-image-dataset generate the deterministic synthetic image benchmark
    ingest-images      validate, store and CLIP-embed a directory of images
    eval-images        cross-modal Precision@5: CLIP (A) vs CLIP + caption fusion (B)
    calibrate-images   fit CLIP score calibration on dev, check it on test
    bench              latency benchmark: text, image and unified search
    optimize           measured precision/latency sweep; picks the config for the target
    smoke              functional checks against the running API
    report             write reports/VERIFICATION.md from the saved reports
    reset              clear the main index and media directory (--yes)
    stats              what is in the index
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from mmsearch.config import Settings, get_settings

log = logging.getLogger("mmsearch.cli")
REPORTS = Path(__file__).resolve().parents[2] / "reports"


def _write_report(kind: str, payload: dict, tag: str | None = None) -> Path:
    REPORTS.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = REPORTS / f"{kind}{'_' + tag if tag else ''}_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nreport written to {path}")
    return path


@asynccontextmanager
async def _repo(s: Settings):
    from mmsearch.db.pool import create_pool, init_db
    from mmsearch.db.repository import Repository

    await init_db(s.database_url)
    pool = await create_pool(s.database_url, 1, 4, s.ef_search)
    try:
        yield Repository(pool)
    finally:
        await pool.close()


# ---------------------------------------------------------------- commands

async def cmd_init_db(args, s: Settings) -> None:  # noqa: ANN001
    from mmsearch.db.pool import init_db
    await init_db(s.database_url)
    print("schema ready")


def cmd_download_models(args, s: Settings) -> None:  # noqa: ANN001
    from huggingface_hub import snapshot_download

    ignore = ["*.onnx", "onnx/*", "openvino/*", "*.h5", "*.msgpack", "*.ot",
              "tf_model*", "flax_model*", "rust_model*", "*.tflite"]
    manifest = {}
    models = [(s.bi_encoder_model, s.bi_encoder_revision),
              (s.cross_encoder_model, s.cross_encoder_revision)]
    if s.enable_images:
        models.append((s.clip_model, s.clip_revision))
    for name, rev in models:
        path = snapshot_download(name, revision=rev, ignore_patterns=ignore)
        manifest[name] = {"requested_revision": rev, "commit": Path(path).name}
        print(f"{name}: {path}")
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    home.mkdir(parents=True, exist_ok=True)
    (home / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print("pin these with MMS_BI_ENCODER_REVISION / MMS_CROSS_ENCODER_REVISION / "
          "MMS_CLIP_REVISION")


def cmd_download_dataset(args, s: Settings) -> None:  # noqa: ANN001
    from mmsearch.evaluation.datasets import download_beir
    print(f"dataset at {download_beir(args.name)}")


async def cmd_ingest(args, s: Settings) -> None:  # noqa: ANN001
    from mmsearch.embeddings.bi_encoder import HFTokenizerAdapter
    from mmsearch.ingestion import loaders
    from mmsearch.ingestion.chunker import TokenChunker
    from mmsearch.ingestion.pipeline import IngestionPipeline
    from mmsearch.search.factory import configure_torch, load_bi_encoder

    configure_torch(s.torch_threads)
    docs = {"jsonl": loaders.load_jsonl, "dir": loaders.load_directory,
            "beir": loaders.load_beir_corpus}[args.source](args.path)
    if args.limit:
        import itertools
        docs = itertools.islice(docs, args.limit)
    encoder = load_bi_encoder(s)
    chunker = TokenChunker(HFTokenizerAdapter(encoder.tokenizer), s.chunk_tokens,
                           s.chunk_overlap, s.title_max_tokens)
    async with _repo(s) as repo:
        stats = await IngestionPipeline(repo, encoder, chunker, s).run(
            docs, rebuild_index=args.rebuild_index)
        print(json.dumps(stats.as_dict(), indent=2))
        print(json.dumps(await repo.stats(), indent=2))


async def cmd_build_index(args, s: Settings) -> None:  # noqa: ANN001
    from mmsearch.db.index import build_hnsw, drop_hnsw
    async with _repo(s) as repo, repo.pool.acquire() as conn:
        await drop_hnsw(conn)
        secs = await build_hnsw(conn, s.hnsw_m, s.hnsw_ef_construction,
                                s.index_maintenance_work_mem)
        print(f"index built in {secs:.1f}s")


def _dataset(args):  # noqa: ANN001, ANN202
    from mmsearch.evaluation.datasets import SAMPLE_DIR, load_dataset
    if args.dataset == "sample":
        return load_dataset(args.path or SAMPLE_DIR, "test")
    if not args.path:
        raise SystemExit("--path is required for --dataset beir")
    return load_dataset(args.path, "test")


async def cmd_eval(args, s: Settings) -> None:  # noqa: ANN001
    from mmsearch.evaluation.datasets import split_query_ids
    from mmsearch.evaluation.runner import collect_runs, evaluate, format_result, pick_best
    from mmsearch.search.factory import build_service

    ds = _dataset(args)
    qids = ds.judged_query_ids(args.min_level)
    if args.dataset == "sample":
        splits = {"all": qids}           # 10 queries: splitting would be meaningless
    else:
        dev, test = split_query_ids(qids, s.eval_dev_fraction, s.eval_seed)
        splits = {"dev": dev, "test": test}
    print(f"{ds.name}: {len(qids)} judged queries; " +
          ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    grid_n = sorted({int(x) for x in args.n_grid.split(",")} | {s.rerank_candidates})
    grid_ef = [int(x) for x in args.ef_grid.split(",")] if args.tune else [None]
    max_n = max(grid_n)

    report: dict = {"dataset": ds.name, "min_level": args.min_level,
                    "settings": s.model_dump(), "splits": {k: len(v) for k, v in splits.items()}}
    async with _repo(s) as repo:
        service = build_service(s, repo)
        try:
            await service.warmup(rounds=1)
            t0 = time.perf_counter()
            if args.tune:
                dev, test = splits["dev"], splits["test"]
                sweep = []
                for ef in grid_ef:
                    runs = await collect_runs(service, ds, dev, max_n, ef)
                    for n in grid_n:
                        sweep.append(evaluate(runs, ds, dev, n, min_level=args.min_level,
                                              ef_search=ef, bootstrap=False))
                print("\n== dev sweep ==")
                print(f"{'ef':>6}{'N':>5}{'P@5 A':>9}{'P@5 B':>9}{'gain':>9}{'pool R@N':>10}")
                for r in sweep:
                    print(f"{r.ef_search:>6}{r.n:>5}{r.a['p@5']:>9.4f}{r.b['p@5']:>9.4f}"
                          f"{r.rel_gain_p5:>+9.1%}{r.pool_recall:>10.4f}")
                best = pick_best(sweep)
                print(f"\nselected on dev: N={best.n}, ef_search={best.ef_search}")
                runs = await collect_runs(service, ds, test, max(best.n, 10), best.ef_search)
                final = evaluate(runs, ds, test, best.n, min_level=args.min_level,
                                 ef_search=best.ef_search)
                print(format_result("TEST (held out)", final))
                print(f"\nto use: MMS_RERANK_CANDIDATES={best.n} MMS_EF_SEARCH={best.ef_search}")
                report["dev_sweep"] = [r.summary() for r in sweep]
                report["selected"] = {"n": best.n, "ef_search": best.ef_search}
                report["test"] = final.summary()
            else:
                n = s.rerank_candidates
                for name, ids in splits.items():
                    runs = await collect_runs(service, ds, ids, max(n, 10))
                    res = evaluate(runs, ds, ids, n, min_level=args.min_level)
                    print(format_result(name.upper(), res))
                    report[name] = res.summary()
                    if args.dataset == "sample":
                        report["per_query_p5"] = res.per_query_p5
            report["seconds"] = round(time.perf_counter() - t0, 1)
        finally:
            service.close()
    _write_report("eval", report, args.tag)


async def cmd_ann_recall(args, s: Settings) -> None:  # noqa: ANN001
    import statistics

    from mmsearch.search.factory import configure_torch, load_bi_encoder

    configure_torch(s.torch_threads)
    ds = _dataset(args)
    qids = ds.judged_query_ids()[: args.max_queries]
    encoder = load_bi_encoder(s)
    vecs = encoder.encode([ds.queries[q] for q in qids], is_query=True)
    k = args.k
    rows = []
    async with _repo(s) as repo:
        truth = [{r.chunk_id for r in await repo.exact_search(v, k)} for v in vecs]
        for ef in [int(x) for x in args.ef_grid.split(",")]:
            recalls, times = [], []
            for v, t in zip(vecs, truth, strict=True):
                t0 = time.perf_counter_ns()
                got = await repo.ann_search(v, k, ef)
                times.append((time.perf_counter_ns() - t0) / 1e6)
                recalls.append(len({r.chunk_id for r in got} & t) / k)
            rows.append({"ef_search": ef, "recall": round(statistics.fmean(recalls), 4),
                         "mean_ms": round(statistics.fmean(times), 3)})
    print(f"\nANN recall@{k} vs exact search ({len(qids)} queries)")
    print(f"{'ef_search':>10}{'recall':>9}{'mean ms':>9}")
    for r in rows:
        print(f"{r['ef_search']:>10}{r['recall']:>9.4f}{r['mean_ms']:>9.3f}")
    print("note: ef_search < k caps the result count at ef_search, so recall drops sharply")
    _write_report("ann_recall", {"k": k, "queries": len(qids), "rows": rows})


async def cmd_bench(args, s: Settings) -> None:  # noqa: ANN001
    from mmsearch.benchmarks.latency import (
        DEFAULT_QUERIES,
        bench_http,
        bench_inprocess,
        configs_for,
        environment_info,
        load_queries,
    )

    queries = load_queries(Path(args.queries) if args.queries else DEFAULT_QUERIES)
    report: dict = {"mode": args.mode, "iterations": args.iterations, "warmup": args.warmup,
                    "queries": len(queries), "settings": s.model_dump(),
                    "environment": environment_info(), "target_p95_ms": s.latency_target_ms,
                    "results": {}}
    async with _repo(s) as repo:
        report["index"] = idx = await repo.stats()
    configs = configs_for(idx["chunks"] > 0, s.enable_images and idx["images"] > 0)
    if args.only:
        configs = {k: v for k, v in configs.items() if k in args.only.split(",")}
    if not configs:
        raise SystemExit("nothing to benchmark: ingest text and/or images first")

    if args.mode == "http":
        for label, (mods, rerank) in configs.items():
            report["results"][label] = bench_http(
                args.url, queries, warmup=args.warmup, iterations=args.iterations,
                modalities=mods, rerank=rerank)
    else:
        from mmsearch.search.factory import build_service
        async with _repo(s) as repo:
            service = build_service(s, repo, image_count=idx["images"])
            try:
                await service.warmup()
                for label, (mods, rerank) in configs.items():
                    report["results"][label] = await bench_inprocess(
                        service, queries, warmup=args.warmup, iterations=args.iterations,
                        modalities=mods, rerank=rerank)
            finally:
                service.close()

    def headline(r: dict) -> dict:
        return r.get("total") or r.get("server_total") or r["client_total"]

    for label, r in report["results"].items():
        print(f"\n== {label} ==")
        rows = [("total", headline(r))] + list(r.get("stages", {}).items())
        if "client_total" in r:
            rows.append(("client_total", r["client_total"]))
        for name, st in rows:
            print(f"{name:<14} mean {st['mean']:>7.2f}  p50 {st['p50']:>7.2f}  "
                  f"p95 {st['p95']:>7.2f}  p99 {st['p99']:>7.2f} ms")
    print()
    report["target_met"] = {}
    for label, r in report["results"].items():
        if label.endswith("vector_only"):
            continue
        ok = headline(r)["p95"] < s.latency_target_ms
        report["target_met"][label] = ok
        print(f"{label:<12} p95 {headline(r)['p95']:7.2f} ms vs target "
              f"{s.latency_target_ms:.0f} ms: {'MET' if ok else 'NOT MET'}")
    _write_report(f"bench_{args.mode}", report, args.tag)


def cmd_make_image_dataset(args, s: Settings) -> None:  # noqa: ANN001
    from mmsearch.media.synthetic import generate
    out = generate(args.out, seed=args.seed, instances=args.instances)
    print(f"dataset written to {out}")


async def cmd_ingest_images(args, s: Settings) -> None:  # noqa: ANN001
    from mmsearch.ingestion.images import ImageIngestionPipeline, load_image_directory
    from mmsearch.media.storage import MediaStore
    from mmsearch.search.factory import configure_torch, load_clip

    configure_torch(s.torch_threads)
    clip = load_clip(s)
    async with _repo(s) as repo:
        pipeline = ImageIngestionPipeline(repo, clip, MediaStore(s.media_dir), s)
        stats = await pipeline.ingest_many(load_image_directory(args.path, args.captions))
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v)
                          for k, v in stats.__dict__.items() if k != "errors"}, indent=2))
        for e in stats.errors[:20]:
            print("rejected:", e)
        print(json.dumps(await repo.stats(), indent=2))


async def _image_eval_setup(args, s: Settings, repo):  # noqa: ANN001, ANN202
    from mmsearch.evaluation.datasets import load_dataset, split_query_ids
    from mmsearch.search.factory import build_service

    ds = load_dataset(args.path, "test")
    dev, test = split_query_ids(ds.judged_query_ids(), s.eval_dev_fraction, s.eval_seed)
    image_count = await repo.image_count()
    if not image_count:
        raise SystemExit("no images indexed: run `make ingest-images` first")
    service = build_service(s, repo, image_count=image_count)
    return ds, dev, test, service


async def cmd_eval_images(args, s: Settings) -> None:  # noqa: ANN001
    from mmsearch.evaluation.image_runner import (
        collect_image_runs,
        evaluate_images,
        fit_clip_calibration,
    )
    from mmsearch.evaluation.runner import format_result
    from mmsearch.search.calibration import Platt

    report: dict = {"dataset": str(args.path), "settings": s.model_dump()}
    async with _repo(s) as repo:
        ds, dev, test, service = await _image_eval_setup(args, s, repo)
        try:
            print(f"{ds.name}: {len(dev) + len(test)} queries (dev={len(dev)}, "
                  f"test={len(test)}), {service.image_count} images indexed")
            calib = Platt(s.clip_calib_a, s.clip_calib_b)
            if args.calibrate:
                c = await fit_clip_calibration(service, ds, dev, test, calib)
                calib = c["fitted"]
                print(f"calibration fit on dev: a={calib.a:.3f} b={calib.b:.3f}; test log loss "
                      f"{c['test_log_loss_current']:.4f} -> {c['test_log_loss_fitted']:.4f}")
                report["calibration"] = {**c, "fitted": calib.__dict__}
            n = s.image_candidates
            dev_runs = await collect_image_runs(service, ds, dev, n)
            weights = ([float(w) for w in args.weights.split(",")] if args.tune
                       else [s.caption_weight])
            sweep = [evaluate_images(dev_runs, ds, dev, n=n, weight=w, calibration=calib,
                                     bootstrap=False) for w in weights]
            if args.tune:
                print(f"\n{'w':>6}{'P@5 A':>9}{'P@5 B':>9}{'gain':>9}")
                for r in sweep:
                    print(f"{r.extra['caption_weight']:>6}{r.a['p@5']:>9.4f}{r.b['p@5']:>9.4f}"
                          f"{r.rel_gain_p5:>+9.1%}")
            # Best dev P@5; on a tie prefer the smaller caption weight (less reliance on
            # captions, which real uploads often lack).
            best = max(sweep, key=lambda r: (round(r.b["p@5"], 3), -r.extra["caption_weight"]))
            w = best.extra["caption_weight"]
            test_runs = await collect_image_runs(service, ds, test, n)
            final = evaluate_images(test_runs, ds, test, n=n, weight=w, calibration=calib)
            print(format_result("TEST (held out)", final, "A CLIP", "B fusion", target=None))
            if args.tune or args.calibrate:
                env = [f"MMS_CAPTION_WEIGHT={w}"]
                if args.calibrate:
                    env += [f"MMS_CLIP_CALIB_A={calib.a:.3f}", f"MMS_CLIP_CALIB_B={calib.b:.3f}"]
                print("\nto use: " + " ".join(env))
                report["env"] = env
                if args.write_env:
                    Path(args.write_env).write_text("\n".join(env) + "\n")
                    print(f"settings written to {args.write_env}")
            report["dev_sweep"] = [r.summary() for r in sweep]
            report["test"] = final.summary()
        finally:
            service.close()
    _write_report("eval_images", report)


async def cmd_calibrate_images(args, s: Settings) -> None:  # noqa: ANN001
    from mmsearch.evaluation.image_runner import fit_clip_calibration
    from mmsearch.search.calibration import Platt

    async with _repo(s) as repo:
        ds, dev, test, service = await _image_eval_setup(args, s, repo)
        try:
            c = await fit_clip_calibration(service, ds, dev, test,
                                           Platt(s.clip_calib_a, s.clip_calib_b))
        finally:
            service.close()
    f = c["fitted"]
    print(f"pairs: dev={c['pairs_dev']} test={c['pairs_test']} (base rate {c['base_rate_test']:.3f})")
    print(f"test log loss: current {c['test_log_loss_current']:.4f} -> fitted "
          f"{c['test_log_loss_fitted']:.4f}")
    print(f"\nto use: MMS_CLIP_CALIB_A={f.a:.3f} MMS_CLIP_CALIB_B={f.b:.3f}")
    _write_report("calibrate_images", {**c, "fitted": f.__dict__})


async def cmd_optimize(args, s: Settings) -> None:  # noqa: ANN001
    from mmsearch.benchmarks.latency import DEFAULT_QUERIES, environment_info, load_queries
    from mmsearch.benchmarks.optimize import optimize
    from mmsearch.evaluation.datasets import load_dataset, split_query_ids
    from mmsearch.reranking.cross_encoder import CrossEncoderReranker
    from mmsearch.search.factory import build_service

    ds = load_dataset(args.path, "test")
    dev, test = split_query_ids(ds.judged_query_ids(), s.eval_dev_fraction, s.eval_seed)
    if args.max_dev_queries:
        dev = dev[: args.max_dev_queries]
    queries = load_queries(DEFAULT_QUERIES)
    quant_opts = [False, True] if s.device == "cpu" and not args.no_quantize else [False]
    async with _repo(s) as repo:
        idx = await repo.stats()
        if not idx["chunks"]:
            raise SystemExit("text index is empty: ingest BEIR first")
        service = build_service(s, repo, image_count=idx["images"])
        rerankers = {q: CrossEncoderReranker(s.cross_encoder_model, s.cross_encoder_revision,
                                             s.device, s.rerank_max_length, quantize=q)
                     for q in quant_opts}
        try:
            await service.warmup(rounds=2)
            t0 = time.perf_counter()
            res = await optimize(
                service, s, ds, dev, test, queries,
                n_grid=[int(x) for x in args.n_grid.split(",")],
                max_lengths=[int(x) for x in args.max_lengths.split(",")],
                quantize_options=quant_opts, iterations=args.iterations,
                warmup=args.warmup, rerankers=rerankers)
        finally:
            service.close()
    report = {**res.as_dict(), "target_p95_ms": s.latency_target_ms, "dev_queries": len(dev),
              "test_queries": len(test), "settings": s.model_dump(), "index": idx,
              "environment": environment_info(), "seconds": round(time.perf_counter() - t0, 1)}
    print(f"\nthreads: " + ", ".join(f"{r['threads']}->{r['p95']:.1f}ms" for r in res.threads)
          + f"  (selected {res.best_threads})")
    print(f"{'N':>4}{'len':>5}{'int8':>6}{'dev P@5':>9}{'text p95':>10}{'unified p95':>13}")
    for p in res.grid:
        u = f"{p.unified_p95:.2f}" if p.unified_p95 is not None else "-"
        print(f"{p.n:>4}{p.max_length:>5}{'yes' if p.quantize else 'no':>6}{p.dev_p5:>9.4f}"
              f"{p.text_p95:>10.2f}{u:>13}")
    sel = res.selection
    print(f"\n{sel.reason}")
    if sel.point is not None and res.test:
        print(f"held-out test P@5: A {res.test['a']['p@5']:.4f}  B {res.test['b']['p@5']:.4f}  "
              f"gain {res.test['rel_gain_p5']:+.1%}")
        print("to use: " + " ".join(res.env))
        if args.write_env:
            Path(args.write_env).write_text("\n".join(res.env) + "\n")
            print(f"settings written to {args.write_env}")
    _write_report("optimize", report)


def cmd_smoke(args, s: Settings) -> None:  # noqa: ANN001
    from mmsearch.smoke import run_smoke
    checks = run_smoke(args.url)
    width = max(len(c["name"]) for c in checks)
    for c in checks:
        print(f"{'PASS' if c['ok'] else 'FAIL'}  {c['name']:<{width}}  {c['detail']}")
    passed = sum(c["ok"] for c in checks)
    print(f"\n{passed}/{len(checks)} checks passed")
    _write_report("smoke", {"url": args.url, "checks": checks})
    if passed != len(checks):
        raise SystemExit(1)


def cmd_report(args, s: Settings) -> None:  # noqa: ANN001
    from mmsearch.reporting import build_markdown
    md = build_markdown(REPORTS)
    out = REPORTS / "VERIFICATION.md"
    out.write_text(md)
    print(md)
    print(f"written to {out}")


async def cmd_reset(args, s: Settings) -> None:  # noqa: ANN001
    import shutil
    if not args.yes:
        raise SystemExit("refusing to clear the index without --yes")
    async with _repo(s) as repo, repo.pool.acquire() as conn:
        await conn.execute("TRUNCATE documents, chunks, image_embeddings RESTART IDENTITY CASCADE")
    media = Path(s.media_dir)
    if media.exists():
        shutil.rmtree(media)
    print("cleared documents, chunks, images and the media directory")


async def cmd_stats(args, s: Settings) -> None:  # noqa: ANN001
    async with _repo(s) as repo:
        print(json.dumps(await repo.stats(), indent=2))


# ---------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mmsearch", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db")
    sub.add_parser("download-models")
    d = sub.add_parser("download-dataset")
    d.add_argument("--name", default="nfcorpus")

    i = sub.add_parser("ingest")
    i.add_argument("--source", choices=["jsonl", "dir", "beir"], required=True)
    i.add_argument("--path", required=True)
    i.add_argument("--rebuild-index", action="store_true",
                   help="drop HNSW before loading and rebuild after (use for bulk loads)")
    i.add_argument("--limit", type=int, default=0)

    sub.add_parser("build-index")

    for name in ("eval", "ann-recall"):
        e = sub.add_parser(name)
        e.add_argument("--dataset", choices=["sample", "beir"], default="sample")
        e.add_argument("--path")
        e.add_argument("--min-level", type=int, default=1,
                       help="minimum relevance grade counted as relevant")
        if name == "eval":
            e.add_argument("--tune", action="store_true",
                           help="sweep N and ef_search on dev, report the choice on test")
            e.add_argument("--n-grid", default="10,15,20,25,30,40,50")
            e.add_argument("--ef-grid", default="100,200")
            e.add_argument("--tag", help="label in the report filename")
        else:
            e.add_argument("--k", type=int, default=50)
            e.add_argument("--ef-grid", default="20,40,64,100,200,400")
            e.add_argument("--max-queries", type=int, default=200)

    g = sub.add_parser("make-image-dataset")
    g.add_argument("--out", default="data/images_synth")
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--instances", type=int, default=4, help="images per (colour, shape)")

    ii = sub.add_parser("ingest-images")
    ii.add_argument("--path", required=True, help="directory of images")
    ii.add_argument("--captions", help="JSONL {_id, caption}; default <path>/../captions.jsonl")

    for name in ("eval-images", "calibrate-images"):
        ei = sub.add_parser(name)
        ei.add_argument("--path", default="data/images_synth",
                        help="dataset dir with queries.jsonl + qrels/test.tsv")
        if name == "eval-images":
            ei.add_argument("--tune", action="store_true", help="sweep caption weight on dev")
            ei.add_argument("--weights", default="0,0.25,0.5,0.75,1.0")
            ei.add_argument("--calibrate", action="store_true",
                            help="fit CLIP calibration on dev before evaluating")
            ei.add_argument("--write-env", help="write the tuned MMS_* settings to this file")

    b = sub.add_parser("bench")
    b.add_argument("--mode", choices=["inprocess", "http"], default="inprocess")
    b.add_argument("--only", help="comma-separated subset: text,image,text+image,"
                                  "text+image_vector_only")
    b.add_argument("--url", default="http://localhost:8000")
    b.add_argument("--iterations", type=int, default=1000)
    b.add_argument("--warmup", type=int, default=50)
    b.add_argument("--queries")
    b.add_argument("--tag", help="label in the report filename, e.g. baseline / final")

    o = sub.add_parser("optimize", help="measure precision and latency, pick the best config "
                                         "under the latency target")
    o.add_argument("--path", default="data/beir/nfcorpus")
    o.add_argument("--n-grid", default="10,15,20,25")
    o.add_argument("--max-lengths", default="256,192,128")
    o.add_argument("--no-quantize", action="store_true")
    o.add_argument("--iterations", type=int, default=200)
    o.add_argument("--warmup", type=int, default=20)
    o.add_argument("--max-dev-queries", type=int, default=0)
    o.add_argument("--write-env", help="write the selected MMS_* settings to this file")

    sm = sub.add_parser("smoke", help="functional checks against a running API")
    sm.add_argument("--url", default="http://localhost:8000")

    sub.add_parser("report", help="write reports/VERIFICATION.md from the reports")

    r = sub.add_parser("reset", help="clear the main index and media (needs --yes)")
    r.add_argument("--yes", action="store_true")

    sub.add_parser("stats")
    return p


COMMANDS = {
    "init-db": cmd_init_db, "download-models": cmd_download_models,
    "download-dataset": cmd_download_dataset, "ingest": cmd_ingest,
    "build-index": cmd_build_index, "eval": cmd_eval, "ann-recall": cmd_ann_recall,
    "bench": cmd_bench, "stats": cmd_stats,
    "make-image-dataset": cmd_make_image_dataset, "ingest-images": cmd_ingest_images,
    "eval-images": cmd_eval_images, "calibrate-images": cmd_calibrate_images,
    "optimize": cmd_optimize, "smoke": cmd_smoke, "report": cmd_report, "reset": cmd_reset,
}


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    s = get_settings()
    logging.basicConfig(level=s.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    fn = COMMANDS[args.command]
    result = fn(args, s)
    if asyncio.iscoroutine(result):
        asyncio.run(result)


if __name__ == "__main__":
    main()
