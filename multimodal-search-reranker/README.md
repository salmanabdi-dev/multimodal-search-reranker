# Multimodal Search & Reranker

Two-stage semantic search over **documents and images** from one natural-language query.

```
                 ┌─ MiniLM bi-encoder → pgvector HNSW (chunks) → N documents ─┐
query ──────────┤                                                              ├─ cross-encoder (ONE batch:
                 └─ CLIP text tower  → pgvector HNSW (images) → M images ────┘   passages + image captions)
      → per-modality ranking → calibrate to relevance probabilities → merge → top k → FastAPI
```

Text is retrieved with a MiniLM **bi-encoder** and reranked with an ms-marco **cross-encoder**. Images are embedded with **CLIP** at ingestion. A query is embedded with CLIP's text tower and searched against a separate **HNSW** image index. Images are then reranked by fusing calibrated CLIP similarity with the cross-encoder's score on their caption, when they have one. Both lists are calibrated onto a common relevance scale and merged.

Tech: Python 3.11, PyTorch, Hugging Face Transformers (MiniLM, ms-marco cross-encoder, CLIP ViT-B/32), FastAPI, PostgreSQL 16, pgvector, Docker.

How every part works and why it exists is explained in [docs/COMPONENTS.md](docs/COMPONENTS.md).

## Quick start: demo (Docker, about 15 minutes)

Requirements: Docker with Compose v2.24 or newer, about 6 GB free disk, and internet access for the first build.

```bash
make demo   # build (bakes all three models into the image), start db + api,
            # ingest the 46-document text sample and the 96-image synthetic set
```

Open http://localhost:8000 and try *a red circle* (images first), *what are the symptoms of iron deficiency* (documents first), or *something blue*. The OpenAPI UI is at http://localhost:8000/docs.

```bash
# unified search (default: both modalities)
curl -s localhost:8000/search -H 'content-type: application/json' \
  -d '{"query": "a red circle", "top_k": 5}' | python -m json.tool
# one modality, stage 1 only
curl -s "localhost:8000/search?q=something+blue&modalities=image&rerank=false"
# upload an image (caption optional; it enables caption reranking)
curl -s -F file=@photo.jpg -F caption="a dog on a beach" localhost:8000/images
curl -s localhost:8000/images/<document_id> -o out.jpg
```

Each search result carries `modality`, a calibrated `relevance` (comparable across modalities), `vector_score` and `vector_rank` (stage 1, within its modality), and `rerank_score`. Image results add `image_url`, `width` and `height`. The `Server-Timing` header breaks the request into `embed_text`, `embed_clip`, `retrieve`, `rerank`, `merge` and `total`.

**Demo script for an interview** (after `make demo`, or after `make verify`, which leaves the full NFCorpus + image index in place):

1. Search *a red circle* with both result types checked. Images come first, and each shows its relevance, CLIP similarity and caption score. Point out the movement notes: reranking reorders within each type.
2. Search *what are the symptoms of iron deficiency*. Documents come first; the same merge puts them on top because calibrated relevance, not raw scores, decides.
3. Tick "Show vector-only results alongside" to compare stage 1 with stage 2 side by side.
4. Open "Add an image to the index" and upload any photo with a caption. It is searchable on the next query; uploading it again reports it as already indexed.
5. Open `reports/VERIFICATION.md` for the measured precision and latency, and http://localhost:8000/docs for the API contract.

Upgrading an existing installation: run `make up`. The API applies `sql/002_images.sql` at start-up, so existing data volumes are upgraded in place, and the rebuilt image contains CLIP.

## End-to-end verification (one command)

```bash
make verify          # about 45-90 minutes on a laptop CPU; asks once before clearing the index
make verify-quick    # smaller sweeps and fewer iterations, about 20-30 minutes; numbers are indicative only
```

`scripts/verify.sh` runs everything inside the containers, in this order:
1. build and start the stack;
2. unit and integration tests;
3. reset the main index, then ingest BEIR NFCorpus and the synthetic images;
4. NFCorpus Precision@5 (tuned on dev, reported on the held-out test split);
5. image retrieval evaluation, with CLIP calibration and caption-weight tuning;
6. HNSW recall against exact search;
7. baseline latency for text, image and unified search;
8. the measured optimisation sweep;
9. apply the selected settings and recreate the API;
10. final precision and latency, in-process and over HTTP;
11. functional checks against the running API;
12. add the demo documents;
13. write the report.

It produces:

| File | Contents |
|---|---|
| `reports/VERIFICATION.md` | the summary: tests, P@5 for text and images, HNSW recall, latency before and after optimisation (mean/p50/p95/p99 per configuration, per-stage breakdown), selected settings, environment |
| `reports/*.json` | raw results each table is generated from (`eval_tune_*`, `eval_final_*`, `eval_images_*`, `bench_inprocess_baseline_*`, `bench_inprocess_final_*`, `bench_http_final_*`, `optimize_*`, `smoke_*`, `ann_recall_*`) |
| `reports/junit_unit.xml`, `reports/junit_integration.xml` | test results |
| `reports/verify_steps.tsv`, `reports/verify_<stamp>.log` | status and duration of every step, and the full log |

Every number in `VERIFICATION.md` is read from those files by `mmsearch report`, which can regenerate it at any time. A failing non-critical step (a test suite, say) is recorded and the run continues. A failing build, readiness check or ingestion stops the run. Your previous `.env` is kept as `.env.before-verify`.

**How the optimisation works.** `mmsearch optimize` first measures p95 at each torch thread count. Thread count changes speed only, never results. It then measures dev-split P@5 *and* text and unified p95 for every combination of N ∈ {10, 15, 20, 25}, reranker token limit ∈ {256, 192, 128}, and int8 quantisation on or off. It selects the most precise configuration whose p95 is under 65 ms, with unified p95 as the constraint since that is the API default. Only that configuration is scored on the held-out test split. If nothing meets the target, it selects the fastest configuration and says so in the report.

**What to expect from the +28% target.** NFCorpus is a medical corpus with many partially relevant documents per query. The default cross-encoder (ms-marco MiniLM-L6) was trained on web search queries, and published BEIR results show cross-encoder gains on NFCorpus that are real but modest. The report states the measured gain, its confidence interval, and whether the target was met; treat that number as the answer.

If the gain falls short, the legitimate levers are a stronger reranker and a deeper candidate pool. For example, set `MMS_CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2` in `.env` and re-run `make verify`; the build bakes whichever models `.env` names. Each lever costs latency, which the sweep measures. Weakening stage 1 to inflate the relative gain is not a legitimate lever.

## Reproducing the evaluation (Precision@5, A vs B)

The real evaluation uses BEIR NFCorpus: 3.6k documents, 323 test queries, graded relevance. Its many relevant documents per query are what make Precision@5 a meaningful metric.

```bash
make download-nfcorpus   # ~ 10 MB into data/beir/nfcorpus
make ingest-nfcorpus     # chunk, embed, store, build HNSW (a few minutes on CPU)
make tune                # sweep N and ef_search on the dev split, report the choice on held-out test
make eval                # evaluate the currently configured N on dev and test
make ann-recall          # HNSW recall vs exact search across ef_search values
```

`make tune` prints the dev sweep, the selected configuration, and the held-out test result. The test result includes P@5 for A and B, the relative gain against the +28% target, a paired bootstrap 95% confidence interval, and candidate-pool recall. To serve with the chosen values, put the printed `MMS_RERANK_CANDIDATES` and `MMS_EF_SEARCH` into `.env`, then run `docker compose up -d api`.

`make eval-sample` runs the same comparison on the 8-query sample set. That set is good for intuition and useless as a headline number.

## Reproducing the image evaluation (cross-modal Precision@5)

```bash
make ingest-images      # generate data/images_synth deterministically (seed 7), then ingest it
make eval-images        # A = CLIP alone vs B = CLIP + caption fusion, held-out test split
make tune-images        # fit CLIP calibration and the caption weight on dev, report on test
make calibrate-images   # calibration only: Platt fit on dev, log loss checked on test
```

The synthetic set has 96 images (6 colours × 4 shapes × 4 instances, varied in size, position, rotation and background) and 34 queries: 24 colour-and-shape queries, 6 colour queries and 4 shape queries. Relevance labels are exact. Captions are deliberately partial: a third mention only the colour, a third only the shape, a third are missing. Fusion therefore gets some evidence and never the answer.

The set regenerates byte-for-byte from its seed (a unit test checks this). It is a controlled test of the cross-modal plumbing and ranking logic, not a measure of real-photo quality. To evaluate on real photos, put them in the same layout (`images/`, `captions.jsonl`, `queries.jsonl`, `qrels/test.tsv`) and pass `--path`.

## Reproducing the latency benchmark

```bash
make bench        # in-process: 50 warm-up + 1000 measured searches per configuration
make bench-http   # same, through HTTP (client time + server time from Server-Timing)
```

Four configurations run on the same fixed query file: `text`, `image`, `text+image` (the unified API default) and `text+image_vector_only`. Configurations whose data isn't ingested are skipped. `--only text+image` runs one. Reports print mean, p50, p95 and p99 per stage and in total, and compare each reranked configuration's p95 with the 65 ms target.

The unified path adds one CLIP text forward pass (about 5 to 10 ms on CPU) and an image ANN query, which overlaps with the text one. Reranking adds a few short caption pairs to the existing cross-encoder batch rather than a second model call. On a slow CPU the unified p95 is the configuration most likely to exceed 65 ms. `MMS_IMAGE_CANDIDATES` and `MMS_RERANK_CANDIDATES` are the first levers; the text-only path costs exactly what it did before images existed. Each run also writes `reports/bench_*.json` with CPU model, core count, library versions, model names and every setting, so two runs can be compared honestly.

If p95 is over target on your machine, try these levers in this order. Re-run `make eval` after each one so you know what it cost in precision.

1. `MMS_TORCH_THREADS` equal to your physical core count (it is set inside the container; check `docker compose exec api nproc`).
2. `MMS_RERANK_CANDIDATES=20`, or lower. Latency is roughly linear in N.
3. `MMS_RERANK_MAX_LENGTH=192`.
4. `MMS_RERANK_QUANTIZE=true` for int8 dynamic quantisation of the cross-encoder's linear layers.
5. A GPU: install a CUDA torch wheel in the Dockerfile and set `MMS_DEVICE=cuda`.

On Apple Silicon, make sure Docker builds a native arm64 image. Running an amd64 image under emulation is several times slower.

## Results

Measured results are not written into this README, because they depend on the hardware they were measured on. `make verify` writes them to `reports/VERIFICATION.md`, every number traceable to a raw report file in `reports/`. Quote the numbers from there, together with the environment section that names the CPU.

## Tests

```bash
make test               # unit tests (no DB or models needed)
make test-integration   # real Postgres + all three models against a separate mms_test database
make lint
```

Unit tests cover chunking, metrics, chunk collapse, and the two-stage service. They also cover the verification tooling: optimiser selection, report generation from JUnit and JSON reports, and the HTTP smoke checks against a mock API. For multimodal they also cover image validation (corrupt, truncated, oversized, decompression-bomb and unsupported files), content-addressed storage and path traversal, and calibration fitting. On the search side: caption fusion, cross-modal merging, the unified service, and in-batch and cross-upload deduplication. Finally the deterministic dataset, and every API route including uploads.

The unified-service tests use a fake CLIP that embeds images by their colour and queries by colour words, so the real retrieval, fusion and merge code is exercised end to end with no model.

The integration suite (`tests/integration/test_multimodal.py`) ingests the text sample plus 48 synthetic images. Most go through the bulk pipeline; the red ones are uploaded over HTTP. It then checks:
- duplicate and invalid uploads, and ID clashes with text documents;
- that HNSW agrees with exact search;
- text-to-image retrieval for every colour, including the uploaded ones;
- combined search ranks images first for a visual query and documents first for a textual one, with results sorted by calibrated relevance;
- text-only search is unchanged;
- a P@5 floor on cross-modal quality.

## Running without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
docker compose up -d db                 # or any Postgres 16 with pgvector installed
python -m mmsearch.cli download-models
python -m mmsearch.cli init-db
python -m mmsearch.cli ingest --source jsonl --path data/sample/corpus.jsonl --rebuild-index
python -m mmsearch.cli make-image-dataset --out data/images_synth
python -m mmsearch.cli ingest-images --path data/images_synth/images
uvicorn mmsearch.api.main:app --port 8000
```

## Configuration

Every setting lives in `src/mmsearch/config.py` and can be overridden with an `MMS_`-prefixed environment variable (see `.env.example`). The ones that matter most:

| Setting | Default | Effect |
|---|---|---|
| `MMS_CANDIDATE_CHUNKS` | 50 | K, chunks fetched from HNSW (automatically at least 2×N) |
| `MMS_EF_SEARCH` | 100 | HNSW search beam; automatically raised to at least K |
| `MMS_RERANK_CANDIDATES` | 25 | N, documents scored by the cross-encoder |
| `MMS_RERANK_MAX_LENGTH` | 256 | tokens per (query, passage) pair |
| `MMS_TORCH_THREADS` | 4 | intra-op threads for both models |
| `MMS_BI_ENCODER_MODEL` | all-MiniLM-L6-v2 | changing the dimension also requires changing `vector(384)` in `sql/001_init.sql` |
| `MMS_CROSS_ENCODER_MODEL` | ms-marco-MiniLM-L-6-v2 | any model change in `.env` is baked into the image by `docker compose up -d --build` |
| `MMS_ENABLE_IMAGES` | true | false = the text-only system, CLIP not loaded |
| `MMS_CLIP_MODEL` | openai/clip-vit-base-patch32 | 512-d; must match `vector(512)` in `sql/002_images.sql` |
| `MMS_IMAGE_CANDIDATES` | 20 | M, images fetched from the image HNSW index and fused |
| `MMS_CAPTION_WEIGHT` | 0.5 | image relevance = w·caption + (1−w)·CLIP; tune with `make tune-images` |
| `MMS_CLIP_CALIB_A`, `_B` | 40, −10 | CLIP cosine → probability; fit with `make calibrate-images` |
| `MMS_MAX_IMAGE_BYTES`, `MMS_MAX_IMAGE_PIXELS` | 10 MB, 40 MP | upload limits |

Model revisions default to `main`. `make build` writes the resolved commit hashes to `/models/manifest.json` inside the image. Copy them into `MMS_BI_ENCODER_REVISION`, `MMS_CROSS_ENCODER_REVISION` and `MMS_CLIP_REVISION` to pin the models exactly.

## Repository layout

```
sql/                    migrations: 001 text schema, 002 images (applied at start-up, idempotent)
src/mmsearch/
  config.py             all settings
  db/                   pool + pgvector codec, repository (all SQL), HNSW management
  embeddings/           MiniLM bi-encoder, mean pooling, CLIP encoder
  media/                image validation + content-addressed storage, synthetic dataset
  ingestion/            text: loaders, cleaning, chunker, pipeline; images: pipeline
  retrieval/            stage 1: text HNSW + chunk collapse; image HNSW
  reranking/            stage 2: cross-encoder; caption fusion for images
  search/               unified service, calibration, cross-modal merge
  api/                  FastAPI app, routes, schemas, static search page
  evaluation/           datasets, metrics, text and image A-vs-B runners
  benchmarks/           latency benchmark (text, image, unified)
  cli.py                every command (python -m mmsearch.cli --help)
data/sample/            hand-labelled sample set (BEIR layout)
data/benchmark/         fixed latency query set
tests/unit, tests/integration
```

## Design decisions and trade-offs

These are the questions worth being ready for. The full reasoning for each component is in [docs/COMPONENTS.md](docs/COMPONENTS.md).

**Why a second vector table instead of putting images in `chunks`?** CLIP vectors are 512-d and MiniLM's are 384-d; they live in unrelated spaces and cannot share a column or an index. Images are still rows in `documents` (`modality = 'image'`), so IDs, metadata and deletion work the same for both.

**Why is the image HNSW index created up front, when the text one is rebuilt after bulk loads?** Text arrives in bulk, so one build after loading is fastest. Images arrive one upload at a time and must be searchable immediately, so the graph is maintained incrementally.

**Why not a visual cross-encoder for images?** Models like BLIP's image-text matching head are the true analogue of the text cross-encoder. But they run the vision tower per (query, image) pair at query time, which costs hundreds of milliseconds on CPU and breaks the latency target. Caption fusion reuses the text cross-encoder, inside the batch it already runs, so it adds only a few milliseconds. The cost: uncaptioned images get no reranking beyond CLIP. The natural next step would generate captions offline at ingestion with a captioning model, moving the expensive vision-language work out of the query path.

**How can text and image results share one ranking?** Not by raw score: a cross-encoder logit and a CLIP cosine are on unrelated scales. Each modality's final score is calibrated into a relevance probability. The cross-encoder's logits are already approximately log-odds; CLIP cosines get a Platt fit on dev data. The lists are then merged by probability. Merging by rank alone (e.g. reciprocal rank fusion) would interleave modalities even when one of them has nothing relevant.

**How is the extra latency contained?** Three ways. The text ANN query is in flight while CLIP encodes the query. Captions join the existing cross-encoder batch. And the image branch is skipped entirely when no images are indexed.

**What protects the upload endpoint?** Several checks run in order:
- a size cap enforced while reading;
- the format sniffed from the bytes, never from the filename or Content-Type;
- the pixel count checked from the header before decoding, which stops decompression bombs;
- one guarded full decode;
- content-addressed paths, never derived from user input;
- atomic writes;
- external IDs restricted to a safe character set and never allowed to overwrite a text document.

**What would change for production scale?** Images would go to object storage rather than a local volume. Ingestion would move to a queue with batch CLIP workers. Captions would be generated at ingestion. Calibration would be fit on real click or label data. And there'd be a GPU, or ONNX/int8 builds of all three models.
