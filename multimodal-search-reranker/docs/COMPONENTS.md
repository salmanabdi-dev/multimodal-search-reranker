# How the system works, component by component

Each section answers four questions: what it is, why it is needed, how it works, and how it connects to the rest. Sections 1 to 11 describe the text system. Sections 12 to 17 describe the multimodal layer added on top of it, which leaves the text path's behaviour and cost unchanged.

---

## 1. Database and schema (`sql/001_init.sql`, `db/`)

**What.** PostgreSQL with the pgvector extension, two tables (`documents` and `chunks`), a connection pool, and a repository module that holds every SQL statement in the project.

**Why.** We need durable storage for text and vectors, plus a fast way to ask "which stored vectors are closest to this one?" pgvector adds a `vector` column type, distance operators, and approximate-nearest-neighbour indexes to an ordinary database. Metadata, text and vectors therefore live in one transactional store instead of being split across a database and a separate vector service.

**How.** A document is the unit a user wants back. A chunk is the unit we embed, because models have a token limit and a long document must be split. Each chunk row stores its text, its token count, a `vector(384)` embedding, and the name of the model that produced it. The `embedding_model` column is a safety check. The API refuses to start if the configured model differs from the one that built the index, because vectors from different models live in unrelated spaces. Comparing them would return confident nonsense.

`db/pool.py` registers pgvector's asyncpg codec on every connection, so numpy arrays are sent as binary vectors. It also sets the session's `hnsw.ef_search`. `db/repository.py` contains the ANN query. That query wraps the vector scan in a CTE, so Postgres runs `ORDER BY embedding <=> $1 LIMIT K` against the HNSW index before joining to `documents`.

**Connects to.** Ingestion writes through `upsert_document_with_chunks`. Stage-1 retrieval reads through `ann_search`. The ANN recall tool compares `ann_search` against `exact_search`.

## 2. HNSW index (`db/index.py`)

**What.** A Hierarchical Navigable Small World graph over all chunk vectors, built with `vector_cosine_ops`.

**Why.** Exact nearest-neighbour search compares the query against every vector, so its cost grows linearly with the corpus. HNSW makes search roughly logarithmic, at the cost of occasionally missing a true neighbour, which is why it is called approximate.

**How.** Every vector becomes a node linked to about `m = 16` near neighbours, and the graph has a few sparse upper layers for long jumps. A search starts at the top layer, greedily walks toward the query, and drops a layer each time it can't get closer. On the bottom layer it keeps a candidate list of size `ef_search` and returns the best K. A larger `ef_search` explores more of the graph, which gives better recall and slower queries.

`ef_construction = 128` is the same idea applied while building the graph. pgvector cannot return more rows than `ef_search`, so the code forces `ef_search ≥ K`. The index is dropped before a bulk load and built afterwards, because building one graph from complete data is much faster than updating it thousands of times. `make ann-recall` measures the trade-off directly against exact search.

**Connects to.** Ingestion builds the index. Every stage-1 query uses it.

## 3. Bi-encoder (`embeddings/`)

**What.** `sentence-transformers/all-MiniLM-L6-v2`, a 6-layer, 22M-parameter BERT-style model, loaded with plain `transformers`. It maps any text to a 384-dimensional unit vector.

**Why.** We need a numeric representation of meaning so that "how do I recover from jet lag" lands near a passage about circadian rhythms even when they share no words. It must also be cheap enough to run once per query.

**How.** The tokenizer turns text into token ids. The transformer produces one vector per token. `mean_pool` averages those vectors, skipping padding positions; that is how this model was trained. `l2_normalize` scales the result to length 1, which makes cosine similarity equal to a dot product.

The model encodes queries and documents independently. Independence is what lets us embed documents once at ingestion, index them, and pay only one small forward pass per query. Within a batch, texts are sorted by length so padding is minimal. An integration test checks that our vectors match `sentence-transformers` output to within 1e-4.

**Connects to.** Ingestion uses it for documents. The search service uses it for queries. The `Encoder` protocol in `base.py` is the seam where an image encoder would plug in.

## 4. Ingestion (`ingestion/`)

**What.** Loaders (JSONL, a directory of `.txt`/`.md` files, BEIR), text cleaning, a token-window chunker, and a pipeline that embeds and stores everything in batches.

**Why.** Raw documents arrive in various formats and lengths. The model needs clean text in pieces no longer than 256 tokens, and the database needs everything stored consistently and idempotently.

**How.** `clean_text` normalises Unicode, removes control characters, and collapses whitespace while keeping paragraph breaks. The chunker counts tokens with the bi-encoder's own tokenizer. Counting characters wouldn't work: the model truncates by tokens, so a character-based chunk could lose its tail silently.

Windows are 256 tokens including special tokens and the title, with a 32-token overlap so a sentence on a boundary appears whole somewhere. Every chunk is embedded as `"<title>. <chunk>"` via `passage_text`, because a mid-document chunk often doesn't say what the document is about. The reranker scores exactly the same string.

The pipeline processes 128 documents per batch. It embeds all chunks of the batch together, then writes the batch in one transaction. Re-ingesting a document replaces its chunks, so running ingestion twice is safe.

**Connects to.** It consumes the bi-encoder and writes through the repository. Afterwards it builds the HNSW index.

## 5. Stage 1: vector retrieval (`retrieval/vector_search.py`)

**What.** A function that turns a query vector into an ordered list of candidate documents.

**Why.** The cross-encoder is too expensive to run over the whole corpus. Stage 1's job is recall: put the right documents somewhere in a short list, quickly.

**How.** It asks HNSW for K chunks, where K is at least 2×N, then collapses chunks into documents. Rows arrive sorted by distance, so the first chunk seen for a document is its best one, and later chunks of the same document are dropped. The result is up to N `Candidate`s, each carrying its vector similarity and its stage-1 rank. The rank is kept so the API, the UI and the evaluation can show how far reranking moved each result.

**Connects to.** It sits between the bi-encoder and the cross-encoder. Its output is both system A's answer (the top 5) and the reranker's input.

## 6. Stage 2: cross-encoder reranking (`reranking/cross_encoder.py`)

**What.** `cross-encoder/ms-marco-MiniLM-L-6-v2`, a BERT-style classifier trained on MS MARCO to output one relevance logit for a `[CLS] query [SEP] passage [SEP]` input.

**Why.** A bi-encoder compresses a whole passage into one vector before it ever sees the query, so it blurs details. "Iron skillet seasoning" and "iron deficiency symptoms" both sit near "iron". A cross-encoder reads the query and the passage together, so every query token can attend to every passage token. It can tell which meaning of "iron" the passage is about. That precision is where the Precision@5 gain comes from.

**How.** All N pairs are tokenised into one padded batch with `longest_first` truncation at 256 tokens, then pushed through the model in a single forward pass. A single batched pass is much faster on CPU than N separate calls. Optional int8 dynamic quantisation shrinks the linear layers for faster CPU inference. The scores are pointwise, meaning each pair is scored independently, so reranking the first N of a larger pool gives exactly the same scores. The evaluation relies on this property to sweep N cheaply.

**Connects to.** It takes stage 1's candidates. Its sorted output, truncated to 5, is the API response and system B in the evaluation.

## 7. Search service and API (`search/`, `api/`)

**What.** `SearchService.search_unified` runs the whole query path, for whichever modalities the request asks for, with per-stage timing. (Section 17 covers the image half.) FastAPI exposes it as `POST /search` and `GET /search`, alongside `POST /images`, `GET /images/{id}`, `/health`, `/ready`, `/stats`, and a small search page at `/`.

**Why.** One orchestration point means the API, the evaluation and the benchmark all execute identical code, so the numbers you measure are the numbers you serve.

**How.** Model forward passes run on a dedicated single-thread executor. That keeps the async event loop free while avoiding two forward passes fighting over the same CPU cores. PyTorch releases the GIL inside its kernels.

At start-up, the FastAPI lifespan opens the DB pool, checks that the index matches the configured model, loads both models, sets torch thread counts, and runs warm-up queries through the full pipeline. The first-request costs (memory allocation, kernel selection) are paid before traffic arrives, and `/ready` only reports ready after that. Each response includes a `Server-Timing` header so latency can be broken down per stage, even from a browser.

**Connects to.** This is the top of the architecture. Everything above feeds it.

## 8. Evaluation (`evaluation/`)

**What.** Dataset loading in BEIR layout, ranking metrics, and a runner comparing A (vector only) with B (vector plus reranker).

**Why.** A claimed improvement is only meaningful if it is measured fairly. That means the same candidates for both systems, tuning separated from reporting, and some estimate of noise.

**How.** For each query the runner retrieves the largest candidate pool once and scores it with the cross-encoder once. A is the top 5 by vector score. B(N) reranks the first N candidates and takes the top 5. The only difference between A and B is reranking.

Precision@5 is the number of relevant documents in the top 5, divided by 5, averaged over queries. The test is run at document level, after the chunk collapse. Queries are split deterministically into 40% dev and 60% test. `--tune` sweeps N and `ef_search` on dev only and reports the selected configuration on test, so the headline number has not been fitted to the queries it is measured on. Reports also include nDCG@10, MRR@10, strict P@5 (grade 2 only), Recall@N of the candidate pool (the ceiling on what reranking can achieve), and a paired bootstrap confidence interval on the gain.

**Connects to.** It calls `SearchService.retrieve` and `SearchService.rerank`, the same code the API uses.

## 9. Latency benchmark (`benchmarks/latency.py`)

**What.** A repeatable measurement of mean, p50, p95 and p99 latency, overall and per stage, for reranked and vector-only search.

**Why.** "Under 65 ms" has to be defined and measured the same way every time, or it means nothing.

**How.** A fixed query file is visited in a fixed order. The service warms up, then the benchmark sends 50 more warm-up requests and 1,000 measured ones at concurrency 1, with no caching. In-process mode times `SearchService.search` directly. HTTP mode adds the network and JSON layer, and records both client time and the server's own total from `Server-Timing`. Every report records hardware, versions, models and settings. The target is evaluated on p95, not the mean, because the tail is what users notice.

**Connects to.** It measures the same service the API serves.

## 10. Docker (`Dockerfile`, `docker-compose.yml`)

**What.** Two services. `db` is the official `pgvector/pgvector:pg16` image, which applies `sql/` on first start. `api` is Python 3.11 with CPU PyTorch, the package installed, and both models downloaded into the image at build time.

**Why.** It gives one command to a working system, with the same versions on every machine. Baking the models into the image makes start-up offline and deterministic; the API container runs with `HF_HUB_OFFLINE=1`.

**Connects to.** The Makefile runs every CLI command inside the `api` container, so evaluation and benchmarks use exactly the deployed environment.

## 11. Tests (`tests/`)

Unit tests run without a database or models. They cover chunk budgets and overlap, the text-cleaning rules, every metric, chunk-to-document collapse, the `ef_search ≥ K` rule, the service's reranking and timing behaviour (using a keyword-matching fake reranker), the A-vs-B runner, the dataset split, and the HTTP API (using a stub service).

Integration tests run against a separate `mms_test` database with the real models. They cover ingestion, idempotent re-ingestion, HNSW matching exact search on a small corpus, end-to-end relevance, reranking not hurting precision on the sample set, and parity with `sentence-transformers`.


---

# The multimodal layer

## 12. CLIP encoder (`embeddings/clip_encoder.py`)

**What.** `openai/clip-vit-base-patch32`: a vision transformer and a text transformer, each ending in a projection into one shared 512-d space.

**Why.** MiniLM only understands text. To let a typed query find an image, the query and the image must land in the same vector space, and nothing in the text system can do that.

**How.** CLIP was trained on about 400 million (image, caption) pairs with a contrastive objective. In each batch, the true pair's vectors are pulled together and every other pairing is pushed apart. After training, "a red circle" typed by a user and a picture of a red circle have a high cosine similarity.

Each tower runs independently, so CLIP is a cross-modal *bi-encoder*: image vectors are computed once at ingestion and indexed, and a query costs one short text forward pass. The text tower truncates at 77 tokens, which is fine for queries. Outputs are L2-normalised, as with MiniLM, so pgvector's cosine distance applies unchanged.

**Connects to.** Image ingestion uses the image tower; unified search uses the text tower. Both go through the same inference thread as the other models.

## 13. Image validation and storage (`media/storage.py`)

**What.** `decode_image` turns untrusted bytes into a validated RGB image. `MediaStore` keeps the file on disk.

**Why.** An upload endpoint is an attack surface. The usual failures are files that aren't images, huge files, and tiny files that decode to gigabytes of pixels (decompression bombs). Path traversal is a risk too, if any user input reaches a file path.

**How.** Checks happen cheapest-first:
1. The byte size is capped, and the API stops reading one byte past the limit.
2. Pillow opens the header only, which gives the real format (sniffed from the bytes, not trusted from the filename or Content-Type) and the dimensions.
3. The pixel count is checked against the limit before any decoding happens.
4. One full decode runs in a guarded block and produces RGB. This also handles PNG transparency, palettes, and the first frame of a GIF.

Files are stored content-addressed at `<sha[:2]>/<sha>.<ext>`. The path comes entirely from the hash, never from the user. Identical bytes are stored once, and every write goes to a temp file first and is then renamed, so a crash cannot leave a partial image.

**Connects to.** Image ingestion calls it. `GET /images/{id}` resolves stored paths through `MediaStore.resolve`, which refuses anything outside the media directory. The files never change, so they are served with immutable cache headers.

## 14. Image ingestion (`ingestion/images.py`, `sql/002_images.sql`)

**What.** A pipeline that runs validate, store, CLIP-embed, and DB write, either for one upload or for a directory in batches.

**Why.** Images need the same guarantees text has: idempotent re-ingestion, batched model calls, and model provenance recorded per vector.

**How.** An image is a row in `documents` (`modality = 'image'`) plus a row in `image_embeddings`. That second row holds the SHA-256, the file path, MIME type, dimensions, optional caption, the 512-d vector, and the model name. A separate table is needed because the dimension differs from the text vectors.

Before embedding, each image's hash is looked up. Known bytes return the existing document, updating its caption if a new one was given, with no CLIP call. Duplicates within one batch are embedded once. Two concurrent uploads of the same new image race on the unique hash; the loser retries and finds the winner's row.

An image may not take over a text document's `external_id`; the API returns 409 for that. The image HNSW index is created with the table and maintained incrementally, so an upload is searchable as soon as the request returns. In the API, CLIP runs on the service's inference thread, so an upload never blocks the event loop and never runs a forward pass concurrently with a search.

**Connects to.** `POST /images` and the `ingest-images` CLI both use it. The upload route also bumps the service's cached image count, which lets search skip the image branch entirely while the index is empty.

## 15. Image retrieval and reranking (`retrieval/image_search.py`, `reranking/image_reranker.py`)

**What.** Stage 1 takes the CLIP query vector, searches the image HNSW index, and returns M candidates. Stage 2 is caption fusion.

**Why.** CLIP alone is a bi-encoder with the same blind spots as MiniLM: it compresses the whole image into one vector before seeing the query. A second stage should add evidence, as the cross-encoder does for text.

**How.** One image is one vector, so stage 1 is a plain HNSW query with no collapse step. A true image cross-encoder, such as BLIP's image-text matching head, would run its vision tower per (query, image) pair at query time. That costs hundreds of milliseconds per query on CPU.

Caption fusion instead reuses the text cross-encoder on the image's caption, inside the same batched forward pass as the text passages. Relevance is `w · sigmoid(caption logit) + (1 − w) · calibrated CLIP`, or calibrated CLIP alone when there is no caption. So a caption adds evidence without overriding the image: an uncaptioned image with a strong CLIP match stays above a well-captioned weak one, and a unit test pins that behaviour.

**Connects to.** It feeds the merge step (section 16). The image evaluation compares CLIP order (A) against fused order (B) on the same candidates.

## 16. Calibration and cross-modal merging (`search/calibration.py`, `search/unified.py`)

**What.** Platt scaling, which maps each modality's score to a relevance probability, and a merge over those probabilities.

**Why.** A cross-encoder logit of 3 and a CLIP cosine of 0.29 cannot be sorted against each other; they are on unrelated scales. Merging raw scores lets whichever modality emits bigger numbers win. Merging by rank alone (reciprocal rank fusion, say) interleaves modalities even when one of them has nothing relevant.

**How.** The ms-marco cross-encoder was trained with binary cross-entropy, so `sigmoid(logit)` is already roughly a probability. CLIP cosines are not, so they get `p = sigmoid(a · cos + b)`, with `a` and `b` fitted by a two-parameter logistic regression (Newton's method). The fit uses every (query, image) pair from an exact scan, on dev queries only, and is checked on test by log loss (`make calibrate-images`). Bi-encoder cosines get the same treatment for when reranking is off.

Each modality is first ranked by its own best signal. The lists are then merged by probability with a stable sort, so the order within a modality never changes.

**Connects to.** It is the last step of the unified search path, and the reason a visual query puts images first while a factual question puts documents first.

## 17. The unified query path (`search/service.py`)

```
embed_text (MiniLM)     ──► text HNSW query ┐ (in flight while CLIP runs)
embed_clip (CLIP text)  ──► image HNSW query ┤
                                             ▼
rerank: ONE cross-encoder batch = N passages + captions of the M images that have one
        → text sorted by logit; images by fusion
merge:  calibrated probabilities → top k
```

The two encoders run back to back on the single inference thread; they share the same cores, so running them in parallel wouldn't help. Each ANN query starts as soon as its vector exists, so the text database round trip overlaps with CLIP's forward pass. Requesting only `text` runs exactly the original path, and requesting only `image` skips MiniLM.

The benchmark measures the four configurations (text, image, unified, unified without reranking) on the same fixed query file. Section 9's reproducibility rules apply to all of them.


## 18. Verification and measured optimisation (`scripts/verify.sh`, `benchmarks/optimize.py`, `reporting.py`, `smoke.py`)

**What.** One command that builds the stack, tests it, evaluates and benchmarks it, optimises it against the latency target, and writes a report in which every number is traceable to a raw result file.

**Why.** "28% better" and "under 65 ms" are claims. They need a reproducible procedure behind them, and one where the configuration that is reported is the one that is served.

**How.** The optimiser treats latency as a constraint and precision as the objective. Thread count affects latency only, so it is chosen first. Then every (N, token limit, int8) point gets a dev-split P@5 and a measured p95, using the same pointwise-scoring trick as the evaluation, so each reranker variant scores the pool once per query.

The winner is the most precise point under the target, and only the winner is scored on the held-out test split. Its settings are merged into `.env` and the container is *recreated*, not restarted, because a restart would keep the old environment. The final benchmarks therefore measure what is actually deployed. The smoke checks then exercise each capability over HTTP against that container.

**Connects to.** It uses the same service, evaluation runners, and benchmark code as everything else. It adds no separate measurement path.
