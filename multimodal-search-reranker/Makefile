# Everything runs inside the api container so results are reproducible across machines.
DC  = docker compose
RUN = $(DC) exec api python -m mmsearch.cli

.PHONY: env up down logs build ingest-sample eval-sample download-nfcorpus ingest-nfcorpus \
        eval tune ann-recall images-synth ingest-images eval-images tune-images \
        calibrate-images demo optimize smoke report verify verify-quick bench bench-http stats \
        test test-integration lint clean

env:
	@test -f .env || cp .env.example .env

build: env
	$(DC) build

up: env
	$(DC) up -d --build
	@echo "API: http://localhost:8000  (UI at /, docs at /docs)"

down:
	$(DC) down

logs:
	$(DC) logs -f api

# --- small hand-labelled corpus (seconds) ---
ingest-sample:
	$(RUN) ingest --source jsonl --path data/sample/corpus.jsonl --rebuild-index
	$(DC) restart api

eval-sample:
	$(RUN) eval --dataset sample

# --- BEIR NFCorpus (the real evaluation) ---
download-nfcorpus:
	$(RUN) download-dataset --name nfcorpus

ingest-nfcorpus:
	$(RUN) ingest --source beir --path data/beir/nfcorpus --rebuild-index
	$(DC) restart api

tune:
	$(RUN) eval --dataset beir --path data/beir/nfcorpus --tune

eval:
	$(RUN) eval --dataset beir --path data/beir/nfcorpus

ann-recall:
	$(RUN) ann-recall --dataset beir --path data/beir/nfcorpus

# --- images: deterministic synthetic benchmark (96 images, 34 queries) ---
images-synth:
	$(RUN) make-image-dataset --out data/images_synth --seed 7

ingest-images: images-synth
	$(RUN) ingest-images --path data/images_synth/images
	$(DC) restart api

eval-images:
	$(RUN) eval-images --path data/images_synth

tune-images:
	$(RUN) eval-images --path data/images_synth --tune --calibrate

calibrate-images:
	$(RUN) calibrate-images --path data/images_synth

# Everything an interview demo needs: text sample + images, then the UI
demo: up ingest-sample ingest-images
	@echo "open http://localhost:8000 and try: 'a red circle', 'iron deficiency', 'something blue'"

bench:
	$(RUN) bench --mode inprocess

bench-http:
	$(RUN) bench --mode http --url http://localhost:8000

optimize:
	$(RUN) optimize --path data/beir/nfcorpus --write-env reports/optimized.env

smoke:
	$(RUN) smoke --url http://localhost:8000

report:
	$(RUN) report

# Full end-to-end verification (build, tests, evals, benchmarks, optimisation, report).
# Takes roughly 45-90 minutes on a laptop CPU. Writes reports/VERIFICATION.md.
verify: env
	./scripts/verify.sh

verify-quick: env
	./scripts/verify.sh --quick

stats:
	$(RUN) stats

test:
	$(DC) exec api pytest -q

test-integration:
	$(DC) exec api pytest -q -m integration

lint:
	$(DC) exec api ruff check src tests

clean:
	$(DC) down -v
