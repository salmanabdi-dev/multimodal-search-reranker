#!/usr/bin/env bash
# End-to-end verification: build, test, evaluate, benchmark, optimise, report.
#
#   ./scripts/verify.sh            # asks before clearing the main index
#   ./scripts/verify.sh --yes      # non-interactive
#   ./scripts/verify.sh --quick    # smaller sweeps/iterations (smoke-level numbers only)
#
# Everything runs inside the api container, so the numbers describe the deployed system.
# Output: reports/VERIFICATION.md (summary), reports/*.json (raw results),
#         reports/junit_*.xml (tests), reports/verify_<stamp>.log (full log),
#         reports/verify_steps.tsv (status and duration of every step).
set -uo pipefail
cd "$(dirname "$0")/.."

YES=0; QUICK=0
for a in "$@"; do
  case "$a" in
    --yes|-y) YES=1 ;;
    --quick) QUICK=1 ;;
    *) echo "unknown option $a"; exit 2 ;;
  esac
done

mkdir -p reports
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="reports/verify_${STAMP}.log"
STEPS="reports/verify_steps.tsv"
exec > >(tee -a "$LOG") 2>&1
printf "step\tstatus\tseconds\n" > "$STEPS"

DC="docker compose"
RUN="$DC exec -T api python -m mmsearch.cli"
FAILED=0

if [ "$QUICK" = 1 ]; then
  BENCH_ITERS=200; OPT_ARGS="--n-grid 10,20 --max-lengths 256,128 --iterations 60 --max-dev-queries 60"
else
  BENCH_ITERS=1000; OPT_ARGS=""
fi

step() {  # step <name> <critical:0|1> <command...>
  local name="$1" critical="$2"; shift 2
  echo; echo "================================================================"
  echo "== $name"; echo "================================================================"
  local t0=$SECONDS
  "$@"; local rc=$?
  local dt=$((SECONDS - t0))
  if [ $rc -eq 0 ]; then printf "%s\tok\t%s\n" "$name" "$dt" >> "$STEPS"
  else
    printf "%s\tFAILED(%s)\t%s\n" "$name" "$rc" "$dt" >> "$STEPS"; FAILED=1
    if [ "$critical" = 1 ]; then echo "critical step failed: $name"; summary; exit 1; fi
  fi
  return 0
}

summary() {
  echo; echo "== step summary"; column -t -s $'\t' "$STEPS" 2>/dev/null || cat "$STEPS"
  echo; echo "full log: $LOG"
}

wait_ready() {
  echo "waiting for /ready (model loading + warm-up)..."
  for _ in $(seq 1 180); do
    if curl -fsS http://localhost:8000/ready >/dev/null 2>&1; then echo "ready"; return 0; fi
    sleep 5
  done
  echo "API did not become ready in 15 minutes"; $DC logs --tail 80 api; return 1
}

# `docker compose restart` keeps the old environment; settings written to .env only take
# effect when the container is recreated. Recreating keeps the bind-mounted data.
restart_api() { $DC up -d --force-recreate api && wait_ready; }

apply_env() {  # merge KEY=VALUE lines from $1 into .env (backup kept once)
  [ -f "$1" ] || { echo "missing $1"; return 1; }
  [ -f .env.before-verify ] || cp .env .env.before-verify
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    key="${line%%=*}"
    grep -v "^${key}=" .env > .env.tmp || true
    mv .env.tmp .env
    echo "$line" >> .env
    echo "applied $line"
  done < "$1"
}

preflight() {
  command -v docker >/dev/null || { echo "docker not found"; return 1; }
  $DC version || return 1
  [ -f .env ] || cp .env.example .env
  { echo "host: $(uname -a)"; docker version --format 'docker {{.Server.Version}} ({{.Server.Os}}/{{.Server.Arch}})';
    $DC version; } > reports/host.txt
  cat reports/host.txt
}

# Ask up front, not 10+ minutes in after the test suites.
if [ "$YES" != 1 ]; then
  read -r -p "This run clears the main index (documents + images) for a clean evaluation. Continue? [y/N] " ans
  [ "$ans" = "y" ] || [ "$ans" = "Y" ] || { echo "aborted"; exit 1; }
fi
reset_index() { $RUN reset --yes; }

ingest_beir() { $RUN download-dataset --name nfcorpus && \
                $RUN ingest --source beir --path data/beir/nfcorpus --rebuild-index; }
ingest_images() { $RUN make-image-dataset --out data/images_synth --seed 7 && \
                  $RUN ingest-images --path data/images_synth/images; }
apply_optimised() { apply_env reports/optimized.env && apply_env reports/images.env && restart_api; }
# Added after evaluation so the 46 out-of-domain sample docs never contaminate NFCorpus
# results. The HNSW index is maintained incrementally, so no rebuild is needed.
demo_data() { $RUN ingest --source jsonl --path data/sample/corpus.jsonl && restart_api; }

# ----------------------------------------------------------------------------- run
step "preflight"                          1 preflight
step "1. build + start (docker compose)"  1 $DC up -d --build
step "1. api ready"                       1 wait_ready
step "3. unit tests"                      0 $DC exec -T api pytest -q --junitxml=reports/junit_unit.xml
step "3. integration tests"               0 $DC exec -T api pytest -q -m integration -rA \
                                               --junitxml=reports/junit_integration.xml
step "2. reset main index"                1 reset_index
step "2. ingest BEIR NFCorpus"            1 ingest_beir
step "2. ingest synthetic images"         1 ingest_images
step "2. restart api on full index"       1 restart_api
step "4. NFCorpus P@5 (tune dev, test)"   0 $RUN eval --dataset beir --path data/beir/nfcorpus --tune --tag tune
step "5. image retrieval evaluation"      0 $RUN eval-images --path data/images_synth --tune --calibrate \
                                               --write-env reports/images.env
step "2. HNSW recall vs exact"            0 $RUN ann-recall --dataset beir --path data/beir/nfcorpus
step "6. latency baseline"                0 $RUN bench --mode inprocess --iterations "$BENCH_ITERS" --tag baseline
step "8. optimise (measured sweep)"       0 $RUN optimize --path data/beir/nfcorpus --write-env reports/optimized.env $OPT_ARGS
step "8. apply optimised settings"        0 apply_optimised
step "4. NFCorpus P@5 (final settings)"   0 $RUN eval --dataset beir --path data/beir/nfcorpus --tag final
step "6. latency final (in-process)"      0 $RUN bench --mode inprocess --iterations "$BENCH_ITERS" --tag final
step "6. latency final (HTTP)"            0 $RUN bench --mode http --url http://localhost:8000 --iterations "$BENCH_ITERS" --tag final
step "2. end-to-end API checks"           0 $RUN smoke --url http://localhost:8000
step "demo data (sample docs)"            0 demo_data
step "10. write VERIFICATION.md"          0 $RUN report

summary
if [ "$FAILED" = 1 ]; then echo "some steps FAILED; see reports/VERIFICATION.md and $LOG"; exit 1; fi
echo "all steps passed; summary in reports/VERIFICATION.md"
