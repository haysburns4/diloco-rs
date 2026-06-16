#!/usr/bin/env bash
# Run ONE experiment config: a DiLoCo cluster and a compute-matched synchronous
# baseline, both from the same theta^(0), into a single output directory. Writes
# metrics_diloco.csv, metrics_sync.csv, the shared init.safetensors, and a
# manifest.json capturing the full config + git SHA + corpus hash for provenance.
#
# This is the reusable unit that scripts/run_sweep.sh calls once per
# (config, seed). It does no plotting — scripts/plot_sweep.py aggregates the
# directory tree afterward.
#
# Config comes from environment variables (all have defaults except OUTDIR):
#   OUTDIR         output directory for this run (required)
#   NUM_WORKERS    number of workers / simulated ranks       (default 2)
#   ROUNDS         outer rounds                              (default 20)
#   INNER_STEPS    inner steps per round = K                 (default 50)
#   SEED           experiment seed (data stream)             (default 0)
#   DATA_SHARDING  iid | non-iid-contiguous                  (default iid)
#   CORPUS         corpus path                               (default data/input.txt)
#   BATCH_SIZE     sequences per worker per step             (default 16)
#   INNER_LR       inner AdamW learning rate                 (default 1e-3)
#   VAL_FRAC       fraction held out for validation          (default 0.1)
#   EVAL_BATCHES   val batches per eval (0 = whole val set)   (default 0)
#   ADDR           coordinator listen/dial address           (default 127.0.0.1:7070)
set -euo pipefail

cd "$(dirname "$0")/.."

OUTDIR="${OUTDIR:?set OUTDIR to the output directory for this run}"
NUM_WORKERS="${NUM_WORKERS:-2}"
ROUNDS="${ROUNDS:-20}"
INNER_STEPS="${INNER_STEPS:-50}"
SEED="${SEED:-0}"
DATA_SHARDING="${DATA_SHARDING:-iid}"
CORPUS="${CORPUS:-data/input.txt}"
BATCH_SIZE="${BATCH_SIZE:-16}"
INNER_LR="${INNER_LR:-1e-3}"
VAL_FRAC="${VAL_FRAC:-0.1}"
EVAL_BATCHES="${EVAL_BATCHES:-0}"
ADDR="${ADDR:-127.0.0.1:7070}"

mkdir -p "${OUTDIR}"
INIT="${OUTDIR}/init.safetensors"
DILOCO_CSV="${OUTDIR}/metrics_diloco.csv"
SYNC_CSV="${OUTDIR}/metrics_sync.csv"
# Fresh theta^(0) for this run; both DiLoCo and the baseline load it.
rm -f "${INIT}"

# --- DiLoCo cluster ----------------------------------------------------------
./target/release/coordinator \
    --listen "${ADDR}" \
    --world-size "${NUM_WORKERS}" \
    --corpus "${CORPUS}" \
    --init "${INIT}" &
COORD_PID=$!
cleanup() { kill "${COORD_PID}" 2>/dev/null || true; }
trap cleanup EXIT

WORKER_PIDS=()
for ((rank = 0; rank < NUM_WORKERS; rank++)); do
    # Only rank 0 writes metrics (it evaluates the synced global model). The
    # `${arr[@]+...}` guard keeps an empty array safe under `set -u` on bash 3.2.
    METRICS_ARG=()
    if [[ "${rank}" -eq 0 ]]; then
        METRICS_ARG=(--metrics "${DILOCO_CSV}")
    fi
    ./target/release/worker \
        --rank "${rank}" \
        --world-size "${NUM_WORKERS}" \
        --coordinator "${ADDR}" \
        --rounds "${ROUNDS}" \
        --inner-steps "${INNER_STEPS}" \
        --inner-lr "${INNER_LR}" \
        --batch-size "${BATCH_SIZE}" \
        --val-frac "${VAL_FRAC}" \
        --eval-batches "${EVAL_BATCHES}" \
        --seed "${SEED}" \
        --data-sharding "${DATA_SHARDING}" \
        --corpus "${CORPUS}" \
        ${METRICS_ARG[@]+"${METRICS_ARG[@]}"} &
    WORKER_PIDS+=($!)
done

status=0
for pid in "${WORKER_PIDS[@]}"; do
    if ! wait "${pid}"; then status=1; fi
done
cleanup
trap - EXIT
if [[ "${status}" -ne 0 ]]; then
    echo "DiLoCo run failed for ${OUTDIR}" >&2
    exit "${status}"
fi

# --- Synchronous baseline (same compute, same theta^(0)) ---------------------
./target/release/baseline \
    --world-size "${NUM_WORKERS}" \
    --rounds "${ROUNDS}" \
    --inner-steps "${INNER_STEPS}" \
    --inner-lr "${INNER_LR}" \
    --batch-size "${BATCH_SIZE}" \
    --val-frac "${VAL_FRAC}" \
    --eval-batches "${EVAL_BATCHES}" \
    --seed "${SEED}" \
    --data-sharding "${DATA_SHARDING}" \
    --corpus "${CORPUS}" \
    --init "${INIT}" \
    --metrics "${SYNC_CSV}"

# --- Provenance manifest -----------------------------------------------------
GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="$(git diff --quiet 2>/dev/null && echo false || echo true)"
CORPUS_SHA="$(shasum -a 256 "${CORPUS}" 2>/dev/null | cut -d' ' -f1 || echo unknown)"
TOTAL_INNER_STEPS=$(( ROUNDS * INNER_STEPS ))
cat > "${OUTDIR}/manifest.json" <<JSON
{
  "num_workers": ${NUM_WORKERS},
  "rounds": ${ROUNDS},
  "inner_steps": ${INNER_STEPS},
  "total_inner_steps": ${TOTAL_INNER_STEPS},
  "seed": ${SEED},
  "data_sharding": "${DATA_SHARDING}",
  "corpus": "${CORPUS}",
  "corpus_sha256": "${CORPUS_SHA}",
  "batch_size": ${BATCH_SIZE},
  "inner_lr": ${INNER_LR},
  "val_frac": ${VAL_FRAC},
  "git_sha": "${GIT_SHA}",
  "git_dirty": ${GIT_DIRTY},
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON

echo "ok: ${OUTDIR} (K=${INNER_STEPS} N=${NUM_WORKERS} rounds=${ROUNDS} seed=${SEED} mode=${DATA_SHARDING})"
