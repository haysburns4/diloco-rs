#!/usr/bin/env bash
# Run a full, reproducible DiLoCo-vs-synchronous comparison and summarize it.
#
# 1. The coordinator generates theta^(0) and saves it to runs/init.safetensors.
# 2. A DiLoCo cluster (coordinator + N workers) trains; rank 0 logs per-round
#    metrics to runs/metrics_diloco.csv.
# 3. The synchronous baseline loads the *same* theta^(0) and trains for the same
#    total compute, logging to runs/metrics_sync.csv.
# 4. scripts/analyze.py prints a quantitative summary of the two CSVs.
#
# Both runs share model, corpus, train/val split, AdamW LR, per-rank data seeds,
# evaluation and metrics schema (all from diloco-core), so the only difference
# is how often they synchronize.
#
# Usage: scripts/run_comparison.sh [NUM_WORKERS] [ROUNDS] [INNER_STEPS]
set -euo pipefail

cd "$(dirname "$0")/.."

NUM_WORKERS="${1:-2}"
ROUNDS="${2:-30}"
INNER_STEPS="${3:-50}"
CORPUS="${CORPUS:-data/input.txt}"
ADDR="${ADDR:-127.0.0.1:7000}"

OUT="runs"
INIT="${OUT}/init.safetensors"
DILOCO_CSV="${OUT}/metrics_diloco.csv"
SYNC_CSV="${OUT}/metrics_sync.csv"

mkdir -p "${OUT}"
# Fresh theta^(0) for each comparison so both runs share identical new weights.
rm -f "${INIT}"

echo "Building (release)..."
cargo build --release --bin coordinator --bin worker --bin baseline

echo
echo "=== [1/3] DiLoCo: ${NUM_WORKERS} workers x ${ROUNDS} rounds x ${INNER_STEPS} inner steps ==="
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
        --corpus "${CORPUS}" \
        ${METRICS_ARG[@]+"${METRICS_ARG[@]}"} &
    WORKER_PIDS+=($!)
done

status=0
for pid in "${WORKER_PIDS[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
cleanup
trap - EXIT
if [[ "${status}" -ne 0 ]]; then
    echo "DiLoCo run failed." >&2
    exit "${status}"
fi

echo
echo "=== [2/3] Synchronous baseline: same compute, loading shared theta^(0) ==="
./target/release/baseline \
    --world-size "${NUM_WORKERS}" \
    --rounds "${ROUNDS}" \
    --inner-steps "${INNER_STEPS}" \
    --corpus "${CORPUS}" \
    --init "${INIT}" \
    --metrics "${SYNC_CSV}"

echo
echo "=== [3/3] Analysis ==="
python3 scripts/analyze.py --diloco "${DILOCO_CSV}" --sync "${SYNC_CSV}"
