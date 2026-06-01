#!/usr/bin/env bash
# Launch a full DiLoCo run locally: one coordinator + N workers as separate
# processes on 127.0.0.1. The only thing that changes for a real multi-machine
# deployment is the --coordinator address passed to each worker.
#
# Usage: scripts/run_local.sh [NUM_WORKERS] [ROUNDS]
set -euo pipefail

cd "$(dirname "$0")/.."

NUM_WORKERS="${1:-2}"
ROUNDS="${2:-20}"
CORPUS="${CORPUS:-data/input.txt}"
# Port 7070, not 7000: macOS AirPlay Receiver (Control Center) listens on 7000.
ADDR="${ADDR:-127.0.0.1:7070}"
# Data sharding across workers: iid | non-iid-contiguous (see worker --help).
DATA_SHARDING="${DATA_SHARDING:-iid}"

echo "Building (release)..."
cargo build --release --bin coordinator --bin worker

echo "Starting coordinator on ${ADDR} (world-size=${NUM_WORKERS}, sharding=${DATA_SHARDING})"
./target/release/coordinator \
    --listen "${ADDR}" \
    --world-size "${NUM_WORKERS}" \
    --corpus "${CORPUS}" &
COORD_PID=$!

# Always tear the coordinator down when this script exits.
cleanup() { kill "${COORD_PID}" 2>/dev/null || true; }
trap cleanup EXIT

WORKER_PIDS=()
for ((rank = 0; rank < NUM_WORKERS; rank++)); do
    ./target/release/worker \
        --rank "${rank}" \
        --world-size "${NUM_WORKERS}" \
        --coordinator "${ADDR}" \
        --rounds "${ROUNDS}" \
        --corpus "${CORPUS}" \
        --data-sharding "${DATA_SHARDING}" &
    WORKER_PIDS+=($!)
done

# Wait for every worker; fail the script if any worker fails.
status=0
for pid in "${WORKER_PIDS[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

echo "All workers finished (status=${status})."
exit "${status}"
