#!/usr/bin/env bash
# Run a compute-matched K-sweep (optionally crossed with sharding mode and seeds)
# and lay the results out for scripts/plot_sweep.py.
#
# The central experiment: hold TOTAL compute fixed and vary how often workers
# synchronize. Total inner steps per worker T = rounds * K is held constant, so
# for each K we set rounds = T / K. Then compute = T * N * batch is identical
# across the sweep while communication events = rounds = T / K fall as 1/K. The
# result is the communication-quality frontier.
#
# Layout: runs/<SWEEP>/K<K>_N<N>_<mode>/seed<seed>/{metrics_*.csv,manifest.json,...}
#
# Config via environment variables (defaults target the bundled shakespeare set):
#   SWEEP              sweep name / output subdir          (default k_sweep)
#   TOTAL_INNER_STEPS  T = rounds * K, held constant       (default 2000)
#   KS                 K values to sweep (space-separated)  (default "1 5 25 100")
#   MODES              sharding modes                       (default "iid")
#   SEEDS              seeds (space-separated)              (default "0 1 2")
#   NUM_WORKERS        workers / ranks                      (default 4)
#   CORPUS             corpus path                          (default data/shakespeare.txt)
#   BATCH_SIZE, INNER_LR, VAL_FRAC, EVAL_BATCHES, ADDR  passed through to run_experiment.sh
set -euo pipefail

cd "$(dirname "$0")/.."

SWEEP="${SWEEP:-k_sweep}"
TOTAL_INNER_STEPS="${TOTAL_INNER_STEPS:-2000}"
KS="${KS:-1 5 25 100}"
MODES="${MODES:-iid}"
SEEDS="${SEEDS:-0 1 2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CORPUS="${CORPUS:-data/shakespeare.txt}"

# Cap eval cost by default: the larger corpora have big val sets and we evaluate
# every round. Export the passthrough knobs so run_experiment.sh inherits them.
export EVAL_BATCHES="${EVAL_BATCHES:-50}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export INNER_LR="${INNER_LR:-1e-3}"
export VAL_FRAC="${VAL_FRAC:-0.1}"
export ADDR="${ADDR:-127.0.0.1:7070}"

OUTBASE="runs/${SWEEP}"
mkdir -p "${OUTBASE}"

echo "Building (release)..."
cargo build --release --bin coordinator --bin worker --bin baseline

echo
echo "Sweep '${SWEEP}': T=${TOTAL_INNER_STEPS} N=${NUM_WORKERS} corpus=${CORPUS}"
echo "  K in {${KS}}  modes {${MODES}}  seeds {${SEEDS}}"
echo

for mode in ${MODES}; do
    for k in ${KS}; do
        rounds=$(( TOTAL_INNER_STEPS / k ))
        if [[ "${rounds}" -lt 1 ]]; then
            echo "skip K=${k}: T=${TOTAL_INNER_STEPS} < K, would be 0 rounds" >&2
            continue
        fi
        if (( TOTAL_INNER_STEPS % k != 0 )); then
            echo "note K=${k}: T not divisible by K; using rounds=${rounds} (compute = $(( rounds * k )) inner steps)" >&2
        fi
        for seed in ${SEEDS}; do
            outdir="${OUTBASE}/K${k}_N${NUM_WORKERS}_${mode}/seed${seed}"
            echo "=== K=${k} rounds=${rounds} mode=${mode} seed=${seed} -> ${outdir} ==="
            OUTDIR="${outdir}" \
            NUM_WORKERS="${NUM_WORKERS}" \
            ROUNDS="${rounds}" \
            INNER_STEPS="${k}" \
            SEED="${seed}" \
            DATA_SHARDING="${mode}" \
            CORPUS="${CORPUS}" \
                scripts/run_experiment.sh
        done
    done
done

echo
echo "Sweep complete. Aggregate + plot with:"
echo "  python3 scripts/plot_sweep.py --sweep ${OUTBASE}"
