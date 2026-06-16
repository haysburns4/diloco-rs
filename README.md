# diloco-rs

A Rust implementation of [DiLoCo](https://arxiv.org/abs/2311.08105) (Distributed Low-Communication training), a distributed optimization algorithm that trains neural networks across loosely-connected "islands" of compute while communicating up to 500× less than standard distributed training.

Built to understand the algorithm from the ground up, not as a production framework.


## Background

Standard distributed training requires all devices to synchronize gradients at every optimization step. This demands fast, purpose-built interconnects between GPUs. This works inside a single data center but breaks down across data centers or slower networks.

DiLoCo solves this with a two-timescale approach:

- Each worker trains independently for K local steps using AdamW (the inner optimizer)
- Workers then exchange a single "pseudo-gradient" - the net change in parameters over those K steps
- A shared outer optimizer (Nesterov momentum) applies one update to the global parameters using the averaged pseudo-gradients
- Workers sync to the new global parameters and repeat

Communication happens once every K steps instead of every step. With K = 500, the algorithm communicates 500× less than synchronous training while matching its convergence.


## What this trains

A small character-level transformer on a bundled public-domain text corpus (an excerpt of *Alice's Adventures in Wonderland*, in `data/input.txt`). Swap in any text file with `--corpus`. The architecture and task are intentionally minimal. The point is demonstrating the training algorithm.

- ~0.4M parameters (the default `tiny` config: 2 layers, 128-dim, 4 heads)
- Character-level tokenization (~58-character vocabulary, no tokenizer library needed)
- Next-character prediction

<!--
## Project Status

| Component | Status |
|---|---|
| Single-worker training loop (Candle + AdamW) | ✅ Done |
| Multi-worker process architecture | ✅ Done |
| Tensor serialization + inter-worker communication (safetensors over gRPC) | ✅ Done |
| Outer optimizer (Nesterov on pseudo-gradients) | ✅ Done |
| Full DiLoCo inner/outer loop | ✅ Done |
| Fault tolerance (worker join/leave, round resync) | ✅ Done |
| True non-IID data sharding (`--data-sharding non-iid-contiguous`) | ✅ Done |
| Synchronous baseline for comparison | ✅ Done |
| Experiment harness (compute-matched K-sweep, seeds, BPC frontier plots) | ✅ Done |
<!-- 
| Write-up of DiLoCo's claim, experimental methodology, validation loss curves, communication volume comparison, K-sweep showing the communication-vs-convergence tradeoff, non-IID data results, honest discussion and learnings | 🔲 Planned | 

Synchronous DDP's invariant is that all replicas hold identical weights at all times. For that to hold, every process must apply the same update, so they must agree on the averaged gradient before stepping. If they skipped the all-reduce and each applied only its own g_i, the two copies would immediately diverge and you'd no longer have one model being trained. You'd have two different models. Keeping them in lockstep is what forces communication every step.

DiLoCo's insight is precisely to relax that invariant: let each worker diverge freely for K steps on its own copy, then reconcile once. Tolerating divergence is what buys the K× fewer communication events.
-->


## Build

```bash
cargo build --release
```

Requires a recent stable Rust toolchain ([rustup](https://rustup.rs/)). No system dependencies. `protoc` is supplied by a build dependency, so there's nothing to install for the gRPC codegen.

### Run a local DiLoCo cluster

`scripts/run_local.sh` builds the release binaries and launches one coordinator plus N workers as separate processes on `127.0.0.1`, then tears the coordinator down when the workers finish:

```bash
scripts/run_local.sh [NUM_WORKERS] [ROUNDS]   # defaults: 2 workers, 20 rounds
```

Override the corpus or listen address with environment variables:

```bash
CORPUS=data/input.txt ADDR=127.0.0.1:7070 scripts/run_local.sh 4 50
```

#### IID vs. non-IID data

By default every worker samples from the whole training corpus (`--data-sharding iid`), so workers differ only by RNG seed. To test DiLoCo's robustness to non-IID data, split the corpus into N contiguous chunks, one per worker, with `--data-sharding non-iid-contiguous`. Each worker then trains on a structurally distinct slice; the held-out validation set stays shared. At startup each worker logs its chunk range, size, and a decoded text preview so you can confirm they see different content. The `run_local.sh` and `run_comparison.sh` scripts expose this via the `DATA_SHARDING` env var:

```bash
DATA_SHARDING=non-iid-contiguous scripts/run_local.sh 4 50
```

If a chunk ends up too small to hold even one training window (too many workers for the corpus), the worker fails fast with a message suggesting fewer workers, a larger corpus, or IID mode.

To wire the processes up by hand (e.g. across machines - only the address changes):

```bash
# coordinator: waits for `world-size` workers each round
./target/release/coordinator --listen 0.0.0.0:7070 --world-size 2

# workers: rank 0..N-1, each dialing the coordinator
./target/release/worker --rank 0 --world-size 2 --coordinator <coordinator-host>:7070
./target/release/worker --rank 1 --world-size 2 --coordinator <coordinator-host>:7070
```


## Comparing against a synchronous baseline

The point of DiLoCo is to **match the final loss of standard synchronous
data-parallel training while communicating far less**. To check that claim,
`scripts/run_comparison.sh` runs both and compares them:

```bash
scripts/run_comparison.sh [NUM_WORKERS] [ROUNDS] [INNER_STEPS]   # defaults: 2 30 50
```

It (1) trains a DiLoCo cluster, (2) trains the synchronous baseline (the
`baseline` binary, which simulates N data-parallel workers in one process by
averaging their per-step gradients), and (3) writes `runs/metrics_diloco.csv`
and `runs/metrics_sync.csv`, then prints a quantitative summary (final
validation loss, total communication, and the reduction factor).

What makes the comparison credible:

- **Same everything but synchronization.** Both runs share the model, corpus,
  90/10 train/val split, AdamW learning rate, per-rank data seeds, evaluation,
  and metrics, all from `diloco-core`. They even start from the *same* random
  weights (the coordinator saves `theta^(0)` to a file the baseline loads, since
  Candle can't seed the CPU RNG). The only difference is that DiLoCo syncs once
  every `INNER_STEPS` steps and the baseline syncs every step.
- **Matched compute.** The baseline runs `ROUNDS × INNER_STEPS` global steps, so
  it processes the same total sequences as DiLoCo's `N` workers combined. The
  two curves' x-axis (`total_samples`) lines up point-for-point.
- **Honest communication accounting.** One all-reduce event costs `2 ×
  payload_bytes` per worker (upload local + download reduced), measured from the
  actual safetensors encoding. DiLoCo logs one event per round; the baseline one
  per step. At matched compute the ratio is exactly `INNER_STEPS`.
- **Held-out metric.** Both report validation loss over the same fixed,
  deterministic val windows - not noisy per-shard training loss.

The headline is **validation loss at matched compute vs cumulative bytes
communicated**: DiLoCo reaches the baseline's loss for ~`INNER_STEPS` times less
communication. (Wall-clock is logged too, but on localhost it reflects compute,
not network transfer, so it does *not* capture DiLoCo's real-world advantage.
Communication volume is the credible axis.)


## Experiments: the K-sweep

A single comparison shows DiLoCo matching the baseline at one `K`. The real
question is the **shape of the communication-quality tradeoff**: how far you can
push `K` before quality degrades. `scripts/run_sweep.sh` runs that experiment.

The key design choice is holding total compute constant while varying `K`.
Total inner steps per worker `T = ROUNDS × K` is fixed, so for each `K` the sweep
uses `ROUNDS = T / K`. Then compute (`T × N × batch`) is identical across the
sweep while communication events (`= ROUNDS = T / K`) fall as `1/K`. Each config
is repeated over several seeds for error bars.

```bash
# A larger corpus makes K-sweeps and non-IID meaningful (the bundled
# data/input.txt is too small). data/shakespeare.txt is the default below.
TOTAL_INNER_STEPS=2000 KS="1 5 25 100" SEEDS="0 1 2" NUM_WORKERS=4 \
  scripts/run_sweep.sh

# Aggregate every run and draw the frontier (needs matplotlib; a text summary
# and summary.csv print without it).
pip install matplotlib
python3 scripts/plot_sweep.py --sweep runs/k_sweep
```

Layout and provenance: each run lands in
`runs/<sweep>/K<K>_N<N>_<mode>/seed<s>/` with `metrics_diloco.csv`,
`metrics_sync.csv`, the shared `init.safetensors`, and a `manifest.json`
recording the full config, git SHA, and corpus hash. Any point on the plot
is traceable back to an exact run. `scripts/run_experiment.sh` is the reusable
single-config unit the sweep calls; you can run it directly for one config.

The headline figure (`runs/<sweep>/sweep_plot.png`) plots final held-out
**bits-per-character** (`val_loss / ln 2`, the standard char-LM metric) against
cumulative communication on a log axis: the synchronous baseline sits at the
right, and DiLoCo marches left as `K` grows, staying flat in quality until the
*knee*, the empirical limit of how much communication you can trade away. Cross
it with non-IID data to test robustness:

```bash
SWEEP=k_sweep_noniid MODES="iid non-iid-contiguous" scripts/run_sweep.sh
```

### Scope and honest limitations

This is a mechanism reproduction at toy scale, not a benchmark. The model is
~0.4M parameters on a character corpus. So the result to read off these plots is the qualitative shape of the
tradeoff (and the knee's location), not convergence parity at scale. The
synchronous baseline is an in-process exact gradient average. Communication
volume is computed analytically, not measured on a wire. Wall-clock on a
single machine reflects compute, not network, so it understates DiLoCo's
real-world benefit. Communication volume is the credible axis throughout.


## References

- Douillard et al., *DiLoCo: Distributed Low-Communication Training of Language Models* (2023) - [arxiv.org/abs/2311.08105](https://arxiv.org/abs/2311.08105)
- [Candle](https://github.com/huggingface/candle) - Hugging Face's pure-Rust ML framework
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Andrej Karpathy's minimal GPT (inspiration for the small-scale setup)


## License

MIT