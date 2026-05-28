# diloco-rs

A Rust implementation of [DiLoCo](https://arxiv.org/abs/2311.08105) (Distributed Low-Communication training), a distributed optimization algorithm that trains neural networks across loosely-connected "islands" of compute while communicating up to 500× less than standard distributed training.

Built to understand the algorithm from the ground up, not as a production framework.


## Background

Standard distributed training requires all devices to synchronize gradients at every optimization step — which demands fast, purpose-built interconnects between GPUs. This works inside a single data center but breaks down across data centers or slower networks.

DiLoCo solves this with a two-timescale approach:

- Each worker trains independently for K local steps using AdamW (the inner optimizer)
- Workers then exchange a single "pseudo-gradient" — the net change in parameters over those K steps
- A shared outer optimizer (Nesterov momentum) applies one update to the global parameters using the averaged pseudo-gradients
- Workers sync to the new global parameters and repeat

Communication happens once every K steps instead of every step. With K = 500, the algorithm communicates 500× less than synchronous training while matching its convergence.


## What this trains

A small character-level transformer on a bundled public-domain text corpus (an excerpt of *Alice's Adventures in Wonderland*, in `data/input.txt`). Swap in any text file with `--corpus`. The architecture and task are intentionally minimal — the point is demonstrating the training algorithm.

- ~0.4M parameters (the default `tiny` config: 2 layers, 128-dim, 4 heads)
- Character-level tokenization (~58-character vocabulary, no tokenizer library needed)
- Next-character prediction


## Project Status

| Component | Status |
|---|---|
| Single-worker training loop (Candle + AdamW) | ✅ Done |
| Multi-worker process architecture | ✅ Done |
| Tensor serialization + inter-worker communication (safetensors over gRPC) | ✅ Done |
| Outer optimizer (Nesterov on pseudo-gradients) | ✅ Done |
| Full DiLoCo inner/outer loop | ✅ Done |
| Fault tolerance (worker join/leave, round resync) | ⚙️ WIP |
| True non-IID data sharding (currently seed-offset per rank) | 🔲 Planned |
| Synchronous baseline for comparison | 🔲 Planned |
| Experiments (varying K, non-IID data split) | 🔲 Planned |


## Build

```bash
cargo build --release
```

Requires a recent stable Rust toolchain ([rustup](https://rustup.rs/)). No system dependencies — `protoc` is supplied by a build dependency, so there's nothing to install for the gRPC codegen.

### Run a local DiLoCo cluster

`scripts/run_local.sh` builds the release binaries and launches one coordinator plus N workers as separate processes on `127.0.0.1`, then tears the coordinator down when the workers finish:

```bash
scripts/run_local.sh [NUM_WORKERS] [ROUNDS]   # defaults: 2 workers, 20 rounds
```

Override the corpus or listen address with environment variables:

```bash
CORPUS=data/input.txt ADDR=127.0.0.1:7000 scripts/run_local.sh 4 50
```

To wire the processes up by hand (e.g. across machines — only the address changes):

```bash
# coordinator: waits for `world-size` workers each round
./target/release/coordinator --listen 0.0.0.0:7000 --world-size 2

# workers: rank 0..N-1, each dialing the coordinator
./target/release/worker --rank 0 --world-size 2 --coordinator <coordinator-host>:7000
./target/release/worker --rank 1 --world-size 2 --coordinator <coordinator-host>:7000
```


## References

- Douillard et al., *DiLoCo: Distributed Low-Communication Training of Language Models* (2023) — [arxiv.org/abs/2311.08105](https://arxiv.org/abs/2311.08105)
- [Candle](https://github.com/huggingface/candle) — Hugging Face's pure-Rust ML framework
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Andrej Karpathy's minimal GPT (inspiration for the small-scale setup)


## License

MIT