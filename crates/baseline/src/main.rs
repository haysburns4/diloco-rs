//! Synchronous data-parallel baseline for DiLoCo.
//!
//! Standard distributed training synchronizes every step: each of N workers
//! computes a gradient on its own batch, the gradients are averaged, and one
//! optimizer step is applied to the shared model. We simulate that exactly in a
//! single process — averaging the gradients of N per-shard batches is identical
//! to running N processes with a gradient all-reduce, because the gradient of a
//! mean of losses is the mean of the per-loss gradients. Doing it in-process
//! keeps the code clear and the result mathematically exact; on localhost a
//! real multi-process version's wall-clock wouldn't reflect network cost anyway
//! (the comparison's credible axis is communication *volume*).
//!
//! Everything except the synchronization granularity matches the DiLoCo run:
//! same model, corpus, train/val split, AdamW learning rate, per-rank data
//! seeds (`1234 + rank`), evaluation, and metrics schema (all from
//! `diloco-core`). To match total compute, we run `rounds * inner_steps` global
//! steps — so the baseline processes the same total sequences as DiLoCo's N
//! workers — and evaluate every `inner_steps` steps so the rows line up with
//! DiLoCo's rounds. The only difference is that we communicate every step
//! instead of once per round, which is the whole point.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::{loss, AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use clap::Parser;
use diloco_core::{
    eval_loss, params, train_val_split, CharTokenizer, Config, DataShardingMode, Dataset, GptModel,
    MetricsLogger,
};
use rand::{rngs::StdRng, SeedableRng};
use tracing::info;

#[derive(Parser)]
#[command(about = "Synchronous data-parallel baseline (every step is synchronized)")]
struct Args {
    /// Number of simulated data-parallel workers. Match the DiLoCo run so the
    /// total compute and effective batch size line up.
    #[arg(long)]
    world_size: usize,
    /// Number of outer rounds in the DiLoCo run being compared against.
    #[arg(long, default_value_t = 20)]
    rounds: u64,
    /// DiLoCo's inner steps per round (K). The baseline runs `rounds *
    /// inner_steps` global steps and evaluates every `inner_steps` steps.
    #[arg(long, default_value_t = 50)]
    inner_steps: usize,
    /// AdamW learning rate (must match the DiLoCo worker's inner LR).
    #[arg(long, default_value_t = 1e-3)]
    inner_lr: f64,
    /// Sequences per worker per step.
    #[arg(long, default_value_t = 16)]
    batch_size: usize,
    /// Corpus path (must match the DiLoCo run).
    #[arg(long, default_value = "data/input.txt")]
    corpus: PathBuf,
    /// Fraction of the corpus held out for validation (last `val_frac`).
    #[arg(long, default_value_t = 0.1)]
    val_frac: f64,
    /// Max validation batches per eval (0 = use the whole val set).
    #[arg(long, default_value_t = 0)]
    eval_batches: usize,
    /// Experiment seed. Offsets each simulated worker's data-sampling RNG, mirroring
    /// the DiLoCo worker's `--seed` so a (config, seed) pair draws the same streams.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Training data sharding across the simulated workers: `iid` (each samples
    /// the whole corpus, differing only by seed) or `non-iid-contiguous` (each
    /// gets a distinct contiguous chunk). Matches the DiLoCo worker flag so the
    /// two runs differ only in synchronization granularity.
    #[arg(long, default_value = "iid")]
    data_sharding: DataShardingMode,
    /// Optional path to shared initial parameters (theta^(0)). If present it is
    /// loaded so the baseline starts from the exact same weights as DiLoCo.
    #[arg(long)]
    init: Option<PathBuf>,
    /// Optional CSV path for per-checkpoint metrics.
    #[arg(long)]
    metrics: Option<PathBuf>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = Args::parse();
    let device = Device::Cpu;

    let text = std::fs::read_to_string(&args.corpus)
        .with_context(|| format!("reading corpus from {}", args.corpus.display()))?;
    let tokenizer = CharTokenizer::from_text(&text);
    let tokens = tokenizer.encode(&text);
    let cfg = Config::tiny(tokenizer.vocab_size());
    let (train_tokens, val_tokens) = train_val_split(tokens, args.val_frac);

    // One dataset per simulated worker, sharded exactly like the DiLoCo workers:
    // IID => each sees the whole corpus; non-IID => each its own contiguous chunk.
    let mut datasets = Vec::with_capacity(args.world_size);
    for rank in 0..args.world_size {
        let shard = args
            .data_sharding
            .shard(rank, args.world_size, train_tokens.len());
        info!(
            rank,
            mode = %args.data_sharding,
            chunk_start = shard.start,
            chunk_end = shard.end,
            chunk_tokens = shard.len(),
            preview = %shard.preview(&train_tokens, &tokenizer, 50),
            "training data shard"
        );
        datasets.push(Dataset::from_shard(&train_tokens, shard, cfg.block_size)?);
    }

    let mut varmap = VarMap::new();
    let model = {
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        GptModel::new(&cfg, vb)?
    };

    // Start from the shared theta^(0) if one was provided, so the baseline and
    // DiLoCo converge from identical weights.
    if let Some(path) = &args.init {
        if path.exists() {
            let init = params::load_file(path, &device)?;
            params::load_into_varmap(&mut varmap, &init)?;
            info!(path = %path.display(), "loaded shared initial parameters");
        }
    }

    // One persistent optimizer across the whole run — standard synchronous
    // training, in contrast to DiLoCo which resets the inner optimizer each
    // round because the global parameters change underneath it.
    let mut optimizer = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: args.inner_lr,
            ..Default::default()
        },
    )?;

    // One RNG per simulated worker, seeded exactly like the DiLoCo workers
    // (matching `--seed`) so each "rank" sees the same data stream it would there.
    let mut rngs: Vec<StdRng> = (0..args.world_size)
        .map(|rank| StdRng::seed_from_u64(args.seed.wrapping_mul(10_000) + 1234 + rank as u64))
        .collect();

    let per_worker_bytes = params::allreduce_bytes_per_worker(&params::varmap_tensors(&varmap))?;
    let mut metrics = match &args.metrics {
        Some(path) => Some(MetricsLogger::create(path)?),
        None => None,
    };
    let eval = |model: &GptModel| {
        eval_loss(
            model,
            &val_tokens,
            cfg.block_size,
            args.batch_size,
            args.eval_batches,
            &device,
        )
    };

    info!(
        world_size = args.world_size,
        rounds = args.rounds,
        inner_steps = args.inner_steps,
        total_steps = args.rounds * args.inner_steps as u64,
        "synchronous baseline starting"
    );

    let start = Instant::now();
    if let Some(logger) = metrics.as_mut() {
        let val = eval(&model)?;
        logger.log(0, 0, 0.0, 0, val, f32::NAN)?;
        info!(round = 0u64, val_loss = val, "initial eval");
    }

    let total_steps = args.rounds * args.inner_steps as u64;
    for step in 1..=total_steps {
        // Average the gradients of N per-shard batches via the gradient of the
        // mean loss. Each shard contributes the same number of tokens, so this
        // is the exact synchronous data-parallel update.
        let mut losses = Vec::with_capacity(args.world_size);
        for (rank, rng) in rngs.iter_mut().enumerate() {
            let (inputs, targets) = datasets[rank].batch(args.batch_size, &device, rng)?;
            let logits = model.forward(&inputs)?;
            let (b, t, v) = logits.dims3()?;
            losses.push(loss::cross_entropy(
                &logits.reshape((b * t, v))?,
                &targets.reshape((b * t,))?,
            )?);
        }
        let mut summed = losses[0].clone();
        for l in &losses[1..] {
            summed = (summed + l)?;
        }
        let loss = (summed / args.world_size as f64)?;
        optimizer.backward_step(&loss)?;

        if step % args.inner_steps as u64 == 0 {
            let round = step / args.inner_steps as u64;
            // Same basis as DiLoCo, but synchronizing every step rather than
            // once per round: this is where the K-fold communication gap comes
            // from. total_samples uses `step` here vs `round * inner_steps`
            // there, so both axes line up at matching compute.
            let total_samples = step * args.world_size as u64 * args.batch_size as u64;
            let comm_bytes = step * args.world_size as u64 * per_worker_bytes as u64;
            let train_loss = loss.to_scalar::<f32>()?;
            if let Some(logger) = metrics.as_mut() {
                let val = eval(&model)?;
                logger.log(
                    round,
                    total_samples,
                    start.elapsed().as_secs_f64(),
                    comm_bytes,
                    val,
                    train_loss,
                )?;
                info!(round, step, val_loss = val, "checkpoint eval");
            }
        }
    }

    info!("synchronous baseline finished");
    Ok(())
}
