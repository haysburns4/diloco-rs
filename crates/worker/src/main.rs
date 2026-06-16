use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::{loss, AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use clap::Parser;
use diloco_core::{
    eval_loss, params, train_val_split, CharTokenizer, Config, DataShardingMode, Dataset, GptModel,
    MetricsLogger,
};
use diloco_net::{AllReduceRequest, DilocoClient, InitRequest, MAX_MESSAGE_SIZE};
use rand::{rngs::StdRng, SeedableRng};
use tonic::transport::{Channel, Endpoint};
use tracing::{info, warn};

#[derive(Parser)]
#[command(about = "DiLoCo worker (inner optimizer)")]
struct Args {
    /// This worker's rank in [0, world_size). Drives the data shard.
    #[arg(long)]
    rank: u32,
    /// Total number of workers (informational for the worker).
    #[arg(long)]
    world_size: usize,
    /// Coordinator address, e.g. 127.0.0.1:7000.
    #[arg(long)]
    coordinator: String,
    /// Number of outer rounds to run.
    #[arg(long, default_value_t = 20)]
    rounds: u64,
    /// Inner optimizer steps per round (DiLoCo's K / H).
    #[arg(long, default_value_t = 50)]
    inner_steps: usize,
    /// Inner AdamW learning rate.
    #[arg(long, default_value_t = 1e-3)]
    inner_lr: f64,
    /// Sequences per batch.
    #[arg(long, default_value_t = 16)]
    batch_size: usize,
    /// Corpus path (must match the coordinator's so the vocabulary lines up).
    #[arg(long, default_value = "data/input.txt")]
    corpus: PathBuf,
    /// Fraction of the corpus held out for validation (last `val_frac`).
    #[arg(long, default_value_t = 0.1)]
    val_frac: f64,
    /// Max validation batches per eval (0 = use the whole val set).
    #[arg(long, default_value_t = 0)]
    eval_batches: usize,
    /// Experiment seed. Offsets each worker's data-sampling RNG so repeated runs
    /// draw independent batch streams (for error bars across seeds). Model init
    /// is separately randomized by the coordinator (Candle's CPU RNG can't be
    /// seeded), so distinct seeds give genuinely independent runs.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Training data sharding: `iid` (every worker samples the whole corpus,
    /// differing only by seed) or `non-iid-contiguous` (each worker gets a
    /// distinct contiguous chunk). The held-out validation set is shared either
    /// way. Must match across all workers for the chunks to tile the corpus.
    #[arg(long, default_value = "iid")]
    data_sharding: DataShardingMode,
    /// Optional CSV path for per-round metrics. Only rank 0 writes it; the row
    /// reflects the synced global model and aggregates compute/communication
    /// across all workers.
    #[arg(long)]
    metrics: Option<PathBuf>,
}

async fn connect_with_retry(addr: &str) -> Result<DilocoClient<Channel>> {
    let endpoint: Endpoint = Channel::from_shared(format!("http://{addr}"))?;
    for attempt in 1..=40 {
        match endpoint.connect().await {
            Ok(channel) => {
                return Ok(DilocoClient::new(channel)
                    .max_decoding_message_size(MAX_MESSAGE_SIZE)
                    .max_encoding_message_size(MAX_MESSAGE_SIZE));
            }
            Err(e) => {
                warn!(attempt, error = %e, "coordinator not reachable yet, retrying");
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        }
    }
    anyhow::bail!("could not connect to coordinator at {addr} after retries")
}

#[tokio::main]
async fn main() -> Result<()> {
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
    // Train on the leading split; rank 0 evaluates on the held-out tail. The
    // baseline splits the identical corpus the same way, so the val set matches.
    let (train_tokens, val_tokens) = train_val_split(tokens, args.val_frac);

    // This worker's slice of the *training* corpus. IID => the whole corpus
    // (workers differ only by seed); non-IID => a distinct contiguous chunk, so
    // each worker trains on a genuinely different data distribution.
    let shard = args
        .data_sharding
        .shard(args.rank as usize, args.world_size, train_tokens.len());
    info!(
        rank = args.rank,
        mode = %args.data_sharding,
        chunk_start = shard.start,
        chunk_end = shard.end,
        chunk_tokens = shard.len(),
        preview = %shard.preview(&train_tokens, &tokenizer, 50),
        "training data shard"
    );
    let dataset = Dataset::from_shard(&train_tokens, shard, cfg.block_size)?;

    // Distinct stream per (seed, rank): the seed separates experiment repeats,
    // the rank separates workers within a run.
    let mut rng = StdRng::seed_from_u64(args.seed.wrapping_mul(10_000) + 1234 + args.rank as u64);

    // The model is built with random init, but those values are immediately
    // overwritten by the global parameters fetched from the coordinator, so all
    // workers provably start from the same theta^(0).
    let mut varmap = VarMap::new();
    let model = {
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        GptModel::new(&cfg, vb)?
    };

    info!(
        rank = args.rank,
        world_size = args.world_size,
        rounds = args.rounds,
        inner_steps = args.inner_steps,
        coordinator = %args.coordinator,
        "worker starting"
    );

    let mut client = connect_with_retry(&args.coordinator).await?;

    // Pull the current global parameters and the round to start from. For a
    // fresh start the coordinator returns round=1 and θ⁰. For a restarting
    // worker it returns the live round and the current θ so the worker
    // fast-forwards without re-training rounds it missed.
    let init = client
        .init(InitRequest { rank: args.rank })
        .await
        .context("Init RPC failed")?
        .into_inner();
    let global = params::deserialize(&init.params, &device)?;
    params::load_into_varmap(&mut varmap, &global)?;
    let mut round = init.round;

    if round > 1 {
        warn!(rank = args.rank, round, "resyncing to coordinator round (missed earlier rounds)");
    }

    // Rank 0 records the comparison metrics. The model shares storage with the
    // varmap, so after each `load_into_varmap` it reflects the global weights;
    // evaluating it gives the held-out loss of the synced global model.
    let mut metrics = match (args.rank, &args.metrics) {
        (0, Some(path)) => Some(MetricsLogger::create(path)?),
        _ => None,
    };
    let per_worker_bytes = params::allreduce_bytes_per_worker(&global)?;
    let start = Instant::now();
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
    if let Some(logger) = metrics.as_mut() {
        // Round 0: the shared theta^(0), before any training or communication.
        let val = eval(&model)?;
        logger.log(0, 0, 0.0, 0, val, f32::NAN)?;
        info!(rank = args.rank, round = 0u64, val_loss = val, "initial eval");
    }

    while round <= args.rounds {
        // Fresh inner optimizer each round: the global parameters just changed
        // underneath us, so stale AdamW moments would be meaningless.
        let mut optimizer = AdamW::new(
            varmap.all_vars(),
            ParamsAdamW {
                lr: args.inner_lr,
                ..Default::default()
            },
        )?;

        let mut last_loss = f32::NAN;
        for _ in 0..args.inner_steps {
            let (inputs, targets) = dataset.batch(args.batch_size, &device, &mut rng)?;
            let logits = model.forward(&inputs)?;
            let (b, t, v) = logits.dims3()?;
            let loss =
                loss::cross_entropy(&logits.reshape((b * t, v))?, &targets.reshape((b * t,))?)?;
            optimizer.backward_step(&loss)?;
            last_loss = loss.to_scalar::<f32>()?;
        }

        // All-reduce: send local params, receive the new averaged global.
        // On FailedPrecondition the coordinator has moved on (e.g. a timeout
        // closed the round while this worker was training); call Init to resync
        // to the current round and retry from there.
        let local_bytes = params::serialize(&params::varmap_tensors(&varmap))?;
        let reply = match client
            .all_reduce(AllReduceRequest {
                rank: args.rank,
                round,
                params: local_bytes,
            })
            .await
        {
            Ok(r) => r.into_inner(),
            Err(status) if status.code() == tonic::Code::FailedPrecondition => {
                warn!(
                    rank = args.rank,
                    round,
                    "round mismatch — coordinator moved on; resyncing"
                );
                let resync = client
                    .init(InitRequest { rank: args.rank })
                    .await
                    .context("resync Init RPC failed")?
                    .into_inner();
                let new_global = params::deserialize(&resync.params, &device)?;
                params::load_into_varmap(&mut varmap, &new_global)?;
                round = resync.round;
                // Skip metrics for this iteration; the inner training we just
                // did was on stale params and is discarded.
                continue;
            }
            Err(e) => return Err(e).context(format!("AllReduce RPC failed at round {round}")),
        };

        let new_global = params::deserialize(&reply.params, &device)?;
        params::load_into_varmap(&mut varmap, &new_global)?;

        if let Some(logger) = metrics.as_mut() {
            // Aggregate compute and communication across all workers: each round
            // every worker processes `inner_steps * batch_size` sequences and
            // performs one all-reduce event.
            let total_samples =
                round * args.inner_steps as u64 * args.world_size as u64 * args.batch_size as u64;
            let comm_bytes = round * args.world_size as u64 * per_worker_bytes as u64;
            let val = eval(&model)?;
            logger.log(
                round,
                total_samples,
                start.elapsed().as_secs_f64(),
                comm_bytes,
                val,
                last_loss,
            )?;
            info!(rank = args.rank, round, val_loss = val, "round eval");
        }

        info!(rank = args.rank, round, inner_loss = last_loss, "round complete");
        // Advance to the round the coordinator says to enter next. Under normal
        // operation this equals round + 1; under resync it may jump forward.
        round = reply.round;
    }

    info!(rank = args.rank, "worker finished");
    Ok(())
}
