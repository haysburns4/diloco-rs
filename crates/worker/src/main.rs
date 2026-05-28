use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::{loss, AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use clap::Parser;
use diloco_core::{params, CharTokenizer, Config, Dataset, GptModel};
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
    let dataset = Dataset::new(tokens, cfg.block_size);

    // Different seed per rank => each worker sees a different stream of batches,
    // i.e. data-parallel training across workers.
    let mut rng = StdRng::seed_from_u64(1234 + args.rank as u64);

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

    // Pull theta^(0) and load it into the local model.
    let init = client
        .init(InitRequest { rank: args.rank })
        .await
        .context("Init RPC failed")?
        .into_inner();
    let global = params::deserialize(&init.params, &device)?;
    params::load_into_varmap(&mut varmap, &global)?;

    for round in 1..=args.rounds {
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
        let local_bytes = params::serialize(&params::varmap_tensors(&varmap))?;
        let reply = client
            .all_reduce(AllReduceRequest {
                rank: args.rank,
                round,
                params: local_bytes,
            })
            .await
            .with_context(|| format!("AllReduce RPC failed at round {round}"))?
            .into_inner();
        let new_global = params::deserialize(&reply.params, &device)?;
        params::load_into_varmap(&mut varmap, &new_global)?;

        info!(rank = args.rank, round, inner_loss = last_loss, "round complete");
    }

    info!(rank = args.rank, "worker finished");
    Ok(())
}
