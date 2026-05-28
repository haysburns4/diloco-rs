use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::{loss, AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use diloco_core::{CharTokenizer, Config, Dataset, GptModel};
use rand::{rngs::StdRng, SeedableRng};
use tracing::info;

// Training hyperparameters. Kept here (rather than in core) because they're a
// property of *this* training run, not the model definition.
const STEPS: usize = 1000;
const BATCH_SIZE: usize = 16;
const LEARNING_RATE: f64 = 1e-3;
const SEED: u64 = 42;

// Note: `main` is synchronous because the work is CPU-bound and local. Tokio is
// already a dependency so the DiLoCo networking layer (worker <-> coordinator
// parameter sync) can make this `#[tokio::main]` later without a churn.
fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    // CPU for now
    let device = Device::Cpu;

    // Corpus path is the first CLI arg, defaulting to the bundled sample.
    let corpus_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/input.txt"));
    let text = std::fs::read_to_string(&corpus_path)
        .with_context(|| format!("reading corpus from {}", corpus_path.display()))?;

    let tokenizer = CharTokenizer::from_text(&text);
    let tokens = tokenizer.encode(&text);
    let cfg = Config::tiny(tokenizer.vocab_size());
    info!(
        vocab_size = cfg.vocab_size,
        num_tokens = tokens.len(),
        block_size = cfg.block_size,
        n_layer = cfg.n_layer,
        n_embd = cfg.n_embd,
        "loaded corpus and built config"
    );

    let dataset = Dataset::new(tokens, cfg.block_size);

    // VarMap owns the trainable parameters; VarBuilder hands slices of it to the
    // model constructor under named scopes.
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = GptModel::new(&cfg, vb)?;

    let mut optimizer = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: LEARNING_RATE,
            ..Default::default()
        },
    )?;

    // Seeded RNG so the loss curve is reproducible run-to-run.
    let mut rng = StdRng::seed_from_u64(SEED);

    info!("starting training for {STEPS} steps");
    for step in 1..=STEPS {
        let (inputs, targets) = dataset.batch(BATCH_SIZE, &device, &mut rng)?;

        let logits = model.forward(&inputs)?; // (batch, block, vocab)
        let (b, t, v) = logits.dims3()?;

        // Flatten to (batch*block, vocab) vs (batch*block,) for cross-entropy.
        let loss = loss::cross_entropy(&logits.reshape((b * t, v))?, &targets.reshape((b * t,))?)?;

        optimizer.backward_step(&loss)?;

        if step == 1 || step % 100 == 0 {
            info!(step, loss = loss.to_scalar::<f32>()?, "training");
        }
    }

    info!("training complete");
    Ok(())
}
