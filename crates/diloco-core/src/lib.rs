//! Core building blocks shared by every node in a DiLoCo training run:
//! the character tokenizer, the batch sampler, and the tiny transformer model.

pub mod data;
pub mod eval;
pub mod metrics;
pub mod model;
pub mod outer;
pub mod params;
pub mod tokenizer;

pub use data::{train_val_split, DataShardingMode, Dataset, Shard};
pub use eval::eval_loss;
pub use metrics::MetricsLogger;
pub use model::{Config, GptModel};
pub use outer::OuterOptimizer;
pub use tokenizer::CharTokenizer;
