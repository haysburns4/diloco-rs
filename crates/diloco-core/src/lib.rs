//! Core building blocks shared by every node in a DiLoCo training run:
//! the character tokenizer, the batch sampler, and the tiny transformer model.

pub mod data;
pub mod model;
pub mod tokenizer;

pub use data::Dataset;
pub use model::{Config, GptModel};
pub use tokenizer::CharTokenizer;
