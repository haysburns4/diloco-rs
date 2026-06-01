use std::fmt;
use std::str::FromStr;

use candle_core::{Device, Result, Tensor};
use rand::Rng;

use crate::tokenizer::CharTokenizer;

/// How the training corpus is divided across workers. The held-out validation
/// set is unaffected by this choice — only training data is sharded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataShardingMode {
    /// Every worker samples from the entire training corpus; workers differ only
    /// by their RNG seed. Approximates IID data across workers. (The original
    /// behavior.)
    Iid,
    /// The training corpus is cut into `world_size` contiguous chunks and each
    /// worker samples exclusively from its own chunk, so workers see genuinely
    /// different data distributions — the non-IID setting DiLoCo claims to be
    /// robust to.
    NonIidContiguous,
}

impl DataShardingMode {
    /// The half-open token range `[start, end)` of a `num_tokens`-long training
    /// corpus assigned to `rank` of `world_size`.
    ///
    /// IID returns the whole corpus for every rank, so both modes share a single
    /// downstream code path. For the non-IID case the integer-floor boundaries
    /// tile `[0, num_tokens)` with no gaps or overlap: consecutive chunks meet
    /// exactly, remainder tokens land in the later chunks (each at most one token
    /// larger), and `rank == world_size - 1` ends precisely at `num_tokens`.
    pub fn shard(self, rank: usize, world_size: usize, num_tokens: usize) -> Shard {
        assert!(world_size > 0, "world_size must be positive");
        assert!(
            rank < world_size,
            "rank {rank} out of range for world_size {world_size}"
        );
        match self {
            DataShardingMode::Iid => Shard {
                start: 0,
                end: num_tokens,
            },
            DataShardingMode::NonIidContiguous => Shard {
                start: rank * num_tokens / world_size,
                end: (rank + 1) * num_tokens / world_size,
            },
        }
    }
}

impl fmt::Display for DataShardingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Mirrors `FromStr` so the CLI spelling round-trips.
        let s = match self {
            DataShardingMode::Iid => "iid",
            DataShardingMode::NonIidContiguous => "non-iid-contiguous",
        };
        f.write_str(s)
    }
}

impl FromStr for DataShardingMode {
    // String (not a custom type) so clap's value-parser fallback accepts it: the
    // bound is `Err: Into<Box<dyn Error + Send + Sync>>`, which String satisfies.
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "iid" => Ok(DataShardingMode::Iid),
            "non-iid-contiguous" => Ok(DataShardingMode::NonIidContiguous),
            other => Err(format!(
                "unknown sharding mode '{other}' (expected: iid, non-iid-contiguous)"
            )),
        }
    }
}

/// A worker's half-open `[start, end)` slice of the training corpus.
#[derive(Debug, Clone, Copy)]
pub struct Shard {
    pub start: usize,
    pub end: usize,
}

impl Shard {
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// The first `max_chars` characters of this shard, decoded and with newlines
    /// flattened to spaces — a one-line startup sanity check that workers really
    /// do see different text.
    pub fn preview(&self, tokens: &[u32], tokenizer: &CharTokenizer, max_chars: usize) -> String {
        let end = (self.start + max_chars).min(self.end);
        tokenizer.decode(&tokens[self.start..end]).replace('\n', " ")
    }
}

/// Split a token stream into a training prefix and a validation suffix.
/// `val_frac` is the fraction held out for validation (e.g. 0.1 = last 10%).
/// Every node splits the same corpus the same way, so the train and val sets
/// are identical across the DiLoCo workers and the synchronous baseline.
pub fn train_val_split(tokens: Vec<u32>, val_frac: f64) -> (Vec<u32>, Vec<u32>) {
    assert!(
        (0.0..1.0).contains(&val_frac),
        "val_frac must be in [0, 1), got {val_frac}"
    );
    let val_len = (tokens.len() as f64 * val_frac) as usize;
    let split = tokens.len() - val_len;
    let val = tokens[split..].to_vec();
    let mut train = tokens;
    train.truncate(split);
    (train, val)
}

/// Holds the entire corpus as a flat sequence of token ids and serves random
/// contiguous windows for next-token prediction.
pub struct Dataset {
    tokens: Vec<u32>,
    block_size: usize,
}

impl Dataset {
    pub fn new(tokens: Vec<u32>, block_size: usize) -> Self {
        assert!(
            tokens.len() > block_size + 1,
            "corpus ({} tokens) must be longer than block_size + 1 ({})",
            tokens.len(),
            block_size + 1
        );
        Self { tokens, block_size }
    }

    /// Build the dataset for one worker's `shard` of `train_tokens`. Centralizes
    /// the slice + the too-small-chunk check so the worker and baseline share it.
    ///
    /// Fails fast (rather than the generic panic in [`Dataset::new`] or a
    /// degenerate sampler) when a contiguous chunk can't hold even a single
    /// training window — the typical cause is too many workers for the corpus.
    pub fn from_shard(train_tokens: &[u32], shard: Shard, block_size: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(
            shard.len() >= block_size + 2,
            "shard [{}, {}) has {} tokens, fewer than the {} needed for one \
             training window (block_size {} + 1 target + 1); use fewer workers, \
             a larger corpus, or --data-sharding iid",
            shard.start,
            shard.end,
            shard.len(),
            block_size + 2,
            block_size,
        );
        Ok(Self::new(train_tokens[shard.start..shard.end].to_vec(), block_size))
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Sample a batch of `(inputs, targets)`, each of shape
    /// `(batch_size, block_size)`. `targets` is `inputs` shifted one position
    /// to the right — the standard char-LM objective.
    pub fn batch<R: Rng>(
        &self,
        batch_size: usize,
        device: &Device,
        rng: &mut R,
    ) -> Result<(Tensor, Tensor)> {
        let mut xs = Vec::with_capacity(batch_size * self.block_size);
        let mut ys = Vec::with_capacity(batch_size * self.block_size);
        let max_start = self.tokens.len() - self.block_size - 1;
        for _ in 0..batch_size {
            let start = rng.gen_range(0..=max_start);
            xs.extend_from_slice(&self.tokens[start..start + self.block_size]);
            ys.extend_from_slice(&self.tokens[start + 1..start + 1 + self.block_size]);
        }
        let shape = (batch_size, self.block_size);
        let x = Tensor::from_vec(xs, shape, device)?;
        let y = Tensor::from_vec(ys, shape, device)?;
        Ok((x, y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iid_gives_every_worker_the_whole_corpus() {
        for rank in 0..4 {
            let s = DataShardingMode::Iid.shard(rank, 4, 100);
            assert_eq!((s.start, s.end), (0, 100));
        }
    }

    #[test]
    fn non_iid_chunks_tile_the_corpus_without_gaps() {
        let (n, len) = (3usize, 100usize); // doesn't divide evenly
        let shards: Vec<Shard> = (0..n)
            .map(|r| DataShardingMode::NonIidContiguous.shard(r, n, len))
            .collect();

        // Boundaries meet exactly: starts at 0, ends at len, no gaps or overlap.
        assert_eq!(shards[0].start, 0);
        assert_eq!(shards[n - 1].end, len);
        for pair in shards.windows(2) {
            assert_eq!(pair[0].end, pair[1].start);
        }

        // Every token is covered once, and chunk sizes are balanced to within one.
        let sizes: Vec<usize> = shards.iter().map(Shard::len).collect();
        assert_eq!(sizes.iter().sum::<usize>(), len);
        assert!(sizes.iter().max().unwrap() - sizes.iter().min().unwrap() <= 1);
    }

    #[test]
    fn from_shard_rejects_a_chunk_too_small_for_a_window() {
        let tokens: Vec<u32> = (0..10).collect();
        let shard = Shard { start: 0, end: 4 };
        assert!(Dataset::from_shard(&tokens, shard, 8).is_err());
    }

    #[test]
    fn mode_string_round_trips() {
        for mode in [DataShardingMode::Iid, DataShardingMode::NonIidContiguous] {
            assert_eq!(mode.to_string().parse::<DataShardingMode>().unwrap(), mode);
        }
        assert!("bogus".parse::<DataShardingMode>().is_err());
    }
}
