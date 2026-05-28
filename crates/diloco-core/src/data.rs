use candle_core::{Device, Result, Tensor};
use rand::Rng;

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
