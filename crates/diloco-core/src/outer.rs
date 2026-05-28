//! DiLoCo's outer optimizer, run by the coordinator once per round.
//!
//! Each worker trains locally for K steps starting from the shared global
//! parameters `theta`, producing `theta_i`. The coordinator forms the averaged
//! "pseudo-gradient"
//!
//! ```text
//!   delta = theta - mean_i(theta_i)
//! ```
//!
//! and applies it with SGD + Nesterov momentum (the choice from the DiLoCo
//! paper):
//!
//! ```text
//!   m     = mu * m + delta
//!   theta = theta - lr * (delta + mu * m)
//! ```
//!
//! With `lr = 1`, `mu = 0` and one worker this reduces to "adopt the local
//! params"; averaging across workers with `lr = 1` is plain FedAvg. Momentum is
//! what makes it DiLoCo rather than FedAvg.

use std::collections::HashMap;

use anyhow::{ensure, Result};
use candle_core::Tensor;

pub struct OuterOptimizer {
    momentum: HashMap<String, Tensor>,
    lr: f64,
    mu: f64,
}

impl OuterOptimizer {
    /// Momentum buffers are initialized to zero with the same shapes as the
    /// global parameters.
    pub fn new(global: &HashMap<String, Tensor>, lr: f64, mu: f64) -> Result<Self> {
        let momentum = global
            .iter()
            .map(|(name, t)| Ok((name.clone(), t.zeros_like()?)))
            .collect::<Result<HashMap<_, _>>>()?;
        Ok(Self { momentum, lr, mu })
    }

    /// Apply one outer step in place to `global`, given each worker's locally
    /// trained parameters.
    pub fn step(
        &mut self,
        global: &mut HashMap<String, Tensor>,
        locals: &[HashMap<String, Tensor>],
    ) -> Result<()> {
        ensure!(!locals.is_empty(), "outer step needs at least one worker");
        let n = locals.len() as f64;

        for (name, theta) in global.iter_mut() {
            // Mean of the workers' local parameters for this tensor.
            let mut sum = locals[0]
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("worker 0 missing param {name}"))?
                .clone();
            for local in &locals[1..] {
                let t = local
                    .get(name)
                    .ok_or_else(|| anyhow::anyhow!("a worker is missing param {name}"))?;
                sum = (&sum + t)?;
            }
            let mean = sum.affine(1.0 / n, 0.0)?;

            // delta = theta - mean(theta_i)
            let delta = (&*theta - &mean)?;

            // m = mu * m + delta
            let m = self.momentum.get_mut(name).expect("momentum has every param");
            *m = (&m.affine(self.mu, 0.0)? + &delta)?;

            // theta = theta - lr * (delta + mu * m)   (Nesterov)
            let update = (&delta + &m.affine(self.mu, 0.0)?)?;
            *theta = (&*theta - &update.affine(self.lr, 0.0)?)?;
        }
        Ok(())
    }
}
