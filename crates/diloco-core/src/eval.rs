//! Deterministic held-out evaluation, shared by the DiLoCo workers and the
//! synchronous baseline so the two runs are compared on *exactly* the same
//! metric. Unlike training, this uses no RNG: it walks fixed, non-overlapping
//! windows over the validation tokens, so the result depends only on the model
//! and the val set — identical on every call and across both runs.

use candle_core::{Device, Result, Tensor};
use candle_nn::loss;

use crate::model::GptModel;

/// Mean next-token cross-entropy of `model` over the validation tokens.
///
/// The val stream is cut into back-to-back windows of `block_size` (the same
/// next-character objective as training: targets are inputs shifted by one).
/// Windows are grouped into batches of `batch_size`; `max_batches` caps how
/// many batches are evaluated (`0` = use every window). Returns `NaN` if the
/// val set is too short to form a single window.
pub fn eval_loss(
    model: &GptModel,
    val_tokens: &[u32],
    block_size: usize,
    batch_size: usize,
    max_batches: usize,
    device: &Device,
) -> Result<f32> {
    // Number of non-overlapping windows that fit, leaving room for the +1 shift.
    let n_windows = if val_tokens.len() > block_size {
        (val_tokens.len() - 1) / block_size
    } else {
        0
    };
    if n_windows == 0 {
        return Ok(f32::NAN);
    }
    let n_windows = if max_batches == 0 {
        n_windows
    } else {
        n_windows.min(max_batches * batch_size)
    };

    // Accumulate a token-weighted mean so a short final batch is weighted
    // correctly rather than averaged as if it were full.
    let mut total_loss = 0.0f64;
    let mut total_tokens = 0usize;
    let mut window = 0usize;
    while window < n_windows {
        let batch_windows = batch_size.min(n_windows - window);
        let mut xs = Vec::with_capacity(batch_windows * block_size);
        let mut ys = Vec::with_capacity(batch_windows * block_size);
        for w in window..window + batch_windows {
            let start = w * block_size;
            xs.extend_from_slice(&val_tokens[start..start + block_size]);
            ys.extend_from_slice(&val_tokens[start + 1..start + 1 + block_size]);
        }
        let shape = (batch_windows, block_size);
        let inputs = Tensor::from_vec(xs, shape, device)?;
        let targets = Tensor::from_vec(ys, shape, device)?;

        let logits = model.forward(&inputs)?;
        let (b, t, v) = logits.dims3()?;
        let mean = loss::cross_entropy(&logits.reshape((b * t, v))?, &targets.reshape((b * t,))?)?;

        let n = b * t;
        total_loss += mean.to_scalar::<f32>()? as f64 * n as f64;
        total_tokens += n;
        window += batch_windows;
    }

    Ok((total_loss / total_tokens as f64) as f32)
}
