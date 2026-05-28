use candle_core::{Device, Result, Tensor, D};
use candle_nn::{
    embedding, layer_norm, linear, ops::softmax_last_dim, Embedding, LayerNorm, Linear, Module,
    VarBuilder,
};

/// Hyperparameters for the tiny transformer.
#[derive(Debug, Clone)]
pub struct Config {
    pub vocab_size: usize,
    pub block_size: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
}

impl Config {
    pub fn tiny(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            block_size: 64,
            n_embd: 128,
            n_head: 4,
            n_layer: 2,
        }
    }
}

/// Causal (masked) multi-head self-attention.
struct CausalSelfAttention {
    qkv: Linear,
    proj: Linear,
    n_head: usize,
    head_dim: usize,
}

impl CausalSelfAttention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let n_embd = cfg.n_embd;
        // One fused projection for query, key and value.
        let qkv = linear(n_embd, 3 * n_embd, vb.pp("qkv"))?;
        let proj = linear(n_embd, n_embd, vb.pp("proj"))?;
        Ok(Self {
            qkv,
            proj,
            n_head: cfg.n_head,
            head_dim: n_embd / cfg.n_head,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        let qkv = self.qkv.forward(x)?; // (b, t, 3c)

        // Split the fused projection and reshape each into (b, n_head, t, head_dim).
        let split = |start: usize| -> Result<Tensor> {
            qkv.narrow(D::Minus1, start * c, c)?
                .reshape((b, t, self.n_head, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = split(0)?;
        let k = split(1)?;
        let v = split(2)?;

        // Scaled dot-product attention scores: (b, n_head, t, t).
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let att = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;

        // Causal mask: positions may only attend to themselves and the past.
        let att = att.broadcast_add(&causal_mask(t, x.device())?)?;
        let att = softmax_last_dim(&att)?;

        // Weighted sum of values, then merge heads back to (b, t, c).
        let y = att.matmul(&v)?; // (b, n_head, t, head_dim)
        let y = y.transpose(1, 2)?.contiguous()?.reshape((b, t, c))?;
        self.proj.forward(&y)
    }
}

/// An upper-triangular matrix of -inf above the diagonal, 0 elsewhere. Added to
/// the attention logits before softmax so masked positions get ~0 weight.
fn causal_mask(t: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..t)
        .flat_map(|i| (0..t).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
        .collect();
    Tensor::from_vec(mask, (t, t), device)
}

/// Position-wise feed-forward network (the "MLP" block of a transformer).
struct Mlp {
    fc: Linear,
    proj: Linear,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let fc = linear(cfg.n_embd, 4 * cfg.n_embd, vb.pp("fc"))?;
        let proj = linear(4 * cfg.n_embd, cfg.n_embd, vb.pp("proj"))?;
        Ok(Self { fc, proj })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc.forward(x)?.gelu()?;
        self.proj.forward(&x)
    }
}

/// One pre-norm transformer block: residual attention then residual MLP.
struct Block {
    ln1: LayerNorm,
    attn: CausalSelfAttention,
    ln2: LayerNorm,
    mlp: Mlp,
}

impl Block {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ln1: layer_norm(cfg.n_embd, 1e-5, vb.pp("ln1"))?,
            attn: CausalSelfAttention::new(cfg, vb.pp("attn"))?,
            ln2: layer_norm(cfg.n_embd, 1e-5, vb.pp("ln2"))?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (x + self.attn.forward(&self.ln1.forward(x)?)?)?;
        &x + self.mlp.forward(&self.ln2.forward(&x)?)?
    }
}

/// A minimal GPT-style decoder-only transformer for character-level LM.
pub struct GptModel {
    tok_emb: Embedding,
    pos_emb: Embedding,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    head: Linear,
    block_size: usize,
}

impl GptModel {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let tok_emb = embedding(cfg.vocab_size, cfg.n_embd, vb.pp("tok_emb"))?;
        let pos_emb = embedding(cfg.block_size, cfg.n_embd, vb.pp("pos_emb"))?;
        let blocks = (0..cfg.n_layer)
            .map(|i| Block::new(cfg, vb.pp(format!("block_{i}"))))
            .collect::<Result<Vec<_>>>()?;
        let ln_f = layer_norm(cfg.n_embd, 1e-5, vb.pp("ln_f"))?;
        let head = linear(cfg.n_embd, cfg.vocab_size, vb.pp("head"))?;
        Ok(Self {
            tok_emb,
            pos_emb,
            blocks,
            ln_f,
            head,
            block_size: cfg.block_size,
        })
    }

    /// `idx` is `(batch, seq_len)` of token ids; returns logits of shape
    /// `(batch, seq_len, vocab_size)`.
    pub fn forward(&self, idx: &Tensor) -> Result<Tensor> {
        let (_b, t) = idx.dims2()?;
        assert!(
            t <= self.block_size,
            "sequence length {t} exceeds block_size {}",
            self.block_size
        );

        let tok = self.tok_emb.forward(idx)?; // (b, t, n_embd)
        let pos_ids = Tensor::arange(0u32, t as u32, idx.device())?;
        let pos = self.pos_emb.forward(&pos_ids)?; // (t, n_embd)
        let mut x = tok.broadcast_add(&pos)?;

        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        let x = self.ln_f.forward(&x)?;
        self.head.forward(&x) // (b, t, vocab_size)
    }
}
