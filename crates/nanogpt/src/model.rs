//! Tiny GPT config, weight init and GPU forward pass.

use crate::tensor as t;
use ops::Error;

pub const EPS: f32 = 1e-5;

#[derive(Debug, Clone, Copy)]
pub struct GptConfig {
    pub vocab: usize,
    pub block: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
}

impl GptConfig {
    /// nanoGPT-shaped and small.
    pub fn tiny(vocab: usize) -> Self {
        Self {
            vocab,
            block: 64,
            n_layer: 2,
            n_head: 4,
            n_embd: 64,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }

    pub fn mlp(&self) -> usize {
        4 * self.n_embd
    }
}

/// Seeded xorshift, so a seed always gives the same weights.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Standard normal via Box-Muller.
    pub fn normal(&mut self) -> f32 {
        let denom = (1u64 << 53) as f64;
        let u1 = ((self.next_u64() >> 11) as f64 + 1.0) / (denom + 1.0);
        let u2 = ((self.next_u64() >> 11) as f64) / denom;
        ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
    }
}

pub struct LayerWeights {
    pub ln1: Vec<f32>,
    pub wqkv: Vec<f32>,
    pub wproj: Vec<f32>,
    pub ln2: Vec<f32>,
    pub w1: Vec<f32>,
    pub w2: Vec<f32>,
}

pub struct Weights {
    pub tok_emb: Vec<f32>,
    pub pos_emb: Vec<f32>,
    pub layers: Vec<LayerWeights>,
    pub ln_f: Vec<f32>,
    pub lm_head: Vec<f32>,
}

fn rand_vec(rng: &mut Rng, len: usize, std: f32) -> Vec<f32> {
    (0..len).map(|_| rng.normal() * std).collect()
}

impl Weights {
    pub fn random(cfg: &GptConfig, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let c = cfg.n_embd;
        let layers = (0..cfg.n_layer)
            .map(|_| LayerWeights {
                ln1: vec![1.0; c],
                wqkv: rand_vec(&mut rng, 3 * c * c, 0.02),
                wproj: rand_vec(&mut rng, c * c, 0.02),
                ln2: vec![1.0; c],
                w1: rand_vec(&mut rng, cfg.mlp() * c, 0.02),
                w2: rand_vec(&mut rng, c * cfg.mlp(), 0.02),
            })
            .collect();
        Self {
            tok_emb: rand_vec(&mut rng, cfg.vocab * c, 0.02),
            pos_emb: rand_vec(&mut rng, cfg.block * c, 0.02),
            layers,
            ln_f: vec![1.0; c],
            lm_head: rand_vec(&mut rng, cfg.vocab * c, 0.02),
        }
    }
}

/// Logits for every position, on the GPU ops.
pub fn forward(
    w: &Weights,
    cfg: &GptConfig,
    tokens: &[u32],
    seq: usize,
) -> Result<Vec<f32>, Error> {
    let c = cfg.n_embd;
    let heads = cfg.n_head;
    let hd = cfg.head_dim();
    assert!(seq > 0 && seq <= cfg.block, "seq must be in 1..=block");
    assert_eq!(
        tokens.len() % seq,
        0,
        "tokens must be a whole number of sequences"
    );
    let batch = tokens.len() / seq;
    let m = batch * seq;

    let pos_ids: Vec<u32> = (0..batch)
        .flat_map(|_| (0..seq).map(|x| x as u32))
        .collect();
    let tok = t::gather_rows(&w.tok_emb, tokens, c)?;
    let pos = t::gather_rows(&w.pos_emb, &pos_ids, c)?;
    let mut x = t::vec_add(&tok, &pos)?;

    let scale = 1.0 / (hd as f32).sqrt();

    for lw in &w.layers {
        let h1 = t::rms_norm(&x, &lw.ln1, m, c, EPS)?;
        let qkv = t::linear(&h1, &lw.wqkv, m, 3 * c, c)?;

        let mut attn = vec![0.0f32; m * c];
        for b in 0..batch {
            for h in 0..heads {
                let base = b * seq * 3 * c + h * hd;
                let q = &qkv[base..];
                let k = &qkv[base + c..];
                let mut scores = t::gemm(
                    q,
                    k,
                    vec![0.0; seq * seq],
                    seq,
                    seq,
                    hd,
                    3 * c,
                    3 * c,
                    seq,
                    false,
                    true,
                    0.0,
                )?;
                for i in 0..seq {
                    for j in 0..seq {
                        let idx = i * seq + j;
                        scores[idx] = if j > i { -1e30 } else { scores[idx] * scale };
                    }
                }
                let p = t::softmax(&scores, seq, seq)?;
                let v = &qkv[base + 2 * c..];
                let head_out = t::gemm(
                    &p,
                    v,
                    vec![0.0; seq * hd],
                    seq,
                    hd,
                    seq,
                    seq,
                    3 * c,
                    hd,
                    false,
                    false,
                    0.0,
                )?;
                for row in 0..seq {
                    let dst = (b * seq + row) * c + h * hd;
                    attn[dst..dst + hd].copy_from_slice(&head_out[row * hd..(row + 1) * hd]);
                }
            }
        }

        let proj = t::linear(&attn, &lw.wproj, m, c, c)?;
        let x_attn = t::vec_add(&x, &proj)?;
        let h2 = t::rms_norm(&x_attn, &lw.ln2, m, c, EPS)?;
        let g = t::linear(&h2, &lw.w1, m, cfg.mlp(), c)?;
        let f = t::gelu(&g)?;
        let down = t::linear(&f, &lw.w2, m, c, cfg.mlp())?;
        x = t::vec_add(&x_attn, &down)?;
    }

    let hf = t::rms_norm(&x, &w.ln_f, m, c, EPS)?;
    t::linear(&hf, &w.lm_head, m, cfg.vocab, c)
}
