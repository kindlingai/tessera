//! DiffusionGemma: one denoising pass over a block. Sketch.

use crate::core::*;
use crate::library::{Mlp, mlp};

pub struct Config {
    pub layers: usize,
    /// Every n-th layer is global; the rest use a window.
    pub global_every: usize,
    pub window: usize,
    pub block: usize,
    pub kv_heads: usize,
    pub softcap: f32,
    pub embed_scale: f32,
}

#[derive(Weights)]
pub struct Attn {
    #[split(heads)]
    pub q: Linear,
    #[split(heads)]
    pub kv: Linear,
    pub q_norm: Norm,
    pub k_norm: Norm,
    #[split(rows)]
    pub o: Linear,
}

/// Norms before and after each sub-block.
#[derive(Weights)]
pub struct Block {
    pub pre_attn: Norm,
    pub attn: Attn,
    pub post_attn: Norm,
    pub pre_mlp: Norm,
    pub mlp: Mlp<GeGlu>,
    pub post_mlp: Norm,
}

#[derive(Weights)]
pub struct DGemma {
    #[split(vocab)]
    pub embed: Embed,
    pub blocks: Vec<Block>,
    pub norm: Norm,
}

/// Per-row facts the planner turns into plans.
pub struct RowState {
    pub step: Tensor,
    pub cap: Tensor,
    pub read_only: Tensor,
    /// Grammar or slot constraint: allowed vocabulary columns.
    pub allowed: Plan,
}

/// One pass; the host repeats it until the block unmasks.
pub fn denoise(
    m: &mut Model,
    c: &Config,
    w: &DGemma,
    rs: &RowState,
    canvas: &Tensor,
    sc: &mut Carried<Tensor>,
) -> Tensor {
    let soft = sc.read_or_zero();
    let mut x = m
        .embed(canvas, &w.embed)
        .reduce()
        .scale(c.embed_scale)
        .add(&soft);
    for (l, b) in w.blocks.iter().enumerate() {
        let mask = match l % c.global_every == c.global_every - 1 {
            true => Mask::BlockCausal { block: c.block },
            false => Mask::Window { size: c.window },
        };
        let a = attention(m, c, &b.attn, Layer(l), mask, &x.rms_norm(&b.pre_attn));
        x = x.add(&a.reduce().rms_norm(&b.post_attn));
        let f = mlp(&b.mlp, &x.rms_norm(&b.pre_mlp));
        x = x.add(&f.reduce().rms_norm(&b.post_mlp));
    }
    let logits = x
        .rms_norm(&w.norm)
        .matmul_cols(&w.embed, &rs.allowed)
        .softcap(c.softcap);
    self_condition(m, w, rs, &logits, sc);
    logits
}

/// Feeds the next pass; live only for rows that continue.
fn self_condition(
    m: &mut Model,
    w: &DGemma,
    rs: &RowState,
    logits: &Tensor,
    sc: &mut Carried<Tensor>,
) {
    let last = rs.step.add_one().compare(Cmp::Gt, &rs.cap.sub_one());
    let finishing = last.and(&rs.read_only);
    let live = m.filter(&finishing.not());
    sc.write_rows(&live, logits.softmax().matmul_rows(&w.embed, &rs.allowed));
}

fn attention(m: &mut Model, c: &Config, w: &Attn, l: Layer, mask: Mask, h: &Tensor) -> Tensor {
    let q = h.matmul(&w.q).rms_norm(&w.q_norm).rope_all();
    let kv = h.matmul(&w.kv).norm_keys(&w.k_norm).rope_keys_all();
    let kind = AttnKind::Gqa {
        kv_heads: c.kv_heads,
    };
    // Earlier blocks are cached; the current block is recomputed.
    let cache = m.paged_kv(l, kind, Sync::Eager);
    m.attend(kind, mask, &q, &kv, &cache, None).matmul(&w.o)
}

/// Unmask the most confident positions; reads return scores.
pub fn unmask(m: &mut Model, logits: &Tensor, count: usize) -> Tensor {
    let confidence = logits.softmax().max_per_row();
    let pick = m.top_k(&confidence, count);
    m.sample_rows(logits, &pick.ids, Sampler::Gumbel)
}
