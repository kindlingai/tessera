//! DiffusionGemma: denoising passes over a block, as a loop. Sketch.

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
    pub max_steps: usize,
    pub unmask_per_step: usize,
}

#[derive(Weights)]
pub struct Attn {
    #[shape(Hidden, Heads, HeadDim)]
    pub q: Linear,
    #[shape(Hidden, KvHeads, HeadDim)]
    pub kv: Linear,
    pub q_norm: Norm,
    pub k_norm: Norm,
    #[shape(Heads, HeadDim, Hidden)]
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
    #[shape(Vocab, Hidden)]
    pub embed: Embed,
    pub blocks: Vec<Block>,
    pub norm: Norm,
}

/// Per-row request facts; the step count comes from the loop.
pub struct RowState {
    pub cap: Tensor,
    pub read_only: Tensor,
    /// Grammar or slot constraint: allowed vocabulary columns.
    pub allowed: Plan,
}

/// Denoises until rows unmask, reach their cap, or the bound.
pub fn generate(m: &mut Model, c: &Config, w: &DGemma, rs: &RowState, canvas: Tensor) -> Tensor {
    // A read-only row at its cap feeds no later pass.
    let last = |_: &_, it: &Iter| at_cap(it, rs).and(&rs.read_only);
    let until = Until {
        max: c.max_steps,
        last: Box::new(last),
    };
    let start = (canvas, Carried::empty());
    let (canvas, _) = m.repeat(until, start, |m, (canvas, mut sc), it| {
        let logits = denoise(m, c, w, rs, &canvas, &mut sc, it);
        let canvas = unmask(m, &logits, &canvas, c.unmask_per_step);
        let more = canvas.has_masked().and(&at_cap(it, rs).not());
        Next {
            carry: (canvas, sc),
            more,
        }
    });
    canvas
}

fn at_cap(it: &Iter, rs: &RowState) -> Tensor {
    it.step.add_one().compare(Cmp::Gt, &rs.cap.sub_one())
}

/// One pass over the block.
pub fn denoise(
    m: &mut Model,
    c: &Config,
    w: &DGemma,
    rs: &RowState,
    canvas: &Tensor,
    sc: &mut Carried<Tensor>,
    it: &Iter,
) -> Tensor {
    let soft = sc.read_or_zero();
    let mut x = m
        .embed(canvas, &w.embed)
        .reduce()
        .scale(c.embed_scale)
        .add(&soft);
    for (i, b) in w.blocks.iter().enumerate() {
        let l = Layer(i);
        let mask = match i % c.global_every == c.global_every - 1 {
            true => Mask::BlockCausal { block: c.block },
            false => Mask::Window { size: c.window },
        };
        let a = attention(m, c, &b.attn, l, mask, &x.rms_norm(&b.pre_attn));
        // The writer is what joins the residual: after its post-norm.
        let a = a.reduce().rms_norm(&b.post_attn);
        x = x.add(&m.site(Site::Writer(l, Sublayer::Attn), a));
        let f = mlp(&b.mlp, &x.rms_norm(&b.pre_mlp))
            .reduce()
            .rms_norm(&b.post_mlp);
        x = x.add(&m.site(Site::Writer(l, Sublayer::Mlp), f));
        x = m.site(Site::Residual(l), x);
    }
    let logits = x
        .rms_norm(&w.norm)
        .matmul_cols(&w.embed, &rs.allowed)
        .softcap(c.softcap);
    self_condition(m, w, rs, &logits, sc, it);
    logits
}

/// Feeds the next pass; a row's last pass skips it.
fn self_condition(
    m: &mut Model,
    w: &DGemma,
    rs: &RowState,
    logits: &Tensor,
    sc: &mut Carried<Tensor>,
    it: &Iter,
) {
    let live = m.filter(&it.last.not());
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

/// Fills the most confident masked positions of the canvas.
pub fn unmask(m: &mut Model, logits: &Tensor, canvas: &Tensor, count: usize) -> Tensor {
    let confidence = logits.softmax().max_per_row().masked_only(canvas);
    let pick = m.top_k(&confidence, count);
    let tokens = m.sample_rows(logits, &pick.ids, Sampler::Gumbel);
    canvas.place(&pick.ids, &tokens)
}
