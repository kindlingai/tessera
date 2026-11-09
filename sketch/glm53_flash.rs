//! GLM-5.3-Flash: KDA linear attention beside full attention. Sketch.

use crate::core::*;
use crate::delta::{Delta, DeltaShape, delta};
use crate::glm53::{AttnShape, Indexer, Mla, attention};
use crate::library::{Moe, moe};

pub struct Config {
    pub layers: usize,
    /// Which layers use full attention; the rest use KDA.
    pub full: LayerSet,
    pub top_k: usize,
    pub attn: AttnShape,
    /// Kind is `Kda`.
    pub kda: DeltaShape,
}

pub enum Mixer {
    Full(Mla, Indexer),
    Kda(Delta),
}

#[derive(Weights)]
pub struct Block {
    pub mixer: Mixer,
    pub mlp_norm: Norm,
    pub moe: Moe<SwiGlu>,
}

#[derive(Weights)]
pub struct Flash {
    #[split(vocab)]
    pub embed: Embed,
    pub blocks: Vec<Block>,
    pub norm: Norm,
    #[split(vocab)]
    pub head: Linear,
}

pub fn forward(m: &mut Model, c: &Config, w: &Flash, tokens: &Tensor) -> Tensor {
    let mut x = m.embed(tokens, &w.embed).reduce();
    for (i, b) in w.blocks.iter().enumerate() {
        let l = Layer(i);
        let a = match &b.mixer {
            Mixer::Full(f, ix) => attention(m, &c.attn, f, Some(ix), l, &x),
            Mixer::Kda(k) => delta(m, &c.kda, k, l, &x),
        };
        x = x.add(&m.site(Site::Writer(l, Sublayer::Attn), a.reduce()));
        let f = moe(m, &b.moe, &x.rms_norm(&b.mlp_norm), c.top_k, l);
        x = x.add(&m.site(Site::Writer(l, Sublayer::Mlp), f.reduce()));
        x = m.site(Site::Residual(l), x);
    }
    let h = m.site(Site::PreHead, x.rms_norm(&w.norm));
    h.matmul(&w.head)
}
