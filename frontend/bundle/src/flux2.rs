//! A FLUX.2 transformer from a diffusers directory.

use crate::names::repo;
use crate::pull;
use sensai_checkpoint::File;
use sensai_engine::flux2::{Config, program};
use sensai_engine::{Graph, Pins, Weight};
use serde::Deserialize;
use std::error::Error;
use std::path::Path;

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

/// Fields the graph needs from `transformer/config.json`.
#[derive(Deserialize)]
struct Sizes {
    num_attention_heads: u64,
    attention_head_dim: u64,
    joint_attention_dim: u64,
    num_layers: u64,
    num_single_layers: u64,
    mlp_ratio: f32,
    rope_theta: f32,
    in_channels: u64,
}

pub struct Transformer {
    pub graph: Graph,
    pub weights: Vec<Weight>,
    pub pins: Pins,
}

/// `dir/transformer`, with `text` rows and up to `rows` image rows.
pub fn transformer(dir: &Path, [seqs, text, rows]: [u64; 3]) -> Result<Transformer> {
    let dir = dir.join("transformer");
    let s: Sizes = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
    if s.in_channels != sensai_engine::flux2::LATENT {
        return Err(format!("{} latent channels", s.in_channels).into());
    }
    let hidden = s.num_attention_heads * s.attention_head_dim;
    let config = Config {
        hidden,
        heads: s.num_attention_heads,
        ffn: (hidden as f32 * s.mlp_ratio) as u64,
        text_dim: s.joint_attention_dim,
        double: s.num_layers,
        single: s.num_single_layers,
        theta: s.rope_theta,
        ..Config::klein(seqs, text, rows)
    };
    let graph = program(&config);
    let file = File::open(&dir.join("diffusion_pytorch_model.safetensors"))?;
    let rules = repo("flux2").unwrap_or_default();
    let (weights, pins) = pull(&graph, &file, 0, &[&rules])?;
    Ok(Transformer {
        graph,
        weights,
        pins,
    })
}
