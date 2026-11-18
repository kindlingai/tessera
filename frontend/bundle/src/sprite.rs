//! The sprite denoiser and its CLIP text tower.

use crate::sprite_fold::Upstream;
use crate::tokenize::hf::Hf;
use sensai_checkpoint::{File, Offer, Torch};
use sensai_engine::{Graph, PathSeg, Tensor, Weight, bind, clip, sprite};
use serde::Deserialize;
use std::error::Error;
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

pub struct Sprite {
    pub graph: Graph,
    pub weights: Vec<Weight>,
    /// RGB per palette id. Id 0 is transparent.
    pub palette: Vec<[u8; 3]>,
}

pub struct Clip {
    pub graph: Graph,
    pub weights: Vec<Weight>,
    pub tokenizer: Hf,
}

fn dotted(path: &[PathSeg]) -> String {
    let seg = |p: &PathSeg| match p {
        PathSeg::Field(f) => f.clone(),
        PathSeg::Index(i) => i.to_string(),
    };
    path.iter().map(seg).collect::<Vec<_>>().join(".")
}

/// Weights by name, each cut to its first `len(name)` values.
fn weights(
    graph: &Graph,
    get: impl Fn(&str) -> Option<Vec<f32>>,
    name: impl Fn(&[PathSeg]) -> String,
    len: impl Fn(&str) -> Option<usize>,
) -> Result<Vec<Weight>> {
    let find = |p: &[PathSeg]| {
        let name = name(p);
        let mut v = get(&name)?;
        if let Some(n) = len(&name) {
            v.truncate(n);
        }
        Some(Tensor::f32(vec![v.len()], v))
    };
    Ok(bind(graph, find)?.into_iter().map(Weight::Floats).collect())
}

/// The denoiser from any offer of its weight names.
pub fn denoiser(offer: &dyn Offer, config: &sprite::Config) -> Result<Sprite> {
    let graph = sprite::program(config);
    let weights = weights(&graph, |n| offer.floats(n).ok(), dotted, |_| None)?;
    let palette = (offer.floats("palette")?.chunks(3))
        .map(|c| [c[0] as u8, c[1] as u8, c[2] as u8])
        .collect();
    Ok(Sprite {
        graph,
        weights,
        palette,
    })
}

/// A converted `.safetensors` file, or the upstream `.pt`.
pub fn denoiser_offer(path: &Path, config: &sprite::Config) -> Result<Box<dyn Offer>> {
    Ok(match path.extension().is_some_and(|x| x == "pt") {
        true => Box::new(Upstream::new(Torch::open(path)?, config)),
        false => Box::new(File::open(path)?),
    })
}

/// The denoiser file in `dir`: converted, else the upstream `.pt`.
pub fn denoiser_file(dir: &Path) -> Option<PathBuf> {
    let converted = dir.join("sprite.safetensors");
    if converted.is_file() {
        return Some(converted);
    }
    let mut pts: Vec<_> = (std::fs::read_dir(dir).ok()?.flatten())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "pt"))
        .collect();
    pts.sort();
    pts.into_iter().next()
}

/// The upstream `model_config.json`.
#[derive(Deserialize)]
struct UpstreamConfig {
    vocab_size: u64,
    grid_size: u64,
    dim: u64,
    n_heads: u64,
    depth_in: u64,
    depth_core: u64,
    depth_out: u64,
    patch: u64,
    mlp_ratio: f64,
    n_registers: u64,
    text_dim: u64,
    text_len: u64,
    rope_theta: f64,
    color_freqs: u64,
    conv_ffn: bool,
    self_cond: bool,
}

/// `dir/model_config.json` as a config, else tessera-small's.
pub fn sprite_config(dir: &Path, seqs: u64) -> Result<sprite::Config> {
    let path = dir.join("model_config.json");
    if !path.is_file() {
        return Ok(sprite::Config::small(seqs));
    }
    let c: UpstreamConfig = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    let fixed = [
        ("patch", c.patch as f64, 2.0),
        (
            "n_registers",
            c.n_registers as f64,
            sprite::REGISTERS as f64,
        ),
        ("text_len", c.text_len as f64, sprite::TEXT as f64),
        ("head width", (c.dim / c.n_heads.max(1)) as f64, 64.0),
        ("rope_theta", c.rope_theta, 100.0),
        ("color_freqs", c.color_freqs as f64, 6.0),
        ("conv_ffn", c.conv_ffn as u8 as f64, 1.0),
        ("self_cond", c.self_cond as u8 as f64, 1.0),
    ];
    for (name, got, want) in fixed {
        if got != want {
            let at = path.display();
            return Err(format!("{at}: {name} is {got}, the graph takes {want}").into());
        }
    }
    Ok(sprite::Config {
        grid: c.grid_size,
        palette: c.vocab_size,
        hidden: c.dim,
        heads: c.n_heads,
        ffn: (c.dim as f64 * c.mlp_ratio) as u64,
        text_dim: c.text_dim,
        depth: [c.depth_in, c.depth_core, c.depth_out],
        max_seqs: seqs,
    })
}

/// CLIP's weight file in `dir`, safetensors first.
fn clip_weights(dir: &Path) -> Option<PathBuf> {
    let found = ["model.safetensors", "pytorch_model.bin"].map(|n| dir.join(n));
    found.into_iter().find(|p| p.is_file())
}

/// A Hugging Face name for a CLIP text weight path.
fn hf_name(path: &[PathSeg]) -> String {
    let dotted = dotted(path);
    let mut parts: Vec<&str> = dotted.split('.').collect();
    fn field(f: &str) -> &str {
        match f {
            "ln1" => "layer_norm1",
            "ln2" => "layer_norm2",
            "q" => "self_attn.q_proj",
            "k" => "self_attn.k_proj",
            "v" => "self_attn.v_proj",
            "o" => "self_attn.out_proj",
            "fc1" => "mlp.fc1",
            "fc2" => "mlp.fc2",
            other => other,
        }
    }
    let tail = match parts.as_slice() {
        ["embed"] => "embeddings.token_embedding.weight".into(),
        ["pos"] => "embeddings.position_embedding.weight".into(),
        ["norm", end] => format!("final_layer_norm.{end}"),
        ["layers", i, f, end] => format!("encoder.layers.{i}.{}.{end}", field(f)),
        _ => {
            parts.insert(0, "?");
            parts.join(".")
        }
    };
    format!("text_model.{tail}")
}

/// CLIP's text tower in `dir`, cut to `context` tokens.
pub fn text_tower(dir: &Path, seqs: u64, context: u64) -> Result<Clip> {
    let config = clip::Config::base(seqs, context);
    let graph = clip::program(&config);
    let path = clip_weights(dir).ok_or_else(|| format!("no CLIP weights in {}", dir.display()))?;
    let file: Box<dyn Offer> = match path.extension().is_some_and(|x| x == "bin") {
        true => Box::new(Torch::open(&path)?),
        false => Box::new(File::open(&path)?),
    };
    let table = (context * config.stack.hidden) as usize;
    let len = |name: &str| name.ends_with("position_embedding.weight").then_some(table);
    let weights = weights(&graph, |n| file.floats(n).ok(), hf_name, len)?;
    let tokenizer = Hf::open(&dir.join("tokenizer.json"))?;
    Ok(Clip {
        graph,
        weights,
        tokenizer,
    })
}
