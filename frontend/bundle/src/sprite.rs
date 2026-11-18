//! The sprite denoiser and its CLIP text tower.

use crate::sprite_fold::Upstream;
use crate::tokenize::hf::Hf;
use sensai_checkpoint::{File, Offer, Torch};
use sensai_engine::{Graph, PathSeg, Tensor, Weight, bind, clip, sprite};
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
pub fn denoiser(offer: &dyn Offer, seqs: u64) -> Result<Sprite> {
    let config = sprite::Config::small(seqs);
    let graph = sprite::program(&config);
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

/// A converted `.safetensors` file, or the upstream `.pt` folded on read.
pub fn denoiser_offer(path: &Path) -> Result<Box<dyn Offer>> {
    Ok(match path.extension().is_some_and(|x| x == "pt") {
        true => Box::new(Upstream::new(Torch::open(path)?, sprite::Config::small(1))),
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

const CLIP_REPO: &str = "openai/clip-vit-base-patch32";

/// The newest cached snapshot of `repo` that `fits`.
pub fn hub_snapshot(repo: &str, fits: impl Fn(&Path) -> bool) -> Option<PathBuf> {
    let env = |k: &str| std::env::var_os(k).map(PathBuf::from);
    let hub = (env("HF_HUB_CACHE"))
        .or_else(|| env("HF_HOME").map(|h| h.join("hub")))
        .or_else(|| Some(std::env::home_dir()?.join(".cache/huggingface/hub")))?;
    let dir = hub.join(format!("models--{}", repo.replace('/', "--")));
    let mut snapshots: Vec<_> = (std::fs::read_dir(dir.join("snapshots")).ok()?.flatten())
        .map(|e| e.path())
        .filter(|p| fits(p))
        .collect();
    snapshots.sort();
    snapshots.pop()
}

/// CLIP in `dir/clip`, else a Hugging Face cache snapshot.
pub fn clip_dir(dir: &Path) -> Option<PathBuf> {
    let own = dir.join("clip");
    if own.is_dir() {
        return Some(own);
    }
    hub_snapshot(CLIP_REPO, |p| {
        p.join("tokenizer.json").is_file() && clip_weights(p).is_some()
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
