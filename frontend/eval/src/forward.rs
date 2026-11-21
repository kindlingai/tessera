//! Qwen3.5 forward passes: each prompt's next-token picks.

use crate::Failure;
use serde::Serialize;
use std::path::Path;
use std::time::Instant;
use tessera_bundle::Overrides;
use tessera_bundle::qwen35::open;
use tessera_bundle::tokenize::hf::Hf;
use tessera_engine::{DimName, Model, Recipe, Tensor};

/// Picks reported after each prompt's last row.
const TOP: usize = 5;

#[derive(Serialize)]
struct Pick {
    id: u32,
    text: String,
    p: f32,
}

#[derive(Serialize)]
struct Pass {
    tokens: usize,
    /// The first pass, which compiles a new bucket's kernels.
    first_ms: f64,
    /// The second pass, with every row's logits read back.
    again_ms: f64,
    top: Vec<Pick>,
}

#[derive(Serialize)]
pub struct Picks {
    device: String,
    load_s: f64,
    passes: Vec<Pass>,
}

/// Each text in `texts` through `dir`'s text model, twice.
pub fn forward(dir: &Path, texts: &Path) -> Result<Picks, Failure> {
    let settings = tessera_frontend_settings::read();
    let tokenizer = Hf::open(&dir.join("tokenizer.json"))?;
    let texts: Vec<String> = serde_json::from_str(&std::fs::read_to_string(texts)?)?;
    let mut prompts = Vec::with_capacity(texts.len());
    for text in &texts {
        let ids = tokenizer.encode(text)?;
        if ids.is_empty() {
            return Err(format!("{text:?} is empty").into());
        }
        prompts.push(ids);
    }
    // The longest prompt bounds the graph and caps the buckets.
    let longest = prompts.iter().map(Vec::len).max().unwrap_or(1) as u64;
    let start = Instant::now();
    let text = open(dir, [1, longest], &Overrides::default())?;
    let recipe = Recipe {
        pins: text.pins.clone(),
        ..Recipe::default()
    };
    let model = Model::open(text.graph.clone(), &recipe, &settings)?;
    if model.interpreted() {
        return Err("the graph has no plan on this device".into());
    }
    let mut model = model.bucketed(&[DimName::Context]);
    text.each(|at, w| Ok(model.load_one(at, &w)?))?;
    let load_s = start.elapsed().as_secs_f64();
    let mut passes = Vec::with_capacity(prompts.len());
    for ids in &prompts {
        passes.push(pass(&mut model, &tokenizer, ids)?);
    }
    Ok(Picks {
        device: model.device(),
        load_s,
        passes,
    })
}

/// `ids` forward twice, timed, then the last row's picks.
fn pass(model: &mut Model, tokenizer: &Hf, ids: &[u32]) -> Result<Pass, Failure> {
    let shape = vec![1, ids.len()];
    let positions = (0..ids.len() as u32).collect();
    let inputs = [
        Tensor::u32(shape.clone(), ids.to_vec()),
        Tensor::u32(shape, positions),
    ];
    let mut ms = [0.0; 2];
    let mut out = Vec::new();
    for at in &mut ms {
        let start = Instant::now();
        out = model.run(&inputs)?;
        *at = start.elapsed().as_secs_f64() * 1e3;
    }
    let logits = out[0].floats()?;
    let vocab = logits.len() / ids.len();
    let last = &logits[logits.len() - vocab..];
    let most = last.iter().fold(f32::MIN, |m, v| m.max(*v));
    let total: f32 = last.iter().map(|v| (v - most).exp()).sum();
    let mut order: Vec<usize> = (0..vocab).collect();
    order.sort_by(|a, b| last[*b].total_cmp(&last[*a]));
    let mut top = Vec::with_capacity(TOP);
    for &id in &order[..TOP.min(vocab)] {
        top.push(Pick {
            id: id as u32,
            text: tokenizer.decode(&[id as u32]),
            p: (last[id] - most).exp() / total,
        });
    }
    Ok(Pass {
        tokens: ids.len(),
        first_ms: ms[0],
        again_ms: ms[1],
        top,
    })
}
