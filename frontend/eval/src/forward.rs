//! One Qwen3.5 forward pass: the next token's picks.

use crate::Failure;
use serde::Serialize;
use std::path::Path;
use std::time::Instant;
use tessera_bundle::Overrides;
use tessera_bundle::qwen35::open;
use tessera_bundle::tokenize::hf::Hf;
use tessera_engine::{Model, Recipe, Tensor};

/// Picks reported after the prompt's last row.
const TOP: usize = 5;

#[derive(Serialize)]
struct Pick {
    id: u32,
    text: String,
    p: f32,
}

#[derive(Serialize)]
pub struct Picks {
    device: String,
    load_s: f64,
    prompt_tokens: usize,
    /// The second pass, with every row's logits read back.
    forward_ms: f64,
    top: Vec<Pick>,
}

/// `prompt` through `dir`'s text model, warmed, then timed.
pub fn forward(dir: &Path, prompt: &str) -> Result<Picks, Failure> {
    let settings = tessera_frontend_settings::read();
    let tokenizer = Hf::open(&dir.join("tokenizer.json"))?;
    let ids = tokenizer.encode(prompt)?;
    if ids.is_empty() {
        return Err("the prompt is empty".into());
    }
    let start = Instant::now();
    let text = open(dir, [1, ids.len() as u64], &Overrides::default())?;
    let recipe = Recipe {
        pins: text.pins.clone(),
        ..Recipe::default()
    };
    let mut model = Model::open(text.graph.clone(), &recipe, &settings)?;
    if model.interpreted() {
        return Err("the graph has no plan on this device".into());
    }
    text.each(|at, w| Ok(model.load_one(at, &w)?))?;
    let load_s = start.elapsed().as_secs_f64();
    let shape = vec![1, ids.len()];
    let positions = (0..ids.len() as u32).collect();
    let inputs = [
        Tensor::u32(shape.clone(), ids.clone()),
        Tensor::u32(shape, positions),
    ];
    let mut out = Vec::new();
    let mut forward_ms = 0.0;
    for _ in 0..2 {
        let start = Instant::now();
        out = model.run(&inputs)?;
        forward_ms = start.elapsed().as_secs_f64() * 1e3;
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
    Ok(Picks {
        device: model.device(),
        load_s,
        prompt_tokens: ids.len(),
        forward_ms,
        top,
    })
}
