//! Embedding speed over texts or gr8-compatible benchmark scenarios.

use crate::Failure;
use serde::Serialize;
use std::io::Write;
use std::path::Path;
use std::time::Instant;
use tessera_tasks::Embedder;
use tessera_tasks::api::{EmbedInput, EmbedMessage, EmbedOptions, EmbedPart, Image};
use tessera_tasks::{png, vision};

#[derive(Serialize)]
pub struct Speed {
    device: String,
    load_s: f64,
    texts: usize,
    tokens: usize,
    texts_per_s: f64,
}

#[derive(Serialize)]
pub struct Scenario {
    scenario: String,
    batch: usize,
    total_tokens: usize,
    n_images: usize,
    mean_ms: f64,
    p50_ms: f64,
    min_ms: f64,
    tok_per_s: f64,
    emb_per_s: f64,
    emb0_l2: f64,
    emb0_sum: f64,
}

#[derive(Serialize)]
pub struct Bench {
    device: String,
    load_s: f64,
    scenarios: Vec<Scenario>,
}

/// Embeds each text with the release in `model`.
pub fn embed(
    model: &Path,
    texts: &Path,
    vectors: Option<&Path>,
    dimensions: Option<usize>,
) -> Result<Speed, Failure> {
    let settings = tessera_frontend_settings::read();
    let start = Instant::now();
    let mut embedder = Embedder::open(model, &settings, &Default::default(), &Default::default())?;
    let load_s = start.elapsed().as_secs_f64();
    let texts: Vec<String> = serde_json::from_str(&std::fs::read_to_string(texts)?)?;
    let inputs: Vec<EmbedInput> = texts.iter().cloned().map(EmbedInput::Text).collect();
    let options = EmbedOptions {
        dimensions,
        normalize: true,
    };
    let start = Instant::now();
    let got = embedder.embed_inputs(&inputs, options)?;
    let seconds = start.elapsed().as_secs_f64();
    if let Some(path) = vectors {
        let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
        for x in got.vectors.iter().flatten() {
            file.write_all(&x.to_le_bytes())?;
        }
        file.flush()?;
    }
    Ok(Speed {
        device: embedder.device(),
        load_s,
        texts: texts.len(),
        tokens: got.tokens,
        texts_per_s: texts.len() as f64 / seconds,
    })
}

fn synthetic_tokens(batch: usize, len: usize) -> Vec<EmbedInput> {
    let mut out = Vec::with_capacity(batch);
    for b in 0..batch {
        let mut ids = Vec::with_capacity(len);
        let mut s = 0x9e37_79b9u32 ^ ((b as u32).wrapping_mul(2_654_435_761) + len as u32);
        for _ in 0..len {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ids.push(100 + ((s >> 8) % 30_000));
        }
        out.push(EmbedInput::Tokens(ids));
    }
    out
}

fn synthetic_mm(batch: usize, side: usize) -> Vec<EmbedInput> {
    let mut out = Vec::with_capacity(batch);
    for b in 0..batch {
        let rgb = vision::synthetic_rgb(side, side, b + 1);
        let mut rgba = Vec::with_capacity(side * side * 4);
        for px in rgb.pixels.as_chunks::<3>().0 {
            rgba.extend_from_slice(&[px[0], px[1], px[2], 255]);
        }
        let png_bytes = png::encode(&Image {
            width: side,
            height: side,
            rgba,
        });
        out.push(EmbedInput::Messages {
            messages: vec![EmbedMessage {
                role: "user".into(),
                parts: vec![
                    EmbedPart::Image(png_bytes),
                    EmbedPart::Text("Describe the color gradient in this image.".into()),
                ],
            }],
            add_generation_prompt: true,
        });
    }
    out
}

fn scenario_inputs(name: &str) -> Result<(Vec<EmbedInput>, usize), Failure> {
    match name {
        "b1_l64" => Ok((synthetic_tokens(1, 64), 0)),
        "b16_l256" => Ok((synthetic_tokens(16, 256), 0)),
        "b64_l256" => Ok((synthetic_tokens(64, 256), 0)),
        "array_batch_8" => Ok((synthetic_tokens(8, 350), 0)),
        "array_batch_16" => Ok((synthetic_tokens(16, 208), 0)),
        "array_batch_64" => Ok((synthetic_tokens(64, 64), 0)),
        "mm_512" => Ok((synthetic_mm(1, 512), 1)),
        "mm_4x512" => Ok((synthetic_mm(4, 512), 4)),
        other => Err(format!("unknown embed bench scenario `{other}`").into()),
    }
}

/// Runs gr8-aligned embedding benchmark scenarios against `model`.
pub fn bench(
    model: &Path,
    spec: &str,
    warmup: usize,
    iters: usize,
    dimensions: Option<usize>,
) -> Result<Bench, Failure> {
    let settings = tessera_frontend_settings::read();
    let start = Instant::now();
    let mut embedder = Embedder::open(model, &settings, &Default::default(), &Default::default())?;
    let load_s = start.elapsed().as_secs_f64();
    let list = if spec == "all" {
        "b1_l64,b16_l256,b64_l256,mm_512,mm_4x512"
    } else {
        spec
    };
    let options = EmbedOptions {
        dimensions,
        normalize: true,
    };
    let mut scenarios = Vec::new();
    for name in list.split(',').filter(|s| !s.is_empty()) {
        let (inputs, n_images) = scenario_inputs(name)?;
        let mut last = None;
        for _ in 0..warmup {
            last = Some(embedder.embed_inputs(&inputs, options)?);
        }
        let mut times = Vec::with_capacity(iters.max(1));
        for _ in 0..iters.max(1) {
            let t0 = Instant::now();
            last = Some(embedder.embed_inputs(&inputs, options)?);
            times.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        times.sort_by(f64::total_cmp);
        let mean_ms = times.iter().sum::<f64>() / times.len() as f64;
        let p50_ms = times[times.len() / 2];
        let min_ms = times[0];
        let got = last.expect("at least one iteration");
        let (mut emb0_sum, mut norm2) = (0.0f64, 0.0f64);
        if let Some(first) = got.vectors.first() {
            for &v in first {
                let x = f64::from(v);
                emb0_sum += x;
                norm2 += x * x;
            }
        }
        scenarios.push(Scenario {
            scenario: name.into(),
            batch: inputs.len(),
            total_tokens: got.tokens,
            n_images,
            mean_ms,
            p50_ms,
            min_ms,
            tok_per_s: got.tokens as f64 / (p50_ms * 1e-3),
            emb_per_s: inputs.len() as f64 / (p50_ms * 1e-3),
            emb0_l2: norm2.sqrt(),
            emb0_sum,
        });
    }
    Ok(Bench {
        device: embedder.device(),
        load_s,
        scenarios,
    })
}
