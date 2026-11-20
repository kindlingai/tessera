//! Embedding speed over a JSON list of texts.

use crate::Failure;
use sensai_tasks::Embedder;
use serde::Serialize;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

#[derive(Serialize)]
pub struct Speed {
    device: String,
    load_s: f64,
    texts: usize,
    tokens: usize,
    texts_per_s: f64,
}

/// Embeds each text with the release in `model`.
pub fn embed(model: &Path, texts: &Path, vectors: Option<&Path>) -> Result<Speed, Failure> {
    let settings = sensai_frontend_settings::read();
    let start = Instant::now();
    let mut embedder = Embedder::open(model, &settings, &sensai_bundle::Overrides::default())?;
    let load_s = start.elapsed().as_secs_f64();
    let texts: Vec<String> = serde_json::from_str(&std::fs::read_to_string(texts)?)?;
    let mut tokens = 0;
    for t in &texts {
        tokens += embedder.tokens(t)?.len();
    }
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let start = Instant::now();
    let out = embedder.embed(&refs)?;
    let seconds = start.elapsed().as_secs_f64();
    if let Some(path) = vectors {
        let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
        for x in out.iter().flatten() {
            file.write_all(&x.to_le_bytes())?;
        }
        file.flush()?;
    }
    Ok(Speed {
        device: embedder.device(),
        load_s,
        texts: texts.len(),
        tokens,
        texts_per_s: texts.len() as f64 / seconds,
    })
}
