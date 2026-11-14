//! Each weight finds a slice in every model file.

use bundle::names::{repo, resolve};
use checkpoint::{Slice, Stored};
use models::Rotated;
use models::{eg2, qwen3};
use trace::{Graph, Linear};

/// A real file's slices, as `tessera-eval --slices` wrote them.
fn slices(json: &str) -> Vec<Slice> {
    let entries: Vec<serde_json::Value> = serde_json::from_str(json).unwrap();
    entries
        .iter()
        .map(|e| Slice {
            name: e["name"].as_str().unwrap().to_string(),
            shape: (e["shape"].as_array().unwrap().iter())
                .map(|n| n.as_u64().unwrap() as usize)
                .collect(),
            stored: Stored::F32,
        })
        .collect()
}

/// Every weight of `g` claims a slice of `manifest`.
fn resolves(model: &str, g: &Graph, manifest: &str) {
    let slices = slices(manifest);
    let rules = repo(model).unwrap_or_default();
    match resolve(g, &slices, &[&rules]) {
        Ok(claims) => assert_eq!(claims.len(), g.weights.len()),
        Err(e) => panic!("{model}: {e}"),
    }
}

#[test]
fn qwen3_files() {
    let g = qwen3::program(&qwen3::Config::small(1, 16));
    resolves("qwen3", &g, include_str!("manifests/qwen3-hf.json"));
    resolves("qwen3", &g, include_str!("manifests/qwen3-gguf.json"));
}

#[test]
fn eg2_files() {
    let c = eg2::Config::release(1, 16);
    let plain = eg2::program::<Linear>(&c);
    resolves("eg2", &plain, include_str!("manifests/eg2-hf.json"));
    let turned = eg2::program::<Rotated>(&c);
    resolves("eg2", &turned, include_str!("manifests/eg2-exlr8.json"));
}
