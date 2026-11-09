//! Qwen3-0.6B: traced, loaded from its checkpoint, and run.

use checkpoint::File;
use interp::{Cpu, Data, Gpu, Kernels, Tensor, bind, gpu_ready, run};
use models::qwen3::{Config, program, tensor_name};
use ops::testing::{cosine_f32, max_abs_diff};
use std::path::PathBuf;
use trace::{Graph, PathSeg};

/// The interpreter only ever sees the JSON text.
fn graph() -> Graph {
    Graph::from_json(&program(&Config::small(8, 40_960)).to_json()).unwrap()
}

#[test]
fn traces() {
    let g = graph();
    graph::verify(&g).unwrap();
    assert_eq!(g.regions.len(), 2 * 28);
    let count = |w: &graph::Weight| w.ty.shape.elements(&g.dims).unwrap().as_const().unwrap();
    let parameters: u64 = g.weights.iter().map(count).sum();
    assert_eq!(parameters, 596_049_920);
}

/// Read from the directory `TESSERA_QWEN3` names.
struct Case {
    weights: Vec<Tensor>,
    tokens: Vec<u32>,
    want: Vec<f32>,
}

fn case(g: &Graph) -> Option<Case> {
    let dir = PathBuf::from(std::env::var_os("TESSERA_QWEN3")?);
    let file = File::open(&dir.join("model.safetensors")).unwrap();
    let find = |p: &[PathSeg]| Some(Data::F32(file.floats(&tensor_name(p)?).ok()?));
    let tokens = std::fs::read_to_string(dir.join("tokens.txt")).unwrap();
    let floats = std::fs::read(dir.join("logits.f32")).unwrap();
    let wide = |b: &[u8]| f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    Some(Case {
        weights: bind(g, find).unwrap(),
        tokens: tokens
            .split_whitespace()
            .map(|t| t.parse().unwrap())
            .collect(),
        want: floats.chunks_exact(4).map(wide).collect(),
    })
}

fn logits(g: &Graph, c: &Case, k: &mut dyn Kernels) -> Vec<f32> {
    let shape = vec![1, c.tokens.len()];
    let positions = (0..c.tokens.len() as u32).collect();
    let ids = |data| Tensor {
        shape: shape.clone(),
        data: Data::U32(data),
    };
    let inputs = [ids(c.tokens.clone()), ids(positions)];
    let start = std::time::Instant::now();
    let out = run(g, k, &c.weights, 0, &inputs).unwrap();
    eprintln!("ran in {:.1} s", start.elapsed().as_secs_f32());
    out[0].floats().unwrap().to_vec()
}

fn best(row: &[f32]) -> usize {
    let top = |a: &(usize, &f32), b: &(usize, &f32)| a.1.total_cmp(b.1);
    row.iter().enumerate().max_by(top).unwrap().0
}

/// Each position must pick the reference's top token.
fn close(label: &str, got: &[f32], c: &Case) {
    let vocab = got.len() / c.tokens.len();
    let (max, mut cos) = (max_abs_diff(got, &c.want), 1.0f32);
    for (g, w) in got.chunks(vocab).zip(c.want.chunks(vocab)) {
        cos = cos.min(cosine_f32(g, w));
        assert_eq!(best(g), best(w), "{label}: top token differs");
    }
    eprintln!("{label}: max_abs {max:.3e} worst cos {cos:.7}");
    assert!(max < 1e-3 && cos > 0.99999, "{label}: {max:.3e} {cos}");
}

#[test]
fn matches_reference() {
    let g = graph();
    let Some(c) = case(&g) else {
        return eprintln!("TESSERA_QWEN3 is not set; skipped");
    };
    close("cpu", &logits(&g, &c, &mut Cpu), &c);
    if !gpu_ready() {
        return eprintln!("no GPU here; skipped");
    }
    let mut k = Gpu::default();
    close("gpu", &logits(&g, &c, &mut k), &c);
    // SiLU and the gate multiply have no GPU kernel yet.
    assert_eq!(k.fallbacks, 2 * 28, "only the MLP's two stages fall back");
}
