//! Qwen3-0.6B: traced, loaded from its checkpoint, and run.

use checkpoint::File;
use interp::{Cpu, Data, Gpu, Kernels, Tensor, bind, gpu_ready, run};
use models::qwen3::{Config, program, tensor_name};
use ops::testing::{cosine_f32, max_abs_diff};
use std::path::PathBuf;
use std::time::Instant;
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

/// One sequence of `tokens` through the interpreter.
fn logits(g: &Graph, c: &Case, tokens: &[u32], k: &mut dyn Kernels) -> Vec<f32> {
    let shape = vec![1, tokens.len()];
    let positions = (0..tokens.len() as u32).collect();
    let ids = |data| Tensor {
        shape: shape.clone(),
        data: Data::U32(data),
    };
    let inputs = [ids(tokens.to_vec()), ids(positions)];
    let start = Instant::now();
    let out = run(g, k, &c.weights, 0, &inputs).unwrap();
    let seconds = start.elapsed().as_secs_f32();
    eprintln!("interpreter, {} tokens: {seconds:.3} s", tokens.len());
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
    close("cpu", &logits(&g, &c, &c.tokens, &mut Cpu), &c);
    if !gpu_ready() {
        return eprintln!("no GPU here; skipped");
    }
    let mut k = Gpu::default();
    close("gpu", &logits(&g, &c, &c.tokens, &mut k), &c);
    // SiLU and the gate multiply have no GPU kernel yet.
    assert_eq!(k.fallbacks, 2 * 28, "only the MLP's two stages fall back");
    #[cfg(target_os = "macos")]
    plan::check(&g, &c);
}

/// The same model as a plan, wholly on Metal.
#[cfg(target_os = "macos")]
mod plan {
    use super::*;
    use lower::lower;
    use lower::metal::{Feed, Runner};
    use std::collections::BTreeMap;
    use trace::DimName::{Context, Seqs};

    /// Lowers for `tokens`, loads the weights, runs, and prints timings.
    fn timed(g: &Graph, c: &Case, tokens: &[u32]) -> Vec<f32> {
        let sizes = [(Seqs, 1), (Context, tokens.len() as u64)];
        let start = Instant::now();
        let runner = Runner::new(lower(g, 0, &sizes).unwrap()).unwrap();
        let built = start.elapsed().as_secs_f64();
        for (at, w) in c.weights.iter().enumerate() {
            runner.load(at as u32, w.floats().unwrap()).unwrap();
        }
        let ready = start.elapsed().as_secs_f64() - built;
        let positions: Vec<u32> = (0..tokens.len() as u32).collect();
        let feeds = [Feed::Ids(tokens), Feed::Ids(&positions)];
        // The first run pays for anything Metal sets up lazily.
        runner.run(&feeds).unwrap();
        let start = Instant::now();
        let mut out = runner.run(&feeds).unwrap();
        let whole = start.elapsed().as_secs_f64();
        let steps = runner.run_timed(&feeds).unwrap();
        let waited: f64 = steps.iter().sum();
        let (n, launches) = (tokens.len(), steps.len());
        eprintln!("plan, {n} tokens: {whole:.3} s in one command buffer");
        eprintln!("  built in {built:.3} s, weights uploaded in {ready:.3} s");
        eprintln!("  {launches} launches take {waited:.3} s waited on one by one");
        let mut by: BTreeMap<&str, (usize, f64)> = BTreeMap::new();
        for (step, seconds) in runner.plan().steps.iter().zip(&steps) {
            let (count, total) = by.entry(step.kernel.name()).or_default();
            (*count, *total) = (*count + 1, *total + seconds);
        }
        for (name, (count, seconds)) in by {
            eprintln!("  {name:8} {count:4} launches {seconds:.3} s");
        }
        out.remove(0)
    }

    pub fn check(g: &Graph, c: &Case) {
        close("plan", &timed(g, c, &c.tokens), c);
        // Longer than the reference, so compare with the interpreter.
        let long: Vec<u32> = c.tokens.iter().cycle().take(512).copied().collect();
        let got = timed(g, c, &long);
        let want = logits(g, c, &long, &mut Gpu::default());
        let off = max_abs_diff(&got, &want);
        eprintln!("plan vs interpreter, 512 tokens: max_abs {off:.3e}");
        assert!(off < 1e-3, "the plan and the interpreter disagree");
    }
}
