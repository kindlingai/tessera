//! EmbeddingGemma 2: traced, loaded from its checkpoint, and checked.

// Linking these registers their lowering rules.
use kernels as _;
use linear as _;

use calibrate::measure;
use checkpoint::File;
use gpukit::testing::{cosine_f32, max_abs_diff};
use interp::{Tensor, bind, run};
use lower::{Device, Plan, Run, Settings, lower, open};
use models::eg2::{Config, probe, program, weight};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;
use trace::DimName::{Context, Seqs};
use trace::Graph;

/// The release sizes, cut to `layers`.
fn config(layers: u64) -> Config {
    let mut c = Config::release(1, 1024);
    c.stack.layers = layers;
    c
}

/// The interpreter only ever sees the JSON text.
fn graph(g: Graph) -> Graph {
    Graph::from_json(&g.to_json()).unwrap()
}

#[test]
fn traces() {
    let g = graph(program(&config(24)));
    graph::verify(&g).unwrap();
    assert_eq!(g.regions.len(), 2 * 24);
    let count = |w: &graph::Weight| w.ty.shape.elements(&g.dims).unwrap().as_const().unwrap();
    let parameters: u64 = g.weights.iter().map(count).sum();
    assert_eq!(parameters, 271_022_080);
}

/// The reference's ids and outputs for one text.
struct Case {
    name: String,
    tokens: Vec<u32>,
    embedding: Vec<f32>,
    /// The mean residual after each layer, `[Layers, Hidden]`.
    layer_means: Vec<f32>,
}

fn floats(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    let wide = |b: &[u8; 4]| f32::from_le_bytes(*b);
    bytes.as_chunks::<4>().0.iter().map(wide).collect()
}

/// The cases under the directory `TESSERA_EG2` names.
fn cases(dir: &Path) -> Vec<Case> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir.join("cases")).unwrap() {
        let path = entry.unwrap().path();
        let tokens = std::fs::read_to_string(path.join("tokens.txt")).unwrap();
        out.push(Case {
            name: path.file_name().unwrap().to_string_lossy().into_owned(),
            tokens: tokens
                .split_whitespace()
                .map(|t| t.parse().unwrap())
                .collect(),
            embedding: floats(&path.join("embedding.f32")),
            layer_means: floats(&path.join("layer_means.f32")),
        });
    }
    out.sort_by_key(|a| a.tokens.len());
    out
}

/// Weights for `g` from the checkpoint in `dir`.
fn weights(g: &Graph, c: &Config, dir: &Path) -> Vec<Tensor> {
    let file = File::open(&dir.join("model.safetensors")).unwrap();
    let read = |name: &str| file.floats(name).ok();
    let find = |p: &[trace::PathSeg]| {
        let v = weight(c, p, &read)?;
        Some(Tensor::f32(vec![v.len()], v))
    };
    bind(g, find).unwrap()
}

/// Tokens and positions for one text, `[1, len]` each.
fn inputs(tokens: &[u32]) -> [Tensor; 2] {
    let shape = vec![1, tokens.len()];
    let positions = (0..tokens.len() as u32).collect();
    let ids = |data| Tensor::u32(shape.clone(), data);
    [ids(tokens.to_vec()), ids(positions)]
}

/// The output's mean over the tokens, on the interpreter.
fn pooled(g: &Graph, w: &[Tensor], tokens: &[u32]) -> Vec<f32> {
    let start = Instant::now();
    let out = run(g, w, 0, &inputs(tokens)).unwrap();
    let seconds = start.elapsed().as_secs_f32();
    eprintln!("  interpreter, {} tokens: {seconds:.2} s", tokens.len());
    mean_rows(out[0].floats().unwrap(), tokens.len())
}

/// The mean of `rows` rows.
fn mean_rows(out: &[f32], rows: usize) -> Vec<f32> {
    let width = out.len() / rows;
    let mut mean = vec![0.0f64; width];
    for row in out.chunks(width) {
        mean.iter_mut()
            .zip(row)
            .for_each(|(m, v)| *m += f64::from(*v));
    }
    mean.iter().map(|m| (m / rows as f64) as f32).collect()
}

/// How often a plan launches each kernel.
fn kernels(plan: &Plan) -> BTreeMap<&'static str, usize> {
    let mut by = BTreeMap::new();
    for step in &plan.steps {
        *by.entry(step.kernel.name()).or_default() += step.launches().len();
    }
    by
}

/// Every case through the forward entry on this machine's GPU.
fn check_device(settings: &Settings, g: &Graph, w: &[Tensor], cases: &[Case]) {
    let mut device = Device::guess(settings);
    let mut runner: Option<Box<dyn Run>> = None;
    let mut learned = 0;
    for c in cases {
        let sizes = [(Seqs, 1), (Context, c.tokens.len() as u64)];
        let feeds = inputs(&c.tokens);
        let views = [feeds[0].view(), feeds[1].view()];
        let r = match runner.as_mut() {
            Some(r) => r,
            None => {
                let plan = lower(g, 0, &sizes, &device).unwrap();
                let mut r = open(plan, settings).unwrap();
                for (at, t) in w.iter().enumerate() {
                    r.load(at as u32, t.view()).unwrap();
                }
                device = Device::stored(&*r, settings).unwrap();
                runner.insert(r)
            }
        };
        // Launches with no stored sample are timed under each rule.
        learned += measure(&mut **r, (g, 0), &sizes, &views, &mut device, settings)
            .unwrap()
            .len();
        let plan = lower(g, 0, &sizes, &device).unwrap();
        eprintln!("  {} launches: {:?}", plan.steps.len(), kernels(&plan));
        r.replan(&plan).unwrap();
        r.feed(&views).unwrap();
        let (slot, steps) = (r.plan().outputs[0].slot, r.plan().steps.len());
        let start = Instant::now();
        r.submit(0..steps).unwrap();
        let out = r.read(slot).unwrap().floats().unwrap().to_vec();
        let cold = start.elapsed().as_secs_f32();
        // Warm: the plan ten more times, each read back.
        let start = Instant::now();
        for _ in 0..10 {
            r.submit(0..steps).unwrap();
            r.read(slot).unwrap();
        }
        let warm = start.elapsed().as_secs_f32() / 10.0;
        let (name, n) = (r.device(), c.tokens.len());
        let rate = n as f32 / warm;
        eprintln!("  {name}, {n} tokens: cold {cold:.3} s, warm {warm:.4} s, {rate:.0} tokens/s");
        // Seconds by kernel, each step timed alone on the device.
        let mut by: BTreeMap<&str, f64> = BTreeMap::new();
        for (i, step) in r.plan().steps.clone().iter().enumerate() {
            *by.entry(step.kernel.name()).or_default() += r.repeat(i, 1).unwrap();
        }
        let mut by: Vec<_> = by.into_iter().collect();
        by.sort_by(|a, b| b.1.total_cmp(&a.1));
        let parts: Vec<String> = by
            .iter()
            .map(|(k, s)| format!("{k} {:.1} ms", s * 1e3))
            .collect();
        eprintln!("    {}", parts.join(", "));
        let got = unit(&mean_rows(&out, c.tokens.len()));
        close(&format!("{} on the device", c.name), &got, &c.embedding);
    }
    if let Some(path) = Device::path(settings, &device.name) {
        device.save(&path).unwrap();
        eprintln!(
            "  timed {learned} launches; samples stored in {}",
            path.display()
        );
    }
}

fn unit(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| x / norm).collect()
}

/// `got` must match `want` closely.
fn close(label: &str, got: &[f32], want: &[f32]) {
    let (max, cos) = (max_abs_diff(got, want), cosine_f32(got, want));
    eprintln!("{label}: max_abs {max:.3e} cos {cos:.7}");
    assert!(cos > 0.9999, "{label}: cos {cos}");
}

/// Each layer's mean residual, for the shortest case.
fn check_layers(dir: &Path, c: &Case) {
    for layers in [1, 2, 5, 6, 12, 24] {
        let cfg = config(layers);
        let g = graph(probe(&cfg));
        let w = weights(&g, &cfg, dir);
        let got = pooled(&g, &w, &c.tokens);
        let hidden = got.len();
        let at = (layers as usize - 1) * hidden;
        let want = &c.layer_means[at..at + hidden];
        close(&format!("{} after layer {layers}", c.name), &got, want);
    }
}

#[test]
fn matches_reference() {
    let Some(dir): Option<PathBuf> = settings::read().eg2 else {
        return eprintln!("TESSERA_EG2 is not set; skipped");
    };
    let cases = cases(&dir);
    if settings::read().gpu().is_none() {
        check_layers(&dir, &cases[0]);
    }
    let cfg = config(24);
    let g = graph(program(&cfg));
    let w = weights(&g, &cfg, &dir);
    for c in &cases {
        let got = unit(&pooled(&g, &w, &c.tokens));
        close(&c.name, &got, &c.embedding);
    }
    let settings = settings::read();
    if settings.gpu().is_none() {
        return eprintln!("no GPU here; skipped");
    }
    check_device(&settings, &g, &w, &cases);
}
