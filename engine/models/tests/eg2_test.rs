//! EmbeddingGemma 2: traced, loaded from its checkpoint, and checked.

// Linking these registers their lowering rules.
use kernels as _;
use linear as _;

use bundle::names::{Rule, repo};
use calibrate::measure;
use checkpoint::exlr8::Release;
use checkpoint::{File, Offer};
use codegen::trellis;
use gpukit::testing::{cosine_f32, max_abs_diff};
use interp::{Tensor, run};
use lower::form::F32_FORM;
use lower::{Device, Pins, Plan, Run, Settings, Weight, lower, open};
use models::Rotated;
use models::eg2::{Config, probe, program};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;
use trace::DimName::{Context, Seqs};
use trace::Graph;
use trace::Linear;

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
    let g = graph(program::<Linear>(&config(24)));
    graph::verify(&g).unwrap();
    assert_eq!(g.regions.len(), 2 * 24);
    let count = |w: &graph::Weight| w.ty.shape.elements(&g.dims).unwrap().as_const().unwrap();
    let parameters: u64 = g.weights.iter().map(count).sum();
    assert_eq!(parameters, 271_002_648);
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

/// The graph's weights from `offer`, as the file stores them.
fn pulled(g: &Graph, c: &Config, offer: &dyn Offer) -> Vec<Weight> {
    let rules: Vec<Rule> = repo("eg2").unwrap();
    bundle::pull(g, offer, c.stack.layers, &[&rules]).unwrap().0
}

/// Every weight as floats, packed ones decoded.
fn plain(g: &Graph, weights: &[Weight]) -> Vec<Tensor> {
    let wide = |(w, gw): (&Weight, &graph::Weight)| match w {
        Weight::Floats(t) => t.clone(),
        Weight::Packed(f, bytes) => {
            let axis = |e: &graph::DimExpr| e.value(&g.dims, &|_| None).unwrap() as usize;
            let shape: Vec<usize> = gw.ty.shape.0.iter().map(axis).collect();
            let n = f.elements(bytes.len());
            Tensor::f32(shape, (0..n).map(|i| f.unpack(bytes, i)).collect())
        }
    };
    weights.iter().zip(&g.weights).map(wide).collect()
}

/// Weights for `g` from the bf16 checkpoint in `dir`.
fn weights(g: &Graph, c: &Config, dir: &Path) -> Vec<Tensor> {
    let file = File::open(&dir.join("model.safetensors")).unwrap();
    plain(g, &pulled(g, c, &file))
}

/// The exlr8 release: matrices against bf16, embeddings against the reference.
fn check_exlr8(dir: &Path, g: &Graph, c: &Config, cases: &[Case]) -> Exlr8 {
    let release = match std::fs::read_to_string(dir.join("exlr8.repo")) {
        Ok(repo) => {
            let (repo, revision) = repo.trim().split_once('@').unwrap_or((repo.trim(), "main"));
            eprintln!("exlr8: streaming {repo} at {revision}");
            Release::remote(repo, revision)
        }
        Err(_) => Release::open(&dir.join("exlr8")),
    };
    let file = File::open(&dir.join("model.safetensors")).unwrap();
    for name in [
        "language_model.layers.0.self_attn.q_proj.weight",
        "language_model.layers.0.mlp.down_proj.weight",
        "language_model.embedding_projection.weight",
        "language_model.embed_tokens.weight",
    ] {
        let start = Instant::now();
        let got = release.read(name).unwrap();
        let seconds = start.elapsed().as_secs_f32();
        let want = file.floats(name).unwrap();
        let cos = cosine_f32(&got, &want);
        eprintln!(
            "exlr8 {name}: {} values in {seconds:.2} s, cos {cos:.5}",
            got.len()
        );
        assert!(cos > 0.95, "{name}: cos {cos}");
    }
    let start = Instant::now();
    let w = plain(g, &pulled(g, c, &release));
    eprintln!(
        "exlr8: every weight decoded in {:.1} s",
        start.elapsed().as_secs_f32()
    );
    // The codes before their rotations, rotated in the graph.
    let turned = graph(program::<Rotated>(c));
    let stored = pulled(&turned, c, &release);
    let sides = plain(&turned, &stored);
    let mut embeddings = Vec::new();
    for c in cases {
        let got = unit(&pooled(g, &w, &c.tokens, 1));
        let cos = cosine_f32(&got, &c.embedding);
        eprintln!("exlr8 {}: cos {cos:.5} against the bf16 reference", c.name);
        assert!(cos > 0.98, "{}: cos {cos}", c.name);
        let rotated = unit(&pooled(&turned, &sides, &c.tokens, 1));
        let same = cosine_f32(&rotated, &got);
        eprintln!(
            "exlr8 {} rotated in the graph: cos {same:.7} against decoded",
            c.name
        );
        assert!(
            same > 0.99999,
            "{}: rotated in the graph differs, cos {same}",
            c.name
        );
        embeddings.push(rotated);
    }
    Exlr8 {
        graph: turned,
        weights: sides,
        stored,
        embeddings,
    }
}

/// The rotated graph, its weights, and the interpreter's embeddings.
struct Exlr8 {
    graph: Graph,
    weights: Vec<Tensor>,
    /// The same weights as the release stores them.
    stored: Vec<Weight>,
    embeddings: Vec<Vec<f32>>,
}

/// Tokens and positions for one text, `[1, len]` each.
fn inputs(tokens: &[u32]) -> [Tensor; 2] {
    let shape = vec![1, tokens.len()];
    let positions = (0..tokens.len() as u32).collect();
    let ids = |data| Tensor::u32(shape.clone(), data);
    [ids(tokens.to_vec()), ids(positions)]
}

/// The output's mean over its `rows`, on the interpreter.
fn pooled(g: &Graph, w: &[Tensor], tokens: &[u32], rows: usize) -> Vec<f32> {
    let start = Instant::now();
    let out = run(g, w, 0, &inputs(tokens)).unwrap();
    let seconds = start.elapsed().as_secs_f32();
    eprintln!("  interpreter, {} tokens: {seconds:.2} s", tokens.len());
    mean_rows(out[0].floats().unwrap(), rows)
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

/// Pins, and packed bytes by weight, for the device.
struct Stored<'a> {
    pins: Pins,
    packed: &'a dyn Fn(u32) -> Option<Vec<u8>>,
}

impl Default for Stored<'_> {
    fn default() -> Self {
        Self {
            pins: Pins::all(F32_FORM),
            packed: &|_| None,
        }
    }
}

/// Every case through the forward entry on this machine's GPU.
fn check_device(
    settings: &Settings,
    g: &Graph,
    w: &[Tensor],
    cases: &[Case],
    stored: &Stored,
) -> Vec<Vec<f32>> {
    let mut embeddings = Vec::new();
    let mut device = Device::guess(settings).pinned(stored.pins.clone());
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
                    let path = &g.weights[at].path;
                    let loaded = match (stored.packed)(at as u32) {
                        Some(bytes) => r.load_packed(at as u32, &bytes),
                        None => r.load(at as u32, t.view()),
                    };
                    loaded.unwrap_or_else(|e| panic!("weight {at} {path:?}: {e}"));
                }
                device = Device::stored(&*r, settings).unwrap();
                device = device.pinned(stored.pins.clone());
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
        embeddings.push(unit(&out));
    }
    if let Some(path) = Device::path(settings, &device.name) {
        device.save(&path).unwrap();
        eprintln!(
            "  timed {learned} launches; samples stored in {}",
            path.display()
        );
    }
    embeddings
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
        let g = graph(probe::<Linear>(&cfg));
        let w = weights(&g, &cfg, dir);
        let got = pooled(&g, &w, &c.tokens, c.tokens.len());
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
    let g = graph(program::<Linear>(&cfg));
    let w = weights(&g, &cfg, &dir);
    for c in &cases {
        let got = unit(&pooled(&g, &w, &c.tokens, 1));
        close(&c.name, &got, &c.embedding);
    }
    let quantized = (dir.join("exlr8").is_dir() || dir.join("exlr8.repo").is_file())
        .then(|| check_exlr8(&dir, &g, &cfg, &cases[..1]));
    let settings = settings::read();
    if settings.gpu().is_none() {
        return eprintln!("no GPU here; skipped");
    }
    let got = check_device(&settings, &g, &w, &cases, &Stored::default());
    for (c, got) in cases.iter().zip(&got) {
        close(&format!("{} on the device", c.name), got, &c.embedding);
    }
    let Some(q) = quantized else { return };
    let got = check_device(
        &settings,
        &q.graph,
        &q.weights,
        &cases[..1],
        &Stored::default(),
    );
    for (want, got) in q.embeddings.iter().zip(&got) {
        close("exlr8 rotated on the device", got, want);
    }
    // The codes on the device, decoded as they are read.
    let codes = |at: u32| match &q.stored[at as usize] {
        Weight::Packed(_, bytes) => Some(bytes.clone()),
        Weight::Floats(_) => None,
    };
    let stored = Stored {
        pins: Pins::all(F32_FORM).pin("*.w", trellis::form_for(4).unwrap()),
        packed: &codes,
    };
    let got = check_device(&settings, &q.graph, &q.weights, &cases[..1], &stored);
    for (want, got) in q.embeddings.iter().zip(&got) {
        close("exlr8 codes on the device", got, want);
    }
}
