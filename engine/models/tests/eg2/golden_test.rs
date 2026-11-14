//! EmbeddingGemma 2: layer means, the exlr8 release and timing.

use super::{GOLDEN, release};
use crate::common::{Case, Reference, floats, interpreted, kernels, loaded, seconds};
use crate::common::{table, tokens, unit};
use bundle::names::{Rule, repo};
use calibrate::measure;
use checkpoint::exlr8::Release;
use checkpoint::{File, Offer};
use gpukit::testing::cosine_f32;
use interp::Tensor;
use lower::{Device, Run, Settings, Weight, lower};
use models::eg2::{Config, probe};
use std::path::Path;
use std::time::Instant;
use trace::Graph;
use trace::Linear;

/// The release sizes, cut to `layers`.
fn config(layers: u64) -> Config {
    let mut c = Config::release(1, 1024);
    c.stack.layers = layers;
    c
}

/// The cases under the directory `TESSERA_EG2` names.
fn cases(dir: &Path) -> Vec<Case> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir.join("cases")).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy();
        let tokens = tokens(&path.join("tokens.txt"));
        out.push(Case::one(
            &name,
            tokens,
            floats(&path.join("embedding.f32")),
        ));
    }
    out.sort_by_key(|a| a.tokens.len());
    out
}

/// The reference the suite's golden test reads.
pub fn read(g: &Graph) -> Option<Reference> {
    let dir = settings::read().eg2?;
    Some(Reference {
        weights: weights(g, &config(24), &dir),
        cases: cases(&dir),
    })
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

/// The exlr8 release: its embeddings against the reference.
fn check_exlr8(dir: &Path, g: &Graph, c: &Config, cases: &[Case]) {
    let release = match std::fs::read_to_string(dir.join("exlr8.repo")) {
        Ok(repo) => {
            let (repo, revision) = repo.trim().split_once('@').unwrap_or((repo.trim(), "main"));
            eprintln!("exlr8: streaming {repo} at {revision}");
            Release::remote(repo, revision)
        }
        Err(_) => Release::open(&dir.join("exlr8")),
    };
    let start = Instant::now();
    let w = plain(g, &pulled(g, c, &release));
    eprintln!(
        "exlr8: every weight decoded in {:.1} s",
        start.elapsed().as_secs_f32()
    );
    for c in cases {
        let got = unit(&pooled(g, &w, &c.tokens, 1));
        let cos = cosine_f32(&got, &c.want);
        eprintln!("exlr8 {}: cos {cos:.5} against the bf16 reference", c.name);
        assert!(cos > 0.98, "{}: cos {cos}", c.name);
    }
}

/// The output's mean over its `rows`, on the interpreter.
fn pooled(g: &Graph, w: &[Tensor], tokens: &[u32], rows: usize) -> Vec<f32> {
    mean_rows(&interpreted(g, w, 1, tokens), rows)
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

/// Every case through the forward entry on this machine's GPU.
fn check_device(settings: &Settings, g: &Graph, w: &[Tensor], cases: &[Case]) -> Vec<Vec<f32>> {
    let mut embeddings = Vec::new();
    let mut device = Device::guess(settings);
    let mut runner: Option<Box<dyn Run>> = None;
    let mut learned = 0;
    for c in cases {
        let sizes = c.sizes();
        let feeds = c.inputs();
        let views = [feeds[0].view(), feeds[1].view()];
        let r = match runner.as_mut() {
            Some(r) => r,
            None => {
                let plan = lower(g, 0, &sizes, &device).unwrap();
                let r = loaded(plan, g, w, settings);
                device = Device::stored(&*r, settings).unwrap();
                runner.insert(r)
            }
        };
        // Launches with no stored sample are timed under each rule.
        learned += measure(&mut **r, (g, 0), &sizes, &views, &mut device, settings)
            .unwrap()
            .len();
        let plan = lower(g, 0, &sizes, &device).unwrap();
        let by = kernels(&plan);
        eprintln!("  {} launches: {by:?}", by.values().sum::<usize>());
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
        let steps = seconds(&mut **r);
        table(r.plan(), &steps);
        embeddings.push(out);
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

/// Each layer's mean residual, for the shortest case.
fn check_layers(dir: &Path, c: &Case) {
    let means = dir.join("cases").join(&c.name).join("layer_means.f32");
    let layer_means = floats(&means);
    for layers in [1, 2, 5, 6, 12, 24] {
        let cfg = config(layers);
        // The suite does not check probe's JSON.
        let g = Graph::from_json(&probe::<Linear>(&cfg).to_json()).unwrap();
        let w = weights(&g, &cfg, dir);
        let got = pooled(&g, &w, &c.tokens, c.tokens.len());
        let hidden = got.len();
        let at = (layers as usize - 1) * hidden;
        let want = &layer_means[at..at + hidden];
        GOLDEN
            .tolerance
            .check(&format!("{} after layer {layers}", c.name), &got, want, 1);
    }
}

#[test]
fn reference_checks() {
    let g = release();
    let Some(reference) = read(&g) else {
        return eprintln!("TESSERA_EG2 is not set; skipped");
    };
    let settings = settings::read();
    let dir = settings.eg2.clone().unwrap();
    let cases = &reference.cases;
    if settings.gpu().is_none() {
        check_layers(&dir, &cases[0]);
    }
    if dir.join("exlr8").is_dir() || dir.join("exlr8.repo").is_file() {
        check_exlr8(&dir, &g, &config(24), &cases[..1]);
    }
    if settings.gpu().is_none() {
        return eprintln!("no GPU here; skipped");
    }
    let got = check_device(&settings, &g, &reference.weights, cases);
    for (c, got) in cases.iter().zip(&got) {
        GOLDEN.close(&format!("{} on the device", c.name), got, c);
    }
}
