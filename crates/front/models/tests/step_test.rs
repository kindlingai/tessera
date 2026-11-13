//! A step entry's kept rows against whole forward passes.

// Linking the crate registers its rules.
use kernels as _;

use engine::settings;
use gpukit::testing::{Noise, max_abs_diff};
use interp::{Store, Tensor, run, run_kept};
use lower::{Device, Run, lower, open, rules};
use models::qwen3::{Config, program, stepper};
use std::ops::Range;
use trace::DimName::{Context, Rows, Seqs};
use trace::Graph;

const SEQS: usize = 2;
const LEN: usize = 7;

fn weights(g: &Graph, noise: &mut Noise) -> Vec<Tensor> {
    let size = |e: &graph::DimExpr| e.poly(&g.dims).unwrap().as_const().unwrap() as usize;
    let fill = |w: &graph::Weight| {
        let shape: Vec<usize> = w.ty.shape.0.iter().map(size).collect();
        let data = (0..shape.iter().product()).map(|_| noise.draw()).collect();
        Tensor::f32(shape, data)
    };
    g.weights.iter().map(fill).collect()
}

/// Each sequence's tokens and positions over `span`.
fn inputs(tokens: &[u32], span: Range<usize>) -> [Tensor; 2] {
    feeds(tokens, SEQS, span)
}

fn feeds(tokens: &[u32], seqs: usize, span: Range<usize>) -> [Tensor; 2] {
    let shape = vec![seqs, span.len()];
    let rows = |of: &dyn Fn(usize, usize) -> u32| {
        let all = (0..seqs).flat_map(|s| span.clone().map(move |t| of(s, t)));
        Tensor::u32(shape.clone(), all.collect())
    };
    [rows(&|s, t| tokens[s * LEN + t]), rows(&|_, t| t as u32)]
}

fn best(row: &[f32]) -> u32 {
    let top = |a: &(usize, &f32), b: &(usize, &f32)| a.1.total_cmp(b.1);
    row.iter().enumerate().max_by(top).unwrap().0 as u32
}

#[test]
fn steps_match_forward() {
    let c = Config::toy(4, 16);
    let (whole, step) = (program(&c), stepper(&c));
    assert_eq!(whole.weights, step.weights);
    assert_eq!(step.kept.len(), 2 * 2, "keys and values per layer");
    let mut noise = Noise(3);
    let weights = weights(&whole, &mut noise);
    let draw = |_| ((noise.draw() + 1.0) * 16.0) as u32;
    let tokens: Vec<u32> = (0..SEQS * LEN).map(draw).collect();
    let want = run(&whole, &weights, 0, &inputs(&tokens, 0..LEN)).unwrap();
    let want = want[0].floats().unwrap();
    let vocab = want.len() / (SEQS * LEN);
    // Three rows at once, then shorter launches.
    let mut store = Store::default();
    let mut got = vec![0.0; want.len()];
    for span in [0..3, 3..4, 4..5, 5..7] {
        let feeds = inputs(&tokens, span.clone());
        let out = run_kept(&step, &weights, &mut store, 0, &feeds).unwrap();
        let (ids, logits) = (out[0].indices().unwrap(), out[1].floats().unwrap());
        assert_eq!(out[0].shape, [SEQS, span.len()]);
        assert_eq!(store.rows(0), span.end);
        for (at, row) in logits.chunks(vocab).enumerate() {
            let (s, t) = (at / span.len(), span.start + at % span.len());
            assert_eq!(ids[at], best(row) as usize);
            got[(s * LEN + t) * vocab..][..vocab].copy_from_slice(row);
        }
    }
    let off = max_abs_diff(&got, want);
    assert!(off < 1e-5, "steps drift from the forward pass by {off:e}");
}

#[test]
fn launches_start_empty() {
    let c = Config::toy(4, 16);
    let step = stepper(&c);
    let weights = weights(&step, &mut Noise(5));
    let tokens = [1; SEQS * LEN];
    let feeds = inputs(&tokens, 0..2);
    // Without a store, every launch is a first one.
    let once = run(&step, &weights, 0, &feeds).unwrap();
    let again = run(&step, &weights, 0, &feeds).unwrap();
    assert_eq!(once, again);
    // Rows past the context bound are refused.
    let mut store = Store::default();
    let long = inputs(&tokens, 0..LEN);
    for _ in 0..2 {
        run_kept(&step, &weights, &mut store, 0, &long).unwrap();
    }
    let third = run_kept(&step, &weights, &mut store, 0, &long);
    assert!(third.is_err(), "21 rows exceed a context of 16");
}

/// Runs one launch as a plan. Returns ids, logits.
fn launch(
    g: &Graph,
    runner: &mut Option<Box<dyn Run>>,
    weights: &[Tensor],
    feeds: &[Tensor; 2],
    device: &Device,
) -> (Vec<u32>, Vec<f32>) {
    let [rows, past] = [feeds[0].shape[1], feeds[1].indices().unwrap()[0]];
    let sizes = [
        (Seqs, 1),
        (Rows, rows as u64),
        (Context, (past + rows) as u64),
    ];
    let plan = lower(g, 0, &sizes, device).unwrap();
    let [ids, logits] = [plan.outputs[0].slot, plan.outputs[1].slot];
    match runner {
        Some(r) => r.replan(&plan).unwrap(),
        None => {
            // Weights load once. Later plans reuse them.
            let r = runner.insert(open(plan, &settings()).unwrap());
            for (at, w) in weights.iter().enumerate() {
                r.load(at as u32, w.view()).unwrap();
            }
        }
    }
    let r = runner.as_mut().unwrap();
    r.feed(&[feeds[0].view(), feeds[1].view()]).unwrap();
    r.submit(0..r.plan().steps.len()).unwrap();
    let logits = r.read(logits).unwrap().floats().unwrap().to_vec();
    (r.ids(ids).unwrap(), logits)
}

#[test]
fn plans_keep_rows() {
    let step = stepper(&Config::toy(4, 16));
    let weights = weights(&step, &mut Noise(9));
    let tokens: Vec<u32> = (0..LEN as u32).map(|t| (t * 7 + 3) % 32).collect();
    let spans = [0..3, 3..4, 4..5, 5..7];
    let favor = |r: &&lower::Rule| Device::guess(&settings()).favoring(r.name);
    for device in rules().iter().map(favor) {
        let (mut store, mut runner) = (Store::default(), None);
        for span in spans.clone() {
            let feeds = feeds(&tokens, 1, span);
            let want = run_kept(&step, &weights, &mut store, 0, &feeds).unwrap();
            let (ids, logits) = launch(&step, &mut runner, &weights, &feeds, &device);
            let off = max_abs_diff(&logits, want[1].floats().unwrap());
            assert!(off < 1e-4, "the plan is off by {off:e}");
            let want: Vec<u32> = want[0]
                .indices()
                .unwrap()
                .iter()
                .map(|i| *i as u32)
                .collect();
            assert_eq!(ids, want);
        }
    }
}
