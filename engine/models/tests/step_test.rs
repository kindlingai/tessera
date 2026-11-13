//! A step entry's ragged rows against whole forward passes.

// Linking the crate registers its rules.
use kernels as _;

use gpukit::testing::{Noise, max_abs_diff};
use interp::{Store, Tensor, run, run_kept};
use lower::{Device, Run, lower, open, rules};
use models::qwen3::{Config, program, stepper};
use settings::read as settings;
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

/// Every sequence's tokens and positions, `[Seqs, Len]` each.
fn whole(tokens: &[u32]) -> [Tensor; 2] {
    let shape = vec![SEQS, LEN];
    let positions = (0..SEQS).flat_map(|_| 0..LEN as u32).collect();
    [
        Tensor::u32(shape.clone(), tokens.to_vec()),
        Tensor::u32(shape, positions),
    ]
}

/// One launch's rows: a slot and a position each.
fn feeds(tokens: &[u32], rows: &[(usize, usize)]) -> [Tensor; 3] {
    let shape = vec![rows.len()];
    let column = |of: &dyn Fn(usize, usize) -> u32| {
        Tensor::u32(
            shape.clone(),
            rows.iter().map(|(s, p)| of(*s, *p)).collect(),
        )
    };
    [
        column(&|s, p| tokens[s * LEN + p]),
        column(&|_, p| p as u32),
        column(&|s, _| s as u32),
    ]
}

fn best(row: &[f32]) -> u32 {
    let top = |a: &(usize, &f32), b: &(usize, &f32)| a.1.total_cmp(b.1);
    row.iter().enumerate().max_by(top).unwrap().0 as u32
}

/// Rows mixing both sequences at different positions, in three launches.
fn launches() -> [Vec<(usize, usize)>; 3] {
    [
        vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)],
        vec![(1, 2), (0, 3), (1, 3)],
        vec![(1, 4), (1, 5), (1, 6), (0, 4), (0, 5), (0, 6)],
    ]
}

#[test]
fn steps_match_forward() {
    let c = Config::toy(4, 16);
    let (forward, step) = (program(&c), stepper(&c));
    assert_eq!(forward.weights, step.weights);
    assert_eq!(step.kept.len(), 2 * 2, "keys and values per layer");
    let mut noise = Noise(3);
    let weights = weights(&forward, &mut noise);
    let draw = |_| ((noise.draw() + 1.0) * 16.0) as u32;
    let tokens: Vec<u32> = (0..SEQS * LEN).map(draw).collect();
    let want = run(&forward, &weights, 0, &whole(&tokens)).unwrap();
    let want = want[0].floats().unwrap();
    let vocab = want.len() / (SEQS * LEN);
    let mut store = Store::default();
    let mut got = vec![0.0; want.len()];
    for rows in launches() {
        let out = run_kept(&step, &weights, &mut store, 0, &feeds(&tokens, &rows)).unwrap();
        let (ids, logits) = (out[0].indices().unwrap(), out[1].floats().unwrap());
        assert_eq!(out[0].shape, [rows.len()]);
        for (at, row) in logits.chunks(vocab).enumerate() {
            let (s, p) = rows[at];
            assert_eq!(ids[at], best(row) as usize);
            got[(s * LEN + p) * vocab..][..vocab].copy_from_slice(row);
        }
    }
    assert_eq!(store.rows(0), LEN);
    let off = max_abs_diff(&got, want);
    assert!(off < 1e-5, "steps drift from the forward pass by {off:e}");
}

#[test]
fn launches_start_empty() {
    let c = Config::toy(4, 16);
    let step = stepper(&c);
    let weights = weights(&step, &mut Noise(5));
    let tokens = [1; SEQS * LEN];
    let inputs = feeds(&tokens, &[(0, 0), (0, 1)]);
    // Without a store, every launch is a first one.
    let once = run(&step, &weights, 0, &inputs).unwrap();
    let again = run(&step, &weights, 0, &inputs).unwrap();
    assert_eq!(once, again);
    // A row past the context bound is refused.
    let past = [[1u32], [16], [0]].map(|v| Tensor::u32(vec![1], v.to_vec()));
    assert!(run_kept(&step, &weights, &mut Store::default(), 0, &past).is_err());
}

/// Runs one launch as a plan. Returns ids, logits.
fn launch(
    g: &Graph,
    runner: &mut Option<Box<dyn Run>>,
    weights: &[Tensor],
    inputs: &[Tensor; 3],
    device: &Device,
) -> (Vec<u32>, Vec<f32>) {
    let rows = inputs[0].shape[0];
    let furthest = inputs[1]
        .indices()
        .unwrap()
        .iter()
        .max()
        .copied()
        .unwrap_or(0);
    let sizes = [
        (Seqs, SEQS as u64),
        (Rows, rows as u64),
        (Context, furthest as u64 + 1),
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
    let views: Vec<_> = inputs.iter().map(Tensor::view).collect();
    r.feed(&views).unwrap();
    r.submit(0..r.plan().steps.len()).unwrap();
    let logits = r.read(logits).unwrap().floats().unwrap().to_vec();
    (r.ids(ids).unwrap(), logits)
}

#[test]
fn plans_keep_rows() {
    let step = stepper(&Config::toy(4, 16));
    let weights = weights(&step, &mut Noise(9));
    let tokens: Vec<u32> = (0..(SEQS * LEN) as u32).map(|t| (t * 7 + 3) % 32).collect();
    let favor = |r: &&lower::Rule| Device::guess(&settings()).favoring(r.name);
    for device in rules().iter().map(favor) {
        let (mut store, mut runner) = (Store::default(), None);
        for rows in launches() {
            let inputs = feeds(&tokens, &rows);
            let want = run_kept(&step, &weights, &mut store, 0, &inputs).unwrap();
            let (ids, logits) = launch(&step, &mut runner, &weights, &inputs, &device);
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
