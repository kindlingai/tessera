//! A step entry's ragged rows against whole forward passes.

use super::{Spec, loaded};
use sensai_interp::{Store, Tensor, run, run_kept};
use sensai_lower::{Cache, Device, View};
use sensai_run::Run;
use sensai_run::testing::runs;
use sensai_testkit::{Noise, argmax, cosine_f32, max_abs_diff, weights};
use sensai_trace::DimName::{Context, Rows, Seqs};
use sensai_trace::Graph;

const SEQS: usize = 2;
const LEN: usize = 7;

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

/// Rows mixing both sequences at different positions, in three launches.
fn launches() -> [Vec<(usize, usize)>; 3] {
    [
        vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)],
        vec![(1, 2), (0, 3), (1, 3)],
        vec![(1, 4), (1, 5), (1, 6), (0, 4), (0, 5), (0, 6)],
    ]
}

/// The small step graph, launch by launch, against `small`.
pub fn steps(spec: &Spec) {
    let (Some(small), Some(stepper)) = (spec.small, &spec.stepper) else {
        return;
    };
    let (forward, step) = (small(), (stepper.small)());
    assert_eq!(forward.weights, step.weights);
    let mut noise = Noise(3);
    let weights = weights(&forward, &mut noise);
    let draw = |_| ((noise.draw() + 1.0) * 16.0) as u32;
    let tokens: Vec<u32> = (0..SEQS * LEN).map(draw).collect();
    let want = run(&forward, &weights, 0, &whole(&tokens)).unwrap();
    let want = want[0].floats().unwrap();
    let vocab = want.len() / (SEQS * LEN);
    let mut store = Store::default();
    let mut got = vec![0.0; want.len()];
    let mut each = Vec::new();
    for rows in launches() {
        let out = run_kept(&step, &weights, &mut store, 0, &feeds(&tokens, &rows)).unwrap();
        each.push(out[1].floats().unwrap().to_vec());
        let (ids, logits) = (out[0].indices().unwrap(), out[1].floats().unwrap());
        assert_eq!(out[0].shape, [rows.len()]);
        for (at, row) in logits.chunks(vocab).enumerate() {
            let (s, p) = rows[at];
            assert_eq!(ids[at], argmax(row));
            got[(s * LEN + p) * vocab..][..vocab].copy_from_slice(row);
        }
    }
    for kept in 0..step.kept.len() {
        assert!(store.len(kept) >= LEN, "kept rows {kept}");
    }
    let off = max_abs_diff(&got, want);
    assert!(
        off < stepper.drift,
        "steps drift from the forward pass by {off:e}"
    );
    planned(spec.name, &step, &weights, &tokens, &each);
}

/// The same launches as plans on each device.
fn planned(name: &str, step: &Graph, weights: &[Tensor], tokens: &[u32], want: &[Vec<f32>]) {
    for settings in runs() {
        let device = Device::guess(&settings);
        let mut plans = Cache::default();
        let mut runner: Option<Box<dyn Run>> = None;
        // Kept rows span the whole context, as buffers hold them.
        let context = step
            .dims
            .iter()
            .find(|d| d.name == Context)
            .map(|d| d.size.max());
        for (rows, want) in launches().iter().zip(want) {
            let sizes = [
                (Seqs, SEQS as u64),
                (Rows, rows.len() as u64),
                (Context, context.unwrap_or(LEN as u64)),
            ];
            let plan = plans.lower(step, 0, &sizes, &device).unwrap();
            let runner = match &mut runner {
                Some(runner) => {
                    runner.replan(&plan).unwrap();
                    runner
                }
                None => runner.insert(loaded(plan.clone(), step, weights, &settings)),
            };
            let tensors = feeds(tokens, rows);
            let views: Vec<View> = tensors.iter().map(Tensor::view).collect();
            runner.feed(&views).unwrap();
            runner.submit(0..runner.plan().steps.len()).unwrap();
            let out = runner.read(runner.plan().outputs[1].slot).unwrap();
            let got = &out.floats().unwrap()[..want.len()];
            let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let (off, cos) = (max_abs_diff(got, want) / scale, cosine_f32(got, want));
            let label = format!("{name} step of {} rows on {}", rows.len(), runner.device());
            assert!(off < 1e-3 && cos > 0.9999, "{label}: off {off:e} cos {cos}");
        }
    }
}
