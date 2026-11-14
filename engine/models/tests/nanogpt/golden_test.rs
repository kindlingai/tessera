//! The nanogpt, run from JSON alone, against recorded logits.

use super::config;
use gpukit::testing::{Noise, max_abs_diff};
use interp::{Tensor, run};
use lower::{Device, lower, open};
use models::nanogpt::program;
use settings::read as settings;
use trace::DimName::{Context, Seqs};
use trace::Graph;

/// Two sequences of five tokens.
const TOKENS: [u32; 10] = [1, 5, 2, 7, 0, 3, 3, 6, 4, 1];

/// What a separately written forward pass gave for them.
const WANT: [f32; 80] = [
    -0.5819097,
    0.9464245,
    -0.25651923,
    0.44374448,
    0.42700422,
    0.5043015,
    -0.17512685,
    -0.19725302,
    -0.3801139,
    1.0426403,
    -0.8008181,
    0.24591343,
    0.43158793,
    0.3266746,
    0.2642794,
    -0.4526164,
    -0.3201371,
    0.41653788,
    -0.5565316,
    0.41563302,
    0.20271699,
    0.5936879,
    0.1861455,
    -0.17288274,
    -0.47850513,
    0.8474452,
    -1.0114443,
    0.5201106,
    0.2744798,
    0.15248644,
    0.4764801,
    -0.3163423,
    -0.075451165,
    0.7937894,
    -0.5690401,
    0.5649995,
    0.42009616,
    -0.044843018,
    0.5187617,
    0.053624272,
    -0.96142167,
    0.9565887,
    -0.4553787,
    0.72382903,
    0.45841226,
    0.19055681,
    -0.088227674,
    0.053602763,
    -0.8020006,
    0.6133259,
    -0.6831398,
    0.8884844,
    0.39828232,
    0.29584026,
    0.1484291,
    0.111393385,
    -0.205567,
    0.8246002,
    -0.5287255,
    0.55457497,
    0.41442686,
    0.058069207,
    0.40274623,
    0.09468585,
    -0.6795289,
    0.4615848,
    -0.64473045,
    0.8350158,
    0.33846933,
    -0.10223823,
    0.079992265,
    0.23728693,
    -0.13590321,
    1.0223426,
    0.096325606,
    0.22274908,
    0.40421975,
    -0.09247392,
    -0.28796703,
    0.14493969,
];

/// The interpreter only ever sees the JSON text.
fn graph() -> Graph {
    Graph::from_json(&program(&config()).to_json()).unwrap()
}

/// The weights `WANT` was recorded with, in graph order.
fn weights(g: &Graph) -> Vec<Tensor> {
    let mut noise = Noise(1337);
    let mut make = |w: &graph::Weight| {
        let size = |e: &trace::DimExpr| e.poly(&g.dims).unwrap().as_const().unwrap() as usize;
        let shape: Vec<usize> = w.ty.shape.0.iter().map(size).collect();
        // Norm weights sit near one. The rest sit near zero.
        let value = |noise: &mut Noise| match shape.len() {
            1 => 1.0 + 0.1 * noise.draw(),
            _ => 0.3 * noise.draw(),
        };
        let values = (0..shape.iter().product()).map(|_| value(&mut noise));
        Tensor::f32(shape.clone(), values.collect())
    };
    g.weights.iter().map(&mut make).collect()
}

fn positions() -> Vec<u32> {
    (0..2).flat_map(|_| 0..5).collect()
}

fn close(label: &str, got: &[f32]) {
    let off = max_abs_diff(got, &WANT);
    eprintln!("{label}: max_abs {off:.3e}");
    assert!(off < 1e-5, "{label} is off by {off:e}");
}

#[test]
fn cpu_matches_record() {
    let g = graph();
    let ids = |data: Vec<u32>| Tensor::u32(vec![2, 5], data);
    let inputs = [ids(TOKENS.to_vec()), ids(positions())];
    let out = run(&g, &weights(&g), 0, &inputs).unwrap();
    close("cpu", out[0].floats().unwrap());
}

#[test]
fn plan_matches_record() {
    let g = graph();
    let plan = lower(
        &g,
        0,
        &[(Seqs, 2), (Context, 5)],
        &Device::guess(&settings()),
    )
    .unwrap();
    let mut runner = open(plan, &settings()).unwrap();
    for (at, w) in weights(&g).iter().enumerate() {
        runner.load(at as u32, w.view()).unwrap();
    }
    let ids = |data: Vec<u32>| Tensor::u32(vec![2, 5], data);
    let inputs = [ids(TOKENS.to_vec()), ids(positions())];
    let out = runner.run(&[inputs[0].view(), inputs[1].view()]).unwrap();
    close("plan", out[0].floats().unwrap());
}
