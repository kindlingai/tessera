//! The per-model suite: what every model test runs alike.

use gpukit::testing::{Noise, cosine_f32, max_abs_diff};
use interp::{Tensor, run};
use lower::form::named;
use lower::{Device, Pins, View, lower, open};
use settings::read as settings;
use trace::DimName::{Context, Seqs};
use trace::{DType, DimName, Graph, TensorType};

/// One model as the suite sees it.
pub struct Spec {
    pub name: &'static str,
    /// The forward graph, small enough for the host executor.
    pub small: fn() -> Graph,
    /// Forms its weights may be stored in, by name.
    pub forms: &'static [&'static str],
}

/// The tests every model runs: `common::suite!("name", small)`.
macro_rules! suite {
    ($name:literal, $small:expr) => {
        const SPEC: common::Spec = common::Spec {
            name: $name,
            small: $small,
            forms: &["f32", "bf16", "q8_0", "q4_0", "nvfp4"],
        };

        #[test]
        fn faux_weights() {
            common::faux(&SPEC);
        }
    };
}
pub(crate) use suite;

/// Rows the faux run feeds.
const ROWS: u64 = 8;

/// Random floats, or ids below 8, shaped like `ty`.
fn fill(g: &Graph, ty: &TensorType, noise: &mut Noise) -> Tensor {
    let sizes = [(Seqs, 1), (Context, ROWS)];
    let size = |name: DimName| sizes.iter().find(|(n, _)| *n == name).map(|(_, s)| *s);
    let dim = |e: &graph::DimExpr| e.poly(&g.dims).unwrap().eval(&size).unwrap() as usize;
    let shape: Vec<usize> = ty.shape.0.iter().map(dim).collect();
    let mut values = (0..shape.iter().product()).map(|_| noise.draw());
    match ty.dtype {
        DType::U32 => Tensor::u32(shape, values.map(|v| ((v + 1.0) * 4.0) as u32).collect()),
        _ => Tensor::f32(shape, values.by_ref().collect()),
    }
}

/// Random weights in every form: device against interpreter.
pub fn faux(spec: &Spec) {
    let g = (spec.small)();
    let mut noise = Noise(17);
    let weights: Vec<Tensor> = (g.weights.iter())
        .map(|w| fill(&g, &w.ty, &mut noise))
        .collect();
    let ports = &g.entries[0].inputs;
    let inputs: Vec<Tensor> = ports.iter().map(|p| fill(&g, &p.ty, &mut noise)).collect();
    let feeds: Vec<View> = inputs.iter().map(Tensor::view).collect();
    let settings = settings();
    for form in spec.forms {
        let f = named(form).unwrap_or_else(|| panic!("{form} is a registered form"));
        // The interpreter sees the weights as the form keeps them.
        let kept: Vec<Tensor> = (weights.iter())
            .map(|t| Tensor::f32(t.shape.clone(), f.roundtrip(t.floats().unwrap())))
            .collect();
        let want = run(&g, &kept, 0, &inputs).unwrap();
        let device = Device::guess(&settings).pinned(Pins::all(f));
        let plan = lower(&g, 0, &[(Seqs, 1), (Context, ROWS)], &device).unwrap();
        let mut runner = open(plan, &settings).unwrap();
        for (at, w) in kept.iter().enumerate() {
            runner.load(at as u32, w.view()).unwrap();
        }
        let got = runner.run(&feeds).unwrap().remove(0);
        let (got, want) = (got.floats().unwrap(), want[0].floats().unwrap());
        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let (off, cos) = (max_abs_diff(got, want) / scale, cosine_f32(got, want));
        eprintln!(
            "{} as {form} on {}: off {off:.2e}, cos {cos:.7}",
            spec.name,
            runner.device()
        );
        assert!(
            off < 1e-3 && cos > 0.9999,
            "{} as {form}: off {off:e} cos {cos}",
            spec.name
        );
    }
}
