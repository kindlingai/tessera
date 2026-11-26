use crate::dense::TILED_RULE;
use tessera_graph::{Act, Dim, DimName::Rows};
use tessera_interp::Tensor;
use tessera_lower::{Device, Pick, Settings};
use tessera_testkit::{Noise, weights};

/// CPU row stages after a product match the interpreter.
#[test]
fn cpu_row_stages() {
    let settings = Settings {
        backend: Pick::Cpu,
        ..tessera_run::testing::host()
    };
    let sizes = [(Rows, 3)];
    let acts = [
        Act::GeluTanh,
        Act::Silu,
        Act::Sigmoid,
        Act::Relu,
        Act::Softplus,
        Act::Exp,
        Act::Tanh,
        Act::SignedSqrt,
    ];
    for act in acts {
        let g = tessera_testkit::graphs::finished(&[Dim::bounded(Rows, 4)], [32, 32], act);
        let mut noise = Noise(9);
        let stored = weights(&g, &mut noise);
        let ports = g.entries[0].inputs.iter();
        let inputs: Vec<Tensor> = ports
            .map(|p| tessera_testkit::fill(&g, &p.ty, &sizes, &mut noise).unwrap())
            .collect();
        let want = tessera_interp::run(&g, &stored, 0, &inputs).unwrap();
        let device = Device::guess(&settings).favoring(TILED_RULE);
        let plan = tessera_lower::lower(&g, 0, &sizes, &device).unwrap();
        assert!(plan.steps.iter().any(|s| s.rule == TILED_RULE));
        let mut runner = tessera_run::open(plan, &settings).unwrap();
        for (at, w) in stored.iter().enumerate() {
            runner.load(at as u32, w.view()).unwrap();
        }
        let feeds: Vec<_> = inputs.iter().map(Tensor::view).collect();
        let got = runner.run(&feeds).unwrap().remove(0);
        let pairs = got.floats().unwrap().iter().zip(want[0].floats().unwrap());
        let off = pairs.map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(off < 1e-4, "{act:?}: off by {off:e}");
    }
}
