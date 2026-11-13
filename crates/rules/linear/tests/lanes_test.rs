//! The lane schedule against the CPU interpreter, on the GPU.

// The other matmul rules come from here.
use kernels as _;
use linear as _;

use gpukit::testing::{Noise, backend_ready, max_abs_diff};
use graph::DimName::{self, Ffn, Hidden, Rows};
use interp::{Tensor, run};
use lower::{Device, Dialect, Source, lower, open, rules};
use trace::{
    Act, DType, Dim, EntryKind, Graph, Linear, Model, Port, Role, Size, TensorType, Weights,
};

#[derive(Weights)]
struct Layer {
    #[shape(Hidden, Ffn)]
    w: Linear,
}

/// A graph's work on its input and the product.
type Finish = fn(&trace::Tensor, trace::Tensor) -> trace::Tensor;

/// `k` inputs to `n` outputs, over any number of rows.
fn linear(k: u64, n: u64) -> Graph {
    finished(k, n, |_, product| product)
}

/// The same, with stages after the matmul.
fn finished(k: u64, n: u64, finish: Finish) -> Graph {
    let fixed = |name, n| Dim {
        name,
        size: Size::Fixed(n),
    };
    let size = Size::Bounded(64);
    let rows = Dim { name: Rows, size };
    let m = Model::new(vec![rows, fixed(Hidden, k), fixed(Ffn, n)]);
    let w: Layer = trace::declare(&m);
    let ty = TensorType::new([Rows, Hidden], DType::F32);
    let role = Role::Logits;
    let [x] = m.entry(EntryKind::Forward, [Port { role, ty }]);
    m.finish([(Role::Logits, &finish(&x, x.matmul(&w.w)))])
}

fn noise(shape: Vec<usize>, noise: &mut Noise) -> Tensor {
    let data = (0..shape.iter().product()).map(|_| noise.draw()).collect();
    Tensor::f32(shape, data)
}

#[test]
fn rule_registers() {
    assert!(rules().iter().any(|r| r.name == "lanes"));
    let g = linear(52, 7);
    let sizes = |m| [(Rows, m)];
    let plan = lower(&g, 0, &sizes(3), &Device::guess().favoring("lanes")).unwrap();
    let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
    assert_eq!(names, ["lanes"]);
    // Past its row limit, another rule takes the site.
    let wide = lower(&g, 0, &sizes(33), &Device::guess().favoring("lanes")).unwrap();
    assert!(wide.steps.iter().all(|s| s.kernel.name() != "lanes"));
    // A favored rule keeps its node from a hiding cover.
    let residual = finished(48, 48, |x, product| x.add(&product));
    let tiled = Device::guess().favoring("matmul");
    let kept = lower(&residual, 0, &sizes(3), &tiled).unwrap();
    let names: Vec<&str> = kept.steps.iter().map(|s| s.kernel.name()).collect();
    assert_eq!(names, ["matmul", "map"]);
    // Weights come in wide loads, under a hashed entry name.
    let Source::Made(text) = plan.steps[0].kernel.source() else {
        panic!("lanes are generated");
    };
    for dialect in [Dialect::Metal, Dialect::Cuda] {
        let text = text.render(dialect);
        assert!(text.contains("f_f32x4(w + at + i)") && text.contains("linear_f32_"));
    }
}

#[test]
fn lanes_match_cpu() {
    if !backend_ready() {
        return eprintln!("no GPU here; skipped");
    }
    let mut draws = Noise(23);
    // Widths that lanes do not divide, and one they do.
    let plain = [(52, 7), (4, 3), (1024, 96)].map(|(k, n)| (k, n, linear(k, n)));
    // The sum as an operand, then as the chain's input.
    let residual: Finish = |x, product| x.add(&product);
    let gate: Finish = |x, product| product.act(Act::Silu).mul(x);
    let fused = [residual, gate].map(|finish| (48, 48, finished(48, 48, finish)));
    for (k, n, g) in plain.into_iter().chain(fused) {
        let rows: DimName = Rows;
        let weights = [noise(vec![n as usize, k as usize], &mut draws)];
        for m in [1usize, 5] {
            let x = noise(vec![m, k as usize], &mut draws);
            let want = run(&g, &weights, 0, std::slice::from_ref(&x)).unwrap();
            let device = Device::guess().favoring("lanes");
            let plan = lower(&g, 0, &[(rows, m as u64)], &device).unwrap();
            // Stages after the matmul ride in the same launch.
            let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
            assert_eq!(names, ["lanes"]);
            assert_eq!(plan.steps[0].covers.len(), g.nodes.len() - 2);
            let mut runner = open(plan).unwrap();
            runner.load(0, weights[0].view()).unwrap();
            let got = runner.run(&[x.view()]).unwrap().remove(0);
            let off = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
            assert!(off < 1e-3, "{m} rows of {k} to {n}: off by {off:e}");
        }
    }
}
