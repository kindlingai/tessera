//! The lane schedule against the CPU interpreter, on the GPU.

use graph::DimName::{self, Ffn, Hidden, Rows};
use interp::{Cpu, Data, Tensor, run};
use lower::{Device, Dialect, Feed, Source, lower, open, rules};
use ops::testing::{Noise, gpu_available, max_abs_diff};
use trace::{DType, Dim, EntryKind, Graph, Linear, Model, Port, Role, Size, TensorType, Weights};

#[derive(Weights)]
struct Layer {
    #[shape(Hidden, Ffn)]
    w: Linear,
}

/// `k` inputs to `n` outputs, over any number of rows.
fn linear(k: u64, n: u64) -> Graph {
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
    m.finish([(Role::Logits, &x.matmul(&w.w))])
}

fn noise(shape: Vec<usize>, noise: &mut Noise) -> Tensor {
    let data = (0..shape.iter().product()).map(|_| noise.draw()).collect();
    Tensor::f32(shape, data)
}

#[test]
fn rule_registers() {
    assert!(rules().iter().any(|r| r.name == "lanes"));
    assert_eq!((linear::lanes(8), linear::lanes(1024)), (1, 64));
    let g = linear(50, 7);
    let sizes = |m| [(Rows, m)];
    let plan = lower(&g, 0, &sizes(3), &Device::guess().favoring("lanes")).unwrap();
    let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
    assert_eq!(names, ["lanes", "sums"]);
    // Past its row limit, another rule takes the site.
    let wide = lower(&g, 0, &sizes(33), &Device::guess().favoring("lanes")).unwrap();
    assert!(wide.steps.iter().all(|s| s.kernel.name() != "lanes"));
    // Three rows take the body that shares each weight.
    let Source::Made(text) = plan.steps[0].kernel.source() else {
        panic!("lanes are generated");
    };
    for dialect in [Dialect::Metal, Dialect::Cuda] {
        let text = text.render(dialect);
        assert!(text.contains("float c = w[at + i];") && text.contains("linear_rows_f32("));
    }
}

#[test]
fn lanes_match_cpu() {
    if !gpu_available() {
        return eprintln!("no GPU here; skipped");
    }
    let mut draws = Noise(23);
    // Widths that lanes do not divide, and one they do.
    let cases: [(u64, u64, DimName); 3] = [(50, 7, Rows), (1, 3, Rows), (1024, 96, Rows)];
    for (k, n, rows) in cases {
        let g = linear(k, n);
        let weights = [noise(vec![n as usize, k as usize], &mut draws)];
        for m in [1usize, 5] {
            let x = noise(vec![m, k as usize], &mut draws);
            let want = run(&g, &mut Cpu, &weights, 0, std::slice::from_ref(&x)).unwrap();
            let device = Device::guess().favoring("lanes");
            let plan = lower(&g, 0, &[(rows, m as u64)], &device).unwrap();
            assert_eq!(plan.steps[0].kernel.name(), "lanes");
            let runner = open(plan).unwrap();
            runner.load(0, weights[0].floats().unwrap()).unwrap();
            let Data::F32(fed) = &x.data else {
                unreachable!("noise is floats");
            };
            let got = runner.run(&[Feed::Floats(fed)]).unwrap().remove(0);
            let off = max_abs_diff(&got, want[0].floats().unwrap());
            assert!(off < 1e-3, "{m} rows of {k} to {n}: off by {off:e}");
        }
    }
}
