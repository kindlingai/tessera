//! The tiled matmul against the CPU interpreter, on the GPU.

// The other matmul rules come from here.
use kernels as _;
use linear as _;

use gpukit::testing::{Noise, backend_ready, max_abs_diff};
use graph::DimName::{Ffn, Hidden, Rows};
use interp::{Tensor, run};
use lower::{Device, Dialect, Pins, Source, lower, open, round, rules};
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
    let size = Size::Bounded(256);
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
    assert!(rules().iter().any(|r| r.name == "tiled"));
    let g = linear(72, 50);
    let plan = lower(&g, 0, &[(Rows, 37)], &Device::guess().favoring("tiled")).unwrap();
    let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
    assert_eq!(names, ["tiled"]);
    let Source::Made(text) = plan.steps[0].kernel.source() else {
        panic!("tiled is generated");
    };
    for dialect in [Dialect::Metal, Dialect::Cuda] {
        let text = text.render(dialect);
        assert!(text.contains("TILE_MMA(acc_t[i][j], ta[i], tb[j])"));
    }
}

#[test]
fn tiles_match_cpu() {
    if !backend_ready() {
        return eprintln!("no GPU here; skipped");
    }
    let mut draws = Noise(29);
    // Sizes off the block, on it, and beyond it.
    let sizes = [
        (1, 3, 4),
        (37, 50, 72),
        (64, 64, 64),
        (130, 96, 1024),
        (256, 512, 1024),
    ];
    for (m, n, k) in sizes {
        let g = linear(k, n);
        let x = noise(vec![m as usize, k as usize], &mut draws);
        for form in [DType::F32, DType::Bf16] {
            // The reference sees the weights as the kernel stores them.
            let w = noise(vec![n as usize, k as usize], &mut draws);
            let rounded = w.floats().unwrap().iter().map(|x| round(form, *x));
            let weights = [Tensor::f32(w.shape.clone(), rounded.collect())];
            let want = run(&g, &weights, 0, std::slice::from_ref(&x)).unwrap();
            let device = Device::guess().favoring("tiled");
            let device = device.pinned(Pins::all(form));
            let plan = lower(&g, 0, &[(Rows, m)], &device).unwrap();
            let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
            let name = match form {
                DType::Bf16 => "tiled_bf16",
                _ => "tiled",
            };
            assert_eq!(names, [name]);
            let mut runner = open(plan).unwrap();
            runner.load(0, weights[0].view()).unwrap();
            let got = runner.run(&[x.view()]).unwrap().remove(0);
            let off = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
            assert!(off < 1e-3, "{m}x{n}x{k} {form:?}: off by {off:e}");
            let seconds = runner.run_timed(&[x.view()]).unwrap()[0];
            let rate = 2.0 * (m * n * k) as f64 / seconds / 1e12;
            eprintln!("{m}x{n}x{k} {form:?}: {rate:.2} TFLOP/s, off by {off:.2e}");
        }
    }
}
