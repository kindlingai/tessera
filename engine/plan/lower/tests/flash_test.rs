//! Attention on tiles against the CPU interpreter, on the GPU.

// Linking the crates registers their rules.
use kernels as _;
use linear as _;

use gpukit::testing::{Noise, max_abs_diff};
use graph::DimName::{Context, HeadDim, Heads, Rows, Seqs};
use graph::{DType, Dim, Graph, Size, TensorType};
use interp::{Tensor, run};
use lower::{Device, lower, open};
use settings::read as settings;
use trace::{EntryKind, Mask, Model, Port, Role};

/// Causal attention over `heads` query heads of `width`.
fn attention(heads: u64, width: u64) -> Graph {
    let dim = |name, size| Dim { name, size };
    let bounded = [(Seqs, 4), (Rows, 128), (Context, 256)];
    let mut dims: Vec<Dim> = bounded.map(|(n, max)| dim(n, Size::Bounded(max))).into();
    dims.extend([
        dim(Heads, Size::Fixed(heads)),
        dim(HeadDim, Size::Fixed(width)),
    ]);
    let m = Model::new(dims);
    let port = |rows| Port {
        role: Role::Tokens,
        ty: TensorType::new([Seqs, rows, Heads, HeadDim], DType::F32),
    };
    let [q, k, v] = m.entry(
        EntryKind::Forward,
        [port(Rows), port(Context), port(Context)],
    );
    let out = m.attend(Mask::Causal, 0.125, [&q, &k, &v], None);
    m.finish([(Role::Logits, &out)])
}

fn noise(shape: Vec<usize>, noise: &mut Noise) -> Tensor {
    let data = (0..shape.iter().product()).map(|_| noise.draw()).collect();
    Tensor::f32(shape, data)
}

#[test]
fn flash_matches_cpu() {
    let mut draws = Noise(41);
    // Rows off, on and past the block, with older keys.
    let cases = [
        (1, 1, 1),
        (1, 1, 38),
        (2, 7, 7),
        (1, 33, 33),
        (2, 70, 93),
        (1, 128, 128),
    ];
    for (heads, width) in [(2u64, 16u64), (4, 64), (2, 128)] {
        let g = attention(heads, width);
        for (seqs, rows, keys) in cases {
            let (h, w) = (heads as usize, width as usize);
            let q = noise(vec![seqs, rows, h, w], &mut draws);
            let k = noise(vec![seqs, keys, h, w], &mut draws);
            let v = noise(vec![seqs, keys, h, w], &mut draws);
            let inputs = [q, k, v];
            let want = run(&g, &[], 0, &inputs).unwrap();
            let sizes = [
                (Seqs, seqs as u64),
                (Rows, rows as u64),
                (Context, keys as u64),
            ];
            let device = Device::guess(&settings()).favoring("flash");
            let plan = lower(&g, 0, &sizes, &device).unwrap();
            let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
            assert_eq!(names, ["map", "flash"], "{seqs}x{rows}x{keys} at {width}");
            let mut runner = open(plan, &settings()).unwrap();
            let views: Vec<_> = inputs.iter().map(|t| t.view()).collect();
            let got = runner.run(&views).unwrap().remove(0);
            let off = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
            assert!(
                off < 1e-3,
                "{seqs}x{rows}x{keys} at {width}: off by {off:e}"
            );
        }
    }
}
