//! Checks native Metal batched matmul kernels against the CPU interpreter.

use tessera_kernels as _;
use tessera_metal_kernels as _;

use tessera_graph::DimName::{Heads, Hidden, Rows};
use tessera_graph::{DType, Dim, DimName, EntryKind, Role, TensorType};
use tessera_interp::{Tensor, run};
use tessera_lower::{Device, lower};
use tessera_run::open;
use tessera_run::testing::runs;
use tessera_testkit::{Noise, max_abs_diff};
use tessera_trace::{Model, Port};

const OUT: DimName = DimName::named("Out");

fn batched_graph(batches: u64, n: u64, k: u64) -> tessera_trace::Graph {
    let m = Model::new(vec![
        Dim::fixed(Heads, batches),
        Dim::bounded(Rows, 128),
        Dim::fixed(OUT, n),
        Dim::fixed(Hidden, k),
    ]);
    let [x, w] = m.entry(
        EntryKind::Forward,
        [
            Port {
                role: Role::Logits,
                ty: TensorType::new([Heads, Rows, Hidden], DType::F32),
            },
            Port {
                role: Role::Logits,
                ty: TensorType::new([Heads, OUT, Hidden], DType::F32),
            },
        ],
    );
    let y = x.matmul_rows(&w);
    m.finish([(Role::Logits, &y)])
}

#[test]
fn batched_metal_matches_cpu() {
    let mut draws = Noise(97);
    // Decode (m=1..8), MMA prefill (m=17, 32, 65), and non-aligned edges.
    let cases = [
        (4usize, 1, 19, 128),
        (4, 4, 33, 256),
        (2, 8, 65, 512),
        (3, 17, 37, 67),
        (2, 32, 64, 128),
        (2, 65, 35, 132),
    ];
    for (batches, rows, n, k) in cases {
        let g = batched_graph(batches as u64, n as u64, k as u64);
        let x = Tensor::f32(
            vec![batches, rows, k],
            (0..batches * rows * k).map(|_| draws.draw()).collect(),
        );
        let w = Tensor::f32(
            vec![batches, n, k],
            (0..batches * n * k).map(|_| draws.draw()).collect(),
        );
        let want = run(&g, &[], 0, &[x.clone(), w.clone()]).unwrap();
        for settings in runs() {
            let dev = Device::guess(&settings);
            let plan = lower(&g, 0, &[(Rows, rows as u64)], &dev).unwrap();
            let kname = plan.steps[0].kernel.name();
            if settings.gpu().is_some() {
                let want_name = if rows >= 16 {
                    "batched_metal_mma"
                } else {
                    "batched_metal"
                };
                assert_eq!(kname, want_name, "r={rows} k={k} picked {kname}");
            }
            let mut runner = open(plan, &settings).unwrap();
            let got = runner.run(&[x.view(), w.view()]).unwrap().remove(0);
            let (got, want) = (got.floats().unwrap(), want[0].floats().unwrap());
            let scale = want.iter().fold(1.0f32, |m, v| m.max(v.abs()));
            let off = max_abs_diff(got, want) / scale;
            assert!(off < 1e-4, "b={batches} r={rows} n={n} k={k}: off {off:e}");
        }
    }
}
