//! CPU parity tests for Metal argmax and small-k top-k.

use tessera_kernels as _;
use tessera_metal_kernels as _;

use tessera_graph::DimName::{Context, Rows, TopK};
use tessera_graph::{DType, Dim, Placement, TensorType};
use tessera_interp::{Tensor, run};
use tessera_lower::{Backend, Device, Settings, lower};
use tessera_run::open;
use tessera_run::testing::runs;
use tessera_testkit::Noise;
use tessera_trace::{EntryKind, Model, Port, Role};

fn metal_settings() -> Option<Settings> {
    runs().into_iter().find(|s| s.gpu() == Some(Backend::Metal))
}

fn stress_data(rows: usize, width: usize) -> Vec<f32> {
    let mut noise = Noise(97);
    let mut out = Vec::with_capacity(rows * width);
    for r in 0..rows {
        match r % 6 {
            0 => out.extend((0..width).map(|_| noise.draw())),
            1 => out.extend(std::iter::repeat_n(f32::NEG_INFINITY, width)),
            2 => {
                for i in 0..width {
                    let v = match i % 7 {
                        0 => -0.0,
                        1 => 0.0,
                        2 => f32::NAN,
                        3 => f32::NEG_INFINITY,
                        _ => ((i % 13) as f32) - 6.0,
                    };
                    out.push(v);
                }
            }
            3 => out.extend(std::iter::repeat_n(2.5f32, width)),
            4 => {
                for i in 0..width {
                    out.push(if i % 97 == 5 {
                        f32::INFINITY
                    } else if i % 5 == 0 {
                        f32::NAN
                    } else {
                        -f32::MAX / ((i % 11 + 1) as f32)
                    });
                }
            }
            _ => out.extend((0..width).map(|i| (width - i) as f32 * 0.01)),
        }
    }
    out
}

#[test]
fn argmax_metal_matches_cpu() {
    let Some(settings) = metal_settings() else {
        return;
    };
    let cases: &[(usize, usize, usize, &str)] = &[
        // Single-launch argmax (small width or many rows).
        (1, 1, 1, "argmax_row_metal"),
        (6, 7, 1, "argmax_row_metal"),
        (6, 513, 1, "argmax_row_metal"),
        (32, 16_384, 1, "argmax_row_metal"),
        // Split-width argmax (few rows, large vocab).
        (1, 32_771, 1, "argmax_fin_metal"),
        (4, 131_072, 1, "argmax_fin_metal"),
        (10, 16_384, 1, "argmax_fin_metal"),
        // Single-launch shared-memory small-k top-k.
        (6, 256, 8, "topk_small_metal"),
        (4, 513, 16, "topk_small_metal"),
        (6, 2048, 32, "topk_small_metal"),
        (3, 5, 8, "topk_small_metal"),
        // Two-stage lazy chunk-pop wide top-k.
        (6, 4_097, 8, "topk_select_metal"),
        (3, 65_539, 32, "topk_select_metal"),
        (1, 131_072, 8, "topk_select_metal"),
    ];
    for &(rows, width, k, expected) in cases {
        let model = Model::new(vec![
            Dim::fixed(TopK, k as u64),
            Dim::bounded(Context, width as u64),
            Dim::bounded(Rows, rows as u64),
        ]);
        let [x] = model.entry(
            EntryKind::Forward,
            [Port {
                role: Role::Tokens,
                ty: TensorType::new([Rows, Context], DType::F32),
            }],
        );
        let g = model.finish([(Role::Tokens, &x.top_k(TopK))]);
        let data = stress_data(rows, width);
        let input = Tensor::f32(vec![rows, width], data);
        let want = run(&g, &[], 0, std::slice::from_ref(&input)).unwrap();

        let sizes = [(Rows, rows as u64), (Context, width as u64)];
        let device = Device::guess(&settings);
        let plan = lower(&g, 0, &sizes, &device).unwrap();
        let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
        assert!(
            names.contains(&expected),
            "rows={rows} width={width} k={k}: expected {expected} in {names:?}"
        );

        let mut runner = open(plan, &settings).unwrap();
        let got = runner.run(&[input.view()]).unwrap().remove(0);
        assert_eq!(
            got.indices().unwrap(),
            want[0].indices().unwrap(),
            "rows={rows} width={width} k={k}"
        );
    }

    // Split rank verifies nonzero `first` and `whole > width`.
    for &(width, k) in &[(512usize, 2usize), (512, 8), (16_384, 2), (16_384, 8)] {
        let model = Model::new(vec![
            Dim::fixed(TopK, k as u64),
            Dim::bounded(Context, width as u64),
            Dim::bounded(Rows, 2),
        ]);
        let mut ty = TensorType::new([Rows, Context], DType::F32);
        ty.placement = Placement::Split { axis: 1 };
        let [x] = model.entry(
            EntryKind::Forward,
            [Port {
                role: Role::Tokens,
                ty,
            }],
        );
        let g = model.finish([(Role::Tokens, &x.top_k(TopK))]);
        let device = Device::guess(&settings).ranked(vec![0, 1]).as_rank(1);
        let plan = lower(&g, 0, &[(Rows, 2), (Context, width as u64)], &device).unwrap();
        let half = width / 2;
        let local_k = k / 2;
        let mut data = vec![f32::NEG_INFINITY; 2 * width];
        for i in 0..local_k.min(4) {
            data[10 + i] = 100.0 - i as f32;
        }
        let input = Tensor::f32(vec![2, width], data);
        let mut runner = open(plan, &settings).unwrap();
        let got = runner.run(&[input.view()]).unwrap().remove(0);
        let idx: Vec<u32> = got
            .bytes()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect();
        assert_eq!(idx[0], (half + 10) as u32, "width={width} k={k}");
        assert_eq!(idx[local_k], width as u32, "row 1 all -inf should be whole");
    }
}
