//! Parity and planner tests for large-`k` radix `TopK`.

use tessera_attention as _;
use tessera_kernels as _;
use tessera_linear as _;

use tessera_graph::DimName::{Context, Rows, Seqs, TopK};
use tessera_graph::{Dim, Graph, Placement};
use tessera_interp::{Tensor, run};
use tessera_lower::{Device, Dialect, Limits, Platform, Source, View, lower};
use tessera_run::open;
use tessera_run::testing::{host, runs};
use tessera_testkit::graphs::{self, floats};
use tessera_testkit::{Noise, fill, max_abs_diff};
use tessera_trace::{EntryKind, Model, Role};

fn top_k_graph(rows: u64, width: u64, k: u64, split: bool) -> Graph {
    let m = Model::new(vec![
        Dim::bounded(Rows, rows),
        Dim::bounded(Context, width),
        Dim::fixed(TopK, k),
    ]);
    let mut ty = floats([Rows, Context]);
    if split {
        ty.ty.placement = Placement::Split { axis: 1 };
    }
    let [x] = m.entry(EntryKind::Forward, [ty]);
    m.finish([(Role::Tokens, &x.top_k(TopK))])
}

fn check_parity(rows: usize, width: usize, k: usize, data: Vec<f32>, want_kernel: &str) {
    let g = top_k_graph(rows as u64, width as u64, k as u64, false);
    let sizes = [(Rows, rows as u64), (Context, width as u64)];
    let input = Tensor::f32(vec![rows, width], data);
    let want = run(&g, &[], 0, std::slice::from_ref(&input)).unwrap();
    for settings in runs() {
        let device = Device::guess(&settings);
        let plan = lower(&g, 0, &sizes, &device).unwrap();
        assert_eq!(plan.steps.len(), 1);
        assert_eq!(plan.steps[0].kernel.name(), want_kernel);
        let mut runner = open(plan, &settings).unwrap();
        let got = runner.run(&[input.view()]).unwrap().remove(0);
        assert_eq!(
            got.indices().unwrap(),
            want[0].indices().unwrap(),
            "rows={rows} width={width} k={k} backend={:?}",
            settings.backend
        );
    }
}

/// Builds 10 edge-case rows of `width` floats.
fn edge_rows(width: usize, k: usize) -> Vec<f32> {
    let mut noise = Noise(42);
    let mut out = Vec::with_capacity(10 * width);
    // Row 0: random noise.
    out.extend((0..width).map(|_| noise.draw()));
    // Row 1: all negative infinity.
    out.extend(std::iter::repeat_n(f32::NEG_INFINITY, width));
    // Row 2: sparse finite values and NaNs.
    for i in 0..width {
        let v = if i % (width / 10).max(1) == 3 {
            (i as f32) * 0.25 - 5.0
        } else if i % 7 == 0 {
            f32::NAN
        } else {
            f32::NEG_INFINITY
        };
        out.push(v);
    }
    // Row 3: heavy ties across the threshold.
    for i in 0..width {
        let v = if i < k / 2 {
            5.0
        } else if i < k * 2 {
            1.0
        } else {
            -1.0
        };
        out.push(v);
    }
    // Row 4: signed zeros, infinities and NaNs.
    for i in 0..width {
        let v = match i % 8 {
            0 => -0.0,
            1 => 0.0,
            2 => f32::NAN,
            3 => f32::NEG_INFINITY,
            4 if i < 24 => f32::INFINITY,
            _ => -0.0,
        };
        out.push(v);
    }
    // Row 5: negatives and subnormals.
    let min_sub = f32::from_bits(1);
    for i in 0..width {
        let v = match i % 7 {
            0 => -f32::MAX,
            1 => -1e20,
            2 => -min_sub,
            3 => -0.0,
            4 => min_sub,
            5 => -((i + 1) as f32),
            _ => -1.0 / (i + 1) as f32,
        };
        out.push(v);
    }
    // Row 6: all equal.
    out.extend(std::iter::repeat_n(3.5f32, width));
    // Row 7: strictly ascending.
    out.extend((0..width).map(|i| i as f32));
    // Row 8: strictly descending.
    out.extend((0..width).map(|i| (width - i) as f32));
    // Row 9: exact top-k split triggering early exit.
    for i in 0..width {
        out.push(if i < k { 100.0 + i as f32 } else { -10.0 });
    }
    out
}

#[test]
fn radix_parity() {
    // Edge cases on odd width with k = 64.
    check_parity(10, 1025, 64, edge_rows(1025, 64), "radix_top_k");
    // Width <= k cases.
    let mut noise = Noise(9);
    let small: Vec<f32> = (0..4 * 128).map(|_| noise.draw()).collect();
    check_parity(4, 128, 256, small, "radix_top_k");
    let eq: Vec<f32> = (0..2 * 2048).map(|_| noise.draw()).collect();
    check_parity(2, 2048, 2048, eq, "radix_top_k");
    // Multi-row batch.
    let batch: Vec<f32> = (0..64 * 1024).map(|_| noise.draw()).collect();
    check_parity(64, 1024, 64, batch, "radix_top_k");
    // GLM-5.3 scale: k = 2048 at 5000 and 131072.
    let mid: Vec<f32> = (0..4 * 5000).map(|_| noise.draw()).collect();
    check_parity(4, 5000, 2048, mid, "radix_top_k");
    let mut big = Vec::with_capacity(2 * 131_072);
    big.extend((0..131_072).map(|_| noise.draw()));
    for i in 0..131_072 {
        big.push(if i % 40 == 0 {
            ((i % 17) as f32) - 4.0
        } else {
            f32::NEG_INFINITY
        });
    }
    check_parity(2, 131_072, 2048, big, "radix_top_k");
    // Low-mantissa values test the third 10-bit pass.
    let base = 1.0f32.to_bits();
    let fine: Vec<f32> = (0..4096u32)
        .map(|i| f32::from_bits(base + ((i.wrapping_mul(2654435761)) & 0x3ff)))
        .collect();
    check_parity(1, 4096, 128, fine, "radix_top_k");
    // Uncached key path: k = 4096.
    let wide_k: Vec<f32> = (0..2 * 6000)
        .map(|i| if i % 11 == 0 { f32::NAN } else { noise.draw() })
        .collect();
    check_parity(2, 6000, 4096, wide_k, "radix_top_k_4096");
}

#[test]
fn radix_plan_and_split() {
    // Planner selects radix_top_k with O(N) work and zero scratch.
    let g = top_k_graph(1, 131_072, 2048, false);
    let sizes = [(Rows, 1), (Context, 131_072)];
    let plan = lower(&g, 0, &sizes, &Device::guess(&host())).unwrap();
    assert_eq!(plan.steps.len(), 1);
    assert_eq!(plan.steps[0].kernel.name(), "radix_top_k");
    let work = plan.steps[0].kernel.work(2048);
    assert!(work < 2_000_000.0, "expected O(N) work, got {work}");
    // All backends including WGSL render cleanly.
    for k in [2048u64, 4096, 8192] {
        let g = top_k_graph(1, 16_384, k, false);
        let plan = lower(
            &g,
            0,
            &[(Rows, 1), (Context, 16_384)],
            &Device::guess(&host()),
        )
        .unwrap();
        let Source::Made(body) = plan.steps[0].kernel.source() else {
            panic!("expected codegen body");
        };
        let cuda = body.text().render(Dialect::Cuda);
        let metal = body.text().render(Dialect::Metal);
        let wgsl = body.wgsl().expect("wgsl renders").text;
        assert!(cuda.contains("atomicAdd") && metal.contains("atomic_fetch_add_explicit"));
        assert!(wgsl.contains("atomicAdd") && wgsl.contains("atomicLoad"));
    }
    // 32 KB Metal falls back to top_k for k = 8192.
    let metal = Platform::assumed(Limits {
        group_bytes: 32 * 1024,
        ..Limits::assumed()
    });
    let g8 = top_k_graph(1, 16_384, 8192, false);
    let p8 = lower(
        &g8,
        0,
        &[(Rows, 1), (Context, 16_384)],
        &Device::default().on(metal),
    )
    .unwrap();
    assert_eq!(p8.steps[0].kernel.name(), "top_k");
    // Split rank exercises nonzero `first` and `whole > width`.
    let gs = top_k_graph(2, 256, 128, true);
    let s = host();
    let dev = Device::guess(&s).ranked(vec![0, 1]).as_rank(1);
    let plan = lower(&gs, 0, &[(Rows, 2), (Context, 256)], &dev).unwrap();
    assert_eq!(plan.steps[0].kernel.name(), "radix_top_k");
    let mut data = vec![f32::NEG_INFINITY; 2 * 256];
    for (i, slot) in data.iter_mut().enumerate().take(80) {
        *slot = i as f32;
    }
    for (i, slot) in data[128..148].iter_mut().enumerate() {
        *slot = i as f32;
    }
    let input = Tensor::f32(vec![2, 256], data);
    let mut runner = open(plan, &s).unwrap();
    let got = runner.run(&[input.view()]).unwrap().remove(0);
    let out: Vec<u32> = got
        .bytes()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect();
    assert_eq!(out[0], 128 + 79);
    assert_eq!(out[63], 128 + 16);
    assert_eq!(out[64], 128 + 19);
    assert_eq!(out[64 + 20], 256);
}

#[test]
fn radix_indexed_attention() {
    let dims = vec![Dim::bounded(Seqs, 2), Dim::bounded(Context, 256)];
    let g = graphs::indexed(&dims, [2, 16, 64]);
    let sizes = [(Seqs, 1), (Context, 160)];
    let mut noise = Noise(17);
    let inputs: Vec<Tensor> = g.entries[0]
        .inputs
        .iter()
        .map(|p| fill(&g, &p.ty, &sizes, &mut noise).unwrap())
        .collect();
    let want = run(&g, &[], 0, &inputs).unwrap();
    for settings in runs() {
        let plan = lower(&g, 0, &sizes, &Device::guess(&settings)).unwrap();
        assert!(plan.steps.iter().any(|s| s.kernel.name() == "radix_top_k"));
        let mut runner = open(plan, &settings).unwrap();
        let feeds: Vec<View> = inputs.iter().map(Tensor::view).collect();
        let got = runner.run(&feeds).unwrap().remove(0);
        let diff = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
        assert!(diff < 1e-4, "off by {diff:e}");
    }
}

#[test]
fn take_narrow_parity() {
    use tessera_graph::DimName::Hidden;
    use tessera_graph::{Act, DType, TensorType};
    use tessera_lower::Pins;
    use tessera_lower::form::{BF16_FORM, F32_FORM, FP8_FORM, Load as Stored};
    use tessera_trace::Port;

    let cases: [(usize, usize, &'static dyn Stored, &'static dyn Stored, f32); 6] = [
        (64, 8, F32_FORM, F32_FORM, 1e-5),
        (64, 6, F32_FORM, F32_FORM, 1e-5),
        (64, 8, BF16_FORM, BF16_FORM, 2e-2),
        (64, 8, BF16_FORM, F32_FORM, 2e-2),
        (64, 8, F32_FORM, BF16_FORM, 2e-2),
        (64, 8, FP8_FORM, F32_FORM, 2e-1),
    ];
    for (width, picks, act, out_form, tol) in cases {
        let m = Model::new(vec![
            Dim::bounded(Rows, 16),
            Dim::fixed(Hidden, width as u64),
            Dim::fixed(TopK, picks as u64),
        ]);
        let p_x = Port {
            role: Role::Logits,
            ty: TensorType::new([Rows, Hidden], DType::F32),
        };
        let p_ids = Port {
            role: Role::Tokens,
            ty: TensorType::new([Rows, TopK], DType::U32),
        };
        let [x_in, ids_in] = m.entry(EntryKind::Forward, [p_x, p_ids]);
        let g = m.finish([(Role::Logits, &x_in.act(Act::Silu).take(&ids_in))]);
        let rows = 4usize;
        let sizes = [(Rows, rows as u64)];
        let x = fill(&g, &g.entries[0].inputs[0].ty, &sizes, &mut Noise(61)).unwrap();
        // Include out-of-range ids to verify zero fill.
        let ids_vec: Vec<u32> = (0..rows * picks)
            .map(|i| {
                if i % 7 == 0 {
                    width as u32 + 5
                } else {
                    (i * 13 % width) as u32
                }
            })
            .collect();
        let ids = Tensor::u32(vec![rows, picks], ids_vec);
        let want = run(&g, &[], 0, &[x.clone(), ids.clone()]).unwrap();
        let dev = Device::guess(&host()).pinned(Pins::all(act).activations(act).outputs(out_form));
        let plan = lower(&g, 0, &sizes, &dev).unwrap();
        let step = plan
            .steps
            .iter()
            .find(|s| s.kernel.name() == "take")
            .expect("take step");
        assert!(!plan.steps.iter().any(|s| s.kernel.name() == "widen"));
        assert!(!plan.steps.iter().any(|s| s.kernel.name() == "narrow"));
        if let Source::Made(body) = step.kernel.source() {
            let want_entry = match (act.name(), out_form.name()) {
                ("f32", "f32") => "plan_take".into(),
                (a, b) => format!("plan_take_{a}_{b}"),
            };
            assert_eq!(body.entry, want_entry);
        }
        let feeds = [x.view(), ids.view()];
        for settings in runs() {
            let mut runner = open(plan.clone(), &settings).unwrap();
            let got = runner.run(&feeds).unwrap();
            let diff = max_abs_diff(got[0].floats().unwrap(), want[0].floats().unwrap());
            assert!(
                diff < tol,
                "picks={picks} {}->{} diff={diff:e}",
                act.name(),
                out_form.name()
            );
        }
    }
}
