//! Verifies native Metal GQA split-K decode and FlashAttention-2 prefill.

use tessera_attention as _;
use tessera_kernels as _;
use tessera_metal_attention as _;

use tessera_graph::Dim;
use tessera_graph::DimName::{Context, Rows, Segments, Seqs};
use tessera_interp::{Store, Tensor, run_kept};
use tessera_lower::form::{BF16_FORM, FP8_FORM, Load};
use tessera_lower::kept::{self, Layout};
use tessera_lower::{Backend, Device, Pick, Pins, Plan, Settings, View, lower};
use tessera_run::testing::runs;
use tessera_run::{Run, open};
use tessera_testkit::{Noise, graphs, max_abs_diff};

fn build_rows(segs: &[(u32, u32, u32)]) -> (Vec<u32>, Vec<u32>) {
    let (mut sl, mut po) = (Vec::new(), Vec::new());
    for &(slot, start, count) in segs {
        for i in 0..count {
            sl.push(slot);
            po.push(start + i);
        }
    }
    (sl, po)
}

fn eval_plan(
    state: &mut Option<Box<dyn Run>>,
    plan: Plan,
    settings: &Settings,
    inputs: &[Tensor],
) -> Tensor {
    if let Some(exec) = state.as_mut() {
        exec.replan(&plan).unwrap();
    } else {
        let mut exec = open(plan.clone(), settings).unwrap();
        if let Some(tbl) = plan.table {
            let rev: Vec<u8> = (0..tbl.len as u32)
                .rev()
                .flat_map(u32::to_le_bytes)
                .collect();
            exec.held_mut().upload(tbl, &rev).unwrap();
        }
        *state = Some(exec);
    }
    let views: Vec<View> = inputs.iter().map(Tensor::view).collect();
    state.as_mut().unwrap().run(&views).unwrap().remove(0)
}

#[test]
fn attn_metal_matches() {
    let Some(settings) = runs()
        .into_iter()
        .find(|s| matches!(s.backend, Pick::Gpu(Backend::Metal)))
    else {
        return;
    };
    let mut draws = Noise(83);
    let fill = |shape: Vec<usize>, d: &mut Noise| {
        let len: usize = shape.iter().product();
        Tensor::f32(shape, (0..len).map(|_| d.draw()).collect())
    };
    let layouts: [&'static dyn Layout; 2] = [&kept::Dense, &kept::PAGED16];
    let forms: [(Option<&'static dyn Load>, f32); 3] = [
        (None, 5e-3),
        (Some(BF16_FORM), 3e-2),
        (Some(FP8_FORM), 5e-2),
    ];
    for &(heads, kv_heads, width) in &[(4u64, 2u64, 64u64), (8, 2, 128), (6, 1, 256)] {
        let g =
            graphs::gqa_attend_rows(&[Dim::bounded(Rows, 64)], [heads, kv_heads, width], 128, 4);
        for (kv_form, tol) in forms {
            let pins = kv_form.map_or_else(Pins::default, |f| Pins::default().kept(f));
            for layout in layouts {
                let mut store = Store::default();
                let mut runner = None;
                for segs in [
                    &[(0u32, 0u32, 24u32), (2, 0, 18)][..],
                    &[(0, 24, 1), (2, 18, 1)][..],
                ] {
                    let (slots, positions) = build_rows(segs);
                    let n = slots.len();
                    let ctx = segs.iter().map(|&(_, s, c)| (s + c) as u64).max().unwrap();
                    let inputs = [
                        fill(vec![n, heads as usize, width as usize], &mut draws),
                        fill(vec![n, kv_heads as usize, width as usize], &mut draws),
                        fill(vec![n, kv_heads as usize, width as usize], &mut draws),
                        Tensor::u32(vec![n], slots),
                        Tensor::u32(vec![n], positions),
                    ];
                    let want = run_kept(&g, &[], &mut store, 0, &inputs).unwrap();
                    let sizes = [
                        (Seqs, 4),
                        (Rows, n as u64),
                        (Context, ctx),
                        (Segments, segs.len() as u64),
                    ];
                    let mut device = Device::guess(&settings).pinned(pins.clone());
                    device.kept = layout;
                    let plan = lower(&g, 0, &sizes, &device).unwrap();
                    let want_k = if n > 16 {
                        "attn_metal_pf"
                    } else {
                        "attn_metal_dec"
                    };
                    let deferred = n > 16
                        && layout.name() == "dense"
                        && width == 128
                        && kv_form.is_none_or(|f| f.name() == "bf16");
                    if !deferred {
                        assert!(plan.steps.iter().any(|s| s.kernel.name() == want_k));
                    }
                    let got = eval_plan(&mut runner, plan, &settings, &inputs);
                    let off = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
                    assert!(off < tol, "d={width} {} n={n}: off {off:e}", layout.name());
                }
            }
        }
    }
    // Split-K decode path at long context.
    let g_long = graphs::gqa_attend_rows(&[Dim::bounded(Rows, 600)], [8, 1, 256], 600, 2);
    let mut store = Store::default();
    let mut runner = None;
    for &(start, count) in &[(0u32, 550u32), (550u32, 1u32)] {
        let (slots, positions) = build_rows(&[(0, start, count)]);
        let n = slots.len();
        let inputs = [
            fill(vec![n, 8, 256], &mut draws),
            fill(vec![n, 1, 256], &mut draws),
            fill(vec![n, 1, 256], &mut draws),
            Tensor::u32(vec![n], slots),
            Tensor::u32(vec![n], positions),
        ];
        let want = run_kept(&g_long, &[], &mut store, 0, &inputs).unwrap();
        let sizes = [
            (Seqs, 2),
            (Rows, n as u64),
            (Context, (start + count) as u64),
            (Segments, 1),
        ];
        let mut device = Device::guess(&settings).pinned(Pins::default().kept(FP8_FORM));
        device.kept = &kept::PAGED16;
        let plan = lower(&g_long, 0, &sizes, &device).unwrap();
        if count == 1 {
            let names: Vec<_> = plan.steps.iter().map(|s| s.kernel.name()).collect();
            assert!(
                names.contains(&"attn_metal_dec"),
                "expected attn_metal_dec, got {names:?}"
            );
            assert!(
                names.contains(&"attn_metal_reduce"),
                "expected attn_metal_reduce, got {names:?}"
            );
        } else {
            assert!(
                plan.steps
                    .iter()
                    .any(|s| s.kernel.name() == "attn_metal_pf")
            );
        }
        let got = eval_plan(&mut runner, plan, &settings, &inputs);
        let off = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
        assert!(off < 5e-2, "split-K n={n}: off {off:e}");
    }
}
