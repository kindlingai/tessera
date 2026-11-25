//! Parity, narrow-form, and safety tests for `add_rms_norm`.

use tessera_kernels as _;

use tessera_graph::DimName::{Hidden, Rows};
use tessera_graph::{DType, Dim, Graph, TensorType};
use tessera_interp::{Tensor, run};
use tessera_lower::form::{BF16_FORM, F32_FORM, FP8_FORM};
use tessera_lower::{Device, Limits, Pins, Platform, Site, View, lower};
use tessera_run::open;
use tessera_run::testing::runs;
use tessera_testkit::{Noise, fill, max_abs_diff};
use tessera_trace::{Act, EntryKind, Model, Norm, Port, Role, Weights, declare};

#[derive(Weights)]
struct NormWeight {
    #[shape(Hidden)]
    w: Norm,
}

/// GPU platform with a chip model for cost-based lowering.
fn gpu_device() -> Device {
    let gpu = Platform {
        backend: "cuda".into(),
        name: "Test GPU".into(),
        units: 128,
        bandwidth: 1e12,
        ..Platform::assumed(Limits::assumed())
    };
    Device::default().on(gpu)
}

/// Two-output residual add and RMS norm graph.
fn two_output_graph(width: u64, weighted: bool, add3: bool) -> Graph {
    let m = Model::new(vec![Dim::bounded(Rows, 64), Dim::fixed(Hidden, width)]);
    let nw: NormWeight = declare(&m);
    let port = || Port {
        role: Role::Logits,
        ty: TensorType::new([Rows, Hidden], DType::F32),
    };
    let [x, a, b] = m.entry(EntryKind::Forward, [port(), port(), port()]);
    let h = if add3 { x.add(&a.add(&b)) } else { x.add(&a) };
    let normed = if weighted {
        h.rms_norm(&nw.w, 1e-5)
    } else {
        h.rms(1e-5)
    };
    // Both residual and normed values are live outputs.
    m.finish([(Role::Logits, &normed), (Role::Logits, &h)])
}

/// One-output residual add and RMS norm graph.
fn one_output_graph(width: u64, weighted: bool, add3: bool) -> Graph {
    let m = Model::new(vec![Dim::bounded(Rows, 64), Dim::fixed(Hidden, width)]);
    let nw: NormWeight = declare(&m);
    let port = || Port {
        role: Role::Logits,
        ty: TensorType::new([Rows, Hidden], DType::F32),
    };
    let [x, a, b] = m.entry(EntryKind::Forward, [port(), port(), port()]);
    let h = if add3 { x.add(&a.add(&b)) } else { x.add(&a) };
    let normed = if weighted {
        h.rms_norm(&nw.w, 1e-5)
    } else {
        h.rms(1e-5)
    };
    m.finish([(Role::Logits, &normed)])
}

/// Runs `g` against `tessera_interp` and checks output parity.
fn assert_parity(
    g: &Graph,
    rows: u64,
    device: &Device,
    tol: f32,
    check_plan: impl FnOnce(&tessera_lower::Plan),
) {
    let sizes = [(Rows, rows)];
    let mut noise = Noise(42);
    let weights: Vec<Tensor> = g
        .weights
        .iter()
        .map(|w| fill(g, &w.ty, &sizes, &mut noise).unwrap())
        .collect();
    let inputs: Vec<Tensor> = g.entries[0]
        .inputs
        .iter()
        .map(|p| fill(g, &p.ty, &sizes, &mut noise).unwrap())
        .collect();
    let want = run(g, &weights, 0, &inputs).unwrap();
    let feeds: Vec<View> = inputs.iter().map(Tensor::view).collect();
    let plan = lower(g, 0, &sizes, device).unwrap();
    check_plan(&plan);
    for settings in runs() {
        let mut runner = open(plan.clone(), &settings).unwrap();
        for (i, w) in weights.iter().enumerate() {
            runner.load(i as u32, w.view()).unwrap();
        }
        let got = runner.run(&feeds).unwrap();
        assert_eq!(got.len(), want.len());
        for (idx, (g_out, w_out)) in got.iter().zip(&want).enumerate() {
            let diff = max_abs_diff(g_out.floats().unwrap(), w_out.floats().unwrap());
            assert!(diff < tol, "output {idx} off by {diff:e} (tol {tol:e})");
        }
    }
}

#[test]
fn two_output_parity() {
    let device = gpu_device();
    // Single-row block mode at GLM-5.3 hidden width 6144.
    let g_wide = two_output_graph(6144, true, false);
    assert_parity(&g_wide, 1, &device, 1e-4, |plan| {
        assert_eq!(
            plan.steps.len(),
            1,
            "expected 1 fused launch: {:?}",
            plan.steps
        );
        assert_eq!(plan.steps[0].rule, "add_rms_norm");
        assert_eq!(plan.steps[0].kernel.name(), "block_add_norm_save");
        assert_eq!(plan.steps[0].also.len(), 1);
    });
    // Multi-row block mode (4 rows x 1024).
    let g_mid = two_output_graph(1024, true, false);
    assert_parity(&g_mid, 4, &device, 1e-4, |plan| {
        assert_eq!(plan.steps.len(), 1);
        assert_eq!(plan.steps[0].kernel.name(), "block_add_norm_save");
        assert_eq!(plan.steps[0].also.len(), 1);
    });
    // Multi-row lane mode (20 rows x 512).
    let g_lane = two_output_graph(512, true, false);
    assert_parity(&g_lane, 20, &device, 1e-4, |plan| {
        assert_eq!(plan.steps.len(), 1);
        assert_eq!(plan.steps[0].kernel.name(), "lane_add_norm_save");
        assert_eq!(plan.steps[0].also.len(), 1);
    });
    // Small width lane mode, unweighted rms.
    let g_small = two_output_graph(64, false, false);
    assert_parity(&g_small, 1, &device, 1e-4, |plan| {
        assert_eq!(plan.steps.len(), 1);
        assert_eq!(plan.steps[0].kernel.name(), "lane_add_rms_save");
        assert_eq!(plan.steps[0].also.len(), 1);
    });
}

#[test]
fn add3_norm_parity() {
    let device = gpu_device();
    // Three-addend two-output MoE residual pattern.
    let g_moe = two_output_graph(1024, true, true);
    assert_parity(&g_moe, 2, &device, 1e-4, |plan| {
        assert_eq!(
            plan.steps.len(),
            1,
            "expected 1 fused add3 launch: {:?}",
            plan.steps
        );
        assert_eq!(plan.steps[0].rule, "add_rms_norm");
        assert_eq!(plan.steps[0].kernel.name(), "block_add3_norm_save");
        assert_eq!(plan.steps[0].also.len(), 1);
    });
    // Three-addend one-output final norm pattern.
    let g_final = one_output_graph(1024, true, true);
    assert_parity(&g_final, 1, &device, 1e-4, |plan| {
        assert_eq!(plan.steps.len(), 1);
        assert_eq!(plan.steps[0].kernel.name(), "block_add3_norm");
        assert!(plan.steps[0].also.is_empty());
    });
}

#[test]
fn one_output_parity() {
    let device = gpu_device();
    let g = one_output_graph(64, true, false);
    assert_parity(&g, 3, &device, 1e-4, |plan| {
        assert_eq!(plan.steps.len(), 1);
        assert_eq!(plan.steps[0].kernel.name(), "lane_add_norm");
        assert!(plan.steps[0].also.is_empty());
    });
}

#[test]
fn narrow_form_parity() {
    // Mixed bf16 activations with f32 residual stream.
    let pins_mixed = Pins::all(BF16_FORM)
        .activations(BF16_FORM)
        .site(Site::Residual, F32_FORM);
    let device_mixed = gpu_device().pinned(pins_mixed);
    let g = two_output_graph(512, true, false);
    assert_parity(&g, 2, &device_mixed, 2e-2, |plan| {
        let fused = plan.steps.iter().find(|s| s.rule == "add_rms_norm");
        assert!(fused.is_some(), "expected add_rms_norm in {:?}", plan.steps);
    });
    // All bf16 outputs so pass 2 recomputes in f32.
    let pins_bf16 = Pins::all(BF16_FORM)
        .activations(BF16_FORM)
        .outputs(BF16_FORM);
    let device_bf16 = gpu_device().pinned(pins_bf16);
    assert_parity(&g, 2, &device_bf16, 5e-2, |plan| {
        assert_eq!(
            plan.steps.len(),
            1,
            "expected 1 bf16 fused launch: {:?}",
            plan.steps
        );
        assert_eq!(plan.steps[0].rule, "add_rms_norm");
        assert_eq!(plan.steps[0].also.len(), 1);
    });
    // Fp8 activations with f32 residual stream.
    let pins_fp8 = Pins::all(BF16_FORM)
        .activations(FP8_FORM)
        .site(Site::Residual, F32_FORM);
    let device_fp8 = gpu_device().pinned(pins_fp8);
    assert_parity(&g, 2, &device_fp8, 2e-1, |plan| {
        let fused = plan.steps.iter().find(|s| s.rule == "add_rms_norm");
        assert!(
            fused.is_some(),
            "expected add_rms_norm for fp8 in {:?}",
            plan.steps
        );
    });
}

#[test]
fn early_reader_parity() {
    // An earlier outside reader of `h` blocks fusion safely.
    let m = Model::new(vec![Dim::bounded(Rows, 8), Dim::fixed(Hidden, 64)]);
    let nw: NormWeight = declare(&m);
    let port = || Port {
        role: Role::Logits,
        ty: TensorType::new([Rows, Hidden], DType::F32),
    };
    let [x, a] = m.entry(EntryKind::Forward, [port(), port()]);
    let h = x.add(&a);
    let early = h.act(Act::Silu);
    let normed = h.rms_norm(&nw.w, 1e-5);
    let g = m.finish([(Role::Logits, &early.add(&normed))]);
    let device = gpu_device();
    assert_parity(&g, 2, &device, 1e-4, |plan| {
        assert!(
            plan.steps.iter().all(|s| s.rule != "add_rms_norm"),
            "must not absorb h past earlier reader: {:?}",
            plan.steps
        );
    });
}
