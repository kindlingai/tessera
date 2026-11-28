use crate::mm::{propose_mm, propose_mm_mx8};
use crate::{cuda_source, hip_source, prefill_source, trellis_source};
use tessera_graph::DimName::Rows;
use tessera_graph::{Act, DType, Dim, Graph, Node, NodeId, Op, Port, Role, Row, Stage, TensorType};
use tessera_kernels as _;
use tessera_linear as _;
use tessera_lower::form::{BF16_FORM, named};
use tessera_lower::{Device, Knobs, Limits, Pins, Platform, Source, Tree, Wants, explain, lower};
use tessera_testkit::graphs::linear;

fn mm_entry(g: &Graph, rows: u64, form: &str) -> Option<(&'static str, &'static str, usize)> {
    mm_shaped(g, rows, form, Knobs::default())
}

fn mm_draft(
    g: &Graph,
    rows: u64,
    form: &str,
    knobs: Knobs,
    mx8: bool,
) -> Option<(&'static str, &'static str, usize, tessera_lower::Load)> {
    let form = named(form).expect("known form");
    let dev = Device::default().pinned(Pins::all(form));
    let parts = dev.division();
    let mut tree = Tree::upto(
        g,
        0,
        Wants::ALL,
        &[(Rows, rows)],
        &[],
        &dev.pins,
        dev.kept,
        &parts,
        0,
    )
    .expect("tree builds");
    tree.knobs = knobs;
    let root = tree.outputs[0].0;
    let drafts = if mx8 {
        propose_mm_mx8(&tree, root)?
    } else {
        propose_mm(&tree, root)?
    };
    let d = &drafts[usize::from(mx8)];
    let Source::Fixed { entry, .. } = d.kernel.source() else {
        panic!("fast_mm uses fixed source");
    };
    let load = d.kernel.load(0).expect("load model");
    Some((d.kernel.name(), entry, d.reads.len(), load))
}

fn mm_shaped(
    g: &Graph,
    rows: u64,
    form: &str,
    knobs: Knobs,
) -> Option<(&'static str, &'static str, usize)> {
    let (name, entry, reads, _) = mm_draft(g, rows, form, knobs, false)?;
    Some((name, entry, reads))
}

pub(crate) fn push_node(g: &mut Graph, op: Op, args: Vec<NodeId>, ty: TensorType) -> NodeId {
    let id = NodeId(g.nodes.len() as u32);
    g.nodes.push(Node {
        op,
        args,
        ty,
        numerics: None,
    });
    id
}

pub(crate) fn with_stage(mut g: Graph, stage: Stage, extra_ty: Option<TensorType>) -> Graph {
    let mat = g.entries[0].outputs[0].node;
    let ty = g.nodes[mat.0 as usize].ty.clone();
    let mut args = vec![mat];
    if let Some(ety) = extra_ty {
        let port = g.entries[0].inputs.len() as u32;
        let extra = push_node(
            &mut g,
            Op::Input { entry: 0, port },
            Vec::new(),
            ety.clone(),
        );
        g.entries[0].inputs.push(Port {
            role: Role::Logits,
            ty: ety,
        });
        args.push(extra);
    }
    let out = push_node(&mut g, Op::Row(Row(vec![stage])), args, ty);
    g.entries[0].outputs[0].node = out;
    g
}

#[test]
fn linear_mm_entries() {
    let cuda = cuda_source();
    let hip = hip_source();
    assert!(!hip.is_empty());
    for entry in [
        "tessera_mm_bf16(",
        "tessera_mm_bf16_ks2(",
        "tessera_mm_bf16_sk(",
        "tessera_stacked_lanes_tc(",
        "tessera_mm_mxfp8_r8(",
        "tessera_mm_mxfp8_r8_sk(",
        "tessera_mm_mxfp8_r16_sk(",
        "tessera_mm_reduce(",
        "tessera_mm_mxfp8_r16(",
        "tessera_mm_mxfp8_r64(",
        "tessera_mm_mxfp8_r64_sk(",
    ] {
        assert!(cuda.contains(entry), "missing {entry}");
    }
    for entry in [
        "dtr_prep(",
        "dtr_pre_k4_t8(",
        "dtr_pre_k4_t16(",
        "dtr_pre_gen8(",
        "dtr_pre_gen16(",
        "tessera_mm_mxfp8_prep(",
        "tessera_mm_mxfp8_mx8(",
        "tessera_mm_mxfp8_mx8_sk(",
    ] {
        assert!(prefill_source().contains(entry), "missing {entry}");
    }
    for entry in [
        "DTR_DEF(dtr_k4_t4,",
        "DTR_GEN_DEF(dtr_gen4,",
        "dtr_reduce(",
        "ptr_gen(",
    ] {
        assert!(trellis_source().contains(entry), "missing {entry}");
    }
}

#[test]
fn fast_mm_selection() {
    let rows = [Dim::bounded(Rows, 128)];
    // Wide-N bf16 picks 16x32, narrow-N picks split-K 16x16.
    assert_eq!(
        mm_entry(&linear(&rows, 4096, 12288), 16, "bf16"),
        Some(("fast_mm_bf16", "tessera_mm_bf16", 2))
    );
    assert_eq!(
        mm_entry(&linear(&rows, 12288, 4096), 16, "bf16"),
        Some(("fast_mm_bf16", "tessera_mm_bf16_ks2", 2))
    );
    assert_eq!(
        mm_entry(&linear(&rows, 36864, 1536), 8, "bf16"),
        Some(("fast_mm_bf16_sk", "tessera_mm_bf16_sk", 2))
    );
    let mut one = Knobs::default();
    one.parse("fast_mm_bf16_sk:splits=1").unwrap();
    assert_eq!(
        mm_shaped(&linear(&rows, 36864, 1536), 8, "bf16", one),
        Some(("fast_mm_bf16", "tessera_mm_bf16_ks2", 2))
    );
    // Narrow MXFP8 r8 uses split-K, wide N stays single-pass.
    assert_eq!(
        mm_entry(&linear(&rows, 6144, 576), 8, "mxfp8"),
        Some(("fast_mm_mxfp8_sk", "tessera_mm_mxfp8_r8_sk", 2))
    );
    assert_eq!(
        mm_entry(&linear(&rows, 6144, 4096), 8, "mxfp8"),
        Some(("fast_mm_mxfp8", "tessera_mm_mxfp8_r8", 2))
    );
    assert_eq!(
        mm_entry(&linear(&rows, 12288, 6144), 16, "mxfp8"),
        Some(("fast_mm_mxfp8", "tessera_mm_mxfp8_r16", 2))
    );
    assert_eq!(
        mm_entry(&linear(&rows, 12288, 6144), 24, "mxfp8"),
        Some(("fast_mm_mxfp8_sk", "tessera_mm_mxfp8_r64_sk", 2))
    );
    assert_eq!(
        mm_entry(&linear(&rows, 6144, 576), 72, "mxfp8"),
        Some(("fast_mm_mxfp8_sk", "tessera_mm_mxfp8_r64_sk", 2))
    );
    assert_eq!(
        mm_entry(&linear(&rows, 2048, 4096), 256, "mxfp8"),
        Some(("fast_mm_mxfp8", "tessera_mm_mxfp8_r64", 2))
    );
    // The splits knob also sets r64 splits.
    for (splits, rows_n, kind, entry) in [
        (2, 256, "fast_mm_mxfp8_sk", "tessera_mm_mxfp8_r64_sk"),
        (1, 24, "fast_mm_mxfp8", "tessera_mm_mxfp8_r64"),
    ] {
        let mut set = Knobs::default();
        set.parse(&format!("fast_mm_mxfp8_sk:splits={splits}"))
            .unwrap();
        assert_eq!(
            mm_shaped(&linear(&rows, 12288, 6144), rows_n, "mxfp8", set),
            Some((kind, entry, 2))
        );
    }
    // One row and unaligned K decline. Tall M is accepted.
    assert_eq!(mm_entry(&linear(&rows, 6144, 576), 1, "mxfp8"), None);
    assert!(mm_entry(&linear(&rows, 6144, 576), 4096, "mxfp8").is_some());
    assert_eq!(mm_entry(&linear(&rows, 256, 64), 8, "mxfp8"), None);
    // Fused epilogues: residual, bias, SwiGLU, mul, silu.
    let base = linear(&rows, 4096, 12288);
    let out_ty = base.nodes[base.entries[0].outputs[0].node.0 as usize]
        .ty
        .clone();
    let bias_ty = TensorType::new(vec![out_ty.shape.0[1].clone()], DType::F32);
    let add_res = with_stage(base.clone(), Stage::Add, Some(out_ty.clone()));
    let add_bias = with_stage(base.clone(), Stage::Add, Some(bias_ty));
    let silu_only = with_stage(base.clone(), Stage::Act(Act::Silu), None);
    let swiglu = with_stage(silu_only.clone(), Stage::Mul, Some(out_ty.clone()));
    let mul_only = with_stage(base, Stage::Mul, Some(out_ty));
    for (g, reads) in [
        (add_res, 3),
        (add_bias, 3),
        (swiglu, 3),
        (mul_only, 3),
        (silu_only, 2),
    ] {
        assert_eq!(
            mm_entry(&g, 16, "bf16"),
            Some(("fast_mm_bf16", "tessera_mm_bf16", reads))
        );
        assert_eq!(
            mm_entry(&g, 8, "mxfp8"),
            Some(("fast_mm_mxfp8", "tessera_mm_mxfp8_r8", reads))
        );
    }
}

/// Fast-path diagnostics report rule and hardware rejections accurately.
#[test]
fn lowering_report_diagnostics() {
    let rows = [Dim::bounded(Rows, 1 << 21)];
    let cuda = Platform {
        backend: "cuda".into(),
        ..Platform::host(Limits::assumed())
    };
    let cuda_no_bf16 = Platform {
        backend: "cuda".into(),
        ..Platform::assumed(Limits::assumed())
    };
    let metal = Platform {
        backend: "metal".into(),
        ..Platform::assumed(Limits::assumed())
    };
    let mxfp8 = named("mxfp8").expect("mxfp8 form");

    // Aligned m=1 bf16 hits fast_gemv with no warning.
    let g_ok = linear(&rows, 512, 256);
    let dev = Device::default()
        .on(cuda.clone())
        .pinned(Pins::all(BF16_FORM));
    let r = explain(&g_ok, 0, &[(Rows, 1)], &dev).unwrap();
    assert!(!r.has_fallbacks(), "{}", r.format(10));
    assert!(r.warning("--explain-lowering").is_none());

    // Unaligned k=768 mxfp8 misses fast_gemv and emits boot warning.
    let g_unaligned = linear(&rows, 768, 256);
    let dev_mxfp8 = Device::default().on(cuda.clone()).pinned(Pins::all(mxfp8));
    let r = explain(&g_unaligned, 0, &[(Rows, 1)], &dev_mxfp8).unwrap();
    assert!(r.has_fallbacks());
    let msg = r.format(10);
    assert!(msg.contains("fast_gemv"), "{msg}");
    assert!(msg.contains("k=768 is not a multiple of 1024"), "{msg}");
    let warn = r.warning("--explain-lowering").expect("warning");
    assert!(
        warn.contains("warning: lowering fell off fast path"),
        "{warn}"
    );
    assert!(warn.contains("--explain-lowering"), "{warn}");

    // 4096 rows stay fast. The launch grid bounds fast_mm.
    let r = explain(&g_ok, 0, &[(Rows, 4096)], &dev).unwrap();
    assert!(!r.has_fallbacks(), "{}", r.format(10));
    let msg = explain(&g_ok, 0, &[(Rows, 1 << 20)], &dev)
        .unwrap()
        .format(9);
    assert!(msg.contains("grid height 65536 exceeds 65535"), "{msg}");

    // Missing bf16 tensor tiles reports hardware mismatch via Needs.
    let dev_no_tile = Device::default()
        .on(cuda_no_bf16)
        .pinned(Pins::all(BF16_FORM));
    let r = explain(&g_ok, 0, &[(Rows, 8)], &dev_no_tile).unwrap();
    assert!(r.has_fallbacks());
    let msg = r.format(10);
    assert!(msg.contains("fast_mm"), "{msg}");
    assert!(msg.contains("requires bf16 tensor tiles"), "{msg}");

    // Metal ignores CUDA-only fast_gemv to avoid false boot warnings.
    let dev_metal = Device::default().on(metal).pinned(Pins::all(mxfp8));
    let r = explain(&g_unaligned, 0, &[(Rows, 1)], &dev_metal).unwrap();
    assert!(!r.has_fallbacks(), "{}", r.format(10));

    // Blackwell sm_121 picks native MXFP8 MMA, sm_90 stays on r64.
    let g_mx = linear(&rows, 6144, 6144);
    let dyn_limits = Limits {
        group_bytes: 48 * 1024,
        dynamic_group_bytes: 99 * 1024,
        ..Limits::assumed()
    };
    let mut sm121 = Platform {
        backend: "cuda".into(),
        arch: "sm_121".into(),
        name: "NVIDIA GB10".into(),
        units: 48,
        bandwidth: 273e9,
        ..Platform::host(dyn_limits)
    };
    sm121.row.as_mut().unwrap().mxfp8_tile = Some([16, 8, 32]);
    let sm90 = Platform {
        backend: "cuda".into(),
        arch: "sm_90".into(),
        name: "NVIDIA H100".into(),
        units: 132,
        bandwidth: 3350e9,
        ..Platform::host(dyn_limits)
    };
    let p121 = lower(
        &g_mx,
        0,
        &[(Rows, 256)],
        &Device::default().on(sm121.clone()).pinned(Pins::all(mxfp8)),
    )
    .unwrap();
    let names121: Vec<&str> = p121.steps.iter().map(|s| s.kernel.name()).collect();
    assert_eq!(names121, ["fast_mm_mxfp8_prep", "fast_mm_mxfp8_mx8"]);
    let g_kva = linear(&rows, 6144, 576);
    let pkva = lower(
        &g_kva,
        0,
        &[(Rows, 2048)],
        &Device::default().on(sm121.clone()).pinned(Pins::all(mxfp8)),
    )
    .unwrap();
    let kva_names: Vec<&str> = pkva.steps.iter().map(|s| s.kernel.name()).collect();
    assert_eq!(kva_names, ["fast_mm_mxfp8_prep", "fast_mm_mxfp8_mx8"]);
    // Two MXFP8 projections share one prep via CSE.
    let mut g_pair = linear(&rows, 6144, 2048);
    let x = g_pair.nodes[g_pair.entries[0].outputs[0].node.0 as usize].args[0];
    let mut w2 = g_pair.weights[0].clone();
    w2.path.push(tessera_graph::PathSeg::Field("b".into()));
    let w2_ty = w2.ty.clone();
    let wid = g_pair.weights.len() as u32;
    let w2_node = push_node(&mut g_pair, Op::Weight(wid), Vec::new(), w2_ty);
    g_pair.weights.push(w2);
    let y2_ty = g_pair.nodes[g_pair.entries[0].outputs[0].node.0 as usize]
        .ty
        .clone();
    let y2 = push_node(&mut g_pair, Op::Matmul, vec![x, w2_node], y2_ty);
    let mut out2 = g_pair.entries[0].outputs[0].clone();
    out2.node = y2;
    g_pair.entries[0].outputs.push(out2);
    let ppair = lower(
        &g_pair,
        0,
        &[(Rows, 2048)],
        &Device::default().on(sm121).pinned(Pins::all(mxfp8)),
    )
    .unwrap();
    let preps = ppair
        .steps
        .iter()
        .filter(|s| s.kernel.name() == "fast_mm_mxfp8_prep")
        .count();
    let mx8s = ppair
        .steps
        .iter()
        .filter(|s| s.kernel.name() == "fast_mm_mxfp8_mx8")
        .count();
    assert_eq!((preps, mx8s), (1, 2));
    let p90 = lower(
        &g_mx,
        0,
        &[(Rows, 256)],
        &Device::default().on(sm90).pinned(Pins::all(mxfp8)),
    )
    .unwrap();
    let names90: Vec<&str> = p90.steps.iter().map(|s| s.kernel.name()).collect();
    assert_eq!(names90, ["fast_mm_mxfp8"]);
}

#[test]
fn fast_mm_loads() {
    let rows = [Dim::bounded(Rows, 512)];
    let cases = [
        (4096, 12288, 16, "bf16", "tessera_mm_bf16", false),
        (12288, 4096, 16, "bf16", "tessera_mm_bf16_ks2", false),
        (36864, 1536, 8, "bf16", "tessera_mm_bf16_sk", false),
        (6144, 4096, 8, "mxfp8", "tessera_mm_mxfp8_r8", true),
        (6144, 576, 8, "mxfp8", "tessera_mm_mxfp8_r8_sk", true),
        (12288, 6144, 16, "mxfp8", "tessera_mm_mxfp8_r16", true),
        (6144, 576, 16, "mxfp8", "tessera_mm_mxfp8_r16_sk", true),
        (2048, 4096, 256, "mxfp8", "tessera_mm_mxfp8_r64", true),
        (12288, 6144, 24, "mxfp8", "tessera_mm_mxfp8_r64_sk", true),
        (4096, 6144, 4, "int4_g32", "tessera_mm_int4_r16", false),
        (4096, 512, 8, "int4_g32", "tessera_mm_int4_r16_sk", false),
        (4096, 6144, 256, "int4_g32", "tessera_mm_int4_r64", false),
        (4096, 2048, 32, "int4_g32", "tessera_mm_int4_r64_sk", false),
    ];
    for (k, n, m, form, want_entry, want_e4m3) in cases {
        let (_, entry, _, load) = mm_draft(&linear(&rows, k, n), m, form, Knobs::default(), false)
            .expect("matched fast_mm");
        assert_eq!(entry, want_entry);
        assert!(load.trip.bf16_flops > 0.0, "{entry}");
        assert!(load.trip.loads > 0.0 && load.trip.width > 0.0, "{entry}");
        assert_eq!(load.trip.e4m3 > 0.0, want_e4m3, "{entry}");
    }
    let mut k_bf16_wide = Knobs::default();
    k_bf16_wide.parse("fast_mm_bf16:split_groups=0").unwrap();
    assert_eq!(
        mm_shaped(&linear(&rows, 12288, 4096), 16, "bf16", k_bf16_wide),
        Some(("fast_mm_bf16", "tessera_mm_bf16", 2))
    );
    let mut k_bf16_split = Knobs::default();
    k_bf16_split
        .parse("fast_mm_bf16:split_groups=2048")
        .unwrap();
    assert_eq!(
        mm_shaped(&linear(&rows, 4096, 12288), 16, "bf16", k_bf16_split),
        Some(("fast_mm_bf16", "tessera_mm_bf16_ks2", 2))
    );
    let g_mx = linear(&rows, 6144, 6144);
    assert!(mm_draft(&g_mx, 32, "mxfp8", Knobs::default(), true).is_none());
    let mut k_mx_low = Knobs::default();
    k_mx_low.parse("fast_mm_mxfp8_mx8:min_rows=32").unwrap();
    assert_eq!(
        mm_draft(&g_mx, 32, "mxfp8", k_mx_low, true).map(|(n, e, _, _)| (n, e)),
        Some(("fast_mm_mxfp8_mx8", "tessera_mm_mxfp8_mx8"))
    );
    let mut k_mx_high = Knobs::default();
    k_mx_high.parse("fast_mm_mxfp8_mx8:min_rows=256").unwrap();
    assert!(mm_draft(&g_mx, 128, "mxfp8", k_mx_high, true).is_none());
    let mut k_mx_sk = Knobs::default();
    k_mx_sk.parse("fast_mm_mxfp8_sk:splits=4").unwrap();
    k_mx_sk.parse("fast_mm_mxfp8_mx8_sk:splits=1").unwrap();
    assert_eq!(
        mm_draft(&g_mx, 256, "mxfp8", k_mx_sk, true).map(|(n, e, _, _)| (n, e)),
        Some(("fast_mm_mxfp8_mx8", "tessera_mm_mxfp8_mx8"))
    );
    let mut k_mx_sk2 = Knobs::default();
    k_mx_sk2.parse("fast_mm_mxfp8_mx8_sk:splits=2").unwrap();
    assert_eq!(
        mm_draft(&g_mx, 256, "mxfp8", k_mx_sk2, true).map(|(n, e, _, _)| (n, e)),
        Some(("fast_mm_mxfp8_mx8_sk", "tessera_mm_mxfp8_mx8_sk"))
    );

    let mut k_blk = Knobs::default();
    k_blk.parse("fast_mm_bf16:min_blocks=3").unwrap();
    let g_blk = linear(&rows, 4096, 12288);
    let (_, e_base, _, _) = mm_draft(&g_blk, 16, "bf16", Knobs::default(), false).unwrap();
    let (_, e_blk, _, _) = mm_draft(&g_blk, 16, "bf16", k_blk.clone(), false).unwrap();
    assert_eq!(e_base, e_blk);
    let pins = Pins::all(tessera_lower::form::named("bf16").unwrap());
    let parts = Device::default().division();
    let mut tree = Tree::upto(
        &g_blk,
        0,
        Wants::ALL,
        &[(Rows, 16)],
        &[],
        &pins,
        tessera_lower::kept::DENSE,
        &parts,
        0,
    )
    .unwrap();
    tree.knobs = k_blk;
    let d_blk = &propose_mm(&tree, tree.outputs[0].0).unwrap()[0];
    let Source::Fixed { text, .. } = d_blk.kernel.source() else {
        panic!("expected fixed source");
    };
    assert!(text[tessera_lower::Backend::Cuda].contains("#define MM_BF16_BLK 3"));
}

#[test]
fn bf16_inputs_skip_widen() {
    let mut g = linear(&[Dim::bounded(Rows, 64)], 4096, 4096);
    g.entries[0].inputs[0].ty.dtype = DType::Bf16;
    for n in &mut g.nodes {
        if matches!(n.op, Op::Input { .. }) {
            n.ty.dtype = DType::Bf16;
        }
    }
    let limits = Limits {
        group_bytes: 48 * 1024,
        dynamic_group_bytes: 99 * 1024,
        ..Limits::assumed()
    };
    let mut sm121 = Platform {
        backend: "cuda".into(),
        arch: "sm_121".into(),
        name: "NVIDIA GB10".into(),
        units: 48,
        bandwidth: 273e9,
        ..Platform::host(limits)
    };
    sm121.row.as_mut().unwrap().mxfp8_tile = Some([16, 8, 32]);
    let cuda = Device::default().on(sm121);
    for (w_form, rows) in [
        ("bf16", 1),
        ("bf16", 16),
        ("mxfp8", 1),
        ("mxfp8", 16),
        ("mxfp8", 64),
        ("int4_g32", 1),
        ("int4_g32", 16),
        ("trellis_k4", 1),
        ("trellis_k4", 64),
    ] {
        let pins = Pins::all(named(w_form).expect("form"))
            .site(tessera_lower::Site::Matmul, BF16_FORM)
            .outputs(BF16_FORM);
        let dev = cuda.clone().pinned(pins);
        let plan = lower(&g, 0, &[(Rows, rows)], &dev).expect("lowers");
        let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
        assert!(
            !names
                .iter()
                .any(|n| n.contains("widen") || n.contains("narrow")),
            "{w_form} rows={rows}: unexpected cast in {names:?}"
        );
        let has_x = plan.steps.iter().any(|s| {
            matches!(
                s.kernel.source(),
                Source::Fixed { text, .. } if text[tessera_lower::Backend::Cuda].contains("#define X_BF16 1")
            )
        });
        let has_y = plan.steps.iter().any(|s| {
            matches!(
                s.kernel.source(),
                Source::Fixed { text, .. } if text[tessera_lower::Backend::Cuda].contains("#define Y_BF16 1")
            )
        });
        assert!(
            has_x && has_y,
            "{w_form} rows={rows}: missing X/Y_BF16 in {names:?}"
        );
    }
}
