use crate::mm::propose_mm;
use crate::{cuda_source, hip_source, prefill_source};
use tessera_graph::DimName::Rows;
use tessera_graph::{Act, DType, Dim, Graph, Node, NodeId, Op, Port, Role, Row, Stage, TensorType};
use tessera_lower::form::named;
use tessera_lower::{Device, Knobs, Pins, Source, Tree, Wants};
use tessera_testkit::graphs::linear;

fn mm_entry(g: &Graph, rows: u64, form: &str) -> Option<(&'static str, &'static str, usize)> {
    mm_shaped(g, rows, form, Knobs::default())
}

fn mm_shaped(
    g: &Graph,
    rows: u64,
    form: &str,
    knobs: Knobs,
) -> Option<(&'static str, &'static str, usize)> {
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
    let drafts = propose_mm(&tree, root)?;
    let d = &drafts[0];
    let Source::Fixed { entry, .. } = d.kernel.source() else {
        panic!("fast_mm uses fixed source");
    };
    Some((d.kernel.name(), entry, d.reads.len()))
}

fn push_node(g: &mut Graph, op: Op, args: Vec<NodeId>, ty: TensorType) -> NodeId {
    let id = NodeId(g.nodes.len() as u32);
    g.nodes.push(Node {
        op,
        args,
        ty,
        numerics: None,
    });
    id
}

fn with_stage(mut g: Graph, stage: Stage, extra_ty: Option<TensorType>) -> Graph {
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
        "DTR_DEF(dtr_k4_t4,",
        "dtr_gen4(",
    ] {
        assert!(cuda.contains(entry), "missing {entry}");
    }
    for entry in [
        "dtr_prep(",
        "dtr_pre_k4_t8(",
        "dtr_pre_k4_t16(",
        "dtr_pre_gen8(",
        "dtr_pre_gen16(",
    ] {
        assert!(prefill_source().contains(entry), "missing {entry}");
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
