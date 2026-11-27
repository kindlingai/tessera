use crate::stacked::{FastStacked, stacked_needs, stacked_source, stacked_source_for};
use tessera_graph::DimName::Rows;
use tessera_graph::{
    DType, Dim, Node, NodeId, Op, Pairs, Port, Role, Row, Stage, TensorType, View,
};
use tessera_kernels as _;
use tessera_linear as _;
use tessera_lower::form::BF16_FORM;
use tessera_lower::{Device, Limits, Pins, Platform, Source, lower};

fn stacked_kernel(rows: usize, n: usize, k: usize, wide: bool) -> FastStacked {
    FastStacked {
        len: 16 * rows * n,
        batches: 16,
        rows,
        n,
        k,
        block: n * k,
        first: 0,
        step: if wide { n } else { k },
        x_stride: [rows * k, k],
        o_stride: [rows * n, n],
        wide,
        tiled64_min_rows: 32,
        min_blocks: None,
        rope: None,
        form: BF16_FORM,
    }
}

#[test]
fn stacked_64_picks_prefill() {
    let lanes = stacked_kernel(2048, 256, 512, false);
    assert!(lanes.lanes_64());
    assert!(lanes.tiled_64());
    let Source::Fixed { entry, .. } = stacked_source_for(&lanes) else {
        panic!("expected fixed source");
    };
    assert_eq!(entry, "tessera_stacked_lanes_64");
    assert_eq!(
        stacked_needs(&lanes).group_bytes,
        64 * 18 * 4 * 2 + 64 * 40 * 2
    );

    let wide = stacked_kernel(2048, 512, 192, true);
    assert!(wide.wide_64());
    assert!(wide.tiled_64());
    let Source::Fixed { entry, .. } = stacked_source_for(&wide) else {
        panic!("expected fixed source");
    };
    assert_eq!(entry, "tessera_stacked_wide_64");

    let decode_lanes = stacked_kernel(1, 256, 512, false);
    assert!(!decode_lanes.tiled_64());
    let Source::Fixed { entry, .. } = stacked_source_for(&decode_lanes) else {
        panic!("expected fixed source");
    };
    assert_eq!(entry, "tessera_stacked_lanes_tc");

    let decode_wide = stacked_kernel(1, 512, 192, true);
    assert!(!decode_wide.tiled_64());
    let Source::Fixed { entry, .. } = stacked_source_for(&decode_wide) else {
        panic!("expected fixed source");
    };
    assert_eq!(entry, "tessera_stacked_wide");
}

#[test]
fn stacked_source_has_64() {
    let src = stacked_source();
    assert!(src.contains("tessera_stacked_wide"));
    assert!(src.contains("tessera_stacked_lanes_64"));
    assert!(src.contains("tessera_stacked_wide_64"));
    assert!(src.contains("st_rope_tail"));
}

#[test]
fn stacked_fuses_join_rope() {
    let mut g = tessera_testkit::graphs::blocks(&[Dim::bounded(Rows, 256)], [16, 128, 512]);
    let heads = g
        .dims
        .iter()
        .find(|d| d.name.as_str() == "Heads")
        .unwrap()
        .name;
    let latent = g
        .dims
        .iter()
        .find(|d| d.name.as_str() == "Latent")
        .unwrap()
        .name;
    let rot_ty = TensorType::new(vec![Rows.into(), heads.into(), 64u64.into()], DType::F32);
    let pos_ty = TensorType::new(vec![Rows.into()], DType::U32);
    let rot_in = NodeId(g.nodes.len() as u32);
    g.nodes.push(Node {
        op: Op::Input { entry: 0, port: 2 },
        args: vec![],
        ty: rot_ty.clone(),
        numerics: None,
    });
    let pos_in = NodeId(g.nodes.len() as u32);
    g.nodes.push(Node {
        op: Op::Input { entry: 0, port: 3 },
        args: vec![],
        ty: pos_ty.clone(),
        numerics: None,
    });
    g.entries[0].inputs.push(Port {
        role: Role::Logits,
        ty: rot_ty.clone(),
    });
    g.entries[0].inputs.push(Port {
        role: Role::Logits,
        ty: pos_ty,
    });
    let keys = g.entries[0].outputs[0].node;
    let perm_ty = TensorType::new(vec![Rows.into(), heads.into(), latent.into()], DType::F32);
    let perm = NodeId(g.nodes.len() as u32);
    g.nodes.push(Node {
        op: Op::View(View::Permute(vec![1, 0, 2])),
        args: vec![keys],
        ty: perm_ty,
        numerics: None,
    });
    let rope = NodeId(g.nodes.len() as u32);
    g.nodes.push(Node {
        op: Op::Row(Row(vec![Stage::Rope {
            theta: 1e4,
            pairs: Pairs::Adjacent,
            turned: None,
        }])),
        args: vec![rot_in, pos_in],
        ty: rot_ty,
        numerics: None,
    });
    let out_ty = TensorType::new(
        vec![Rows.into(), heads.into(), (512u64 + 64u64).into()],
        DType::F32,
    );
    let joined = NodeId(g.nodes.len() as u32);
    g.nodes.push(Node {
        op: Op::View(View::Concat),
        args: vec![perm, rope],
        ty: out_ty,
        numerics: None,
    });
    g.entries[0].outputs[0].node = joined;

    let cuda = Platform {
        backend: "cuda".into(),
        arch: "sm_90".into(),
        name: "NVIDIA H100".into(),
        units: 132,
        bandwidth: 3350e9,
        ..Platform::host(Limits::assumed())
    };
    let dev = Device::default().on(cuda).pinned(Pins::all(BF16_FORM));
    for (rows, want) in [
        (1u64, "tessera_stacked_wide"),
        (64u64, "tessera_stacked_wide_64"),
    ] {
        let plan = lower(&g, 0, &[(Rows, rows)], &dev).unwrap();
        let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
        assert!(!names.contains(&"join_rope"), "rows={rows} names={names:?}");
        let fused = plan
            .steps
            .iter()
            .find(|s| s.kernel.name() == "fast_stacked_wide")
            .unwrap();
        assert_eq!(fused.reads.len(), 4);
        let Source::Fixed { entry, .. } = fused.kernel.source() else {
            panic!("expected fixed source");
        };
        assert_eq!(entry, want);
    }
}
