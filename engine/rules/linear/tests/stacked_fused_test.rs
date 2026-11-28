//! Lowering and launch-arg tests for CUDA stacked_fused BF16 matmul.

use tessera_kernels as _;
use tessera_linear as _;

use tessera_graph::DimName::Rows;
use tessera_graph::{
    DType, Dim, DimExpr, Entry, EntryKind, Graph, Node, NodeId, Op, Output, Placement, Port, Role,
    Shape, TensorType, View, Weight,
};
use tessera_lower::form::BF16_FORM;
use tessera_lower::{Device, Limits, Pins, Platform, Shape as GridShape, Source, lower};

fn tensor(dtype: DType, dims: Vec<DimExpr>) -> TensorType {
    TensorType {
        dtype,
        shape: Shape(dims),
        placement: Placement::Replicated,
    }
}

fn push(g: &mut Graph, op: Op, args: Vec<NodeId>, ty: TensorType) -> NodeId {
    let id = NodeId(g.nodes.len() as u32);
    g.nodes.push(Node {
        op,
        args,
        ty,
        numerics: None,
    });
    id
}

struct Case {
    batches: u64,
    n: u64,
    k: u64,
    x_width: u64,
    x_off: u64,
    swapped: bool,
    peel_in: bool,
    peel_out: bool,
    x_dtype: DType,
}

fn stacked_graph(c: &Case) -> Graph {
    let mut g = Graph::new(vec![Dim::bounded(Rows, 4096)]);
    let (w_rows, w_cols) = if c.swapped { (c.k, c.n) } else { (c.n, c.k) };
    let w_ty = tensor(DType::F32, vec![(c.batches * w_rows).into(), w_cols.into()]);
    g.weights.push(Weight {
        path: vec![],
        ty: w_ty.clone(),
    });
    let x_dims = if c.peel_in {
        vec![Rows.into(), c.batches.into(), c.x_width.into()]
    } else {
        vec![c.batches.into(), Rows.into(), c.x_width.into()]
    };
    let x_in = push(
        &mut g,
        Op::Input { entry: 0, port: 0 },
        vec![],
        tensor(c.x_dtype, x_dims.clone()),
    );
    let mut x = x_in;
    if c.x_width != c.k || c.x_off != 0 {
        let sliced_dims = if c.peel_in {
            vec![Rows.into(), c.batches.into(), c.k.into()]
        } else {
            vec![c.batches.into(), Rows.into(), c.k.into()]
        };
        x = push(
            &mut g,
            Op::View(View::Slice {
                axis: 2,
                start: c.x_off.into(),
                len: c.k.into(),
            }),
            vec![x],
            tensor(c.x_dtype, sliced_dims),
        );
    }
    if c.peel_in {
        x = push(
            &mut g,
            Op::View(View::Permute(vec![1, 0, 2])),
            vec![x],
            tensor(c.x_dtype, vec![c.batches.into(), Rows.into(), c.k.into()]),
        );
    }
    let w_leaf = push(&mut g, Op::Weight(0), vec![], w_ty);
    let mut w = push(
        &mut g,
        Op::View(View::Reshape(Shape(vec![
            c.batches.into(),
            w_rows.into(),
            w_cols.into(),
        ]))),
        vec![w_leaf],
        tensor(
            DType::F32,
            vec![c.batches.into(), w_rows.into(), w_cols.into()],
        ),
    );
    if c.swapped {
        w = push(
            &mut g,
            Op::View(View::Permute(vec![0, 2, 1])),
            vec![w],
            tensor(DType::F32, vec![c.batches.into(), c.n.into(), c.k.into()]),
        );
    }
    let mm_ty = tensor(DType::F32, vec![c.batches.into(), Rows.into(), c.n.into()]);
    let mut out = push(&mut g, Op::Matmul, vec![x, w], mm_ty);
    if c.peel_out {
        let out_ty = tensor(DType::F32, vec![Rows.into(), c.batches.into(), c.n.into()]);
        out = push(
            &mut g,
            Op::View(View::Permute(vec![1, 0, 2])),
            vec![out],
            out_ty,
        );
    }
    g.entries.push(Entry {
        kind: EntryKind::Step,
        inputs: vec![Port {
            role: Role::Tokens,
            ty: tensor(c.x_dtype, x_dims),
        }],
        outputs: vec![Output {
            role: Role::Tokens,
            node: out,
        }],
    });
    g
}

fn cuda_platform() -> Platform {
    Platform {
        backend: "cuda".into(),
        arch: "sm_121".into(),
        name: "NVIDIA GB10".into(),
        units: 48,
        bandwidth: 273e9,
        ..Platform::host(Limits::assumed())
    }
}

#[test]
fn stacked_fused_variants() {
    let dev = Device::default()
        .on(cuda_platform())
        .pinned(Pins::all(BF16_FORM))
        .favoring("stacked_fused");
    let cases = [
        (false, 64u64, "stacked_fused_mma", "tessera_sf_mma", 256),
        (true, 64, "stacked_fused_mma_t", "tessera_sf_mma_t", 256),
        (false, 8, "stacked_fused_mma16", "tessera_sf_mma16", 128),
        (true, 8, "stacked_fused_mma16_t", "tessera_sf_mma16_t", 128),
        (false, 1, "stacked_fused_lanes", "tessera_sf_lanes", 256),
        (true, 1, "stacked_fused_wide", "tessera_sf_wide", 256),
    ];
    for (swapped, rows, want_name, want_entry, want_threads) in cases {
        let g = stacked_graph(&Case {
            batches: 16,
            n: 128,
            k: 512,
            x_width: 576,
            x_off: 64,
            swapped,
            peel_in: !swapped,
            peel_out: swapped,
            x_dtype: DType::F32,
        });
        let plan = lower(&g, 0, &[(Rows, rows)], &dev).unwrap();
        assert_eq!(plan.steps.len(), 1, "{want_name}: {:?}", plan.steps.len());
        let step = &plan.steps[0];
        assert_eq!(step.kernel.name(), want_name);
        let Source::Fixed { entry, .. } = step.kernel.source() else {
            panic!("{want_name}: expected fixed CUDA source");
        };
        assert_eq!(entry, want_entry);
        let launches = step.kernel.launches(step);
        assert_eq!(launches.len(), 1);
        let GridShape::Grid { threads, .. } = launches[0].shape else {
            panic!("{want_name}: expected grid launch");
        };
        assert_eq!(threads, want_threads);
        let want_args = if want_name == "stacked_fused_lanes" {
            17
        } else {
            16
        };
        assert_eq!(launches[0].args.len(), want_args, "{want_name}");
    }
}

#[test]
fn stacked_fused_guards() {
    let g_peeled = stacked_graph(&Case {
        batches: 16,
        n: 128,
        k: 512,
        x_width: 512,
        x_off: 0,
        swapped: false,
        peel_in: true,
        peel_out: false,
        x_dtype: DType::F32,
    });
    // Non-bf16-tile CUDA GPUs must reject all stacked_fused kernels.
    let cuda_no_bf16 = Platform {
        backend: "cuda".into(),
        ..Platform::assumed(Limits::assumed())
    };
    let dev_no_bf16 = Device::default()
        .on(cuda_no_bf16)
        .pinned(Pins::all(BF16_FORM))
        .favoring("stacked_fused");
    for rows in [1u64, 8, 64] {
        let plan = lower(&g_peeled, 0, &[(Rows, rows)], &dev_no_bf16).unwrap();
        assert!(
            plan.steps
                .iter()
                .all(|s| !s.kernel.name().starts_with("stacked_fused")),
            "rows={rows}: unexpected stacked_fused without bf16_tile"
        );
    }

    // Metal and ROCm must reject stacked_fused kernels.
    for backend in ["metal", "rocm"] {
        let plat = Platform {
            backend: backend.into(),
            ..Platform::assumed(Limits::assumed())
        };
        let dev = Device::default()
            .on(plat)
            .pinned(Pins::all(BF16_FORM))
            .favoring("stacked_fused");
        let plan = lower(&g_peeled, 0, &[(Rows, 64)], &dev).unwrap();
        assert!(
            plan.steps
                .iter()
                .all(|s| !s.kernel.name().starts_with("stacked_fused")),
            "{backend}: unexpected stacked_fused"
        );
    }

    // Plain stacked matmul stays on default rules.
    let g_plain = stacked_graph(&Case {
        batches: 16,
        n: 128,
        k: 512,
        x_width: 512,
        x_off: 0,
        swapped: false,
        peel_in: false,
        peel_out: false,
        x_dtype: DType::F32,
    });
    let dev_cuda = Device::default()
        .on(cuda_platform())
        .pinned(Pins::all(BF16_FORM))
        .favoring("stacked_fused");
    let plan = lower(&g_plain, 0, &[(Rows, 64)], &dev_cuda).unwrap();
    assert!(
        plan.steps
            .iter()
            .all(|s| !s.kernel.name().starts_with("stacked_fused"))
    );

    // Unaligned slice offset must not trigger stacked_fused.
    let g_unaligned = stacked_graph(&Case {
        batches: 16,
        n: 128,
        k: 512,
        x_width: 520,
        x_off: 2,
        swapped: false,
        peel_in: true,
        peel_out: false,
        x_dtype: DType::F32,
    });
    let plan = lower(&g_unaligned, 0, &[(Rows, 64)], &dev_cuda).unwrap();
    assert!(
        plan.steps
            .iter()
            .all(|s| !s.kernel.name().starts_with("stacked_fused"))
    );
}
