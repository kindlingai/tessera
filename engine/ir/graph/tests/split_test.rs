use tessera_graph::DimName::{Context, HeadDim, Heads, Hidden, KvHeads, Rows, Seqs};
use tessera_graph::{
    Act, Attend, Composite, DType, Dim, DimExpr, Entry, EntryKind, Graph, HALF_ROWS, Mask, Node,
    NodeId, Numerics, Op, Output, Parts, PathSeg, Port, Region, Role, Row, Shape, Stage,
    TensorType, View, Weight, shard, split_batch, trim,
};

#[test]
fn splits_sharded_step() {
    let mut g = Graph::new(vec![
        Dim::bounded(Seqs, 2),
        Dim::bounded(Rows, 16),
        Dim::bounded(Context, 32),
        Dim::fixed(Hidden, 16),
        Dim::fixed(Heads, 2),
        Dim::fixed(KvHeads, 2),
        Dim::fixed(HeadDim, 8),
    ]);
    let w_ty = TensorType::new([Hidden, Hidden], DType::F32);
    for name in ["wq", "wk", "wv", "wo"] {
        g.weights.push(Weight {
            path: vec![PathSeg::Field(name.into())],
            ty: w_ty.clone(),
        });
    }
    let kept_ty = TensorType::new([Seqs, Context, KvHeads, HeadDim], DType::F32);
    g.kept.push(kept_ty.clone());
    g.kept.push(kept_ty.clone());
    g.entries.push(Entry {
        kind: EntryKind::Step,
        inputs: vec![
            Port {
                role: Role::Tokens,
                ty: TensorType::new([Rows, Hidden], DType::F32),
            },
            Port {
                role: Role::Slots,
                ty: TensorType::new([Rows], DType::U32),
            },
            Port {
                role: Role::Positions,
                ty: TensorType::new([Rows], DType::U32),
            },
        ],
        outputs: Vec::new(),
    });
    let add = |g: &mut Graph, op: Op, args: Vec<NodeId>, ty: TensorType, num: bool| {
        g.nodes.push(Node {
            op,
            args,
            ty,
            numerics: num.then(Numerics::default),
        });
        NodeId(g.nodes.len() as u32 - 1)
    };
    let rh = TensorType::new([Rows, Hidden], DType::F32);
    let x = add(
        &mut g,
        Op::Input { entry: 0, port: 0 },
        vec![],
        rh.clone(),
        false,
    );
    let slots = add(
        &mut g,
        Op::Input { entry: 0, port: 1 },
        vec![],
        TensorType::new([Rows], DType::U32),
        false,
    );
    let pos = add(
        &mut g,
        Op::Input { entry: 0, port: 2 },
        vec![],
        TensorType::new([Rows], DType::U32),
        false,
    );
    let r_start = NodeId(g.nodes.len() as u32);
    let wq = add(&mut g, Op::Weight(0), vec![], w_ty.clone(), false);
    let wk = add(&mut g, Op::Weight(1), vec![], w_ty.clone(), false);
    let wv = add(&mut g, Op::Weight(2), vec![], w_ty.clone(), false);
    let wo = add(&mut g, Op::Weight(3), vec![], w_ty, false);
    let q = add(&mut g, Op::Matmul, vec![x, wq], rh.clone(), true);
    let k = add(&mut g, Op::Matmul, vec![x, wk], rh.clone(), true);
    let v = add(&mut g, Op::Matmul, vec![x, wv], rh.clone(), true);
    let qh_ty = TensorType::new([Rows, Heads, HeadDim], DType::F32);
    let kh_ty = TensorType::new([Rows, KvHeads, HeadDim], DType::F32);
    let qh = add(
        &mut g,
        Op::View(View::Reshape(Shape::from([Rows, Heads, HeadDim]))),
        vec![q],
        qh_ty.clone(),
        false,
    );
    let kh = add(
        &mut g,
        Op::View(View::Reshape(Shape::from([Rows, KvHeads, HeadDim]))),
        vec![k],
        kh_ty.clone(),
        false,
    );
    let vh = add(
        &mut g,
        Op::View(View::Reshape(Shape::from([Rows, KvHeads, HeadDim]))),
        vec![v],
        kh_ty,
        false,
    );
    let ka = add(
        &mut g,
        Op::Append(0),
        vec![kh, slots, pos],
        kept_ty.clone(),
        false,
    );
    let va = add(&mut g, Op::Append(1), vec![vh, slots, pos], kept_ty, false);
    let att = add(
        &mut g,
        Op::Attend(Attend {
            mask: Mask::Causal,
            scale: 0.25,
            picked: false,
            ragged: true,
            sinks: false,
            rel_bias: false,
        }),
        vec![qh, ka, va, slots, pos],
        qh_ty,
        true,
    );
    let flat = add(
        &mut g,
        Op::View(View::Reshape(Shape::from([Rows, Hidden]))),
        vec![att],
        rh.clone(),
        false,
    );
    let proj = add(&mut g, Op::Matmul, vec![flat, wo], rh.clone(), true);
    let out = add(
        &mut g,
        Op::Row(Row(vec![Stage::Add])),
        vec![x, proj],
        rh.clone(),
        false,
    );
    let r_end = NodeId(g.nodes.len() as u32);
    let act = add(
        &mut g,
        Op::Row(Row(vec![Stage::Act(Act::Silu)])),
        vec![out],
        rh,
        false,
    );
    g.regions.push(Region {
        kind: Composite::Attention,
        start: r_start,
        end: r_end,
        inputs: g.reads(r_start, r_end),
        outputs: vec![out],
    });
    g.entries[0].outputs.push(Output {
        role: Role::Logits,
        node: act,
    });
    let base = trim(&g, 0).expect("trim");
    let sharded = shard(&base, &Parts::even(2)).expect("shard").graph;
    let split = split_batch(&sharded, 0).expect("split_batch");
    assert!(split.dims.iter().any(|d| d.name == HALF_ROWS));
    let ars = split
        .nodes
        .iter()
        .filter(|n| matches!(&n.op, Op::Row(r) if r.0.contains(&Stage::AllReduce)))
        .count();
    assert_eq!(ars, 2);
    assert_eq!(
        split.nodes.iter().filter(|n| n.op == Op::Append(0)).count(),
        2
    );
    assert_eq!(
        split
            .regions
            .iter()
            .filter(|r| r.kind == Composite::Attention)
            .count(),
        2
    );
    assert_eq!(
        split.entries[0].inputs[0].ty.shape.0[0],
        DimExpr::Const(2) * DimExpr::Dim(HALF_ROWS)
    );
}

#[test]
fn verify_kept_and_shapes() {
    use tessera_graph::{Conv, ErrorKind, infer, verify};
    let push = |g: &mut Graph, op: Op, args: &[NodeId]| -> NodeId {
        let tys: Vec<&TensorType> = args.iter().map(|a| &g.node(*a).ty).collect();
        let ty = infer(g, &op, &tys).unwrap();
        let numerics = op.reduces().then(Numerics::default);
        g.nodes.push(Node {
            op,
            args: args.to_vec(),
            ty,
            numerics,
        });
        NodeId(g.nodes.len() as u32 - 1)
    };
    let mut g = Graph::new(vec![
        Dim::bounded(Seqs, 4),
        Dim::bounded(Rows, 8),
        Dim::bounded(Context, 16),
        Dim::fixed(Hidden, 4),
    ]);
    let row = TensorType::new([Rows, Hidden], DType::F32);
    let idx = TensorType::new([Rows], DType::U32);
    let kv = TensorType::new([Seqs, Context, Hidden], DType::F32);
    let st = TensorType::new(
        vec![DimExpr::Dim(Seqs), DimExpr::Const(3), DimExpr::Dim(Hidden)],
        DType::F32,
    );
    let port = |role, ty| Port { role, ty };
    for _ in 0..2 {
        g.entries.push(Entry {
            kind: EntryKind::Forward,
            inputs: vec![
                port(Role::Tokens, row.clone()),
                port(Role::Slots, idx.clone()),
                port(Role::Positions, idx.clone()),
            ],
            outputs: vec![],
        });
    }
    g.weights.push(Weight {
        path: vec![PathSeg::Field("w".into())],
        ty: TensorType::new([Hidden, Hidden], DType::F32),
    });
    g.kept = vec![kv, st];
    let x0 = push(&mut g, Op::Input { entry: 0, port: 0 }, &[]);
    let s0 = push(&mut g, Op::Input { entry: 0, port: 1 }, &[]);
    let p0 = push(&mut g, Op::Input { entry: 0, port: 2 }, &[]);
    let x1 = push(&mut g, Op::Input { entry: 1, port: 0 }, &[]);
    let s1 = push(&mut g, Op::Input { entry: 1, port: 1 }, &[]);
    let p1 = push(&mut g, Op::Input { entry: 1, port: 2 }, &[]);
    let w = push(&mut g, Op::Weight(0), &[]);
    push(&mut g, Op::Append(0), &[x0, s0, p0]);
    push(&mut g, Op::Append(0), &[x1, s1, p1]);
    let conv = Op::Conv(Conv {
        dilation: 1,
        state: Some(1),
    });
    let c0 = push(&mut g, conv.clone(), &[x0, w, s0, p0]);
    push(&mut g, conv.clone(), &[x1, w, s1, p1]);
    g.nodes[c0.0 as usize].ty.shape =
        Shape(vec![Rows.into(), DimExpr::Const(2) + DimExpr::Const(2)]);
    verify(&g).unwrap();
    let mut dup_app = g.clone();
    push(&mut dup_app, Op::Append(0), &[x0, s0, p0]);
    assert_eq!(
        verify(&dup_app).unwrap_err().error.kind,
        ErrorKind::Reference
    );
    let mut dup_st = g.clone();
    push(&mut dup_st, conv, &[x0, w, s0, p0]);
    assert_eq!(
        verify(&dup_st).unwrap_err().error.kind,
        ErrorKind::Reference
    );
    let mut mixed = g.clone();
    mixed.kept[0] = mixed.kept[1].clone();
    let mix_conv = Op::Conv(Conv {
        dilation: 1,
        state: Some(0),
    });
    push(&mut mixed, mix_conv, &[x0, w, s0, p0]);
    assert_eq!(verify(&mixed).unwrap_err().error.kind, ErrorKind::Reference);
}
