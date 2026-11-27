use crate::*;
use DimName::*;
use Placement::*;

/// A graph holding only inputs of these types.
fn inputs(tys: Vec<TensorType>) -> (Graph, Vec<NodeId>) {
    let (fixed, bounded) = (Dim::fixed, Dim::bounded);
    let dims = vec![
        bounded(Seqs, 4),
        bounded(Rows, 64),
        bounded(Context, 64),
        fixed(Hidden, 8),
        fixed(Heads, 2),
        fixed(HeadDim, 4),
        fixed(Ffn, 32),
    ];
    let mut g = Graph::new(dims);
    let ports = tys.into_iter().map(|ty| Port {
        role: Role::Tokens,
        ty,
    });
    g.entries.push(Entry {
        kind: EntryKind::Forward,
        inputs: ports.collect(),
        outputs: vec![],
    });
    let ids = (0..g.entries[0].inputs.len() as u32)
        .map(|port| {
            let ty = g.entries[0].inputs[port as usize].ty.clone();
            g.nodes.push(Node {
                op: Op::Input { entry: 0, port },
                args: vec![],
                ty,
                numerics: None,
            });
            NodeId(port)
        })
        .collect();
    (g, ids)
}

fn f32s(shape: impl Into<Shape>, at: Placement) -> TensorType {
    TensorType::new(shape, DType::F32).placed(at)
}

fn result(tys: Vec<TensorType>, op: Op) -> Result<TensorType, TypeError> {
    let (g, ids) = inputs(tys);
    let args: Vec<&TensorType> = ids.iter().map(|i| &g.node(*i).ty).collect();
    infer(&g, &op, &args)
}

fn placed(tys: Vec<TensorType>, op: Op) -> Result<Placement, ErrorKind> {
    result(tys, op).map(|t| t.placement).map_err(|e| e.kind)
}

fn add() -> Op {
    Op::Row(Row(vec![Stage::Add]))
}

#[test]
fn broadcast_split_axes() {
    let x = f32s([Rows, Ffn], Split { axis: 1 });
    let bias = f32s([Ffn], Split { axis: 0 });
    assert_eq!(placed(vec![x.clone(), bias], add()), Ok(Split { axis: 1 }));
    let full = f32s([Ffn], Replicated);
    assert_eq!(placed(vec![x, full], add()), Err(ErrorKind::Placement));
    let x = f32s([Seqs, Context, Hidden], Split { axis: 1 });
    let other = f32s([Context, Hidden], Split { axis: 1 });
    assert_eq!(placed(vec![x, other], add()), Err(ErrorKind::Placement));
}

#[test]
fn matmul_row_splits() {
    let x = f32s([Rows, Hidden], Split { axis: 0 });
    let w = f32s([Ffn, Hidden], Replicated);
    assert_eq!(
        placed(vec![x, w.clone()], Op::Matmul),
        Ok(Split { axis: 0 })
    );
    let partial = f32s([Rows, Hidden], Partial);
    assert_eq!(placed(vec![partial.clone(), w], Op::Matmul), Ok(Partial));
    let cast = Op::Cast(DType::Bf16);
    assert_eq!(placed(vec![partial], cast), Err(ErrorKind::Placement));
}

#[test]
fn plain_casts() {
    let cast = |from, to| {
        let x = TensorType::new([Rows, Hidden], from);
        placed(vec![x], Op::Cast(to))
    };
    assert_eq!(cast(DType::F32, DType::U32), Ok(Replicated));
    for (from, to) in [(DType::F32, DType::Bf16), (DType::F16, DType::F32)] {
        assert_eq!(cast(from, to), Err(ErrorKind::DType));
    }
}

#[test]
fn reshape_keeps_split_axis() {
    let heads = Shape::from([Rows, Heads, HeadDim]);
    let x = f32s([Rows, Hidden], Split { axis: 1 });
    let split = Op::View(View::Reshape(heads.clone()));
    assert_eq!(placed(vec![x], split), Ok(Split { axis: 1 }));
    let back = Op::View(View::Reshape(Shape::from([Rows, Hidden])));
    let by_head = f32s(heads.clone(), Split { axis: 1 });
    assert_eq!(placed(vec![by_head], back.clone()), Ok(Split { axis: 1 }));
    let by_dim = f32s(heads, Split { axis: 2 });
    assert_eq!(placed(vec![by_dim], back), Err(ErrorKind::Placement));
}

#[test]
fn decode_and_constants() {
    let q = f32s([Seqs, Rows, Heads, HeadDim], Replicated);
    let kv = f32s([Seqs, Context, Heads, HeadDim], Replicated);
    let attend = |scale| {
        Op::Attend(Attend {
            mask: Mask::Causal,
            scale,
            picked: false,
            ragged: false,
            sinks: false,
            rel_bias: false,
        })
    };
    let tys = vec![q.clone(), kv.clone(), kv.clone()];
    assert!(result(tys.clone(), attend(0.5)).is_ok());
    let err = result(tys, attend(f32::INFINITY)).unwrap_err();
    assert_eq!(err.kind, ErrorKind::Value);
    // Values narrower than keys shape the output.
    let narrow = f32s([Seqs, Context, Heads, Hidden], Replicated);
    let out = result(vec![q, kv, narrow], attend(0.5)).unwrap();
    assert_eq!(out.shape, Shape::from([Seqs, Rows, Heads, Hidden]));
    let row = f32s([Rows, Hidden], Replicated);
    let nan = Op::Row(Row(vec![Stage::Scale(f32::NAN)]));
    assert_eq!(placed(vec![row], nan), Err(ErrorKind::Value));
}

#[test]
fn weights_and_fields_checked() {
    let mut g = crate::verify_test::mlp();
    g.weights[1].ty.shape = Shape::from([Rows]);
    assert_eq!(verify(&g).unwrap_err().error.kind, ErrorKind::Shape);
    let text = crate::verify_test::mlp().to_json();
    let v = format!("\"version\": {VERSION},");
    let extra = text.replacen(&v, &format!("{v} \"extra\": 0,"), 1);
    assert!(matches!(
        Graph::from_json(&extra),
        Err(FormatError::Json(_))
    ));
}

#[test]
fn batched_matmul() {
    let x = f32s([Seqs, Rows, Heads, HeadDim], Replicated);
    let keys = f32s([Seqs, Context, HeadDim], Replicated);
    let out = result(vec![x.clone(), keys], Op::Matmul).unwrap();
    assert_eq!(out.shape, Shape::from([Seqs, Rows, Heads, Context]));
    let other = f32s([Rows, Context, HeadDim], Replicated);
    let batch = placed(vec![x.clone(), other], Op::Matmul);
    assert_eq!(batch, Err(ErrorKind::Shape));
    let split = f32s([Seqs, Context, HeadDim], Split { axis: 1 });
    assert_eq!(placed(vec![x, split], Op::Matmul), Ok(Split { axis: 3 }));
}

#[test]
fn picks_by_index() {
    let scores = f32s([Seqs, Rows, Context], Replicated);
    let top = |k: DimName| Op::Select(Select::TopK { k: k.into() });
    let ids = result(vec![scores.clone()], top(Heads)).unwrap();
    assert_eq!(ids.shape, Shape::from([Seqs, Rows, Heads]));
    assert_eq!(ids.dtype, DType::U32);
    let unsized_k = placed(vec![scores.clone()], top(Rows));
    assert_eq!(unsized_k, Err(ErrorKind::Shape));
    let partial = f32s([Seqs, Rows, Context], Partial);
    assert_eq!(placed(vec![partial], top(Heads)), Err(ErrorKind::Placement));
    let take = Op::Select(Select::Take);
    let taken = result(vec![scores.clone(), ids.clone()], take).unwrap();
    assert_eq!((&taken.shape, taken.dtype), (&ids.shape, DType::F32));
    let masked = result(vec![scores.clone()], Op::Mask(Mask::Causal));
    assert_eq!(masked, Ok(scores));
    let q = f32s([Seqs, Rows, Heads, HeadDim], Replicated);
    let kv = f32s([Seqs, Context, Heads, HeadDim], Replicated);
    let attend = Op::Attend(Attend {
        mask: Mask::Causal,
        scale: 0.5,
        picked: true,
        ragged: false,
        sinks: false,
        rel_bias: false,
    });
    let tys = vec![q, kv.clone(), kv];
    assert_eq!(placed(tys.clone(), attend.clone()), Err(ErrorKind::Arity));
    let listed = tys.into_iter().chain([ids]).collect();
    assert_eq!(placed(listed, attend), Ok(Replicated));
}

#[test]
fn ragged_causal() {
    let q = f32s([Rows, Heads, HeadDim], Replicated);
    let kv = f32s([Seqs, Context, Heads, HeadDim], Replicated);
    let by = TensorType::new([Rows], DType::U32);
    let tys = vec![q, kv.clone(), kv, by.clone(), by];
    let attend = |mask| {
        Op::Attend(Attend {
            mask,
            scale: 0.5,
            picked: false,
            ragged: true,
            sinks: false,
            rel_bias: false,
        })
    };
    assert!(result(tys.clone(), attend(Mask::Causal)).is_ok());
    let mut split = tys.clone();
    split[0].placement = Split { axis: 1 };
    let wrong = placed(split.clone(), attend(Mask::Causal));
    assert_eq!(wrong, Err(ErrorKind::Placement));
    split[1].placement = Split { axis: 2 };
    split[2].placement = Split { axis: 2 };
    let heads = placed(split, attend(Mask::Causal));
    assert_eq!(heads, Ok(Split { axis: 1 }));
    for mask in [Mask::Bidirectional, Mask::Within(4)] {
        assert_eq!(placed(tys.clone(), attend(mask)), Err(ErrorKind::Value));
    }
}

#[test]
fn layout_and_rope() {
    let x = f32s([Seqs, Rows, Heads, HeadDim], Replicated);
    let pos = TensorType::new([Seqs, Rows], DType::U32);
    let rope = Op::Row(Row(vec![Stage::Rope {
        theta: 1e4,
        pairs: Pairs::Halves,
        turned: None,
    }]));
    assert!(result(vec![x.clone(), pos.clone()], rope.clone()).is_ok());
    let rows = TensorType::new([Rows], DType::U32);
    let misled = placed(vec![x.clone(), rows], rope.clone());
    assert_eq!(misled, Err(ErrorKind::Shape));
    let odd = f32s(vec![Seqs.into(), Rows.into(), 3.into()], Replicated);
    assert_eq!(placed(vec![odd, pos], rope), Err(ErrorKind::Shape));
    let joined = result(vec![x.clone(), x], Op::View(View::Concat)).unwrap();
    assert_eq!(joined.shape.0[3], HeadDim + HeadDim);
    let one = f32s(vec![Seqs.into(), 1.into(), Hidden.into()], Replicated);
    let expand = |axis| {
        let to = Rows.into();
        Op::View(View::Expand { axis, to })
    };
    let wide = result(vec![one.clone()], expand(1)).unwrap();
    assert_eq!(wide.shape, Shape::from([Seqs, Rows, Hidden]));
    assert_eq!(placed(vec![one], expand(2)), Err(ErrorKind::Shape));
    let center = Op::Row(Row(vec![Stage::Center]));
    let split = f32s([Rows, Hidden], Split { axis: 1 });
    assert_eq!(placed(vec![split], center), Err(ErrorKind::Placement));
}

#[test]
fn kept_rows_append() {
    let rows = f32s([Rows, Heads, HeadDim], Replicated);
    let kept = f32s([Seqs, Context, Heads, HeadDim], Replicated);
    let wide = f32s([Rows, Heads, Hidden], Replicated);
    let split = rows.clone().placed(Split { axis: 1 });
    let ids = TensorType::new([Rows], DType::U32);
    let (mut g, at) = inputs(vec![rows, wide, split, ids.clone(), ids]);
    g.kept.push(kept.clone());
    let append = |g: &Graph, arg: usize, kept| {
        let args = [at[arg], at[3], at[4]].map(|i| &g.node(i).ty);
        infer(g, &Op::Append(kept), &args).map_err(|e| e.kind)
    };
    assert_eq!(append(&g, 0, 0), Ok(kept.clone()));
    assert_eq!(append(&g, 1, 0), Err(ErrorKind::Shape));
    assert_eq!(append(&g, 2, 0), Err(ErrorKind::Placement));
    assert_eq!(append(&g, 0, 1), Err(ErrorKind::Reference));
    // Head-split rows go to head-split kept rows.
    g.kept[0] = kept.clone().placed(Split { axis: 2 });
    assert_eq!(append(&g, 2, 0), Ok(g.kept[0].clone()));
    g.kept[0] = kept.clone();
    let push = |g: &mut Graph| {
        g.nodes.push(Node {
            op: Op::Append(0),
            args: vec![at[0], at[3], at[4]],
            ty: kept.clone(),
            numerics: None,
        });
        verify(g).map_err(|e| e.error.kind)
    };
    assert_eq!(push(&mut g), Ok(()));
    assert_eq!(push(&mut g), Err(ErrorKind::Reference), "appended twice");
    g.nodes.pop();
    // Kept rows must grow along a bounded axis.
    g.kept[0] = f32s([Seqs, Heads, Heads, HeadDim], Replicated);
    assert_eq!(verify(&g).unwrap_err().error.kind, ErrorKind::Shape);
}

#[test]
fn attend_sinks_infer() {
    let q = f32s([Rows, Heads, HeadDim], Replicated);
    let kv = f32s([Seqs, Context, Heads, HeadDim], Replicated);
    let by = TensorType::new([Rows], DType::U32);
    let sinks = f32s([Heads], Replicated);
    let op = Op::Attend(Attend {
        mask: Mask::Causal,
        scale: 0.5,
        picked: false,
        ragged: true,
        sinks: true,
        rel_bias: false,
    });
    let tys = vec![
        q.clone(),
        kv.clone(),
        kv.clone(),
        by.clone(),
        by.clone(),
        sinks,
    ];
    assert_eq!(placed(tys.clone(), op.clone()), Ok(Replicated));
    // Missing sinks fails arity.
    assert_eq!(placed(tys[..5].to_vec(), op.clone()), Err(ErrorKind::Arity));
    // Wrong shape fails shape check.
    let bad_shape = f32s([HeadDim], Replicated);
    let mut wrong = tys.clone();
    wrong[5] = bad_shape;
    assert_eq!(placed(wrong, op.clone()), Err(ErrorKind::Shape));
    // Head-split queries require head-split sinks.
    let mut split = tys;
    split[0].placement = Split { axis: 1 };
    split[1].placement = Split { axis: 2 };
    split[2].placement = Split { axis: 2 };
    assert_eq!(placed(split.clone(), op.clone()), Err(ErrorKind::Placement));
    split[5].placement = Split { axis: 0 };
    assert_eq!(placed(split, op), Ok(Split { axis: 1 }));
}

/// Ragged attention types, sinks and bias last.
fn biased(sinks: TensorType, win: DimName) -> (Vec<TensorType>, Op) {
    let q = f32s([Rows, Heads, HeadDim], Replicated);
    let kv = f32s([Seqs, Context, Heads, HeadDim], Replicated);
    let by = TensorType::new([Rows], DType::U32);
    let rel = f32s([Rows, Heads, win], Replicated);
    let op = Op::Attend(Attend {
        mask: Mask::Back(2),
        scale: 0.5,
        picked: false,
        ragged: true,
        sinks: true,
        rel_bias: true,
    });
    (vec![q, kv.clone(), kv, by.clone(), by, sinks, rel], op)
}

#[test]
fn attend_bias_infer() {
    let (tys, op) = biased(f32s([Rows, Heads], Replicated), Ffn);
    assert_eq!(placed(tys.clone(), op.clone()), Ok(Replicated));
    assert_eq!(placed(tys[..6].to_vec(), op.clone()), Err(ErrorKind::Arity));
    // Static sinks still fit beside a bias.
    let (fixed, _) = biased(f32s([Heads], Replicated), Ffn);
    assert_eq!(placed(fixed, op.clone()), Ok(Replicated));
    for (at, bad) in [
        (5, f32s([Rows, HeadDim], Replicated)),
        (5, f32s([Seqs, Context, Heads], Replicated)),
        (6, f32s([Rows, Heads], Replicated)),
        (6, f32s([Rows, HeadDim, Ffn], Replicated)),
    ] {
        let mut wrong = tys.clone();
        wrong[at] = bad;
        assert_eq!(placed(wrong, op.clone()), Err(ErrorKind::Shape));
    }
    let mut ints = tys.clone();
    ints[6].dtype = DType::U32;
    assert_eq!(placed(ints, op.clone()), Err(ErrorKind::DType));
    // Head-split queries need head-split sinks and bias.
    let mut split = tys;
    split[0].placement = Split { axis: 1 };
    split[1].placement = Split { axis: 2 };
    split[2].placement = Split { axis: 2 };
    assert_eq!(placed(split.clone(), op.clone()), Err(ErrorKind::Placement));
    split[5].placement = Split { axis: 1 };
    assert_eq!(placed(split.clone(), op.clone()), Err(ErrorKind::Placement));
    split[6].placement = Split { axis: 1 };
    assert_eq!(placed(split, op), Ok(Split { axis: 1 }));
    // Whole sequences carry one sink per query head.
    let q = f32s([Seqs, Context, Heads, HeadDim], Replicated);
    let whole = Op::Attend(Attend {
        mask: Mask::Causal,
        scale: 0.5,
        picked: false,
        ragged: false,
        sinks: true,
        rel_bias: true,
    });
    let sinks = f32s([Seqs, Context, Heads], Replicated);
    let rel = f32s([Seqs, Context, Heads, Ffn], Replicated);
    let tys = vec![q.clone(), q.clone(), q, sinks, rel];
    assert_eq!(placed(tys, whole), Ok(Replicated));
}

/// A step graph attending with `sinks` and a bias.
fn biased_step(sinks: TensorType) -> Graph {
    let (tys, op) = biased(sinks, Ffn);
    let (mut g, ids) = inputs(tys);
    g.entries[0].kind = EntryKind::Step;
    // Slots and positions follow the picks ungathered.
    g.entries[0].inputs[3].role = Role::Candidates;
    g.entries[0].inputs[4].role = Role::Candidates;
    let node = crate::verify_test::push(&mut g, op, &ids);
    g.entries[0].outputs.push(Output {
        role: Role::Logits,
        node,
    });
    g
}

#[test]
fn attend_bias_trims() {
    let g = biased_step(f32s([Rows, Heads], Replicated));
    verify(&g).unwrap();
    assert_eq!(Graph::from_json(&g.to_json()).unwrap(), g);
    let Op::Attend(a) = &g.nodes.last().unwrap().op else {
        panic!("attend is last");
    };
    assert_eq!(
        (a.arity(), a.sinks_arg(), a.rel_bias_arg()),
        (7, Some(5), Some(6))
    );
    // Per-token sinks and bias follow the picked rows.
    let gathers = |g: &Graph| {
        let trimmed = trim(g, 0).unwrap();
        let attend = trimmed.node(trimmed.entries[0].outputs[0].node);
        assert!(matches!(attend.op, Op::Attend(_)));
        assert_eq!(attend.ty.shape.0[0], DimExpr::Dim(Picks));
        let gather = |n: &&Node| n.op == Op::Select(Select::Gather);
        trimmed.nodes.iter().filter(gather).count()
    };
    assert_eq!(gathers(&g), 3);
    // Per-head sinks stay whole.
    assert_eq!(gathers(&biased_step(f32s([Heads], Replicated))), 2);
}
