use crate::*;
use DimName::*;
use Placement::*;

/// A graph holding only inputs of these types.
fn inputs(tys: Vec<TensorType>) -> (Graph, Vec<NodeId>) {
    let fixed = |name, n| Dim {
        name,
        size: Size::Fixed(n),
    };
    let bounded = |name, n| Dim {
        name,
        size: Size::Bounded(n),
    };
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
        })
    };
    let tys = vec![q, kv.clone(), kv];
    assert!(result(tys.clone(), attend(0.5)).is_ok());
    let err = result(tys, attend(f32::INFINITY)).unwrap_err();
    assert_eq!(err.kind, ErrorKind::Value);
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
    });
    let tys = vec![q, kv.clone(), kv];
    assert_eq!(placed(tys.clone(), attend.clone()), Err(ErrorKind::Arity));
    let listed = tys.into_iter().chain([ids]).collect();
    assert_eq!(placed(listed, attend), Ok(Replicated));
}

#[test]
fn layout_and_rope() {
    let x = f32s([Seqs, Rows, Heads, HeadDim], Replicated);
    let pos = TensorType::new([Seqs, Rows], DType::U32);
    let rope = Op::Row(Row(vec![Stage::Rope { theta: 1e4 }]));
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
