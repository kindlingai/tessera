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
