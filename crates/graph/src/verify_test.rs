use crate::*;
use DimName::*;

pub(crate) fn push(g: &mut Graph, op: Op, args: &[NodeId]) -> NodeId {
    let ty = attempt(g, op.clone(), args).unwrap();
    let op_reduces = op.reduces();
    g.nodes.push(Node {
        op,
        args: args.to_vec(),
        ty,
        numerics: op_reduces.then(Numerics::default),
    });
    NodeId(g.nodes.len() as u32 - 1)
}

fn attempt(g: &Graph, op: Op, args: &[NodeId]) -> Result<TensorType, TypeError> {
    let tys: Vec<&TensorType> = args.iter().map(|a| &g.node(*a).ty).collect();
    infer(g, &op, &tys)
}

fn weight(g: &mut Graph, name: &str, ty: TensorType) -> NodeId {
    g.weights.push(Weight {
        path: vec![PathSeg::Field(name.into())],
        ty,
    });
    push(g, Op::Weight(g.weights.len() as u32 - 1), &[])
}

/// Embed, norm, MLP and residual, with chosen weight placements.
pub(crate) fn mlp_with(up_at: Placement, down_at: Placement) -> (Graph, [NodeId; 5]) {
    let dims = vec![
        Dim {
            name: Rows,
            size: Size::Bounded(64),
        },
        Dim {
            name: Hidden,
            size: Size::Fixed(8),
        },
        Dim {
            name: Ffn,
            size: Size::Fixed(32),
        },
        Dim {
            name: Vocab,
            size: Size::Fixed(16),
        },
    ];
    let mut g = Graph::new(dims);
    let tokens = TensorType::new([Rows], DType::U32);
    g.entries.push(Entry {
        kind: EntryKind::Forward,
        inputs: vec![Port {
            role: Role::Tokens,
            ty: tokens,
        }],
        outputs: vec![],
    });
    let t = push(&mut g, Op::Input { entry: 0, port: 0 }, &[]);
    let e = weight(
        &mut g,
        "embed",
        TensorType::new([Vocab, Hidden], DType::F32),
    );
    let n = weight(&mut g, "norm", TensorType::new([Hidden], DType::F32));
    let up_ty = TensorType::new([Ffn, Hidden], DType::F32).placed(up_at);
    let up = weight(&mut g, "up", up_ty);
    let down_ty = TensorType::new([Hidden, Ffn], DType::F32).placed(down_at);
    let down = weight(&mut g, "down", down_ty);
    let x = push(&mut g, Op::Select(Select::Gather), &[e, t]);
    let norm = Op::Row(Row(vec![Stage::RmsNorm { eps: 1e-5 }]));
    let h = push(&mut g, norm, &[x, n]);
    let u = push(&mut g, Op::Matmul, &[h, up]);
    let a = push(&mut g, Op::Row(Row(vec![Stage::Act(Act::GeluTanh)])), &[u]);
    let d = push(&mut g, Op::Matmul, &[a, down]);
    (g, [t, x, h, u, d])
}

pub(crate) fn mlp() -> Graph {
    let (mut g, [_, x, _, _, d]) = mlp_with(Placement::Replicated, Placement::Replicated);
    let y = push(&mut g, Op::Row(Row(vec![Stage::Add])), &[x, d]);
    g.entries[0].outputs.push(Output {
        role: Role::Logits,
        node: y,
    });
    g
}

fn kind(g: &Graph, op: Op, args: &[NodeId]) -> ErrorKind {
    attempt(g, op, args).unwrap_err().kind
}

#[test]
fn mlp_verifies() {
    verify(&mlp()).unwrap();
}

#[test]
fn shape_errors() {
    let (g, [_, x, h, u, _]) = mlp_with(Placement::Replicated, Placement::Replicated);
    let up = NodeId(3);
    assert_eq!(kind(&g, Op::Matmul, &[u, up]), ErrorKind::Shape);
    assert_eq!(
        kind(&g, Op::Row(Row(vec![Stage::Add])), &[x, u]),
        ErrorKind::Shape
    );
    let wide = View::Reshape(Shape::from([Rows, Ffn]));
    assert_eq!(kind(&g, Op::View(wide), &[h]), ErrorKind::Shape);
    let past = View::Slice {
        axis: 1,
        start: 4.into(),
        len: 8.into(),
    };
    assert_eq!(kind(&g, Op::View(past), &[h]), ErrorKind::Shape);
}

#[test]
fn dtype_errors() {
    let (mut g, [t, x, _, _, _]) = mlp_with(Placement::Replicated, Placement::Replicated);
    let e = NodeId(1);
    assert_eq!(
        kind(&g, Op::Select(Select::Gather), &[e, x]),
        ErrorKind::DType
    );
    let ints = push(&mut g, Op::Cast(DType::I32), &[x]);
    assert_eq!(
        kind(&g, Op::Row(Row(vec![Stage::Add])), &[x, ints]),
        ErrorKind::DType
    );
    assert_eq!(
        kind(&g, Op::Row(Row(vec![Stage::Softmax])), &[t]),
        ErrorKind::DType
    );
}

#[test]
fn placement_errors() {
    use Placement::*;
    let (mut g, [_, x, _, u, d]) = mlp_with(Split { axis: 0 }, Split { axis: 1 });
    assert_eq!(g.node(u).ty.placement, Split { axis: 1 });
    assert_eq!(g.node(d).ty.placement, Partial);
    let add = Op::Row(Row(vec![Stage::Add]));
    assert_eq!(kind(&g, add.clone(), &[x, d]), ErrorKind::Placement);
    let norm = Op::Row(Row(vec![Stage::RmsNorm { eps: 1e-5 }]));
    assert_eq!(kind(&g, norm, &[d, NodeId(2)]), ErrorKind::Placement);
    let slice = View::Slice {
        axis: 1,
        start: 0.into(),
        len: 4.into(),
    };
    assert_eq!(kind(&g, Op::View(slice), &[u]), ErrorKind::Placement);
    let whole = push(&mut g, Op::Row(Row(vec![Stage::AllReduce])), &[d]);
    assert_eq!(
        push(&mut g, add, &[x, whole]),
        NodeId(g.nodes.len() as u32 - 1)
    );
    verify(&g).unwrap();
}

#[test]
fn verify_catches_tampering() {
    let mut g = mlp();
    g.nodes[6].ty.shape = Shape::from([Rows, Ffn]);
    let err = verify(&g).unwrap_err();
    assert_eq!((err.node, err.error.kind), (Some(6), ErrorKind::Shape));

    let mut g = mlp();
    g.nodes[6].args[0] = NodeId(9);
    assert_eq!(verify(&g).unwrap_err().error.kind, ErrorKind::Reference);

    let mut g = mlp();
    g.dims.pop();
    assert_eq!(verify(&g).unwrap_err().node, None);
}
