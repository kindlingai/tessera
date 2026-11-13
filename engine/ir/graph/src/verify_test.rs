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
        Dim::bounded(Rows, 64),
        Dim::fixed(Hidden, 8),
        Dim::fixed(Ffn, 32),
        Dim::fixed(Vocab, 16),
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

/// An op on `mlp_with` nodes, and its expected error.
type Bad = (Op, &'static [u32], ErrorKind);

/// Bad ops on whole weights. Node 10 casts x.
fn whole_cases() -> Vec<Bad> {
    use ErrorKind::*;
    let row = |s| Op::Row(Row(vec![s]));
    let past = View::Slice {
        axis: 1,
        start: 4.into(),
        len: 8.into(),
    };
    vec![
        (Op::Matmul, &[7, 3], Shape),
        (row(Stage::Add), &[5, 7], Shape),
        (Op::View(View::Reshape([Rows, Ffn].into())), &[6], Shape),
        (Op::View(past), &[6], Shape),
        (Op::Select(Select::Gather), &[1, 5], DType),
        (row(Stage::Add), &[5, 10], DType),
        (row(Stage::Softmax), &[0], DType),
    ]
}

/// Bad ops with up split on rows, down on columns.
fn split_cases() -> Vec<Bad> {
    let row = |s| Op::Row(Row(vec![s]));
    let slice = View::Slice {
        axis: 1,
        start: 0.into(),
        len: 4.into(),
    };
    vec![
        (row(Stage::Add), &[5, 9], ErrorKind::Placement),
        (
            row(Stage::RmsNorm { eps: 1e-5 }),
            &[9, 2],
            ErrorKind::Placement,
        ),
        (Op::View(slice), &[7], ErrorKind::Placement),
    ]
}

fn refused(g: &Graph, (op, args, want): Bad) {
    let ids: Vec<NodeId> = args.iter().copied().map(NodeId).collect();
    let got = attempt(g, op.clone(), &ids).unwrap_err().kind;
    assert_eq!(got, want, "{op:?} on {args:?}");
}

type Edit = fn(&mut Graph);

/// Edits that break `mlp()`, and the node blamed.
const TAMPER: &[(Edit, Option<u32>)] = &[
    (|g| g.nodes[6].ty.shape = Shape::from([Rows, Ffn]), Some(6)),
    (|g| g.nodes[6].args[0] = NodeId(9), Some(6)),
    (|g| g.dims.truncate(g.dims.len() - 1), None),
];

#[test]
fn bad_ops_refused() {
    use Placement::*;
    verify(&mlp()).unwrap();
    let (mut g, [_, x, ..]) = mlp_with(Replicated, Replicated);
    push(&mut g, Op::Cast(DType::I32), &[x]);
    whole_cases().into_iter().for_each(|c| refused(&g, c));
    let (g, _) = mlp_with(Split { axis: 0 }, Split { axis: 1 });
    split_cases().into_iter().for_each(|c| refused(&g, c));
}

#[test]
fn reduce_makes_partials_whole() {
    use Placement::*;
    let (mut g, [_, x, _, u, d]) = mlp_with(Split { axis: 0 }, Split { axis: 1 });
    assert_eq!(g.node(u).ty.placement, Split { axis: 1 });
    assert_eq!(g.node(d).ty.placement, Partial);
    let whole = push(&mut g, Op::Row(Row(vec![Stage::AllReduce])), &[d]);
    push(&mut g, Op::Row(Row(vec![Stage::Add])), &[x, whole]);
    verify(&g).unwrap();
}

#[test]
fn verify_catches_tampering() {
    for (edit, node) in TAMPER {
        let mut g = mlp();
        edit(&mut g);
        assert_eq!(verify(&g).unwrap_err().node, *node);
    }
}
