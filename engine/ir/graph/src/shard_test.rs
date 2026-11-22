use crate::*;
use DimName::*;
use Placement::*;

/// Input `x`, then weights of these shapes.
fn graph(w: &[[DimName; 2]]) -> Graph {
    let mut g = Graph::new(vec![
        Dim::bounded(Rows, 8),
        Dim::fixed(Hidden, 4),
        Dim::fixed(Ffn, 6),
    ]);
    let ty = |shape: [DimName; 2]| TensorType::new(shape, DType::F32);
    g.entries.push(Entry {
        kind: EntryKind::Forward,
        inputs: vec![Port {
            role: Role::Tokens,
            ty: ty([Rows, Hidden]),
        }],
        outputs: vec![],
    });
    g.nodes.push(Node {
        op: Op::Input { entry: 0, port: 0 },
        args: vec![],
        ty: ty([Rows, Hidden]),
        numerics: None,
    });
    for (i, shape) in w.iter().enumerate() {
        g.weights.push(Weight {
            path: vec![PathSeg::Index(i as u32)],
            ty: ty(*shape),
        });
        g.nodes.push(Node {
            op: Op::Weight(i as u32),
            args: vec![],
            ty: ty(*shape),
            numerics: None,
        });
    }
    g
}

/// Appends `op` over `args`, typed by the rules.
fn push(g: &mut Graph, op: Op, args: &[NodeId]) -> NodeId {
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
}

fn finish(g: &mut Graph, out: NodeId) {
    g.entries[0].outputs.push(Output {
        role: Role::Logits,
        node: out,
    });
    verify(g).unwrap();
}

/// `x + act(x w1) w2`, with `mid` between the matmuls.
fn mlp(mid: Stage) -> Graph {
    let mut g = graph(&[[Ffn, Hidden], [Hidden, Ffn]]);
    let [x, w1, w2] = [NodeId(0), NodeId(1), NodeId(2)];
    let up = push(&mut g, Op::Matmul, &[x, w1]);
    let inner = push(&mut g, Op::Row(Row(vec![mid])), &[up]);
    let down = push(&mut g, Op::Matmul, &[inner, w2]);
    let out = push(&mut g, Op::Row(Row(vec![Stage::Add])), &[x, down]);
    finish(&mut g, out);
    g
}

fn placements(g: &Graph) -> Vec<Placement> {
    g.weights.iter().map(|w| w.ty.placement).collect()
}

#[test]
fn mlp_splits_and_sums() {
    let g = shard(&mlp(Stage::Act(Act::Silu)), &Parts::even(2))
        .unwrap()
        .graph;
    assert_eq!(placements(&g), [Split { axis: 0 }, Split { axis: 1 }]);
    let sums = |n: &Node| n.op == Op::Row(Row(vec![Stage::AllReduce]));
    assert_eq!(g.nodes.iter().filter(|n| sums(n)).count(), 1);
    // Partial down, then whole sum and add.
    assert_eq!(g.nodes[5].ty.placement, Partial);
    assert_eq!(g.nodes[6].ty.placement, Replicated);
    let out = g.entries[0].outputs[0].node;
    assert_eq!((out, g.node(out).ty.placement), (NodeId(7), Replicated));
}

#[test]
fn whole_rows_stay_whole() {
    // A norm between them needs whole rows, so nothing splits.
    let g = shard(&mlp(Stage::Rms { eps: 1e-6 }), &Parts::even(2)).unwrap();
    assert_eq!(placements(&g.graph), [Replicated, Replicated]);
    assert_eq!(g.graph.nodes.len(), 7);
    let held: Vec<u32> = g.whole.iter().map(|w| w.weight).collect();
    assert_eq!(held, [0, 1]);
}

#[test]
fn uneven_stays_whole() {
    // Six columns do not divide into four equal units.
    let g = shard(&mlp(Stage::Act(Act::Silu)), &Parts(vec![3, 1])).unwrap();
    assert_eq!(placements(&g.graph), [Replicated, Replicated]);
    assert!(g.whole[0].why.contains("does not divide"), "{:?}", g.whole);
}

#[test]
fn cut_runs_stay_whole() {
    // Three of six columns per rank cut blocks of two.
    let even = |w, _: &[usize], _, count| w == 0 || count % 2 == 0;
    let g = shard_aligned(&mlp(Stage::Act(Act::Silu)), &Parts::even(2), &even).unwrap();
    assert_eq!(placements(&g.graph), [Replicated, Replicated]);
    assert!(g.whole[0].why.contains("cuts its form"), "{:?}", g.whole);
}

#[test]
fn gather_leaves_region() {
    let mut g = graph(&[[Ffn, Hidden]]);
    let out = push(&mut g, Op::Matmul, &[NodeId(0), NodeId(1)]);
    g.regions.push(Region {
        kind: Composite::Model,
        start: out,
        end: NodeId(out.0 + 1),
        inputs: vec![NodeId(0), NodeId(1)],
        outputs: vec![out],
    });
    finish(&mut g, out);
    let g = shard(&g, &Parts::even(2)).unwrap().graph;
    assert_eq!(placements(&g), [Split { axis: 0 }]);
    let gather = NodeId(out.0 + 1);
    assert_eq!(g.node(gather).op, Op::Row(Row(vec![Stage::AllGather])));
    assert_eq!(g.entries[0].outputs[0].node, gather);
    assert_eq!(
        (g.regions[0].end, &g.regions[0].outputs),
        (gather, &vec![out])
    );
}
