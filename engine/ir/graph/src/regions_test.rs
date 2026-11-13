use crate::verify_test::mlp;
use crate::*;

/// A region's start, end and outputs over `mlp()`.
type Span = (u32, u32, &'static [u32]);

/// Region sets and whether they verify.
const SETS: &[(&[Span], bool)] = &[
    (&[(7, 10, &[9])], true),
    // Node 10 reads 9. Node 9 is not an output.
    (&[(7, 10, &[8])], false),
    (&[(7, 9, &[9])], false),
    (&[(9, 9, &[9])], false),
    (&[(7, 10, &[9]), (7, 9, &[8]), (5, 6, &[5])], true),
    (&[(7, 10, &[9]), (8, 11, &[10])], false),
    (&[(7, 10, &[9]), (7, 10, &[9])], false),
];

const STRICT: Numerics = Numerics {
    accumulate: DType::F32,
    order: Order::Fixed,
    math: Math::Exact,
};

const RELAXED: Numerics = Numerics {
    accumulate: DType::F16,
    order: Order::Any,
    math: Math::Approx,
};

const INT: Numerics = Numerics {
    accumulate: DType::I32,
    ..RELAXED
};

/// A node's numerics, and whether the graph then verifies.
const NUMERICS: &[(usize, Option<Numerics>, bool)] = &[
    (7, None, false),
    (5, Some(STRICT), false),
    (7, Some(RELAXED), true),
    (7, Some(INT), false),
];

/// `mlp()` with regions whose inputs come from `Graph::reads`.
fn with_regions(spans: &[Span]) -> Graph {
    let mut g = mlp();
    for &(start, end, outputs) in spans {
        let (start, end) = (NodeId(start), NodeId(end));
        g.regions.push(Region {
            kind: Composite::Mlp,
            start,
            end,
            inputs: g.reads(start, end),
            outputs: outputs.iter().copied().map(NodeId).collect(),
        });
    }
    g
}

fn expect(g: &Graph, ok: bool, kind: ErrorKind) {
    match verify(g) {
        Ok(()) => assert!(ok, "{:?} verified", g.regions),
        Err(e) => assert_eq!((ok, e.error.kind), (false, kind), "{e}"),
    }
}

fn region_case(&(spans, ok): &(&[Span], bool)) {
    let g = with_regions(spans);
    expect(&g, ok, ErrorKind::Region);
    if ok {
        assert_eq!(Graph::from_json(&g.to_json()).unwrap(), g);
    }
}

fn numerics_case(&(node, numerics, ok): &(usize, Option<Numerics>, bool)) {
    let mut g = mlp();
    g.nodes[node].numerics = numerics;
    expect(&g, ok, ErrorKind::Numerics);
}

#[test]
fn regions_seal() {
    SETS.iter().for_each(region_case);
    let mut g = with_regions(&[(7, 10, &[9])]);
    g.regions[0].inputs.clear();
    expect(&g, false, ErrorKind::Region);
}

#[test]
fn numerics_only_on_reductions() {
    assert_eq!(mlp().nodes[7].numerics, Some(STRICT));
    NUMERICS.iter().for_each(numerics_case);
}
