use crate::verify_test::mlp;
use crate::*;

/// Covers the up, act and down nodes of `mlp()`.
fn sealed() -> Graph {
    let mut g = mlp();
    g.regions.push(region(7, 10, vec![9]));
    g
}

fn region(start: u32, end: u32, outputs: Vec<u32>) -> Region {
    Region {
        kind: Composite::Mlp,
        start: NodeId(start),
        end: NodeId(end),
        inputs: Vec::new(),
        outputs: outputs.into_iter().map(NodeId).collect(),
    }
}

/// Fills in inputs the way the tracer does.
fn add(g: &mut Graph, mut r: Region) {
    r.inputs = g.reads(r.start, r.end);
    g.regions.push(r);
}

fn kind_of(g: &Graph) -> ErrorKind {
    verify(g).unwrap_err().error.kind
}

#[test]
fn sealed_region_verifies() {
    let mut g = mlp();
    add(&mut g, region(7, 10, vec![9]));
    assert_eq!(g.regions[0].inputs, [NodeId(6), NodeId(3), NodeId(4)]);
    verify(&g).unwrap();
    let back = Graph::from_json(&g.to_json()).unwrap();
    assert_eq!(back, g);
}

#[test]
fn inputs_must_match() {
    assert_eq!(kind_of(&sealed()), ErrorKind::Region);
}

#[test]
fn values_cannot_leak() {
    let mut g = mlp();
    add(&mut g, region(7, 10, vec![8]));
    assert_eq!(kind_of(&g), ErrorKind::Region);
    let mut g = mlp();
    add(&mut g, region(7, 9, vec![9]));
    assert_eq!(kind_of(&g), ErrorKind::Region);
    let mut g = mlp();
    add(&mut g, region(9, 9, vec![9]));
    assert_eq!(kind_of(&g), ErrorKind::Region);
}

#[test]
fn regions_nest() {
    let mut g = mlp();
    add(&mut g, region(7, 10, vec![9]));
    add(&mut g, region(7, 9, vec![8]));
    add(&mut g, region(5, 6, vec![5]));
    verify(&g).unwrap();
    add(&mut g, region(8, 11, vec![10]));
    assert_eq!(kind_of(&g), ErrorKind::Region);
    let mut g = mlp();
    add(&mut g, region(7, 10, vec![9]));
    add(&mut g, region(7, 10, vec![9]));
    assert_eq!(kind_of(&g), ErrorKind::Region);
}

#[test]
fn numerics_only_on_reductions() {
    let mut g = mlp();
    assert_eq!(g.nodes[7].numerics, Some(Numerics::default()));
    g.nodes[7].numerics = None;
    let err = verify(&g).unwrap_err();
    assert_eq!((err.node, err.error.kind), (Some(7), ErrorKind::Numerics));

    let mut g = mlp();
    g.nodes[5].numerics = Some(Numerics::default());
    assert_eq!(kind_of(&g), ErrorKind::Numerics);

    let mut g = mlp();
    let relaxed = Numerics {
        accumulate: DType::F16,
        order: Order::Any,
        math: Math::Approx,
    };
    g.nodes[7].numerics = Some(relaxed);
    verify(&g).unwrap();
    g.nodes[7].numerics = Some(Numerics {
        accumulate: DType::I32,
        ..relaxed
    });
    assert_eq!(kind_of(&g), ErrorKind::Numerics);
}
