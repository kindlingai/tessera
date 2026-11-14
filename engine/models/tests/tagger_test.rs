//! spaCy's small English tagger: traced and verified.

use models::tagger::{Config, program};

#[test]
fn traces() {
    let c = Config {
        rows: vec![5000, 1000, 2500, 2500, 50, 50],
        width: 96,
        depth: 4,
        tags: 50,
        max_context: 512,
    };
    let g = trace::Graph::from_json(&program(&c).to_json()).unwrap();
    graph::verify(&g).unwrap();
    let count = |w: &graph::Weight| w.ty.shape.elements(&g.dims).unwrap().as_const().unwrap();
    assert_eq!(g.weights.iter().map(count).sum::<u64>(), 1_570_514);
}
