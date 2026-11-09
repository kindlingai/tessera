//! GLM-5.3 traces, verifies and survives JSON at its released sizes.

use graph::verify;
use models::glm53::{Config, program};
use trace::DimName::*;
use trace::{Composite, Graph, Shape};

#[test]
fn release_traces() {
    let c = Config::release(8, 425_984);
    let g = program(&c);
    let back = Graph::from_json(&g.to_json()).unwrap();
    assert_eq!(back, g);
    verify(&back).unwrap();

    let regions = |kind| g.regions.iter().filter(|r| r.kind == kind).count() as u64;
    assert_eq!(regions(Composite::Attention), c.layers);
    assert_eq!(regions(Composite::Mlp), c.dense_layers);
    assert_eq!(regions(Composite::Moe), c.layers - c.dense_layers);
    // Indexing layers also hand their keys to later layers.
    let shared = g.regions.iter().filter(|r| r.outputs.len() == 2).count();
    assert_eq!(shared, 21);

    let logits = g.node(g.entries[0].outputs[0].node);
    assert_eq!(logits.ty.shape, Shape::from([Seqs, Context, Vocab]));
    let count = |w: &graph::Weight| w.ty.shape.elements(&g.dims).unwrap().as_const().unwrap();
    let parameters: u64 = g.weights.iter().map(count).sum();
    assert_eq!(parameters, 743_377_019_904);
}
