//! Region details the spec does not state.

use super::release;

#[test]
fn mlp_numerics() {
    let g = release();
    let mlp = &g.regions[1];
    let ops = &g.nodes[mlp.start.0 as usize..mlp.end.0 as usize];
    assert_eq!(ops.iter().filter(|n| n.numerics.is_some()).count(), 2);
}
