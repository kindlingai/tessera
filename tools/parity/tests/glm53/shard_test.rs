//! The GLM step graph over ranks matches one rank.

use sensai_graph::{Parts, PathSeg, Placement};
use sensai_models::mla::{RotatedMoe, stepper};
use sensai_parity::step_ranks;

/// Expert sets split. Attention and the indexer stay whole.
#[test]
fn step_over_ranks() {
    let g = stepper::<RotatedMoe>(&sensai_models::small::glm53());
    for ranks in [2, 4] {
        let sharded = step_ranks(&g, &Parts::even(ranks));
        let weights = &sharded.graph.weights;
        let mut whole = Vec::new();
        for w in &sharded.whole {
            let Some(PathSeg::Field(name)) = weights[w.weight as usize].path.last() else {
                panic!("a whole weight without a name");
            };
            whole.push(name.as_str());
        }
        whole.sort();
        whole.dedup();
        assert_eq!(whole, ["kv_a", "o", "q_b", "weights", "wk", "wq"]);
        let sets = weights.iter().filter(|w| w.ty.rank() == 3);
        let split = Placement::Split { axis: 0 };
        assert_eq!(sets.filter(|w| w.ty.placement == split).count(), 6);
        let kept = &sharded.graph.kept;
        assert!(kept.iter().all(|k| k.placement == Placement::Replicated));
    }
}
