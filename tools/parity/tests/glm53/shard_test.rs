//! The GLM step graph over ranks matches one rank.

use tessera_graph::{Parts, PathSeg, Placement};
use tessera_models::mla::{RotatedMoe, stepper};
use tessera_parity::step_ranks;

/// Two heads split over two ranks, not four.
#[test]
fn step_over_ranks() {
    let g = stepper::<RotatedMoe>(&tessera_models::small::glm53());
    for (ranks, heads) in [(2, true), (4, false)] {
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
        let want: &[&str] = if heads { &[] } else { &["kv_b", "o", "q_b"] };
        assert_eq!(whole, want);
        for w in weights {
            let Some(PathSeg::Field(name)) = w.path.last() else {
                continue;
            };
            let want = match name.as_str() {
                "q_b" | "kv_b" if heads => Placement::Split { axis: 0 },
                "o" if heads => Placement::Split { axis: 1 },
                "head" => Placement::Split { axis: 0 },
                "q_b" | "kv_b" | "o" | "kv_a" | "weights" | "wk" | "wq" => Placement::Replicated,
                _ => continue,
            };
            assert_eq!(w.ty.placement, want, "{name}");
        }
        let sets = weights.iter().filter(|w| w.ty.rank() == 3);
        let split = Placement::Split { axis: 0 };
        assert_eq!(sets.filter(|w| w.ty.placement == split).count(), 6);
        let kept = &sharded.graph.kept;
        assert!(kept.iter().all(|k| k.placement == Placement::Replicated));
    }
}
