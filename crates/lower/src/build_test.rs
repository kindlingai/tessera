use crate::{Error, Kernel, MapStage, Plan, lower};
use graph::DimName::{Context, Seqs};
use graph::{Act, Composite, Graph};
use models::qwen3::{Config, program};

fn toy() -> (Graph, Plan) {
    let g = program(&Config::toy(4, 16));
    let plan = lower(&g, 0, &[(Seqs, 2), (Context, 5)]).unwrap();
    (g, plan)
}

#[test]
fn gate_fuses() {
    let (g, plan) = toy();
    let fused = Kernel::Map(vec![MapStage::Act(Act::Silu), MapStage::Mul]);
    let steps: Vec<_> = plan.steps.iter().filter(|s| s.kernel == fused).collect();
    assert_eq!(steps.len(), 2, "one per layer");
    for s in steps {
        assert_eq!((s.covers.len(), s.reads.len()), (2, 2));
        let holds = |r: &&graph::Region| s.covers.iter().all(|c| r.start <= *c && *c < r.end);
        let region = g.regions.iter().find(holds).unwrap();
        assert_eq!(region.kind, Composite::Mlp);
    }
}

#[test]
fn buffers_are_shared() {
    let (g, plan) = toy();
    for s in &plan.steps {
        let in_place = s.reads.iter().any(|r| r.buffer == s.write.buffer);
        assert_eq!(in_place, matches!(s.kernel, Kernel::Rope { .. }));
    }
    let values = plan.weights.len() + plan.inputs.len() + plan.steps.len();
    assert!(
        plan.buffers.len() < values,
        "dead values free their buffers"
    );
    let angles = |s: &&crate::Step| matches!(s.kernel, Kernel::Angles { .. });
    assert_eq!(plan.steps.iter().filter(angles).count(), 1);
    let unsized_context = lower(&g, 0, &[(Seqs, 2)]);
    assert!(matches!(unsized_context, Err(Error::Size(_))));
}
