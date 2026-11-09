use crate::{Device, Error, Kernel, MapStage, Plan, assign, lower, peak, schedule, select};
use graph::DimName::{self, Context, Seqs};
use graph::{Act, Composite, Graph};
use models::qwen3::{Config, program};

const SIZES: [(DimName, u64); 2] = [(Seqs, 2), (Context, 5)];

fn toy(device: &Device) -> (Graph, Plan) {
    let g = program(&Config::toy(4, 16));
    let plan = lower(&g, 0, &SIZES, device).unwrap();
    (g, plan)
}

fn count(plan: &Plan, kernel: &str) -> usize {
    let named = |s: &&crate::Step| s.kernel.name() == kernel;
    plan.steps.iter().filter(named).count()
}

#[test]
fn gate_fuses() {
    let (g, plan) = toy(&Device::guess());
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
    let (g, plan) = toy(&Device::guess());
    for s in &plan.steps {
        let in_place = s.reads.iter().any(|r| r.buffer == s.write.buffer);
        assert_eq!(in_place, s.kernel.overwrites());
    }
    let values = plan.weights.len() + plan.inputs.len() + plan.steps.len();
    assert!(
        plan.buffers.len() < values,
        "dead values free their buffers"
    );
    assert_eq!(count(&plan, "angles"), 1);
    let unsized_context = lower(&g, 0, &[(Seqs, 2)], &Device::guess());
    assert!(matches!(unsized_context, Err(Error::Size(_))));
}

#[test]
fn cost_picks_the_matmul() {
    // Ten rows fit the thin kernel, so cost alone decides.
    let (_, tiled) = toy(&Device::guess().without("thin"));
    let (_, thin) = toy(&Device::guess().without("matmul"));
    assert_eq!((count(&tiled, "thin"), count(&thin, "matmul")), (0, 0));
    assert_eq!(count(&tiled, "matmul"), count(&thin, "thin"));
    let mut measured = Device::guess();
    let cheap = thin.steps.iter().map(|s| (s, 1e-6));
    measured.fit(cheap.chain(tiled.steps.iter().map(|s| (s, 1e-3))));
    let (_, fitted) = toy(&measured);
    assert_eq!(count(&fitted, "matmul"), 0, "the faster kernel wins");
    assert!(measured.cost(&fitted).seconds < measured.cost(&tiled).seconds);
}

#[test]
fn cost_picks_attention() {
    // Dense attention is unproven, so a guess streams.
    let (_, stream) = toy(&Device::guess());
    assert_eq!((count(&stream, "attend"), count(&stream, "scores")), (2, 0));
    let (_, dense) = toy(&Device::guess().without("attend"));
    assert_eq!((count(&dense, "attend"), count(&dense, "join")), (0, 0));
    let per_layer = ["scores", "causal", "values"].map(|k| count(&dense, k));
    assert_eq!(per_layer, [2, 2, 2]);
    let softmax = dense
        .steps
        .iter()
        .find(|s| s.kernel.name() == "causal")
        .unwrap();
    assert_eq!(softmax.write.buffer, softmax.reads[0].buffer, "in place");
}

#[test]
fn order_saves_memory() {
    let g = program(&Config::toy(4, 16));
    let picked = select(&g, 0, &SIZES, &Device::guess()).unwrap();
    let traced: Vec<usize> = (0..picked.ops.len()).collect();
    let order = schedule(&picked);
    let mut sorted = order.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, traced, "every op runs once");
    assert!(peak(&picked, &order) <= peak(&picked, &traced));
    assert!(assign(&picked, &order).scratch() <= assign(&picked, &traced).scratch());
}

#[test]
fn dialects_agree() {
    use crate::{Dialect, Source, source};
    let fused = Kernel::Map(vec![MapStage::Act(Act::Silu), MapStage::Mul]);
    let Source::Made(text) = source(&fused) else {
        panic!("maps are generated");
    };
    let (metal, cuda) = (text.render(Dialect::Metal), text.render(Dialect::Cuda));
    // Buffers, then scalars, in one order for both.
    assert!(metal.contains("device const float *a0 [[buffer(1)]]"));
    assert!(metal.contains("constant u32 &len [[buffer(3)]]"));
    let order = "const float *x,\n    const float *a0,\n    float *out,\n    u32 len\n";
    assert!(cuda.contains(order), "{cuda}");
    assert!(cuda.contains(&format!("__global__ void {}(", text.entry)));
    for body in [&metal, &cuda] {
        assert!(body.contains("v = v * a0[gid];"));
    }
}
