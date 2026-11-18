//! Rules that cover a whole traced region.

use sensai_graph::DimName::{self, Context, Hidden, Rows, Seqs};
use sensai_graph::{Act, Composite, DType, Dim, Graph, NodeId, TensorType};
use sensai_lower::kernels::Map;
use sensai_lower::{Device, Draft, Kernel, Plan, Propose, Read, Rule, T, Tree, lower, sites};
use sensai_run::testing::host;
use sensai_testkit::graphs;
use sensai_trace::{EntryKind, Model, Port, Role};

/// A launch whose cost its rule sets.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "block",
    sizes = [],
    source = Map::plain(Vec::new()).source(),
    launches = Vec::new(),
    work = self.len,
    guess = self.guess,
)]
struct Block {
    len: usize,
    guess: f64,
}

/// Two launches over a region: the first feeds the second.
struct Whole(f64);

impl Propose for Whole {
    fn at(&self, tree: &Tree, at: T) -> Option<Vec<Draft>> {
        let r: sites::Region = tree.site(at)?;
        let (len, guess) = (r.len, self.0);
        let first = Draft::new(Block { len, guess }, r.reads(), len);
        let reads = [Read::Draft(0)].into_iter().chain(r.reads());
        Some(vec![first, Draft::new(Block { len, guess }, reads, len)])
    }

    fn regions(&self) -> bool {
        true
    }
}

const SIZES: [(DimName, u64); 1] = [(Rows, 3)];
const STEP: [(DimName, u64); 3] = [(Seqs, 1), (Rows, 2), (Context, 5)];
const BOUNDED: [Dim; 3] = [
    Dim::bounded(Seqs, 4),
    Dim::bounded(Rows, 16),
    Dim::bounded(Context, 16),
];

/// A region reads `a = x + y` twice. Then an add.
fn graph() -> Graph {
    let m = Model::new(vec![Dim::bounded(Rows, 8), Dim::fixed(Hidden, 16)]);
    let ty = TensorType::new([Rows, Hidden], DType::F32);
    let role = Role::Logits;
    let port = || Port {
        role,
        ty: ty.clone(),
    };
    let [x, y] = m.entry(EntryKind::Forward, [port(), port()]);
    let [inner] = m.composite(Composite::Mlp, || {
        let a = x.add(&y);
        [a.act(Act::Silu).mul(&a)]
    });
    m.finish([(Role::Logits, &inner.add(&x))])
}

fn plan(g: &Graph, sizes: &[(DimName, u64)], guess: f64) -> Plan {
    let rule = Rule::made("whole", module_path!(), Whole(guess));
    lower(g, 0, sizes, &Device::guess(&host()).adding(rule)).unwrap()
}

fn rules(plan: &Plan) -> Vec<&'static str> {
    plan.steps.iter().map(|s| s.rule).collect()
}

#[test]
fn region_rules_cover() {
    let g = graph();
    let region = &g.regions[0];
    let within = |n: &NodeId| region.start <= *n && *n < region.end;
    let cheap = plan(&g, &SIZES, 1e-15);
    assert_eq!(rules(&cheap), ["whole", "whole", "map"]);
    // The second launch reads the first. No other enters.
    assert_eq!(cheap.steps[1].reads[0], cheap.steps[0].write);
    assert!(cheap.steps[2].covers.iter().all(|n| !within(n)));
    let dear = plan(&g, &SIZES, 1.0);
    assert!(dear.steps.iter().all(|s| s.rule != "whole"));
}

#[test]
fn regions_write_kept() {
    let plan = plan(&graphs::appended(&BOUNDED, 16), &STEP, 1e-15);
    // The region writes K and V. No append runs.
    assert_eq!(rules(&plan), ["whole", "whole"]);
    assert_eq!(plan.kept.len(), 2);
    for (_, slot) in &plan.kept {
        assert!(plan.steps[0].reads.contains(slot));
    }
}
