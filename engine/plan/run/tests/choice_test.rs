//! How selection picks among rules: cost, favor and limits.

use tessera_graph::DimName::{self, Hidden, Rows, Seqs};
use tessera_graph::{Act, DType, Dim, Graph, TensorType};
use tessera_lower::kernels::Map;
use tessera_lower::sites;
use tessera_lower::{
    Device, Draft, Kernel, Launch, Limits, Needs, Plan, Platform, Propose, Rule, Shape, Step, T,
    Tree, explain, lower, select,
};
use tessera_run::testing::host;
use tessera_trace::{EntryKind, Model, Port, Role};

/// A map stand-in with its own cost.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "cheap",
    sizes = [],
    source = Map::plain(Vec::new()).source(),
    launches = Vec::new(),
    work = self.len,
    guess = 1e-9,
)]
struct Cheap {
    len: usize,
}

#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "dear",
    sizes = [],
    source = Map::plain(Vec::new()).source(),
    launches = Vec::new(),
    work = self.len,
    guess = 1e-8,
)]
struct Dear {
    len: usize,
}

/// The cheapest, but only for 32 lanes.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "lanes",
    sizes = [],
    source = Map::plain(Vec::new()).source(),
    launches = Vec::new(),
    work = self.len,
    needs = Needs::lanes(),
    guess = 1e-12,
)]
struct Lanes {
    len: usize,
}

fn cheap(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let sites::Chain { reads, len, .. } = tree.site(at)?;
    Some(vec![Draft::new(Cheap { len }, reads, len)])
}

fn dear(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let sites::Chain { reads, len, .. } = tree.site(at)?;
    Some(vec![Draft::new(Dear { len }, reads, len)])
}

fn lanes(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let sites::Chain { reads, len, .. } = tree.site(at)?;
    Some(vec![Draft::new(Lanes { len }, reads, len)])
}

/// A map stand-in whose cost its rule sets.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "priced",
    sizes = [],
    source = Map::plain(Vec::new()).source(),
    launches = Vec::new(),
    work = self.len,
    guess = self.guess,
)]
struct Priced {
    len: usize,
    guess: f64,
}

/// The cheapest, with a grid that grows with its length.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "tall",
    sizes = [],
    source = Map::plain(Vec::new()).source(),
    launches = tall_grid,
    work = self.len,
    guess = 1e-12,
)]
struct Tall {
    len: usize,
}

fn tall_grid(k: &Tall, step: &Step) -> Vec<Launch> {
    let (across, down) = (1, k.len * 400);
    let shape = Shape::Grid {
        across,
        down,
        threads: 256,
    };
    vec![Launch::plain(step, &[], shape)]
}

/// A fast rule for `Tall`.
struct Towers;

impl Propose for Towers {
    fn at(&self, tree: &Tree, at: T) -> Option<Vec<Draft>> {
        let sites::Chain { reads, len, .. } = tree.site(at)?;
        Some(vec![Draft::new(Tall { len }, reads, len)])
    }

    fn fast(&self) -> bool {
        true
    }
}

/// A rule made at run time, its price as data.
struct Price(f64);

impl Propose for Price {
    fn at(&self, tree: &Tree, at: T) -> Option<Vec<Draft>> {
        let sites::Chain { reads, len, .. } = tree.site(at)?;
        let guess = self.0;
        Some(vec![Draft::new(Priced { len, guess }, reads, len)])
    }
}

tessera_lower::rule!("cheap", cheap);
tessera_lower::rule!("dear", dear);
tessera_lower::rule!("lanes", lanes);

const SIZES: [(DimName, u64); 2] = [(Seqs, 2), (Rows, 3)];

/// One chain over rows.
fn chain() -> Graph {
    let m = Model::new(vec![
        Dim::bounded(Seqs, 4),
        Dim::bounded(Rows, 8),
        Dim::fixed(Hidden, 16),
    ]);
    let ty = TensorType::new([Seqs, Rows, Hidden], DType::F32);
    let role = Role::Logits;
    let [x, y] = m.entry(
        EntryKind::Forward,
        [
            Port {
                role,
                ty: ty.clone(),
            },
            Port { role, ty },
        ],
    );
    m.finish([(Role::Logits, &x.add(&y).act(Act::Silu))])
}

fn plan(device: &Device) -> Plan {
    lower(&chain(), 0, &SIZES, device).unwrap()
}

fn names(plan: &Plan) -> Vec<&'static str> {
    plan.steps.iter().map(|s| s.kernel.name()).collect()
}

/// A device where only `cheap` and `dear` apply.
fn pair() -> Device {
    Device::guess(&host()).without("map").without("lanes")
}

#[test]
fn rules_compete() {
    let picked = select(&chain(), 0, &SIZES, &pair()).unwrap();
    assert!(picked.contested.contains("cheap") && picked.contested.contains("dear"));
}

#[test]
fn cost_picks_rule() {
    assert_eq!(names(&plan(&pair())), ["cheap"]);
    // Samples outrank the guesses.
    let (slow, quick) = (plan(&pair()), plan(&pair().favoring("dear")));
    let mut measured = pair();
    measured.fit([(&slow.steps[0], 1e-3), (&quick.steps[0], 1e-6)]);
    let fitted = plan(&measured);
    assert_eq!(names(&fitted), ["dear"]);
    assert!(measured.cost(&fitted).seconds < measured.cost(&slow).seconds);
    // A favored rule wins whatever it costs.
    assert_eq!(names(&plan(&measured.favoring("cheap"))), ["cheap"]);
}

#[test]
fn limits_bar_kernels() {
    let device = Device::guess(&host()).without("map");
    assert_eq!(names(&plan(&device)), ["lanes"]);
    // A 64-lane device runs none of the 32-lane kernels.
    let wide = Limits {
        lanes: 64,
        ..Limits::assumed()
    };
    let wide = device.on(Platform::assumed(wide));
    assert_eq!(names(&plan(&wide)), ["cheap"]);
}

#[test]
fn made_rules_compete() {
    let made = |guess| Rule::made("priced", module_path!(), Price(guess));
    assert_eq!(names(&plan(&pair().adding(made(1e-12)))), ["priced"]);
    assert_eq!(names(&plan(&pair().adding(made(1e-6)))), ["cheap"]);
    let recipe = tessera_lower::Recipe {
        rules: vec![made(1e-6)],
        favor: Some("priced".into()),
        require_favored: true,
        ..Default::default()
    };
    assert_eq!(names(&plan(&pair().with(&recipe))), ["priced"]);
    // Requiring an unused favored rule fails lowering.
    let missing = pair().favoring("unused_rule").requiring_favored(true);
    let err = lower(&chain(), 0, &SIZES, &missing).unwrap_err();
    assert!(matches!(err, tessera_lower::Error::FavoredUnused(r) if r == "unused_rule"));
}

#[test]
fn grids_bar_kernels() {
    let cuda = Platform {
        backend: "cuda".into(),
        ..Platform::host(Limits::assumed())
    };
    let device = Device::default().without("map").without("lanes").on(cuda);
    let device = device.adding(Rule::made("tall", module_path!(), Towers));
    assert_eq!(names(&plan(&device)), ["tall"]);
    // 512 elements need a grid 204800 high.
    let big = [(Seqs, 4), (Rows, 8)];
    assert_eq!(
        names(&lower(&chain(), 0, &big, &device).unwrap()),
        ["cheap"]
    );
    let report = explain(&chain(), 0, &big, &device).unwrap().format(10);
    assert!(
        report.contains("grid height 204800 exceeds 65535"),
        "{report}"
    );
}
