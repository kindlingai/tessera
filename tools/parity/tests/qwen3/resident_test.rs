//! The resident-decoder recipe: one grid per step, as the graph.

use std::path::Path;
use tessera_device::whole;
use tessera_engine::wasm::Module;
use tessera_engine::{Backend, Engine, Row};
use tessera_graph::DimName::{Context, Picks, Rows, Segments, Seqs};
use tessera_graph::{Graph, Parts, shard, trim};
use tessera_lower::{Cluster, Device, Limits, Plan, Platform, Rule, Settings, Weight, kept, lower};
use tessera_models::decoder::{Swiglu, stepper};
use tessera_recipe::Recipe;
use tessera_run::testing::runs;
use tessera_testkit::{Noise, weights};
use tessera_trace::Norm;

const NAME: &str = "resident_decoder";

fn recipe() -> Recipe {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/recipes/resident-decoder");
    Recipe::read(&dir).unwrap()
}

/// The recipe's rule, from its module built to wasm.
fn rule() -> &'static Rule {
    let cache = tessera_testkit::scratch("recipes");
    let bytes = recipe().module_bytes(Some(&cache)).unwrap().unwrap();
    Module::open(&bytes).unwrap().rules[0]
}

/// The lowering choices, with kept rows in `layout`.
fn choices(layout: &'static dyn kept::Layout) -> tessera_lower::Recipe {
    let mut lower = recipe().lower.unwrap();
    lower.rules.push(rule());
    lower.kept = layout;
    lower
}

fn graph() -> Graph {
    stepper::<Norm, Swiglu>(&tessera_models::small::qwen3())
}

/// Rank `rank` of `ranks`' plan for two rows.
fn plan(ranks: u32, rank: u32) -> Plan {
    let mut g = trim(&graph(), 0).unwrap();
    // A device with a compiler, so the kernel text fits.
    let mut device = Device::default().with(&choices(kept::DENSE));
    device.platform = Platform::assumed(Limits::assumed());
    if ranks > 1 {
        g = shard(&g, &Parts::even(ranks as usize)).unwrap().graph;
        device = device.zoned(Cluster::two("a", "b"));
        device = device.ranked((0..ranks).collect()).as_rank(rank);
    }
    let sizes = [
        (Seqs, 4),
        (Rows, 2),
        (Context, 16),
        (Segments, 1),
        (Picks, 2),
    ];
    lower(&g, 0, &sizes, &device).unwrap()
}

/// Resident rules, host sums, and any zone past 0.
fn resident(plan: &Plan) -> (Vec<&'static str>, usize, bool) {
    let resident = plan.steps.iter().filter(|s| s.kernel.resident().is_some());
    let sums = plan
        .steps
        .iter()
        .filter(|s| s.kernel.name() == "all_reduce");
    (
        resident.map(|s| s.rule).collect(),
        sums.count(),
        plan.zoned(),
    )
}

#[test]
fn resident_plans() {
    for (ranks, rank) in [(1, 0), (2, 1)] {
        assert_eq!(resident(&plan(ranks, rank)), (vec![NAME], 0, false));
    }
}

fn row(slot: u32, token: u32, position: u32) -> Row {
    Row {
        slot,
        token,
        position,
    }
}

/// A prompt of 20 rows, then steps over two slots.
fn launches() -> Vec<Vec<Row>> {
    let prompt = (0..20).map(|p| row(0, (p * 7 + 3) % 64, p)).collect();
    let mut all = vec![prompt, vec![row(1, 5, 0), row(1, 9, 1)]];
    for p in 0..4 {
        all.push(vec![row(0, p + 11, 20 + p), row(1, p + 2, 2 + p)]);
    }
    all
}

/// Each launch's ids from the grid, against the graph's.
fn compare(settings: &Settings, layout: &'static dyn kept::Layout, stored: &[Weight]) {
    let open = |lower: &tessera_lower::Recipe| {
        let mut engine = Engine::open(graph(), 64, lower, settings).unwrap();
        engine.load(stored).unwrap();
        engine
    };
    let plain = tessera_lower::Recipe {
        kept: layout,
        ..tessera_lower::Recipe::default()
    };
    let (mut want, mut got) = (open(&plain), open(&choices(layout)));
    let file = tessera_testkit::scratch("resident.grid");
    got.persist(whole::alone(settings.gpu(), &file).unwrap())
        .unwrap();
    for rows in launches() {
        let ids = got.step(&rows).unwrap();
        assert_eq!(
            ids,
            want.step(&rows).unwrap(),
            "{} at {rows:?}",
            layout.name()
        );
    }
    // The plan has no other way to run the step.
    assert_eq!(got.kernels().get(NAME), Some(&1));
}

#[test]
fn resident_matches_graph() {
    let stored = weights(&graph(), &mut Noise(3)).into_iter();
    let stored: Vec<Weight> = stored.map(Weight::Floats).collect();
    // Only CUDA and ROCm run persistent grids.
    let grids = |s: &&Settings| matches!(s.gpu(), Some(Backend::Cuda | Backend::Rocm));
    for settings in runs().iter().filter(grids) {
        compare(settings, kept::DENSE, &stored);
        compare(settings, &kept::PAGED16, &stored);
    }
}
