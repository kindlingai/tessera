//! The resident-decoder recipe: one grid per step, as the graph.

use std::path::Path;
use tessera_device::whole;
use tessera_engine::wasm::Module;
use tessera_engine::{Backend, Engine};
use tessera_graph::Graph;
use tessera_lower::{Rule, Settings, Weight, kept};
use tessera_models::decoder::{Swiglu, stepper};
use tessera_parity::{launches, rank_plan, taken};
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

/// The small model, with room for every launch's positions.
fn graph() -> Graph {
    let mut c = tessera_models::decoder::Config::qwen3_small();
    c.stack.max_context = 64;
    stepper::<Norm, Swiglu>(&c)
}

#[test]
fn resident_plans() {
    for (ranks, rank) in [(1, 0), (2, 1)] {
        assert_eq!(
            taken(&rank_plan(&graph(), &choices(kept::DENSE), ranks, rank)),
            (vec![NAME], 0, false)
        );
    }
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
    for rows in launches(20, 4) {
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
