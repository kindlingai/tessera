//! The glm53-whole recipe: one grid per step, as the graph.

use super::module_test::codes;
use std::path::Path;
use tessera_device::whole;
use tessera_engine::wasm::Module;
use tessera_engine::{Allowed, Backend, Engine, Steps};
use tessera_graph::Graph;
use tessera_lower::form::named;
use tessera_lower::{Pins, Rule, Settings, Weight, kept};
use tessera_models::mla::{RotatedMoe, stepper};
use tessera_parity::{launches, rank_plan, taken};
use tessera_recipe::Recipe;
use tessera_run::testing::runs;
use tessera_testkit::{Noise, weights};

const NAME: &str = "glm53_whole";

fn recipe() -> Recipe {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/recipes/glm53-whole");
    Recipe::read(&dir).unwrap()
}

/// The recipe's rule, from its module built to wasm.
fn rule() -> &'static Rule {
    let cache = tessera_testkit::scratch("recipes");
    let bytes = recipe().module_bytes(Some(&cache)).unwrap().unwrap();
    Module::open(&bytes).unwrap().rules[0]
}

/// Every form, two codebooks, forms by layer. Last pins win.
const PINS: &str = "*q_a=mxfp8 *kv_a=mxfp8 *q_b=int4_g32 *kv_b=bf16 *.o=int4_g32 \
    *.wq=bf16 *.wk=mxfp8 *.weights=bf16 *dense*.gate=int4_g32 *dense*.up=int4_g32 \
    *dense*.down=mxfp8 *router=bf16 *moe*.gate=trellis_k3_mul1 *moe*.up=trellis_k3_mul1 \
    *moe*.down=trellis_k2_mul1 *shared*=trellis_k4_rot moe.1.shared.down=trellis_k3_rot \
    attn.1.q_b=bf16 attn.2.o=f32 embed=bf16 head=bf16";

fn pins() -> Pins {
    let pairs = PINS.split_whitespace().filter_map(|p| p.split_once('='));
    pairs.fold(Pins::default(), |p, (at, form)| {
        p.pin(at, named(form).unwrap())
    })
}

/// The lowering choices, with kept rows in `layout`.
fn choices(layout: &'static dyn kept::Layout) -> tessera_lower::Recipe {
    let mut lower = recipe().lower.unwrap();
    lower.rules.push(rule());
    lower.kept = layout;
    lower.pins = pins();
    lower
}

/// The small model: experts split, masks give ranks.
fn graph() -> Graph {
    let mut c = tessera_models::small::glm53();
    c.stack.ffn = 256;
    c.stack.steps = Steps::Masked;
    stepper::<RotatedMoe>(&c)
}

/// Each weight to load. Codes stay packed.
fn stored() -> Vec<Weight> {
    let pins = pins();
    let mut draws = Noise(3);
    let g = graph();
    let floats = weights(&g, &mut draws);
    let mut out = Vec::new();
    for (w, t) in g.weights.iter().zip(floats) {
        let form = pins.form(&w.path);
        match form.name().starts_with("trellis") {
            true => out.push(Weight::Packed(form, codes(&t.shape, form, &mut draws))),
            false => out.push(Weight::Floats(t)),
        }
    }
    out
}

/// Each launch's ids from the grid, against the graph's.
fn compare(settings: &Settings, layout: &'static dyn kept::Layout, stored: &[Weight]) {
    let open = |lower: &tessera_lower::Recipe| {
        let mut engine = Engine::open(graph(), 16, lower, settings).unwrap();
        engine.load(stored).unwrap();
        engine
    };
    let plain = tessera_lower::Recipe {
        kept: layout,
        pins: pins(),
        ..tessera_lower::Recipe::default()
    };
    let (mut want, mut got) = (open(&plain), open(&choices(layout)));
    let file = tessera_testkit::scratch("glm53-whole.grid");
    got.persist(whole::alone(settings.gpu(), &file).unwrap())
        .unwrap();
    for rows in launches(12, 3) {
        let ids = got.step(&rows).unwrap();
        assert_eq!(
            ids,
            want.step(&rows).unwrap(),
            "{} at {rows:?}",
            layout.name()
        );
        // Each slot's best four: each pass masks the last.
        let mut seen = vec![Vec::new(); want.last(&rows).unwrap().len()];
        for _ in 0..4 {
            let allowed: Vec<Allowed> = seen.iter().cloned().map(Allowed::Except).collect();
            let next = want.last_guided(&rows, &allowed).unwrap();
            assert_eq!(got.last_guided(&rows, &allowed).unwrap(), next, "{rows:?}");
            seen.iter_mut().zip(next).for_each(|(s, id)| s.push(id));
        }
    }
    assert_eq!(got.kernels().get(NAME), Some(&1));
}

/// Host plans, then CUDA runs: the kernel is CUDA text.
#[test]
fn whole_matches_graph() {
    for (ranks, rank) in [(1, 0), (2, 1)] {
        assert_eq!(
            taken(&rank_plan(&graph(), &choices(kept::DENSE), ranks, rank)),
            (vec![NAME], 0, false),
            "{ranks} ranks"
        );
    }
    for settings in runs().iter().filter(|s| s.gpu() == Some(Backend::Cuda)) {
        [kept::DENSE, &kept::PAGED16]
            .into_iter()
            .for_each(|l| compare(settings, l, &stored()));
    }
}
