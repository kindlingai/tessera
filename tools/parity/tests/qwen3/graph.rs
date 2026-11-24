//! The qwen3-graph recipe: a step graph from a wasm module.

use std::path::Path;
use tessera_bundle::Bundle;
use tessera_engine::wasm::Module;
use tessera_engine::{Engine, Event, Pick, Recipe, Request, Settings, Weight};
use tessera_graph::{Graph, PathSeg};
use tessera_interp::Tensor;
use tessera_parity::launches;

/// `small::qwen3()`'s sizes in `config.json` keys.
pub const SMALL: &str = r#"{"vocab_size": 64, "hidden_size": 64, "num_hidden_layers": 2,
    "num_attention_heads": 2, "head_dim": 32, "intermediate_size": 128,
    "num_key_value_heads": 1, "rope_theta": 10000.0, "rms_norm_eps": 1e-6}"#;

/// The recipe's module, built to wasm.
pub fn module() -> Module {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/recipes/qwen3-graph");
    let cache = tessera_testkit::scratch("recipes");
    let recipe = tessera_recipe::Recipe::read(&dir).unwrap();
    Module::open(&recipe.module_bytes(Some(&cache)).unwrap().unwrap()).unwrap()
}

/// `want`'s `weights` in `got`'s order, matched by path.
fn placed(want: &Graph, got: &Graph, weights: &[Tensor]) -> Vec<Weight> {
    let path = |w: &tessera_graph::Weight| PathSeg::dotted(&w.path);
    let mut placed = Vec::new();
    for w in &got.weights {
        let at = want.weights.iter().position(|o| path(o) == path(w));
        let at = at.unwrap_or_else(|| panic!("{} is not in the in-tree graph", path(w)));
        placed.push(Weight::Floats(weights[at].clone()));
    }
    assert_eq!(placed.len(), want.weights.len());
    placed
}

/// Each launch's ids from `got` against `want`'s, on `settings`.
pub fn steps_agree(settings: &Settings, [want, got]: [&Graph; 2], weights: &[Tensor]) {
    let open = |g: &Graph, w: Vec<Weight>| {
        let mut engine = Engine::open(g.clone(), 16, &Recipe::default(), settings).unwrap();
        engine.load(&w).unwrap();
        engine
    };
    let held = weights.iter().cloned().map(Weight::Floats).collect();
    let mut want_engine = open(want, held);
    let mut got_engine = open(got, placed(want, got, weights));
    for rows in launches(10, 4) {
        let ids = got_engine.step(&rows).unwrap();
        assert_eq!(ids, want_engine.step(&rows).unwrap());
    }
}

/// Greedy ids after `prompt` from `bundle`'s graph on the CPU.
pub fn answer(bundle: Bundle, prompt: &str) -> Vec<u32> {
    let settings = Settings {
        backend: Pick::Cpu,
        ..Settings::default()
    };
    let recipe = Recipe {
        pins: bundle.pins.unwrap_or_default(),
        ..Recipe::default()
    };
    let mut engine = Engine::open(bundle.graph, 64, &recipe, &settings).unwrap();
    let load = |at, w: &Weight| engine.load_weight(at, w);
    bundle.weights.each(load).unwrap();
    let request = Request {
        tokens: bundle.tokenizer.encode(prompt).unwrap(),
        max_tokens: 16,
        ..Request::default()
    };
    let mut ids = Vec::new();
    let mut emit = |e| {
        if let Event::Token(id) = e {
            ids.push(id);
        }
    };
    engine.generate(request, &mut emit).unwrap();
    println!("{prompt}{}", bundle.tokenizer.decode(&ids));
    ids
}
