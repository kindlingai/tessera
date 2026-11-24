//! EmbeddingGemma 2.

mod batched_test;
mod config_test;
mod golden_test;
mod rotation_test;
mod rows_test;

use tessera_lower::form::F32_FORM;
use tessera_models::Rotated;
use tessera_models::eg2::{Config, probe, program};
use tessera_parity::{Check, Compare, Golden, Spec, Tolerance, suite};
use tessera_trace::Composite::{Attention, Mlp};
use tessera_trace::{Graph, Linear};

/// Two small layers, one sliding and one full.
fn small() -> Graph {
    program::<Linear>(&tessera_models::eg2::small())
}

fn release() -> Graph {
    program::<Linear>(&Config::release(1, 1024))
}

/// The residual stream after `layers`, its JSON read back.
fn probed(layers: u64) -> Graph {
    let mut c = Config::release(1, 1024);
    c.stack.layers = layers;
    Graph::from_json(&probe::<Linear>(&c).to_json()).unwrap()
}

/// Unit vectors and layer means, so no max_abs bound.
const CLOSE: Tolerance = Tolerance {
    max_abs: f32::INFINITY,
    cos: 0.9999,
    top: false,
};

pub const GOLDEN: Golden = Golden {
    record: None,
    form: F32_FORM,
    checks: &[
        Check {
            name: "embedding",
            compare: Compare::Unit,
            tolerance: CLOSE,
        },
        Check {
            name: "layer_means",
            compare: Compare::Layers(probed, &[1, 2, 5, 6, 12, 24]),
            tolerance: CLOSE,
        },
    ],
    anywhere: false,
};

const SPEC: Spec = Spec {
    name: "eg2",
    release,
    parameters: 271_002_648,
    regions: &[(Attention, 24), (Mlp, 24)],
    stepper: None,
    small: Some(small),
    files: &[
        (include_str!("manifests/hf.json"), release),
        (include_str!("manifests/exlr8.json"), || {
            program::<Rotated>(&Config::release(1, 1024))
        }),
    ],
    golden: Some(GOLDEN),
};

suite!(SPEC);
