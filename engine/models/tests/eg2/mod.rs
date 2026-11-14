//! EmbeddingGemma 2.

mod golden_test;
mod rotation_test;
mod rows_test;

use crate::common::{Check, Compare, Golden, Source, Spec, Tolerance, small_stack, suite};
use lower::form::F32_FORM;
use models::Rotated;
use models::eg2::{Config, probe, program};
use trace::Composite::{Attention, Mlp};
use trace::{Graph, Linear};

/// Two small layers, one sliding and one full.
fn small() -> Graph {
    let c = Config {
        stack: small_stack(256, 128, 256),
        kv_heads: 1,
        full_every: 2,
        full_head_dim: 256,
        per_layer: 128,
        embedding: 128,
        window: 4,
        theta: 1e4,
        full_theta: 1e6,
        eps: 1e-6,
    };
    program::<Linear>(&c)
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
    source: Source::Files(|s| s.eg2.clone()),
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
