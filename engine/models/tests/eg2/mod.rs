//! EmbeddingGemma 2.

mod golden_test;
mod rotation_test;

use crate::common::{Golden, Spec, Tolerance, small_stack, suite};
use lower::form::F32_FORM;
use models::Rotated;
use models::eg2::{Config, program};
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

/// The embedding is a unit vector, so no max_abs bound.
pub const GOLDEN: Golden = Golden {
    read: golden_test::read,
    form: F32_FORM,
    tolerance: Tolerance {
        max_abs: f32::INFINITY,
        cos: 0.9999,
        top: false,
    },
    per_token: false,
    unit: true,
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
