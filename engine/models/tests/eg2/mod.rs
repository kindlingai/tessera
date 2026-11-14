//! EmbeddingGemma 2.

mod golden_test;
mod rotation_test;

use crate::common::{Spec, small_stack, suite};
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
};

suite!(SPEC);
