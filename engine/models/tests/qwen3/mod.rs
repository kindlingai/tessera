//! Qwen3-0.6B.

mod golden_test;
mod step_test;

use crate::common::{Golden, Spec, Tolerance, small_stack, suite};
use lower::form::BF16_FORM;
use models::qwen3::{Config, program};
use trace::Composite::{Attention, Mlp};
use trace::Graph;

/// Two small layers, wide enough for every block form.
fn small() -> Graph {
    let c = Config {
        stack: small_stack(64, 32, 128),
        kv_heads: 1,
        rope_theta: 1e4,
        eps: 1e-6,
    };
    program(&c)
}

fn release() -> Graph {
    program(&Config::small(8, 40_960))
}

/// Plans keep the reference checkpoint in bf16.
pub const GOLDEN: Golden = Golden {
    read: golden_test::read,
    form: BF16_FORM,
    tolerance: Tolerance {
        max_abs: 1e-3,
        cos: 0.99999,
        top: true,
    },
    per_token: true,
    unit: false,
    anywhere: false,
};

const SPEC: Spec = Spec {
    name: "qwen3",
    release,
    parameters: 596_049_920,
    regions: &[(Attention, 28), (Mlp, 28)],
    small: Some(small),
    files: &[
        (include_str!("manifests/hf.json"), release),
        (include_str!("manifests/gguf.json"), release),
    ],
    golden: Some(GOLDEN),
};

suite!(SPEC);
