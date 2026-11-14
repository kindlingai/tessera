//! A small nanogpt.

mod golden;
mod regions_test;

use crate::common::{Golden, Spec, Tolerance, suite};
use lower::form::F32_FORM;
use models::Stack;
use models::nanogpt::{Config, program};
use trace::Composite::{Attention, Mlp};
use trace::Graph;

pub fn config() -> Config {
    let stack = Stack {
        vocab: 8,
        hidden: 8,
        layers: 2,
        heads: 2,
        head_dim: 4,
        ffn: 32,
        max_seqs: 4,
        max_context: 8,
    };
    Config { stack, eps: 1e-5 }
}

pub fn release() -> Graph {
    program(&config())
}

const SPEC: Spec = Spec {
    name: "nanogpt",
    release,
    parameters: 1768,
    regions: &[(Attention, 2), (Mlp, 2)],
    small: None,
    files: &[],
    golden: Some(Golden {
        read: golden::read,
        form: F32_FORM,
        tolerance: Tolerance {
            max_abs: 1e-5,
            cos: 0.0,
            top: false,
        },
        per_token: true,
        unit: false,
        anywhere: true,
    }),
};

suite!(SPEC);
