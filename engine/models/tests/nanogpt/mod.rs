//! A small nanogpt.

mod golden_test;
mod regions_test;

use crate::common::{Spec, suite};
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
};

suite!(SPEC);
