//! Gemma 4 E4B, its text decoder.

mod config_test;

use tessera_models::Stack;
use tessera_models::gemma4::{Config, program, stepper};
use tessera_parity::{DRIFT, Spec, Stepper, suite};
use tessera_trace::Composite::{Attention, Mlp, Model};
use tessera_trace::{Graph, Linear};

/// The released sizes.
fn sizes(max_seqs: u64, max_context: u64) -> Config {
    Config {
        stack: Stack::sized([262_144, 2560, 42, 8, 256, 10_240], max_seqs, max_context),
        kv_heads: 2,
        full_every: 6,
        full_head_dim: 512,
        full_turned: 64,
        per_layer: 256,
        window: 512,
        theta: 1e4,
        full_theta: 1e6,
        first_shared: 24,
        eps: 1e-6,
    }
}

fn small() -> Graph {
    program::<Linear>(&tessera_models::small::gemma4())
}

fn release() -> Graph {
    program::<Linear>(&sizes(1, 1024))
}

const STEPPER: Stepper = Stepper {
    small: || stepper::<Linear>(&tessera_models::small::gemma4()),
    release: || stepper::<Linear>(&sizes(1, 1024)),
    drift: DRIFT,
};

const SPEC: Spec = Spec {
    name: "gemma4",
    release,
    parameters: 7_463_013_418,
    regions: &[(Attention, 42), (Mlp, 42), (Model, 1)],
    small: Some(small),
    stepper: Some(STEPPER),
    files: &[(include_str!("manifests/hf.json"), release)],
    golden: None,
};

suite!(SPEC);
