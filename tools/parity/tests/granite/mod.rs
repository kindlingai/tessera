//! Granite 3.1 1B-A400M: a Llama decoder with routed experts.

mod shard_test;

use tessera_models::decoder::{Config, Moe, program, stepper};
use tessera_parity::{LOGITS, Spec, Stepper, decoder, suite};
use tessera_trace::Composite::{Attention, Layer, Model, Moe as Routed};
use tessera_trace::Graph;

decoder!((), Moe, Config::granite_small(), Config::granite);

const SPEC: Spec = Spec {
    name: "granite",
    release,
    parameters: 1_334_628_352,
    regions: &[(Attention, 24), (Routed, 24), (Layer, 24), (Model, 1)],
    small: Some(small),
    stepper: Some(STEPPER),
    files: &[(include_str!("manifests/hf.json"), release)],
    golden: Some(LOGITS),
};

suite!(SPEC);
