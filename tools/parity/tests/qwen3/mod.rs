//! Qwen3-0.6B.

mod config_test;
mod resident_test;
mod shard_test;

use tessera_models::decoder::{Config, Swiglu, program, stepper};
use tessera_parity::{LOGITS, Spec, Stepper, decoder, suite};
use tessera_trace::Composite::{Attention, Layer, Mlp, Model};
use tessera_trace::{Graph, Norm};

decoder!(Norm, Swiglu, tessera_models::small::qwen3(), Config::qwen3);

const SPEC: Spec = Spec {
    name: "qwen3",
    release,
    parameters: 596_049_920,
    regions: &[(Attention, 28), (Mlp, 28), (Layer, 28), (Model, 1)],
    small: Some(small),
    stepper: Some(STEPPER),
    files: &[
        (include_str!("manifests/hf.json"), release),
        (include_str!("manifests/gguf.json"), release),
    ],
    golden: Some(LOGITS),
};

suite!(SPEC);
