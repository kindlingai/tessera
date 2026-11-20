//! Qwen3-0.6B.

mod shard_test;

use sensai_models::decoder::{Config, Swiglu, program, stepper};
use sensai_parity::{LOGITS, Spec, Stepper, decoder, suite};
use sensai_trace::Composite::{Attention, Layer, Mlp, Model};
use sensai_trace::{Graph, Norm};

decoder!(Norm, Swiglu, sensai_models::small::qwen3(), Config::qwen3);

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
