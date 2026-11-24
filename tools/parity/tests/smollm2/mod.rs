//! SmolLM2-360M: a Llama decoder with tied embeddings.

use tessera_models::decoder::{Config, Swiglu, program, stepper};
use tessera_parity::{LOGITS, Spec, Stepper, decoder, suite};
use tessera_trace::Composite::{Attention, Layer, Mlp, Model};
use tessera_trace::Graph;

decoder!((), Swiglu, Config::smollm2_small(), Config::smollm2);

const SPEC: Spec = Spec {
    name: "smollm2",
    release,
    parameters: 361_821_120,
    regions: &[(Attention, 32), (Mlp, 32), (Layer, 32), (Model, 1)],
    small: Some(small),
    stepper: Some(STEPPER),
    files: &[(include_str!("manifests/hf.json"), release)],
    golden: Some(LOGITS),
};

suite!(SPEC);
