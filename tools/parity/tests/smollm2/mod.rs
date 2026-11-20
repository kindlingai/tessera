//! SmolLM2-360M: a Llama decoder with tied embeddings.

use sensai_models::decoder::{Config, Swiglu, program, stepper};
use sensai_parity::{LOGITS, Spec, Stepper, decoder, suite};
use sensai_trace::Composite::{Attention, Layer, Mlp, Model};
use sensai_trace::Graph;

decoder!((), Swiglu, sensai_models::small::smollm2(), Config::smollm2);

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
