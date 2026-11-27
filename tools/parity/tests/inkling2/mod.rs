//! Inkling-2-Mini: a looped decoder with routed experts.

use tessera_models::inkling2::{Config, Plain, program, stepper};
use tessera_parity::{Spec, Stepper, suite};
use tessera_trace::Composite::{Attention, Model, Moe};
use tessera_trace::Graph;

fn small() -> Graph {
    program::<Plain>(&Config::small())
}

pub fn release() -> Graph {
    program::<Plain>(&Config::mini(1, 1024))
}

const STEPPER: Stepper = Stepper {
    small: || stepper::<Plain>(&Config::small()),
    release: || stepper::<Plain>(&Config::mini(1, 1024)),
    drift: tessera_parity::MLA_DRIFT,
};

const SPEC: Spec = Spec {
    name: "inkling2",
    release,
    parameters: 26_070_385_152,
    regions: &[(Attention, 48), (Moe, 48), (Model, 1)],
    stepper: Some(STEPPER),
    small: Some(small),
    files: &[],
    golden: None,
};

suite!(SPEC);
