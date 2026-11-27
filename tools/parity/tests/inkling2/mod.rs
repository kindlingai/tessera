//! Inkling-2-Mini: a looped decoder with routed experts.

mod looped_test;

use tessera_models::Stack;
use tessera_models::inkling2::{Config, Plain, program, stepper};
use tessera_parity::{Spec, Stepper, suite};
use tessera_trace::Composite::{Attention, Model, Moe};
use tessera_trace::Graph;

/// Two unlooped layers, local and global, for fast rule sweeps.
fn compact() -> Config {
    Config {
        stack: Stack::sized([64, 16, 2, 4, 4, 16], 2, 16),
        global: vec![false, true],
        n_experts: 2,
        top_k: 2,
        loop_passes: 1,
        ..Config::small()
    }
}

fn small() -> Graph {
    program::<Plain>(&compact())
}

pub fn release() -> Graph {
    program::<Plain>(&Config::mini(1, 1024))
}

const STEPPER: Stepper = Stepper {
    small: || stepper::<Plain>(&compact()),
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
