//! A small nanogpt.

mod golden;
mod regions_test;

use tessera_lower::form::F32_FORM;
use tessera_models::nanogpt::{Config, program};
use tessera_parity::{Check, Compare, Golden, Spec, Tolerance, suite};
use tessera_trace::Composite::{Attention, Mlp};
use tessera_trace::Graph;

pub fn config() -> Config {
    tessera_models::nanogpt::small()
}

pub fn release() -> Graph {
    program(&config())
}

const SPEC: Spec = Spec {
    name: "nanogpt",
    release,
    parameters: 1768,
    regions: &[(Attention, 2), (Mlp, 2)],
    stepper: None,
    small: None,
    files: &[],
    golden: Some(Golden {
        record: Some(golden::read),
        form: F32_FORM,
        checks: &[Check {
            name: "logits",
            compare: Compare::Tokens,
            tolerance: Tolerance {
                max_abs: 1e-5,
                cos: 0.0,
                top: false,
            },
        }],
        anywhere: true,
    }),
};

suite!(SPEC);
