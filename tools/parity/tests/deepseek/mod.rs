//! DeepSeek-V3.

use tessera_models::mla::{Config, Moe as Experts, program, stepper};
use tessera_parity::{Spec, Stepper, suite};
use tessera_trace::Composite::{Attention, Layer, Mlp, Model, Moe};
use tessera_trace::Graph;

fn small() -> Graph {
    program::<Experts>(&Config::deepseek_small())
}

pub fn release() -> Graph {
    program::<Experts>(&Config::deepseek_v3(8, 163_840))
}

const STEPPER: Stepper = Stepper {
    small: || stepper::<Experts>(&Config::deepseek_small()),
    release: || stepper::<Experts>(&Config::deepseek_v3(1, 1024)),
    drift: tessera_parity::MLA_DRIFT,
};

const SPEC: Spec = Spec {
    name: "deepseek",
    release,
    parameters: 671_026_419_200,
    regions: &[
        (Attention, 61),
        (Mlp, 3),
        (Moe, 58),
        (Layer, 61),
        (Model, 1),
    ],
    stepper: Some(STEPPER),
    small: Some(small),
    files: &[],
    golden: None,
};

suite!(SPEC);
