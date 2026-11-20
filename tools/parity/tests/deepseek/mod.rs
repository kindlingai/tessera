//! DeepSeek-V3.

use sensai_models::mla::{Config, Moe as Experts, program, stepper};
use sensai_parity::{Spec, Stepper, suite};
use sensai_trace::Composite::{Attention, Mlp, Moe};
use sensai_trace::Graph;

fn small() -> Graph {
    program::<Experts>(&sensai_models::small::deepseek())
}

pub fn release() -> Graph {
    program::<Experts>(&Config::deepseek_v3(8, 163_840))
}

const STEPPER: Stepper = Stepper {
    small: || stepper::<Experts>(&sensai_models::small::deepseek()),
    release: || stepper::<Experts>(&Config::deepseek_v3(1, 1024)),
    drift: sensai_parity::MLA_DRIFT,
};

const SPEC: Spec = Spec {
    name: "deepseek",
    release,
    parameters: 671_026_419_200,
    regions: &[(Attention, 61), (Mlp, 3), (Moe, 58)],
    stepper: Some(STEPPER),
    small: Some(small),
    files: &[],
    golden: None,
};

suite!(SPEC);
