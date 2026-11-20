//! GLM-5.3.

mod module_test;
mod regions_test;
mod shard_test;

use sensai_models::mla::{Config, RotatedMoe, program, stepper};
use sensai_parity::{Spec, Stepper, suite};
use sensai_trace::Composite::{Attention, Layer, Mlp, Model, Moe};
use sensai_trace::Graph;

fn small() -> Graph {
    program::<RotatedMoe>(&sensai_models::small::glm53())
}

pub fn release() -> Graph {
    program::<RotatedMoe>(&Config::glm53(8, 425_984))
}

const STEPPER: Stepper = Stepper {
    small: || stepper::<RotatedMoe>(&sensai_models::small::glm53()),
    release: || stepper::<RotatedMoe>(&Config::glm53(1, 1024)),
    drift: sensai_parity::MLA_DRIFT,
};

const SPEC: Spec = Spec {
    name: "glm53",
    release,
    parameters: 743_849_800_704,
    regions: &[
        (Attention, 78),
        (Mlp, 3),
        (Moe, 75),
        (Layer, 78),
        (Model, 1),
    ],
    stepper: Some(STEPPER),
    small: Some(small),
    files: &[],
    golden: None,
};

suite!(SPEC);
