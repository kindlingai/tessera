//! GLM-5.3.

mod regions_test;

use crate::common::{Spec, suite};
use models::glm53::{Config, program};
use trace::Composite::{Attention, Mlp, Moe};
use trace::Graph;

pub fn release() -> Graph {
    program(&Config::release(8, 425_984))
}

const SPEC: Spec = Spec {
    name: "glm53",
    release,
    parameters: 743_377_019_904,
    regions: &[(Attention, 78), (Mlp, 3), (Moe, 75)],
    small: None,
    files: &[],
};

suite!(SPEC);
