//! spaCy's small English tagger.

use crate::common::{Spec, suite};
use models::tagger::{Config, program};
use trace::Graph;

fn release() -> Graph {
    let c = Config {
        rows: vec![5000, 1000, 2500, 2500, 50, 50],
        width: 96,
        depth: 4,
        tags: 50,
        max_context: 512,
    };
    program(&c)
}

const SPEC: Spec = Spec {
    name: "tagger",
    release,
    parameters: 1_570_514,
    regions: &[],
    small: None,
    files: &[],
    golden: None,
};

suite!(SPEC);
