//! Qwen3-VL-Embedding vision tower and packed text embedding backbone.

use tessera_models::qwen3_vl::{Config, text_program, vision_program};
use tessera_parity::{Spec, suite};
use tessera_trace::Composite::{Attention, Mlp};
use tessera_trace::{Graph, Linear};

mod vision {
    use super::*;

    fn release() -> Graph {
        vision_program::<Linear>(&Config::qwen3_vl_2b(1, 2048, 256).vision)
    }

    const SPEC: Spec = Spec {
        name: "qwen3_vl_vision",
        release,
        parameters: 406_957_056,
        regions: &[(Attention, 24), (Mlp, 24)],
        stepper: None,
        small: None,
        files: &[],
        golden: None,
    };

    suite!(SPEC);
}

mod text {
    use super::*;

    fn release() -> Graph {
        text_program::<Linear>(&Config::qwen3_vl_2b(1, 2048, 256).text)
    }

    const SPEC: Spec = Spec {
        name: "qwen3_vl_text",
        release,
        parameters: 1_720_574_976,
        regions: &[(Attention, 28), (Mlp, 28)],
        stepper: None,
        small: None,
        files: &[],
        golden: None,
    };

    suite!(SPEC);
}
