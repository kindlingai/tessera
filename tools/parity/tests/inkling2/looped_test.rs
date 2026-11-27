//! Compiled step parity on the full looped Inkling-2 small fixture.

use super::{SPEC, STEPPER};
use tessera_models::inkling2::{Config, Plain, program, stepper};
use tessera_parity::{Spec, Stepper};

#[test]
fn looped_steps() {
    const LOOPED: Spec = Spec {
        small: Some(|| program::<Plain>(&Config::small())),
        stepper: Some(Stepper {
            small: || stepper::<Plain>(&Config::small()),
            ..STEPPER
        }),
        ..SPEC
    };
    tessera_parity::steps(&LOOPED);
}
