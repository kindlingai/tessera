use crate::gate::{Capped, Gate};

#[test]
fn capped_refuses_extra() {
    let gate = Capped::new(2);
    assert!(gate.enter(None) && gate.enter(None));
    assert!(!gate.enter(None));
    gate.leave(None);
    assert!(gate.enter(None));
}
