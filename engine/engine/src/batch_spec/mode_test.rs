use super::mode::{Flight, Lone, Mode};

#[test]
fn lone_needs_one_answer() {
    let mode = Lone::default();
    let flight = |active, prefilling| Flight { active, prefilling };
    assert!(mode.solo(flight(0, 0), 1));
    assert!(!mode.solo(flight(0, 0), 2) && !mode.solo(flight(1, 0), 1));
    assert!(mode.promote(flight(1, 0), 8));
    assert!(!mode.promote(flight(1, 0), 7) && !mode.promote(flight(1, 1), 8));
    assert!(mode.tail(2) && !mode.tail(3));
}
