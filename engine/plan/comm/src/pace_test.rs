use crate::pace::{Pace, Spinning, Step};
use std::time::Duration;

#[test]
fn spinning_naps_quiet() {
    let mut pace = Spinning::default();
    (pace.every, pace.awake) = (2, Duration::ZERO);
    std::thread::sleep(Duration::from_millis(1));
    assert_eq!(pace.idle(false), Step::Spin);
    assert_eq!(pace.idle(true), Step::Poll, "pending work keeps it awake");
    pace.awake = Duration::from_secs(3600);
    assert_eq!(pace.idle(false), Step::Spin);
    assert_eq!(pace.idle(false), Step::Poll);
    pace.awake = Duration::ZERO;
    pace.worked();
    assert_eq!(pace.idle(false), Step::Spin);
    std::thread::sleep(Duration::from_millis(1));
    assert_eq!(pace.idle(false), Step::Nap(pace.nap));
}
