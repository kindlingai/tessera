use crate::Asks;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn asks_empty_wakes() {
    let asks = Asks::default();
    let wakes = Arc::new(AtomicUsize::new(0));
    let hit = wakes.clone();
    asks.on_empty(move || _ = hit.fetch_add(1, Ordering::Relaxed));
    let first = asks.hold();
    let second = asks.hold();
    assert!(asks.wanted());
    drop(first);
    assert!(asks.wanted());
    assert_eq!(wakes.load(Ordering::Relaxed), 0);
    drop(second);
    assert!(!asks.wanted());
    assert_eq!(wakes.load(Ordering::Relaxed), 1);
    let again = asks.hold();
    drop(again);
    assert_eq!(wakes.load(Ordering::Relaxed), 2);
}
