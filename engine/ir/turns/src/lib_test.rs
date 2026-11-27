use crate::{Rules, Table, Turn, Turns};
use std::sync::atomic::Ordering;

fn table(budget: Option<u64>) -> Table {
    Table::new(Rules {
        budget,
        learn: budget.is_none(),
        quantum: 4.0,
    })
}

/// Loads seat `at` on a device of 100 bytes.
fn load(t: &mut Table, at: usize, seconds: f64, now: f64) {
    t.settle(now);
    assert_eq!(t.turn(at), Some(Turn::Load));
    t.loaded(at, 60, seconds, 100, now);
}

/// Runs one step of seat `at` that leaves `running` answers.
fn step(t: &mut Table, at: usize, admit: bool, running: usize, now: f64) {
    t.settle(now);
    assert_eq!(t.turn(at), Some(Turn::Step { admit }));
    t.stepped(at, (running > 0, running), 60, now);
}

#[test]
fn fitting_models_alternate() {
    let mut t = table(None);
    let (a, b) = (t.add("a", 0, 40, 0.0), t.add("b", 0, 40, 0.0));
    t.settle(0.0);
    // The first load runs alone and gives the budget.
    assert_eq!((t.turn(a), t.turn(b)), (Some(Turn::Load), None));
    t.loaded(a, 40, 1.0, 200, 1.0);
    t.settle(1.0);
    assert_eq!(t.turn(b), Some(Turn::Load));
    t.loaded(b, 40, 1.0, 200, 2.0);
    t.asked(a, 2.0);
    t.asked(b, 2.0);
    // The seat that ran longest ago steps next.
    step(&mut t, b, true, 1, 2.0);
    assert_eq!(t.turn(b), None);
    step(&mut t, a, true, 1, 2.0);
    step(&mut t, b, true, 1, 2.0);
}

#[test]
fn idle_model_leaves() {
    let mut t = table(Some(100));
    let (a, b) = (t.add("a", 0, 60, 0.0), t.add("b", 0, 60, 0.0));
    load(&mut t, a, 1.0, 0.0);
    t.settle(1.0);
    assert_eq!((t.turn(a), t.turn(b)), (Some(Turn::Leave), None));
    t.left(a, 1.0);
    load(&mut t, b, 1.0, 1.0);
    t.asked(a, 2.0);
    t.settle(2.0);
    assert_eq!(t.turn(b), Some(Turn::Leave));
    t.left(b, 2.0);
    load(&mut t, a, 1.0, 2.0);
    let status = t.status("a").expect("seat a");
    assert_eq!((status.resident, status.loads), (true, 2));
}

#[test]
fn busy_model_drains() {
    let mut t = table(Some(100));
    let a = t.add("a", 0, 60, 0.0);
    load(&mut t, a, 10.0, 0.0);
    t.asked(a, 0.0);
    step(&mut t, a, true, 1, 1.0);
    let b = t.add("b", 0, 60, 1.0);
    t.asked(b, 2.0);
    // Ten load seconds buy forty resident seconds.
    step(&mut t, a, true, 1, 39.0);
    step(&mut t, a, false, 1, 41.0);
    step(&mut t, a, false, 0, 42.0);
    t.settle(42.0);
    assert_eq!(t.turn(a), Some(Turn::Leave));
    t.left(a, 42.0);
    load(&mut t, b, 1.0, 42.0);
}

#[test]
fn priority_takes_memory() {
    let mut t = table(Some(100));
    let a = t.add("a", 0, 60, 0.0);
    load(&mut t, a, 10.0, 0.0);
    t.asked(a, 0.0);
    step(&mut t, a, true, 1, 1.0);
    let b = t.add("b", 1, 60, 1.0);
    // A first load does not move a busy model.
    step(&mut t, a, true, 1, 2.0);
    t.asked(b, 2.0);
    step(&mut t, a, false, 1, 2.0);
}

#[test]
fn lone_model_stays() {
    let mut t = table(Some(100));
    let a = t.add("a", 0, 60, 0.0);
    load(&mut t, a, 1.0, 0.0);
    t.asked(a, 0.0);
    t.settle(0.0);
    assert_eq!(t.turn(a), Some(Turn::Step { admit: true }));
    let stop = t.stop(a);
    assert!(!stop.load(Ordering::Acquire));
    // A second model that asks sends the holder back.
    let b = t.add("b", 0, 30, 1.0);
    t.settle(1.0);
    assert_eq!(t.turn(b), Some(Turn::Load));
    t.loaded(b, 30, 1.0, 100, 1.0);
    t.settle(1.0);
    assert!(!stop.load(Ordering::Acquire));
    t.asked(b, 2.0);
    t.settle(2.0);
    assert!(stop.load(Ordering::Acquire));
}

#[test]
fn idle_seat_returns() {
    let turns = Turns::new(Rules {
        budget: Some(100),
        learn: false,
        quantum: 4.0,
    });
    let a = turns.seat("a", 0, 60);
    assert_eq!(a.next(), Turn::Load);
    a.loaded(60, 1.0, 0);
    a.asker().asked();
    let step = Turn::Step { admit: true };
    assert_eq!(a.next(), step);
    a.stepped((true, 1), 60);
    assert_eq!(a.next(), step);
    a.stepped((false, 0), 60);
    // An idle seat asks the table again.
    let _b = turns.seat("b", 0, 60);
    assert_eq!(a.next(), Turn::Leave);
}

#[test]
fn oversize_model_trims() {
    let mut t = table(Some(100));
    let (a, b) = (t.add("a", 0, 40, 0.0), t.add("b", 0, 40, 0.0));
    t.settle(0.0);
    // The guesses fit. First loads still run singly.
    assert_eq!((t.turn(a), t.turn(b)), (Some(Turn::Load), None));
    t.loaded(a, 60, 1.0, 0, 1.0);
    t.settle(1.0);
    assert_eq!(t.turn(b), Some(Turn::Load));
    t.loaded(b, 70, 1.0, 0, 2.0);
    t.settle(2.0);
    assert_eq!(t.turn(a), Some(Turn::Leave));
    t.left(a, 2.0);
    t.nudge(a);
    t.settle(2.0);
    assert_eq!(t.turn(a), None);
    t.nudge(b);
    t.settle(2.0);
    assert_eq!(t.turn(b), Some(Turn::Step { admit: true }));
}
