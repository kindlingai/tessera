use crate::progress::{Open, Progress, State, Unit};
use std::sync::Arc;

#[test]
fn open_keeps_running_steps() {
    let open = Arc::new(Open::default());
    let progress = Progress::new(open.clone());
    let outer = progress.step("load", Unit::Items, Some(2));
    let inner = outer.progress().step("weights.bin", Unit::Bytes, None);
    inner.set_total(4000);
    let steps = open.steps();
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[1].parent, Some(steps[0].id));
    assert_eq!(steps[0].state, State::Start);
    inner.finish();
    assert_eq!(open.steps().len(), 1);
    drop(outer);
    assert!(open.steps().is_empty());
}

#[test]
fn amount_reads_as_words() {
    let open = Arc::new(Open::default());
    let step = Progress::new(open.clone()).step("get", Unit::Bytes, Some(4_000_000_000));
    std::thread::sleep(std::time::Duration::from_millis(120));
    step.add(1_200_000_000);
    let event = &open.steps()[0];
    assert_eq!(event.amount(), "1.2 GB/4.0 GB");
    assert_eq!(event.fraction(), Some(0.3));
}
