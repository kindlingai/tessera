use super::intake::{Arrival, Feed, Wake};
use std::sync::mpsc;

#[test]
fn feed_keeps_order() {
    let (queue, jobs) = mpsc::channel();
    let mut feed = Feed::new(jobs, Box::new(Arrival::default()));
    assert!(feed.ready(false).is_empty());
    assert_eq!(feed.pull(true), Some(vec![]));
    for job in 0..3 {
        queue.send(job).unwrap();
    }
    assert_eq!(feed.pull(false), Some(vec![0, 1, 2]));
    drop(queue);
    assert_eq!(feed.pull(false), None);
}

#[test]
fn wake_unblocks_feed() {
    let (queue, jobs) = mpsc::channel::<u32>();
    let wake = Wake::default();
    let mut feed = Feed::with_wake(jobs, Box::new(Arrival::default()), wake.clone());
    let w = wake.clone();
    let t = std::thread::spawn(move || w.notify());
    assert_eq!(feed.pull(false), Some(vec![]));
    t.join().unwrap();
    queue.send(7).unwrap();
    wake.notify();
    assert_eq!(feed.pull(false), Some(vec![7]));
    drop(queue);
    wake.notify();
    assert_eq!(feed.pull(false), None);
}
