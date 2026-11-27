use super::intake::{Arrival, Feed};
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
