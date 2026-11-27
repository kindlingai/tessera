//! The order in which queued jobs start.

use std::collections::VecDeque;
use std::sync::mpsc;

/// Chooses the jobs a model starts before each step.
pub(super) trait Intake<J> {
    /// `job` entered the queue.
    fn arrived(&mut self, job: J);
    /// Jobs to start now. `busy`: the model has work.
    fn start(&mut self, busy: bool) -> Vec<J>;
}

/// Starts every job in arrival order.
pub(super) struct Arrival<J>(VecDeque<J>);

impl<J> Default for Arrival<J> {
    fn default() -> Self {
        Self(VecDeque::new())
    }
}

impl<J> Intake<J> for Arrival<J> {
    fn arrived(&mut self, job: J) {
        self.0.push_back(job);
    }

    fn start(&mut self, _busy: bool) -> Vec<J> {
        self.0.drain(..).collect()
    }
}

/// A job queue read through an intake policy.
pub(super) struct Feed<J> {
    jobs: mpsc::Receiver<J>,
    intake: Box<dyn Intake<J>>,
}

impl<J> Feed<J> {
    pub fn new(jobs: mpsc::Receiver<J>, intake: Box<dyn Intake<J>>) -> Self {
        Self { jobs, intake }
    }

    /// Jobs to start now. Never waits.
    pub fn ready(&mut self, busy: bool) -> Vec<J> {
        for job in self.jobs.try_iter() {
            self.intake.arrived(job);
        }
        self.intake.start(busy)
    }

    /// As `ready`, but waits when idle. `None`: the queue closed.
    pub fn pull(&mut self, busy: bool) -> Option<Vec<J>> {
        let jobs = self.ready(busy);
        if busy || !jobs.is_empty() {
            return Some(jobs);
        }
        let job = self.jobs.recv().ok()?;
        self.intake.arrived(job);
        Some(self.ready(false))
    }
}
