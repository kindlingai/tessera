//! Progress of long steps, such as downloads and model loads.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The shortest time between two `Move` events of one step.
const EVERY: Duration = Duration::from_millis(100);

/// A `total` that is not known yet.
const UNKNOWN: u64 = u64::MAX;

static NEXT: AtomicU64 = AtomicU64::new(1);

/// The thing a step counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    Bytes,
    Items,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Start,
    Move,
    Done,
    /// The step ended before `done`, as on an error.
    Stopped,
}

/// One change to one step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Event {
    /// Unique in the process.
    pub id: u64,
    pub parent: Option<u64>,
    pub name: Arc<str>,
    pub unit: Unit,
    pub done: u64,
    pub total: Option<u64>,
    pub state: State,
}

impl Event {
    /// The part done, when the total is known.
    pub fn fraction(&self) -> Option<f64> {
        let total = self.total.filter(|&t| t > 0)?;
        Some((self.done as f64 / total as f64).min(1.0))
    }

    /// The count as words, such as `1.2/4.0 GB`.
    pub fn amount(&self) -> String {
        match (self.unit, self.total) {
            (Unit::Bytes, Some(t)) => format!("{}/{}", size(self.done), size(t)),
            (Unit::Bytes, None) => size(self.done),
            (Unit::Items, Some(t)) => format!("{}/{t}", self.done),
            (Unit::Items, None) => self.done.to_string(),
        }
    }
}

/// Bytes in the largest unit that stays under 1000.
fn size(bytes: u64) -> String {
    let mut value = bytes as f64;
    for unit in ["B", "kB", "MB", "GB"] {
        if value < 1000.0 {
            return format!("{value:.1} {unit}");
        }
        value /= 1000.0;
    }
    format!("{value:.1} TB")
}

/// Takes events. It may get them from many threads.
pub trait Report: Send + Sync {
    fn report(&self, event: &Event);
}

/// Where new steps report. The default reports nowhere.
#[derive(Clone, Default)]
pub struct Progress {
    to: Option<Arc<dyn Report>>,
    parent: Option<u64>,
}

impl Progress {
    pub fn new(to: Arc<dyn Report>) -> Self {
        Self {
            to: Some(to),
            parent: None,
        }
    }

    /// This, also reporting to `more`.
    pub fn also(&self, more: Arc<dyn Report>) -> Self {
        let to: Arc<dyn Report> = match &self.to {
            Some(to) => Arc::new(Each(vec![to.clone(), more])),
            None => more,
        };
        Self {
            to: Some(to),
            parent: self.parent,
        }
    }

    /// Whether events go anywhere.
    pub fn reports(&self) -> bool {
        self.to.is_some()
    }

    /// Starts a step under this one's parent.
    pub fn step(&self, name: impl Into<Arc<str>>, unit: Unit, total: Option<u64>) -> Step {
        let step = Step {
            to: self.to.clone(),
            id: NEXT.fetch_add(1, Ordering::Relaxed),
            parent: self.parent,
            name: name.into(),
            unit,
            done: AtomicU64::new(0),
            total: AtomicU64::new(total.unwrap_or(UNKNOWN)),
            last: Mutex::new(Instant::now()),
            ended: AtomicBool::new(false),
        };
        step.send(State::Start);
        step
    }
}

/// A running step. Many threads may add to one count.
pub struct Step {
    to: Option<Arc<dyn Report>>,
    id: u64,
    parent: Option<u64>,
    name: Arc<str>,
    unit: Unit,
    done: AtomicU64,
    total: AtomicU64,
    /// When the last `Move` went out.
    last: Mutex<Instant>,
    ended: AtomicBool,
}

impl Step {
    /// Where steps inside this one report.
    pub fn progress(&self) -> Progress {
        Progress {
            to: self.to.clone(),
            parent: Some(self.id),
        }
    }

    /// Adds `n` to the count.
    pub fn add(&self, n: u64) {
        self.done.fetch_add(n, Ordering::Relaxed);
        self.moved();
    }

    /// Sets the count.
    pub fn set(&self, done: u64) {
        self.done.store(done, Ordering::Relaxed);
        self.moved();
    }

    pub fn set_total(&self, total: u64) {
        self.total.store(total, Ordering::Relaxed);
        self.moved();
    }

    /// Ends the step as done.
    pub fn finish(self) {
        self.end(State::Done);
    }

    /// Sends `Move` unless one went out in the last `EVERY`.
    fn moved(&self) {
        if self.to.is_none() {
            return;
        }
        let Ok(mut last) = self.last.try_lock() else {
            return;
        };
        if last.elapsed() < EVERY {
            return;
        }
        *last = Instant::now();
        drop(last);
        self.send(State::Move);
    }

    fn end(&self, state: State) {
        if !self.ended.swap(true, Ordering::Relaxed) {
            self.send(state);
        }
    }

    fn send(&self, state: State) {
        let Some(to) = &self.to else {
            return;
        };
        let total = self.total.load(Ordering::Relaxed);
        to.report(&Event {
            id: self.id,
            parent: self.parent,
            name: self.name.clone(),
            unit: self.unit,
            done: self.done.load(Ordering::Relaxed),
            total: (total != UNKNOWN).then_some(total),
            state,
        });
    }
}

impl Drop for Step {
    fn drop(&mut self) {
        self.end(State::Stopped);
    }
}

/// Keeps the open steps, oldest first, for a status answer.
#[derive(Default)]
pub struct Open {
    steps: Mutex<Vec<Event>>,
}

impl Open {
    /// The open steps.
    pub fn steps(&self) -> Vec<Event> {
        self.steps.lock().map(|s| s.clone()).unwrap_or_default()
    }
}

impl Report for Open {
    fn report(&self, event: &Event) {
        let Ok(mut steps) = self.steps.lock() else {
            return;
        };
        let at = steps.iter().position(|s| s.id == event.id);
        match (event.state, at) {
            (State::Done | State::Stopped, Some(at)) => drop(steps.remove(at)),
            (State::Done | State::Stopped, None) => {}
            (_, Some(at)) => steps[at] = event.clone(),
            (_, None) => steps.push(event.clone()),
        }
    }
}

/// Sends each event to every reporter in turn.
struct Each(Vec<Arc<dyn Report>>);

impl Report for Each {
    fn report(&self, event: &Event) {
        for to in &self.0 {
            to.report(event);
        }
    }
}

/// Writes a stderr line as each step starts or stops.
pub struct Log;

impl Report for Log {
    fn report(&self, event: &Event) {
        match event.state {
            State::Start => eprintln!("{}", event.name),
            State::Stopped => eprintln!("{}: stopped at {}", event.name, event.amount()),
            State::Move | State::Done => {}
        }
    }
}
