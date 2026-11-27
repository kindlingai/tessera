//! Which model steps next, and which leaves for another.

use std::cell::Cell;
use std::cmp::Reverse;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

#[cfg(test)]
mod lib_test;

/// How often a waiting thread checks the timed rules.
const TICK: Duration = Duration::from_millis(20);

/// Bytes a second, assumed before the first load.
const RATE: f64 = 2e9;

/// The next action of a model's thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Turn {
    /// Load the weights.
    Load,
    /// Run one step. `admit` is false while the model drains.
    Step { admit: bool },
    /// Drop the weights.
    Leave,
}

/// Where a model's weights are. Times are seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Place {
    Out,
    Loading,
    /// Resident since this time.
    In(f64),
    /// Ends its answers from this time, then leaves.
    Draining(f64),
    Leaving,
}

/// One model's numbers, for `/v1/models`.
#[derive(Clone, Debug, PartialEq)]
pub struct Status {
    pub resident: bool,
    pub bytes: u64,
    /// Seconds the last load took. A guess before the first.
    pub seconds: f64,
    pub loads: u64,
}

struct Seat {
    name: String,
    priority: u32,
    place: Place,
    /// Device bytes when resident. A guess before the first load.
    bytes: u64,
    seconds: f64,
    /// A job may sit in the queue.
    asked: bool,
    /// Answers wait or run on the thread.
    busy: bool,
    /// Answers that hold a slot.
    running: usize,
    /// The first load has not happened.
    prime: bool,
    /// When it began to wait for its weights.
    since: f64,
    /// The step count when it last ran.
    ran: u64,
    /// When it last ran or loaded.
    last: f64,
    loads: u64,
    /// Set while the seat must ask before each step.
    stop: Arc<AtomicBool>,
}

impl Seat {
    fn wants(&self) -> bool {
        self.asked || self.busy
    }

    fn holds(&self) -> bool {
        self.place != Place::Out
    }

    fn going(&self) -> bool {
        matches!(self.place, Place::Draining(_) | Place::Leaving)
    }

    /// Whether the seat can run a step now.
    fn ready(&self) -> bool {
        matches!(self.place, Place::In(_) | Place::Draining(_)) && self.wants()
    }
}

/// The policy numbers.
#[derive(Clone, Copy, Debug)]
pub struct Rules {
    /// The byte limit over all resident models. `None` checks nothing.
    pub budget: Option<u64>,
    /// Take the budget from the first engine's device.
    pub learn: bool,
    /// Resident seconds a busy model keeps per load second.
    pub quantum: f64,
}

/// Every seat, and the one that holds the device.
pub struct Table {
    rules: Rules,
    seats: Vec<Seat>,
    holder: Option<usize>,
    steps: u64,
}

impl Table {
    pub fn new(rules: Rules) -> Self {
        Self {
            rules,
            seats: Vec::new(),
            holder: None,
            steps: 0,
        }
    }

    /// Adds a model that wants its first load.
    pub fn add(&mut self, name: &str, priority: u32, bytes: u64, now: f64) -> usize {
        self.seats.push(Seat {
            name: name.into(),
            priority,
            place: Place::Out,
            bytes,
            seconds: bytes as f64 / RATE,
            asked: false,
            busy: false,
            running: 0,
            prime: true,
            since: now,
            ran: 0,
            last: now,
            loads: 0,
            stop: Arc::new(AtomicBool::new(true)),
        });
        self.seats.len() - 1
    }

    fn fits(&self, bytes: u64) -> bool {
        self.rules.budget.is_none_or(|budget| bytes <= budget)
    }

    /// Reload seconds per byte, lower for an idle seat.
    fn worth(&self, at: usize, now: f64) -> f64 {
        let s = &self.seats[at];
        let idle = if s.wants() { 0.0 } else { now - s.last };
        s.seconds / (1.0 + idle / s.seconds.max(1.0)) / s.bytes.max(1) as f64
    }

    /// Whether `at` may lose its weights to `to`.
    fn yields(&self, at: usize, to: usize, now: f64) -> bool {
        let (s, to) = (&self.seats[at], &self.seats[to]);
        let Place::In(since) = s.place else {
            return false;
        };
        if !s.wants() {
            return true;
        }
        // A first load takes room from idle models only.
        if !to.wants() || to.priority < s.priority {
            return false;
        }
        to.priority > s.priority || now - since >= self.rules.quantum * s.seconds
    }

    /// Sends models out while loaded sizes pass the budget.
    fn trim(&mut self, now: f64) {
        loop {
            let stays = |s: &Seat| s.holds() && !s.going();
            let staying = self.seats.iter().filter(|s| stays(s));
            let (count, used) = staying.fold((0, 0), |(n, sum), s| (n + 1, sum + s.bytes));
            if count < 2 || self.fits(used) {
                return;
            }
            // Idle models go first, then the lowest priority.
            let rank = |i: &usize| {
                let s = &self.seats[*i];
                (s.wants(), s.priority, (self.worth(*i, now) * 1e12) as u64)
            };
            let resident = |i: &usize| matches!(self.seats[*i].place, Place::In(_));
            let Some(at) = (0..self.seats.len()).filter(resident).min_by_key(rank) else {
                return;
            };
            let s = &mut self.seats[at];
            s.place = match s.wants() {
                true => Place::Draining(now),
                false => Place::Leaving,
            };
        }
    }

    /// Gives room to seats that wait for their weights.
    fn place(&mut self, now: f64) {
        let waits = |s: &Seat| s.place == Place::Out && (s.wants() || s.prime);
        let mut order: Vec<usize> = (0..self.seats.len())
            .filter(|&i| waits(&self.seats[i]))
            .collect();
        let key = |s: &Seat| {
            (
                Reverse(s.wants()),
                Reverse(s.priority),
                (s.since * 1e3) as u64,
            )
        };
        order.sort_by_key(|&i| key(&self.seats[i]));
        for to in order {
            let used: u64 = self
                .seats
                .iter()
                .filter(|s| s.holds())
                .map(|s| s.bytes)
                .sum();
            // The first load runs alone. It gives the budget.
            if self.rules.learn {
                if self.seats.iter().all(|s| !s.holds()) {
                    self.seats[to].place = Place::Loading;
                }
                return;
            }
            // First loads have guessed sizes. One runs at a time.
            let loading = self.seats.iter().any(|s| s.place == Place::Loading);
            if self.seats[to].prime && loading {
                continue;
            }
            let need = self.seats[to].bytes;
            if used == 0 || self.fits(used + need) {
                self.seats[to].place = Place::Loading;
                continue;
            }
            let coming: u64 = (self.seats.iter().filter(|s| s.going()))
                .map(|s| s.bytes)
                .sum();
            let mut left = used - coming + need;
            if self.fits(left) {
                return;
            }
            let mut pool: Vec<usize> = (0..self.seats.len())
                .filter(|&i| self.yields(i, to, now))
                .collect();
            pool.sort_by(|&a, &b| self.worth(a, now).total_cmp(&self.worth(b, now)));
            let mut out = Vec::new();
            for at in pool {
                if self.fits(left) {
                    break;
                }
                left -= self.seats[at].bytes;
                out.push(at);
            }
            // A model over the budget may still run alone.
            if self.fits(left) || left == need {
                for at in out {
                    let s = &mut self.seats[at];
                    s.place = match s.wants() {
                        true => Place::Draining(now),
                        false => Place::Leaving,
                    };
                }
            }
            return;
        }
    }

    /// Applies every rule. Call it after each change.
    pub fn settle(&mut self, now: f64) {
        for s in &mut self.seats {
            if let Place::Draining(since) = s.place
                && (s.running == 0 || now - since >= s.seconds)
            {
                s.place = Place::Leaving;
            }
        }
        self.trim(now);
        self.place(now);
        let seats = &self.seats;
        if self.holder.is_none() {
            let next = (0..seats.len()).filter(|&i| seats[i].ready());
            self.holder = next.max_by_key(|&i| (seats[i].priority, Reverse(seats[i].ran)));
        }
        // The holder steps alone until another seat is ready.
        if let Some(at) = self.holder {
            let others = (0..seats.len()).any(|i| i != at && seats[i].ready());
            let stays = matches!(seats[at].place, Place::In(_));
            seats[at].stop.store(others || !stays, Ordering::Release);
        }
    }

    /// The flag that sends seat `at` back to the table.
    pub fn stop(&self, at: usize) -> Arc<AtomicBool> {
        self.seats[at].stop.clone()
    }

    /// The seat's turn, if it has one now.
    pub fn turn(&mut self, at: usize) -> Option<Turn> {
        let held = self.holder == Some(at);
        let turn = match self.seats[at].place {
            Place::Loading => Turn::Load,
            Place::Leaving => Turn::Leave,
            Place::In(_) if held => Turn::Step { admit: true },
            Place::Draining(_) if held => Turn::Step { admit: false },
            _ => return None,
        };
        match turn {
            Turn::Step { .. } => self.seats[at].asked = false,
            _ if held => self.holder = None,
            _ => {}
        }
        Some(turn)
    }

    /// A job entered the seat's queue.
    pub fn asked(&mut self, at: usize, now: f64) {
        let s = &mut self.seats[at];
        if !s.wants() && s.place == Place::Out {
            s.since = now;
        }
        s.asked = true;
    }

    /// Wakes a resident seat without reloading an evicted one.
    pub fn nudge(&mut self, at: usize) {
        let s = &mut self.seats[at];
        if matches!(s.place, Place::In(_) | Place::Draining(_)) {
            s.asked = true;
        }
    }

    /// The seat ran its step. `running` answers hold slots.
    pub fn stepped(&mut self, at: usize, (busy, running): (bool, usize), bytes: u64, now: f64) {
        self.holder = None;
        self.steps += 1;
        let s = &mut self.seats[at];
        (s.busy, s.running, s.bytes) = (busy, running, bytes);
        (s.ran, s.last) = (self.steps, now);
    }

    /// The seat's weights loaded in `seconds`.
    pub fn loaded(&mut self, at: usize, bytes: u64, seconds: f64, memory: u64, now: f64) {
        if std::mem::take(&mut self.rules.learn) && memory > 0 {
            self.rules.budget = Some(memory - memory / 10);
        }
        let s = &mut self.seats[at];
        (s.place, s.bytes, s.seconds) = (Place::In(now), bytes, seconds);
        (s.prime, s.last, s.loads) = (false, now, s.loads + 1);
    }

    /// The seat dropped its weights.
    pub fn left(&mut self, at: usize, now: f64) {
        let s = &mut self.seats[at];
        (s.place, s.since) = (Place::Out, now);
    }

    pub fn status(&self, name: &str) -> Option<Status> {
        let s = self.seats.iter().find(|s| s.name == name)?;
        Some(Status {
            resident: matches!(s.place, Place::In(_) | Place::Draining(_)),
            bytes: s.bytes,
            seconds: s.seconds,
            loads: s.loads,
        })
    }
}

/// The table, shared by every model thread.
pub struct Turns {
    table: Mutex<Table>,
    changed: Condvar,
    start: Instant,
}

impl Turns {
    pub fn new(rules: Rules) -> Arc<Self> {
        Arc::new(Self {
            table: Mutex::new(Table::new(rules)),
            changed: Condvar::new(),
            start: Instant::now(),
        })
    }

    fn now(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    fn lock(&self) -> MutexGuard<'_, Table> {
        self.table.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Changes and settles the table. `own`: the seat's thread calls.
    fn change<R>(&self, own: bool, f: impl FnOnce(&mut Table, f64) -> R) -> R {
        let now = self.now();
        let mut table = self.lock();
        let out = f(&mut table, now);
        table.settle(now);
        let wake = !own || table.seats.len() > 1;
        drop(table);
        if wake {
            self.changed.notify_all();
        }
        out
    }

    /// A seat for a model of about `bytes`.
    pub fn seat(self: &Arc<Self>, name: &str, priority: u32, bytes: u64) -> Seated {
        let add = |t: &mut Table, now| {
            let at = t.add(name, priority, bytes, now);
            (at, t.stop(at))
        };
        let (at, stop) = self.change(false, add);
        let asker = Asker {
            turns: self.clone(),
            at,
        };
        Seated {
            asker,
            stop,
            kept: Cell::new(None),
        }
    }

    pub fn status(&self, name: &str) -> Option<Status> {
        self.lock().status(name)
    }
}

/// Tells a seat of new jobs, from any thread.
#[derive(Clone)]
pub struct Asker {
    turns: Arc<Turns>,
    at: usize,
}

impl Asker {
    /// A job entered the model's queue.
    pub fn asked(&self) {
        self.turns.change(false, |t, now| t.asked(self.at, now));
    }

    /// Wakes the model if resident, without reloading it.
    pub fn nudge(&self) {
        self.turns.change(false, |t, _| t.nudge(self.at));
    }
}

/// A step's result: busy, answers in slots, device bytes.
type Stepped = ((bool, usize), u64);

/// One model's seat, for the model's thread.
pub struct Seated {
    asker: Asker,
    stop: Arc<AtomicBool>,
    /// A busy step the table has not seen yet.
    kept: Cell<Option<Stepped>>,
}

impl Seated {
    pub fn asker(&self) -> Asker {
        self.asker.clone()
    }

    fn change<R>(&self, f: impl FnOnce(&mut Table, usize, f64) -> R) -> R {
        let at = self.asker.at;
        self.asker.turns.change(true, |t, now| f(t, at, now))
    }

    /// Waits for the model's next turn.
    pub fn next(&self) -> Turn {
        if let Some(step) = self.kept.take() {
            // A busy holder steps again with no lock.
            if !self.stop.load(Ordering::Acquire) {
                return Turn::Step { admit: true };
            }
            self.change(|t, at, now| t.stepped(at, step.0, step.1, now));
        }
        let turns = &self.asker.turns;
        let mut table = turns.lock();
        loop {
            table.settle(turns.now());
            if let Some(turn) = table.turn(self.asker.at) {
                return turn;
            }
            let waited = turns.changed.wait_timeout(table, TICK);
            table = waited.unwrap_or_else(|e| e.into_inner()).0;
        }
    }

    /// The model ran its step.
    pub fn stepped(&self, state: (bool, usize), bytes: u64) {
        match state.0 && !self.stop.load(Ordering::Acquire) {
            true => self.kept.set(Some((state, bytes))),
            false => self.change(|t, at, now| t.stepped(at, state, bytes, now)),
        }
    }

    /// The weights loaded. The device has `memory` bytes.
    pub fn loaded(&self, bytes: u64, seconds: f64, memory: u64) {
        self.change(|t, at, now| t.loaded(at, bytes, seconds, memory, now));
    }

    /// The weights dropped.
    pub fn left(&self) {
        self.change(|t, at, now| t.left(at, now));
    }
}
