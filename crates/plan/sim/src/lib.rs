//! Analytic models of GPUs: what a launch should cost.

mod known;
mod nvidia;
mod pool;
mod zone;

use serde::{Deserialize, Serialize};

pub use known::{known, m3_pro, strix_halo};
pub use nvidia::{Nvidia, Sm, gb10};
pub use pool::{Fit, Pool};
pub use zone::{Cluster, Link, Zone, ZoneKind};

/// Where a constant's value comes from.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tag {
    /// Timed on the hardware. The note says how.
    Measured,
    /// Derived from a measurement and an assumption.
    Inferred,
    /// A published or guessed value no probe has checked.
    Assumed,
}

/// One hardware number and its source.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Const {
    pub value: f64,
    pub tag: Tag,
    pub note: &'static str,
}

impl Const {
    pub const fn measured(value: f64, note: &'static str) -> Self {
        let tag = Tag::Measured;
        Self { value, tag, note }
    }

    pub const fn inferred(value: f64, note: &'static str) -> Self {
        let tag = Tag::Inferred;
        Self { value, tag, note }
    }

    pub const fn assumed(value: f64, note: &'static str) -> Self {
        let tag = Tag::Assumed;
        Self { value, tag, note }
    }
}

/// What one thread does on each trip of its loop.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Ops {
    /// Float adds, multiplies and multiply-adds.
    pub float: f64,
    /// Integer and index arithmetic, compares and branches.
    pub int: f64,
    /// Divides, roots, exponentials and the like.
    pub special: f64,
    pub loads: f64,
    pub stores: f64,
    /// The share of loads that open a new cache line.
    pub fresh: f64,
}

/// A launch, as the work and traffic it causes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Load {
    pub threads: usize,
    /// Threads that start together as one group.
    pub group: usize,
    /// Loop trips each thread makes.
    pub trips: f64,
    pub trip: Ops,
    /// Bytes of each buffer the launch reads through.
    pub reads: Vec<f64>,
    pub written: f64,
    /// Bytes one thread waits on at a time.
    pub flight: f64,
}

impl Load {
    /// Float ops every thread together runs.
    pub fn floats(&self) -> f64 {
        self.threads as f64 * self.trips * self.trip.float
    }

    /// Bytes all loads bring. Shared buffers count per load.
    pub fn loaded(&self) -> f64 {
        self.threads as f64 * self.trips * self.trip.loads * WORD
    }
}

/// Bytes one load brings: a float.
const WORD: f64 = 4.0;

/// What bounds a launch's time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Limit {
    /// Starting the launch and its threads.
    Start,
    /// One thread's own ops and waits, in turn.
    Chain,
    /// Every thread's ops against the whole chip's rate.
    Compute,
    /// Instructions against the schedulers' issue slots.
    Issue,
    /// Instructions against one kind of execution unit.
    Pipe,
    /// Bytes that must come from memory.
    Read,
    /// Bytes from cache.
    Cache,
    Write,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Estimate {
    pub seconds: f64,
    pub limit: Limit,
}

impl Estimate {
    /// The slowest of `limits`, after `start` seconds.
    pub(crate) fn slowest(start: f64, limits: &[(Limit, f64)]) -> Self {
        let longer = |a: &&(Limit, f64), b: &&(Limit, f64)| a.1.total_cmp(&b.1);
        let (limit, seconds) = *limits.iter().max_by(longer).expect("some limit");
        match seconds > start {
            true => Self {
                seconds: start + seconds,
                limit,
            },
            false => Self {
                seconds: start + seconds,
                limit: Limit::Start,
            },
        }
    }
}

/// A model of one GPU.
pub trait Chip {
    fn name(&self) -> &str;
    fn estimate(&self, load: &Load) -> Estimate;
    /// The least seconds any kernel needs for the same work.
    fn bound(&self, load: &Load) -> f64;
    /// Every constant the model reads, by name.
    fn constants(&self) -> Vec<(&'static str, Const)>;
}

#[cfg(test)]
mod sim_test;
