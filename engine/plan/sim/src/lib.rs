//! Analytic models of GPUs: what a launch should cost.

mod cost;
mod known;
mod nvidia;
mod pool;
mod zone;

use serde::{Deserialize, Serialize};

pub use cost::{Cost, FIXED, Sample, Timings};
pub use known::{known, m3_pro};
pub use pool::{Fit, Pool};
pub use zone::{Cluster, Link, Zone, ZoneKind};

use sensai_work::Load;

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

/// The bound on a launch's time.
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
    /// Bytes from group memory.
    Shared,
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
pub trait Chip: std::fmt::Debug + Send + Sync {
    fn name(&self) -> &str;
    fn estimate(&self, load: &Load) -> Estimate;
    /// The least seconds any kernel needs for the same work.
    fn bound(&self, load: &Load) -> f64;
    /// Every constant the model reads, by name.
    fn constants(&self) -> Vec<(&'static str, Const)>;
}

#[cfg(test)]
mod sim_test;
