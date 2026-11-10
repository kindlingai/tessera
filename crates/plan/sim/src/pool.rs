//! Any GPU as slow threads behind one memory bus.

use crate::{Chip, Const, Estimate, Limit, Load};
use serde::{Deserialize, Serialize};

/// The constants a probe run can fit on any backend.
#[derive(Clone, Debug, PartialEq)]
pub struct Pool {
    pub name: String,
    /// Seconds an empty launch takes.
    pub floor: Const,
    /// Seconds each started thread adds.
    pub spawn: Const,
    /// Seconds one thread takes per dependent float op.
    pub step: Const,
    /// Seconds one thread takes per exponential or divide.
    pub curve: Const,
    /// Float ops a second, every thread together.
    pub rate: Const,
    /// Seconds a thread waits on a new line.
    pub wait: Const,
    /// Seconds a load beside the last one costs.
    pub near: Const,
    /// Bytes that stay in cache between launches.
    pub cache: Const,
    /// Bytes a second out of cache.
    pub cached: Const,
    /// Bytes a second out of memory, threads interleaved.
    pub read: Const,
    /// The same, each thread along its own stretch.
    pub scan: Const,
    /// Bytes a second into memory.
    pub write: Const,
}

/// A pool's constants as plain numbers, as probes fit them.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct Fit {
    pub floor: f64,
    pub spawn: f64,
    pub step: f64,
    pub curve: f64,
    pub rate: f64,
    pub wait: f64,
    pub near: f64,
    pub cache: f64,
    pub cached: f64,
    pub read: f64,
    pub scan: f64,
    pub write: f64,
}

impl Fit {
    /// Pairs each value with its probe.
    pub fn pool(&self, name: &str) -> Pool {
        let m = Const::measured;
        Pool {
            name: name.into(),
            floor: m(self.floor, "256 threads that do nothing"),
            spawn: m(self.spawn, "65,536 idle threads against 256"),
            step: m(self.step, "a multiply-add loop in 256 threads"),
            curve: m(self.curve, "an exponential loop in 256 threads"),
            rate: m(self.rate, "the multiply-add loop at its best thread count"),
            wait: m(self.wait, "256 threads, each load on a new line of 512 MB"),
            near: m(self.near, "256 threads, each along its own 16 KB"),
            cache: Const::inferred(self.cache, "the small probe buffer; larger sizes unprobed"),
            cached: m(self.cached, "4 MB read again at the best thread count"),
            read: m(self.read, "512 MB read at the best thread count"),
            scan: m(self.scan, "512 MB read in separate stretches"),
            write: m(self.write, "256 MB written at the best thread count"),
        }
    }
}

impl Chip for Pool {
    fn name(&self) -> &str {
        &self.name
    }

    fn estimate(&self, l: &Load) -> Estimate {
        let (threads, t) = (l.threads as f64, &l.trip);
        // The step already holds a loop's own index work.
        let ops = t.float * self.step.value + t.special * self.curve.value;
        // Streams cross lines together, so their waits overlap.
        let fresh = if t.loads > 0.0 { t.fresh } else { 0.0 };
        let waits = fresh * self.wait.value + (1.0 - fresh) * t.loads * self.near.value;
        let chain = l.trips * (ops + waits);
        let compute = threads * l.trips * ops / self.step.value / self.rate.value;
        // Little's law: bytes in flight over the wait.
        let flow = threads * l.flight / self.wait.value;
        let through = |peak: f64| if flow > 0.0 { peak.min(flow) } else { peak };
        let cold = |bytes: &&f64| **bytes > self.cache.value;
        let far: f64 = l.reads.iter().filter(cold).sum();
        let peak = if t.fresh < 1.0 { self.scan } else { self.read };
        let moved = far / through(peak.value) + l.written / self.write.value;
        let start = self.floor.value + self.spawn.value * threads;
        let limits = [
            (Limit::Chain, chain),
            (Limit::Compute, compute),
            (Limit::Read, moved),
            (Limit::Cache, l.loaded() / through(self.cached.value)),
        ];
        let mut slowest = Estimate::slowest(start, &limits);
        if slowest.limit == Limit::Read && l.written > far {
            slowest.limit = Limit::Write;
        }
        slowest
    }

    /// Every op at the chip's rate, every byte at best.
    fn bound(&self, l: &Load) -> f64 {
        let curves = l.floats() / l.trip.float.max(1.0) * l.trip.special;
        let ops = l.floats() + curves * self.curve.value / self.step.value;
        let best = self.read.value.max(self.scan.value);
        let rate = |bytes: &f64| match *bytes > self.cache.value {
            true => bytes / best,
            false => bytes / self.cached.value,
        };
        let moved: f64 = l.reads.iter().map(rate).sum::<f64>() + l.written / self.write.value;
        self.floor.value + (ops / self.rate.value).max(moved)
    }

    fn constants(&self) -> Vec<(&'static str, Const)> {
        vec![
            ("floor", self.floor),
            ("spawn", self.spawn),
            ("step", self.step),
            ("curve", self.curve),
            ("rate", self.rate),
            ("wait", self.wait),
            ("near", self.near),
            ("cache", self.cache),
            ("cached", self.cached),
            ("read", self.read),
            ("scan", self.scan),
            ("write", self.write),
        ]
    }
}
