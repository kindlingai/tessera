//! Any GPU as slow threads behind one memory bus.

use crate::{Chip, Const, Estimate, Limit, Load};

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
