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
    /// Flops a second on the matrix unit. Zero: none.
    pub tile: Const,
    /// Compute units. Zero: every thread resident at once.
    pub cores: Const,
    /// Bytes of registers and group memory one core holds.
    pub store: Const,
    /// Threads one core keeps resident.
    pub resident: Const,
    /// Registers a thread may hold before it spills.
    pub held: Const,
    /// Seconds a thread waits per float from group memory.
    pub shared: Const,
    /// Bytes a second out of one core's group memory.
    pub banked: Const,
    /// Seconds a thread takes per tile flop, in a chain.
    pub mma: Const,
    /// Seconds each spilled register adds to a trip.
    pub spill: Const,
}

/// A pool's constants as plain numbers, fitted from probes.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq)]
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
    #[serde(default)]
    pub tile: f64,
    #[serde(default)]
    pub cores: f64,
    #[serde(default)]
    pub store: f64,
    #[serde(default)]
    pub resident: f64,
    #[serde(default)]
    pub held: f64,
    #[serde(default)]
    pub shared: f64,
    #[serde(default)]
    pub banked: f64,
    #[serde(default)]
    pub mma: f64,
    #[serde(default)]
    pub spill: f64,
}

impl Fit {
    /// Pairs each value with its probe.
    pub fn pool(&self, name: &str) -> Pool {
        let (m, inferred) = (Const::measured, Const::inferred);
        Pool {
            name: name.into(),
            floor: m(self.floor, "256 threads that do nothing"),
            spawn: m(self.spawn, "65,536 idle threads against 256"),
            step: m(self.step, "a multiply-add loop in 256 threads"),
            curve: m(self.curve, "an exponential loop in 256 threads"),
            rate: m(self.rate, "the multiply-add loop at its best thread count"),
            wait: m(self.wait, "256 threads, each load on a new line of 512 MB"),
            near: m(self.near, "256 threads, each along its own 16 KB"),
            cache: inferred(self.cache, "the small probe buffer; larger sizes unprobed"),
            cached: m(self.cached, "4 MB read again at the best thread count"),
            read: m(self.read, "512 MB read at the best thread count"),
            scan: m(self.scan, "512 MB read in separate stretches"),
            write: m(self.write, "256 MB written at the best thread count"),
            tile: m(self.tile, "tile multiplies looping on registers"),
            cores: Const::assumed(self.cores, "the system's count"),
            store: inferred(self.store, "tiled schedules forced in turn"),
            resident: inferred(self.resident, "tiled schedules forced in turn"),
            held: Const::assumed(self.held, "Apple G13: 128 registers a thread"),
            shared: inferred(self.shared, "tiled schedules forced in turn"),
            banked: Const::assumed(self.banked, "128 bytes a cycle"),
            mma: inferred(self.mma, "tiled schedules forced in turn"),
            spill: inferred(self.spill, "tiled schedules forced in turn"),
        }
    }

    /// The same, with residency constants from a known table.
    pub fn filled(mut self, device: &str) -> Fit {
        let Some(known) = crate::known::fit(device) else {
            return self;
        };
        let pairs = [
            (&mut self.cores, known.cores),
            (&mut self.store, known.store),
            (&mut self.resident, known.resident),
            (&mut self.held, known.held),
            (&mut self.shared, known.shared),
            (&mut self.banked, known.banked),
            (&mut self.mma, known.mma),
            (&mut self.spill, known.spill),
        ];
        for (own, table) in pairs {
            if *own == 0.0 {
                *own = table;
            }
        }
        self
    }
}

/// How a launch's groups share the cores.
struct Residency {
    /// Rounds of resident groups the launch takes.
    waves: f64,
    /// Threads resident across the chip.
    threads: f64,
    /// Registers each thread spills.
    spilled: f64,
}

impl Pool {
    /// Groups per core: the least that memory and threads allow.
    fn residency(&self, l: &Load) -> Residency {
        let threads = l.threads as f64;
        if self.cores.value == 0.0 {
            let spilled = 0.0;
            return Residency {
                waves: 1.0,
                threads,
                spilled,
            };
        }
        let (group, bytes) = (l.group.max(1) as f64, l.group_bytes);
        // A group that cannot fit gets fewer registers.
        let room = (self.store.value - bytes) / (4.0 * group);
        let cap = self.held.value.min(room).max(0.0);
        let held = l.trip.held.min(cap);
        let footprint = bytes + 4.0 * group * held;
        let by_store = match footprint > 0.0 {
            true => (self.store.value / footprint).floor(),
            false => f64::INFINITY,
        };
        let by_threads = (self.resident.value / group).floor();
        let fit = by_store.min(by_threads).max(1.0);
        let groups = (threads / group).ceil();
        let per_core = (groups / self.cores.value).ceil();
        Residency {
            waves: (per_core / fit).ceil(),
            threads: threads.min(fit * group * self.cores.value),
            spilled: (l.trip.held - cap).max(0.0),
        }
    }

    /// Seconds the group memory takes to feed the tiles.
    fn banked(&self, l: &Load) -> f64 {
        let rate = self.cores.value * self.banked.value;
        let bytes = 4.0 * l.threads as f64 * l.trips * l.trip.shared;
        if rate > 0.0 { bytes / rate } else { 0.0 }
    }
}

impl Chip for Pool {
    fn name(&self) -> &str {
        &self.name
    }

    fn estimate(&self, l: &Load) -> Estimate {
        let (threads, t) = (l.threads as f64, &l.trip.priced(false));
        // The step already holds a loop's own index work.
        let ops = t.float * self.step.value + t.special * self.curve.value;
        // Streams cross lines together, so their waits overlap.
        let fresh = if t.loads > 0.0 { t.fresh } else { 0.0 };
        let waits = fresh * self.wait.value + (1.0 - fresh) * t.loads * self.near.value;
        let r = self.residency(l);
        let tiles = t.shared * self.shared.value + t.tile_flops * self.mma.value;
        let spills = r.spilled * self.spill.value;
        let chain = r.waves * l.trips * (ops + waits + tiles + spills);
        let compute = threads * l.trips * ops / self.step.value / self.rate.value;
        // Without a matrix unit, tile flops run as floats.
        let tiled = threads * l.trips * t.tile_flops;
        let compute = match self.tile.value > 0.0 {
            true => compute + tiled / self.tile.value,
            false => compute + tiled / self.rate.value,
        };
        // Little's law: bytes in flight over the wait.
        let flow = r.threads * l.flight / self.wait.value;
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
            (Limit::Shared, self.banked(l)),
        ];
        let mut slowest = Estimate::slowest(start, &limits);
        if slowest.limit == Limit::Read && l.written > far {
            slowest.limit = Limit::Write;
        }
        slowest
    }

    /// Every op at the chip's rate, every byte at best.
    fn bound(&self, l: &Load) -> f64 {
        let t = l.trip.priced(false);
        let floats = l.threads as f64 * l.trips * t.float;
        let ops =
            floats + l.threads as f64 * l.trips * t.special * self.curve.value / self.step.value;
        let tiled = l.threads as f64 * l.trips * t.tile_flops;
        let best_rate = self.rate.value.max(self.tile.value);
        let ops = ops + tiled * self.rate.value / best_rate;
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
            ("tile", self.tile),
            ("cores", self.cores),
            ("store", self.store),
            ("resident", self.resident),
            ("held", self.held),
            ("shared", self.shared),
            ("banked", self.banked),
            ("mma", self.mma),
            ("spill", self.spill),
        ]
    }
}
