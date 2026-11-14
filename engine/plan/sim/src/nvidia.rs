//! NVIDIA GPUs: warps on multiprocessors that share a memory bus.

use crate::{Chip, Const, Estimate, Limit, Load};

/// One chip's multiprocessors and memory system.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Sm {
    pub sms: Const,
    /// Warp schedulers per multiprocessor, one issue a cycle each.
    pub schedulers: Const,
    /// 32-bit registers per multiprocessor.
    pub registers: Const,
    /// Registers a thread is assumed to hold.
    pub held: Const,
    /// Threads one multiprocessor keeps resident.
    pub threads: Const,
    /// Thread groups one multiprocessor keeps resident.
    pub groups: Const,
    pub clock: Const,
    /// Cycles one warp instruction holds each pipe.
    pub float: Const,
    pub int: Const,
    pub special: Const,
    pub memory: Const,
    /// Instructions a scheduler issues per cycle, at most.
    pub issue: Const,
    /// Cycles between one lone warp's issues.
    pub lone: Const,
    /// Seconds one thread takes per dependent multiply-add.
    pub chain: Const,
    /// Seconds a warp's load takes when its threads share lines.
    pub gather: Const,
    /// The same when all 32 threads read separate lines.
    pub scatter: Const,
    /// Bytes a second out of memory, with enough in flight.
    pub read: Const,
    /// Bytes a second when each thread reads its own stretch.
    pub scan: Const,
    pub write: Const,
    /// How much reads and writes slow each other.
    pub mixed: Const,
    /// Seconds a load from memory takes.
    pub wait: Const,
    /// Bytes of cache that stay usable.
    pub cache: Const,
    /// Bytes a second from cache to all multiprocessors.
    pub cached: Const,
    /// Seconds a load from cache takes.
    pub near: Const,
    /// Seconds from one launch's end to the next's start.
    pub floor: Const,
    /// Seconds a memory-bound launch takes to fill and drain.
    pub fill: Const,
    /// Peak float32 multiply-adds, in flops a second.
    pub fp32: Const,
    /// Peak fp16 tensor-core rate, in flops a second.
    pub fp16: Const,
    /// Peak 8-bit block-scaled tensor-core rate.
    pub fp8: Const,
    /// One, where one instruction widens an e4m3.
    pub cvt8: Const,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Nvidia {
    pub name: String,
    pub sm: Sm,
}

/// The GB10, as in a DGX Spark.
pub(crate) fn gb10() -> Nvidia {
    let (m, i, a) = (Const::measured, Const::inferred, Const::assumed);
    let sm = Sm {
        sms: m(48.0, "reported by the driver"),
        schedulers: a(4.0, "sm_12x: four warp schedulers, one issue a cycle each"),
        registers: a(65536.0, "sm_12x register file"),
        held: a(32.0, "a small generated kernel; no disassembly read"),
        threads: a(1536.0, "sm_120 and sm_121 limit"),
        groups: a(32.0, "sm_12x"),
        clock: m(2.1e9, "mean clock under sustained load; 2.4e9 is the cap"),
        float: a(1.0, "128 float32 lanes: one cycle per warp instruction"),
        int: m(2.0, "an integer decode loop: 1.85 cycles per instruction"),
        special: a(8.0, "16 special-function lanes"),
        memory: a(1.0, "one issue slot per load or store"),
        issue: a(1.0, "ceiling once enough warps are resident"),
        lone: m(2.8, "one warp per scheduler on a 62-instruction loop"),
        chain: m(3.6e-9, "probe: a multiply-add loop in one thread"),
        gather: m(17e-9, "probe: 256 threads interleaved through 4 MB"),
        scatter: m(123e-9, "probe: 256 threads, each along its own 16 KB"),
        read: m(
            255e9,
            "streaming 16-byte loads, 8 KB or more in flight per SM",
        ),
        scan: m(
            132e9,
            "probe: 4096 threads, each along its own 128 KB of 512 MB",
        ),
        write: m(193e9, "streaming 16-byte stores"),
        mixed: m(1.07, "a copy runs 109 + 109 GB/s"),
        wait: m(433e-9, "pointer chase over 512 MB at 2.1 GHz"),
        cache: m(22e6, "re-reads stay at cache speed up to about 22 MB of 24"),
        cached: m(1460e9, "a 20 MB buffer re-read at 2.1 GHz"),
        near: m(162e-9, "pointer chase over 1 MB at 2.1 GHz"),
        floor: m(2.1e-6, "empty launches back to back, 48 groups"),
        fill: m(
            3.0e-6,
            "single-row products sit this far above their byte floor",
        ),
        fp32: i(25.8e12, "128 lanes x 48 SMs x 2.1 GHz x 2"),
        fp16: m(99.7e12, "m16n8k16 loop, 16 accumulators, at 2.1 GHz"),
        fp8: m(203.6e12, "m16n8k32 block-scaled 8-bit loop at 2.1 GHz"),
        cvt8: a(1.0, "sm_89 on: cvt.rn.f16x2.e4m3x2"),
    };
    Nvidia {
        name: "cuda NVIDIA GB10".into(),
        sm,
    }
}

impl Chip for Nvidia {
    fn name(&self) -> &str {
        &self.name
    }

    fn estimate(&self, l: &Load) -> Estimate {
        let s = &self.sm;
        let group = l.group.max(1) as f64;
        let warps = (group / 32.0).ceil();
        let grid = (l.threads as f64 / group).ceil().max(1.0);
        // How many groups one multiprocessor runs at once.
        let by_registers = (s.registers.value / (s.held.value * warps * 32.0)).floor();
        let by_threads = (s.threads.value / (warps * 32.0)).floor();
        let fit = by_registers.min(by_threads).min(s.groups.value).max(1.0);
        let per_sm = (grid / s.sms.value).ceil();
        let rounds = (per_sm / fit).ceil();
        let resident = fit.min(per_sm);
        let busy = s.sms.value.min(grid);
        let per_scheduler = resident * warps / s.schedulers.value;
        let t = &l.trip.priced(s.cvt8.value > 0.0);
        let pipes = [
            (t.float + 1.0, s.float.value),
            (t.int, s.int.value),
            (t.special, s.special.value),
            (t.loads + t.stores, s.memory.value),
        ];
        let count: f64 = pipes.iter().map(|(n, _)| n * l.trips).sum();
        let rate = s.issue.value.min(per_scheduler / s.lone.value);
        let issue = count * per_scheduler / rate;
        let held = |(n, cycles): &(f64, f64)| n * l.trips * per_scheduler * cycles;
        let pipe = pipes.iter().map(held).fold(0.0, f64::max);
        // Little's law, over the multiprocessors at work.
        let flight = resident * warps * 32.0 * l.flight;
        let through = |peak: f64, wait: f64| match flight > 0.0 {
            true => peak.min(flight * busy / wait),
            false => peak,
        };
        // A warp's load costs more the more lines it touches.
        let apart = (l.threads.min(32) as f64 - 1.0) / 31.0;
        let stall = match t.fresh < 1.0 {
            true => s.gather.value + (s.scatter.value - s.gather.value) * apart,
            false => s.gather.value,
        };
        let chain = l.trips * (t.float * s.chain.value + t.loads * stall);
        let cold = |bytes: &&f64| **bytes > s.cache.value;
        let far: f64 = l.reads.iter().filter(cold).sum();
        let peak = if t.fresh < 1.0 { s.scan } else { s.read };
        let stream = through(peak.value, s.wait.value);
        let mut moved = far / stream + l.written / s.write.value;
        if far > 0.0 && l.written > 0.0 {
            let share = (2.0 * l.written / (far + l.written)).min(1.0);
            moved *= 1.0 + (s.mixed.value - 1.0) * share;
        }
        let cached = l.loaded() / through(s.cached.value, s.near.value);
        let seconds = |cycles: f64| rounds * cycles / s.clock.value;
        let limits = [
            (Limit::Issue, seconds(issue)),
            (Limit::Pipe, seconds(pipe)),
            (Limit::Chain, chain),
            (Limit::Read, moved),
            (Limit::Cache, cached),
        ];
        let capped = far > 0.0 && stream >= peak.value;
        let start = s.floor.value + if capped { s.fill.value } else { 0.0 };
        let mut slowest = Estimate::slowest(start, &limits);
        if slowest.limit == Limit::Read && l.written > far {
            slowest.limit = Limit::Write;
        }
        slowest
    }

    /// Float ops at the float32 peak, bytes at stream rates.
    fn bound(&self, l: &Load) -> f64 {
        let s = &self.sm;
        let rate = |bytes: &f64| match *bytes > s.cache.value {
            true => bytes / s.read.value,
            false => bytes / s.cached.value,
        };
        let moved: f64 = l.reads.iter().map(rate).sum::<f64>() + l.written / s.write.value;
        // A multiply-add counts as two flops.
        let floats = l.threads as f64 * l.trips * l.trip.priced(s.cvt8.value > 0.0).float;
        s.floor.value + (2.0 * floats / s.fp32.value).max(moved)
    }

    fn constants(&self) -> Vec<(&'static str, Const)> {
        let s = &self.sm;
        vec![
            ("sms", s.sms),
            ("schedulers", s.schedulers),
            ("registers", s.registers),
            ("held", s.held),
            ("threads", s.threads),
            ("groups", s.groups),
            ("clock", s.clock),
            ("float", s.float),
            ("int", s.int),
            ("special", s.special),
            ("memory", s.memory),
            ("issue", s.issue),
            ("lone", s.lone),
            ("chain", s.chain),
            ("gather", s.gather),
            ("scatter", s.scatter),
            ("scan", s.scan),
            ("read", s.read),
            ("write", s.write),
            ("mixed", s.mixed),
            ("wait", s.wait),
            ("cache", s.cache),
            ("cached", s.cached),
            ("near", s.near),
            ("floor", s.floor),
            ("fill", s.fill),
            ("fp32", s.fp32),
            ("fp16", s.fp16),
            ("fp8", s.fp8),
            ("cvt8", s.cvt8),
        ]
    }
}
