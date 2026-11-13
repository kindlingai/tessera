//! Small kernels that expose one hardware limit each.

use codegen::prelude::*;
use lower::form::U32_FORM;
use lower::{
    Arg, Cluster, DType, Error, Kernel, KernelRef, Launch, Link, Plan, Run, Settings, Shape, Slot,
    Source, Step, Tensor, Zone, ZoneKind, open,
};
use sim::{Fit, Tag};
use std::time::Instant;

/// The work a probe's threads repeat `trips` times.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A multiply-add, away from memory.
    Spin,
    /// An exponential, away from memory.
    Curve,
    /// `depth` loads at once, threads interleaved through a buffer.
    Read { depth: usize },
    /// A load, each thread along its own stretch.
    Scan,
    /// A store, threads interleaved through a buffer.
    Write,
    /// A load whose address the last load gave.
    Chase,
    /// Tile multiplies on registers, away from memory.
    Tile,
}

/// One launch of a probe kernel.
#[derive(Clone, Copy, Debug)]
pub struct Probe {
    pub kind: Kind,
    pub threads: usize,
    pub trips: usize,
}

/// A probe and the seconds one launch took.
#[derive(Clone, Copy, Debug)]
pub struct Reading {
    pub probe: Probe,
    pub seconds: f64,
}

/// Each probe stops its idle threads first.
#[kernel]
fn probe_spin(x: Buf<F32>, mut out: BufMut<F32>, threads: u32, trips: u32, len: u32) {
    if gid() >= threads {
        return;
    }
    let mut v = x[(gid() % 16) as u64];
    for i in 0..trips {
        v = v * 0.999 + 0.001;
    }
    out[gid() as u64] = v;
}

#[kernel]
fn probe_curve(x: Buf<F32>, mut out: BufMut<F32>, threads: u32, trips: u32, len: u32) {
    if gid() >= threads {
        return;
    }
    let mut v = x[(gid() % 16) as u64];
    for i in 0..trips {
        v = exp(-v);
    }
    out[gid() as u64] = v;
}

/// `depth` loads a trip, threads interleaved through the buffer.
#[kernel]
fn probe_read(
    x: Buf<F32>,
    mut out: BufMut<F32>,
    threads: u32,
    trips: u32,
    len: u32,
    #[build] depth: u32,
) {
    if gid() >= threads {
        return;
    }
    let mut acc = 0.0f32;
    for i in 0..trips {
        let at = i as u64 * depth as u64 * threads as u64 + gid() as u64;
        for d in 0..depth {
            acc += x[at + threads as u64 * d as u64];
        }
    }
    out[gid() as u64] = acc;
}

#[kernel]
fn probe_scan(x: Buf<F32>, mut out: BufMut<F32>, threads: u32, trips: u32, len: u32) {
    if gid() >= threads {
        return;
    }
    let at = gid() as u64 * trips as u64;
    let mut acc = 0.0f32;
    for i in 0..trips {
        acc += x[at + i as u64];
    }
    out[gid() as u64] = acc;
}

#[kernel]
fn probe_write(x: Buf<F32>, mut out: BufMut<F32>, threads: u32, trips: u32, len: u32) {
    if gid() >= threads {
        return;
    }
    for i in 0..trips {
        out[i as u64 * threads as u64 + gid() as u64] = i as f32;
    }
    if trips == 0 {
        out[gid() as u64] = x[(gid() % 16) as u64];
    }
}

/// Each load's address comes from the last.
#[kernel]
fn probe_chase(x: Buf<U32>, mut out: BufMut<U32>, threads: u32, trips: u32, len: u32) {
    if gid() >= threads {
        return;
    }
    let mut at = (gid() * 2654435761) % len;
    for i in 0..trips {
        at = x[at as u64];
    }
    out[gid() as u64] = at;
}

/// Repeats a SIMD group's 32x32 block over 32 staged columns.
#[kernel]
fn probe_tile(x: Buf<F32>, mut out: BufMut<F32>, threads: u32, trips: u32, len: u32) {
    if gid() >= threads {
        return;
    }
    let mut s = shared::<f32>(64 * 36);
    for i in (lid()..64 * 36).by(256) {
        s[i] = x[i as u64];
    }
    barrier();
    let mut acc = tiles(32, 32);
    for it in 0..trips {
        acc.block(&s, 0, &s, 32 * 36, 36, 32);
    }
    let mut sum = 0.0f32;
    acc.each(|r, c, v| {
        sum += v;
    });
    out[gid() as u64] = sum;
}

impl Probe {
    /// Elements the probe reads, then elements it writes.
    fn buffers(&self) -> [usize; 2] {
        let all = self.threads * self.trips;
        match self.kind {
            Kind::Spin | Kind::Curve => [16, self.threads],
            Kind::Read { depth } => [all * depth, self.threads],
            Kind::Scan => [all, self.threads],
            Kind::Write => [16, all.max(self.threads)],
            Kind::Chase => [CHASE, self.threads],
            Kind::Tile => [64 * 36, self.threads],
        }
    }

    /// A plan of this one launch, reading buffer 0.
    fn plan(&self) -> Plan {
        let [read, write] = self.buffers();
        let slot = Slot::floats;
        // A chase reads a table of indices.
        let input = match self.kind {
            Kind::Chase => Slot {
                form: U32_FORM,
                ..slot(0, read)
            },
            _ => slot(0, read),
        };
        let step = Step {
            kernel: KernelRef::new(*self),
            rule: "probe",
            reads: vec![input],
            write: slot(1, write),
            covers: Vec::new(),
            zone: 0,
        };
        Plan {
            buffers: vec![4 * read, 4 * write],
            steps: vec![step],
            inputs: vec![input],
            ..Plan::default()
        }
    }
}

impl Kernel for Probe {
    fn name(&self) -> &'static str {
        "probe"
    }

    fn source(&self) -> Source {
        let mut body = match self.kind {
            Kind::Spin => probe_spin_ir(),
            Kind::Curve => probe_curve_ir(),
            Kind::Read { depth } => probe_read_ir(depth as u32),
            Kind::Scan => probe_scan_ir(),
            Kind::Write => probe_write_ir(),
            Kind::Chase => probe_chase_ir(),
            Kind::Tile => probe_tile_ir(),
        };
        if let Kind::Read { depth } = self.kind {
            body.entry = format!("probe_read{depth}");
        }
        Source::Made(body)
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        let sizes = [self.threads, self.trips, step.reads[0].len].map(Arg::count);
        let buffers = [step.read(0), step.out()];
        vec![Launch {
            args: buffers.into_iter().chain(sizes).collect(),
            shape: Shape::Line(self.threads),
        }]
    }
}

/// Entries in a chase's table: 256 MB.
const CHASE: usize = 1 << 26;

/// Each entry names the next, in one cycle through all.
fn chain() -> Tensor {
    // An odd step and a power-of-two length visit every entry.
    let step = 40_503 * 65_537 + 2;
    let next = |i: usize| (i.wrapping_add(step) % CHASE) as u32;
    Tensor::u32(vec![CHASE], (0..CHASE).map(next).collect())
}

/// Times each probe on this GPU. The fastest run counts.
pub fn read(probes: &[Probe], settings: &Settings) -> Result<Vec<Reading>, Error> {
    let mut runner: Option<Box<dyn Run>> = None;
    let mut table: Option<Tensor> = None;
    let mut readings = Vec::with_capacity(probes.len());
    for &probe in probes {
        let runner = match &mut runner {
            Some(runner) => {
                runner.replan(&probe.plan())?;
                runner
            }
            None => runner.insert(open(probe.plan(), settings)?),
        };
        if probe.kind == Kind::Chase {
            runner.feed(&[table.get_or_insert_with(chain).view()])?;
        }
        // The first run touches every page.
        runner.submit(0..1)?;
        let mut seconds = runner.repeat(0, 1)?;
        // A short launch is noisy, so it gets more tries.
        let tries = if seconds < 1e-3 { 64 } else { 2 };
        for _ in 0..tries {
            seconds = seconds.min(runner.repeat(0, 1)?);
        }
        readings.push(Reading { probe, seconds });
    }
    Ok(readings)
}

/// Bytes that stay in cache, and bytes that cannot.
const SMALL: usize = 1 << 22;
const LARGE: usize = 1 << 29;

/// Elements the host link probe moves at most: 64 MB.
const MOVED: usize = 1 << 24;

/// The fastest of `tries` runs of `go`, in seconds.
fn least(tries: usize, mut go: impl FnMut() -> Result<(), Error>) -> Result<f64, Error> {
    let mut best = f64::INFINITY;
    for _ in 0..tries {
        let start = Instant::now();
        go()?;
        best = best.min(start.elapsed().as_secs_f64());
    }
    Ok(best)
}

/// Times moves between this GPU and the host, both ways.
pub fn links(settings: &Settings) -> Result<Cluster, Error> {
    let slot = |len| Slot::floats(0, len);
    let plan = Plan {
        buffers: vec![4 * MOVED],
        inputs: vec![slot(MOVED)],
        ..Plan::default()
    };
    let mut runner = open(plan, settings)?;
    let zeros = Tensor::zeros(vec![MOVED], DType::F32);
    let mut cluster = Cluster::one(&runner.device());
    let host = cluster.add(Zone {
        name: "host".into(),
        kind: ZoneKind::Cpu,
        parent: None,
    });
    for (from, to) in [(host, 0), (0, host)] {
        let mut go = |len: usize| match from == host {
            true => runner.write(slot(len), zeros.view().slice(0, 0, len)),
            false => runner.read(slot(len)).map(drop),
        };
        // One element shows the latency, and the rest the rate.
        let latency = least(64, || go(1))?;
        let large = least(4, || go(MOVED))?;
        let rate = (4 * MOVED) as f64 / (large - latency).max(1e-9);
        cluster.links.push(Link {
            from,
            to,
            latency,
            rate,
            tag: Tag::Measured,
        });
    }
    Ok(cluster)
}

impl Reading {
    /// Bytes the probe moved through memory.
    fn bytes(&self) -> usize {
        let all = 4 * self.probe.threads * self.probe.trips;
        match self.probe.kind {
            Kind::Read { depth } => all * depth,
            Kind::Scan | Kind::Write => all,
            Kind::Spin | Kind::Curve | Kind::Chase | Kind::Tile => 0,
        }
    }

    fn per_trip(&self) -> f64 {
        self.seconds / self.probe.trips.max(1) as f64
    }
}

/// This device's constants, from `suite`'s readings.
pub fn fit(readings: &[Reading]) -> Fit {
    let of = |kind: Kind, threads: usize, bytes: usize| {
        let same = |r: &&Reading| {
            let p = r.probe;
            p.kind == kind && p.threads == threads && (bytes == 0 || r.bytes() == bytes)
        };
        *readings.iter().find(same).expect("the suite holds it")
    };
    let spin = |threads, idle: bool| {
        let same = |r: &&Reading| {
            let p = r.probe;
            p.kind == Kind::Spin && p.threads == threads && (p.trips == 0) == idle
        };
        *readings.iter().find(same).expect("the suite holds it")
    };
    let fastest = |keep: &dyn Fn(&Reading) -> bool, work: &dyn Fn(&Reading) -> f64| {
        let rates = readings
            .iter()
            .filter(|r| keep(r))
            .map(|r| work(r) / r.seconds);
        rates.fold(0.0, f64::max)
    };
    let bytes = |r: &Reading| r.bytes() as f64;
    let reads = |r: &Reading| matches!(r.probe.kind, Kind::Read { .. });
    let scans = |r: &Reading| r.probe.kind == Kind::Scan;
    let floor = spin(256, true).seconds.min(spin(1, true).seconds);
    let step = (spin(256, false).seconds - floor) / spin(256, false).probe.trips as f64;
    let spawn = (spin(65_536, true).seconds - floor).max(0.0) / 65_536.0;
    let trips = |r: &Reading| (r.probe.threads * r.probe.trips) as f64;
    let wait = of(Kind::Read { depth: 1 }, 256, LARGE).per_trip();
    let near = (of(Kind::Scan, 256, SMALL).per_trip() - step).max(0.0);
    let curve = (of(Kind::Curve, 256, 0).seconds - floor) / 4096.0;
    // A SIMD group's trip does 65536 tile flops.
    let tiles = |r: &Reading| (r.probe.threads / 32 * r.probe.trips) as f64 * 65_536.0;
    Fit {
        floor,
        spawn,
        step,
        curve,
        rate: fastest(&|r| r.probe.kind == Kind::Spin, &trips),
        wait,
        near,
        // The small buffer is the one known to stay cached.
        cache: SMALL as f64,
        cached: fastest(&|r| (reads(r) || scans(r)) && r.bytes() == SMALL, &bytes),
        read: fastest(&|r| reads(r) && r.bytes() == LARGE, &bytes),
        scan: fastest(&|r| scans(r) && r.bytes() == LARGE, &bytes),
        write: fastest(&|r| r.probe.kind == Kind::Write, &bytes),
        tile: fastest(&|r| r.probe.kind == Kind::Tile, &tiles),
    }
}

/// The probes that size up a device's limits.
pub fn suite() -> Vec<Probe> {
    let probe = |kind, threads, trips| Probe {
        kind,
        threads,
        trips,
    };
    let mut all = Vec::new();
    for threads in [1, 256, 65_536] {
        all.push(probe(Kind::Spin, threads, 0));
    }
    for threads in [1, 256, 4096, 65_536, 1 << 20] {
        all.push(probe(Kind::Spin, threads, 4096));
    }
    all.push(probe(Kind::Curve, 256, 4096));
    for bytes in [SMALL, LARGE] {
        for depth in [1, 4] {
            for threads in [256, 4096, 65_536] {
                let trips = bytes / 4 / depth / threads;
                all.push(probe(Kind::Read { depth }, threads, trips));
            }
        }
        for threads in [256, 4096, 65_536] {
            all.push(probe(Kind::Scan, threads, bytes / 4 / threads));
        }
    }
    for threads in [4096, 65_536, 1 << 20] {
        all.push(probe(Kind::Write, threads, (1 << 26) / threads));
    }
    for (threads, trips) in [(1, 100_000), (256, 10_000), (65_536, 1000)] {
        all.push(probe(Kind::Chase, threads, trips));
    }
    for threads in [256, 4096, 65_536] {
        all.push(probe(Kind::Tile, threads, 256));
    }
    all
}
