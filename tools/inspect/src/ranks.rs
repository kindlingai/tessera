//! Ranks by their command-line names, with devices.

use tessera_lower::{Backend, Cluster, Device, Limits, Link, Pins, Platform, Zone, ZoneKind};
use tessera_platform::platform::Asked;
use tessera_sim::Tag;

/// One rank: its hardware and its weight of split axes.
#[derive(Clone, Debug, PartialEq)]
pub struct Spec {
    pub name: String,
    pub backend: &'static str,
    pub arch: String,
    pub part: u32,
    /// Bytes a second to its memory.
    pub bandwidth: f64,
}

/// Machines this tool knows by name.
const PRESETS: [(&str, &str, &str, f64); 4] = [
    ("strix", "rocm", "gfx1151", 256e9),
    ("gb10", "cuda", "sm_121", 273e9),
    ("m3", "metal", "apple9", 100e9),
    ("cpu", "cpu", "cpu", 50e9),
];

/// The zone that runs native kernels on the host.
const CPU: &str = "cpu";

/// `strix:3` or `cuda/sm_90:1`. The part defaults to one.
pub fn spec(text: &str) -> Option<Spec> {
    let (machine, part) = match text.rsplit_once(':') {
        Some((m, p)) => (m, p.parse().ok().filter(|p| *p > 0)?),
        None => (text, 1),
    };
    if let Some((name, backend, arch, bandwidth)) = PRESETS.iter().find(|p| p.0 == machine).copied()
    {
        return Some(Spec {
            name: name.into(),
            backend,
            arch: arch.into(),
            part,
            bandwidth,
        });
    }
    let (backend, arch) = machine.split_once('/')?;
    let gpu = Backend::named(backend).map(Backend::name);
    let backend = gpu.or((backend == CPU).then_some(CPU))?;
    Some(Spec {
        name: machine.into(),
        backend,
        arch: arch.into(),
        part,
        bandwidth: 100e9,
    })
}

impl Spec {
    pub fn kind(&self) -> ZoneKind {
        match self.backend {
            CPU => ZoneKind::Cpu,
            _ => ZoneKind::Gpu,
        }
    }

    /// The hardware from a driver's report, with no device.
    pub fn platform(&self) -> Platform {
        if self.kind() == ZoneKind::Cpu {
            return Platform::host(Limits::host(8));
        }
        Platform::new(Asked {
            backend: self.backend,
            arch: self.arch.clone(),
            name: self.name.clone(),
            driver: "plan only".into(),
            limits: Limits::assumed(),
            units: 0,
            bandwidth: self.bandwidth,
            integrated: true,
            memory: 0,
        })
    }

    /// The GPU its kernels compile for, if any.
    pub fn gpu(&self) -> Option<Backend> {
        Backend::named(self.backend)
    }
}

/// A link between every pair of ranks.
#[derive(Clone, Copy, Debug)]
pub struct Wire {
    pub latency: f64,
    pub rate: f64,
}

/// One zone per rank, each linked to every other.
fn cluster(specs: &[Spec], wire: Wire) -> Cluster {
    let mut c = Cluster::one(&specs[0].name);
    c.zones[0].kind = specs[0].kind();
    for s in &specs[1..] {
        c.add(Zone {
            name: s.name.clone(),
            kind: s.kind(),
            parent: None,
        });
    }
    for from in 0..specs.len() {
        for to in (0..specs.len()).filter(|t| *t != from) {
            c.links.push(Link {
                from,
                to,
                latency: wire.latency,
                rate: wire.rate,
                tag: Tag::Assumed,
            });
        }
    }
    c
}

/// The device rank `rank` lowers for, on its own hardware.
pub fn device(specs: &[Spec], rank: usize, wire: Wire, pins: &Pins) -> Device {
    let parts = specs.iter().map(|s| s.part).collect();
    Device::default()
        .on(specs[rank].platform())
        .pinned(pins.clone())
        .zoned(cluster(specs, wire))
        .ranked((0..specs.len() as u32).collect())
        .uneven(parts)
        .as_rank(rank as u32)
}
