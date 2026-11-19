//! Zones, where steps run and buffers lie, and their links.

use crate::Tag;
use serde::{Deserialize, Serialize};

/// The hardware a zone is.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ZoneKind {
    Gpu,
    Cpu,
    /// A box holding other zones.
    Host,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Zone {
    pub name: String,
    pub kind: ZoneKind,
    /// The zone this one sits inside.
    pub parent: Option<usize>,
}

/// One way bytes get from one zone to another.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct Link {
    pub from: usize,
    pub to: usize,
    /// Seconds a move takes before its bytes count.
    pub latency: f64,
    /// Bytes a second once moving.
    pub rate: f64,
    pub tag: Tag,
}

impl Link {
    pub fn seconds(&self, bytes: f64) -> f64 {
        self.latency + bytes / self.rate
    }
}

/// The zones a plan may run over, and their links.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Cluster {
    pub zones: Vec<Zone>,
    pub links: Vec<Link>,
}

impl Default for Cluster {
    fn default() -> Self {
        Self::one("gpu")
    }
}

impl Cluster {
    /// A single GPU: where every plan ran before zones.
    pub fn one(name: &str) -> Self {
        let gpu = Zone {
            name: name.into(),
            kind: ZoneKind::Gpu,
            parent: None,
        };
        Self {
            zones: vec![gpu],
            links: Vec::new(),
        }
    }

    /// A single CPU, for native kernels.
    pub fn cpu(name: &str) -> Self {
        let mut c = Self::one(name);
        c.zones[0].kind = ZoneKind::Cpu;
        c
    }

    /// Two GPUs linked both ways, at assumed speeds.
    pub fn two(a: &str, b: &str) -> Self {
        let mut cluster = Self::one(a);
        let b = cluster.add(Zone {
            name: b.into(),
            kind: ZoneKind::Gpu,
            parent: None,
        });
        for (from, to) in [(0, b), (b, 0)] {
            cluster.links.push(Link {
                from,
                to,
                latency: 1e-5,
                rate: 1e10,
                tag: Tag::Assumed,
            });
        }
        cluster
    }

    /// `ranks` zones of `kind`, each linked to every other.
    pub fn mesh(ranks: usize, kind: ZoneKind) -> Self {
        let mut cluster = Self::one("rank0");
        cluster.zones[0].kind = kind;
        for r in 1..ranks {
            cluster.add(Zone {
                name: format!("rank{r}"),
                kind,
                parent: None,
            });
        }
        for from in 0..ranks {
            for to in (0..ranks).filter(|t| *t != from) {
                cluster.links.push(Link {
                    from,
                    to,
                    latency: 1e-4,
                    rate: 1e9,
                    tag: Tag::Assumed,
                });
            }
        }
        cluster
    }

    /// Adds a zone. Returns its index.
    pub fn add(&mut self, zone: Zone) -> usize {
        self.zones.push(zone);
        self.zones.len() - 1
    }

    pub fn kind(&self, zone: usize) -> ZoneKind {
        self.zones[zone].kind
    }

    /// A ring sum over `zones`. The slowest link paces rounds.
    pub fn all_reduce(&self, zones: &[usize], bytes: f64) -> Option<f64> {
        let n = zones.len();
        if n < 2 {
            return Some(0.0);
        }
        let share = bytes / n as f64;
        let mut round = 0.0f64;
        for (i, &from) in zones.iter().enumerate() {
            let to = zones[(i + 1) % n];
            round = round.max(self.transfer(from, to, share)?);
        }
        Some(2.0 * (n - 1) as f64 * round)
    }

    /// Seconds to move `bytes`, over the fastest link.
    pub fn transfer(&self, from: usize, to: usize, bytes: f64) -> Option<f64> {
        if from == to {
            return Some(0.0);
        }
        let joins = |l: &&Link| l.from == from && l.to == to;
        let times = self.links.iter().filter(joins).map(|l| l.seconds(bytes));
        times.min_by(f64::total_cmp)
    }
}
