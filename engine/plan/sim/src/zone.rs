//! Zones, where steps run and buffers lie, and their links.

use crate::Tag;
use serde::{Deserialize, Serialize};

/// The hardware a zone is.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
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

    /// Adds a zone. Returns its index.
    pub fn add(&mut self, zone: Zone) -> usize {
        self.zones.push(zone);
        self.zones.len() - 1
    }

    pub fn kind(&self, zone: usize) -> ZoneKind {
        self.zones[zone].kind
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
