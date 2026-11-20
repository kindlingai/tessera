//! Which ranks a rank reaches, and over which ports.

use crate::Error;

/// Slots start with the length, then pad to 16.
pub(crate) const HEADER: usize = 16;

/// An RDMA device and the GID index to send from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Port {
    pub device: String,
    pub gid: u32,
}

/// How the ranks are cabled, one lane per port.
#[derive(Clone, Debug)]
pub enum Fabric {
    /// Every rank reaches every other over the same ports.
    Mesh(Vec<Port>),
    /// Rank r reaches r-1 over `prev` and r+1 over `next`.
    Ring { prev: Vec<Port>, next: Vec<Port> },
}

/// One link to a peer: a queue pair per lane.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Channel {
    pub peer: u32,
    /// Port index per lane. Empty for the rank itself.
    pub ports: Vec<usize>,
    /// The peer's channel that faces this rank.
    pub meets: usize,
}

/// A rank's channels, its send order and its relay.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Wiring {
    pub channels: Vec<Channel>,
    /// Channel indexes, in the order a round posts to them.
    pub sends: Vec<usize>,
    /// Forward what arrives on one channel over another.
    pub relay: Option<(usize, usize)>,
}

impl Fabric {
    /// Every port, in lane order.
    pub fn ports(&self) -> Vec<Port> {
        match self {
            Fabric::Mesh(ports) => ports.clone(),
            Fabric::Ring { prev, next } => [prev.clone(), next.clone()].concat(),
        }
    }

    pub fn lanes(&self) -> usize {
        match self {
            Fabric::Mesh(ports) => ports.len(),
            Fabric::Ring { prev, .. } => prev.len(),
        }
    }

    /// The wiring of `rank` among `ranks`.
    pub(crate) fn wiring(&self, rank: u32, ranks: u32) -> Result<Wiring, Error> {
        let lanes = self.lanes();
        let fits = match self {
            Fabric::Mesh(_) => ranks >= 2,
            Fabric::Ring { prev, next } => (2..=4).contains(&ranks) && prev.len() == next.len(),
        };
        if !fits || lanes == 0 || rank >= ranks {
            let what = format!("rank {rank} of {ranks} over {lanes} lanes");
            return Err(Error::Fit(what));
        }
        Ok(match self {
            Fabric::Mesh(_) => {
                let channel = |peer: u32| Channel {
                    peer,
                    ports: match peer == rank {
                        true => Vec::new(),
                        false => (0..lanes).collect(),
                    },
                    meets: rank as usize,
                };
                Wiring {
                    channels: (0..ranks).map(channel).collect(),
                    sends: (1..ranks).map(|s| ((rank + s) % ranks) as usize).collect(),
                    relay: None,
                }
            }
            Fabric::Ring { .. } => {
                let prev = Channel {
                    peer: (rank + ranks - 1) % ranks,
                    ports: (0..lanes).collect(),
                    meets: 1,
                };
                let next = Channel {
                    peer: (rank + 1) % ranks,
                    ports: (lanes..2 * lanes).collect(),
                    meets: 0,
                };
                Wiring {
                    channels: vec![prev, next],
                    sends: match ranks {
                        2 => vec![1],
                        _ => vec![1, 0],
                    },
                    relay: (ranks == 4).then_some((0, 1)),
                }
            }
        })
    }
}

/// `total` bytes over `lanes`, as (start, length) in 16s.
pub(crate) fn parts(total: usize, lanes: usize) -> Vec<(usize, usize)> {
    let step = (total.div_ceil(lanes) + 15) & !15;
    (0..lanes)
        .map(|lane| {
            let start = (lane * step).min(total);
            (start, ((lane + 1) * step).min(total) - start)
        })
        .collect()
}

/// A rank's message to the others about one queue pair.
#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub(crate) struct Lane {
    pub qpn: u32,
    pub mtu: u32,
    /// The remote key of each region, on this lane's port.
    pub rkeys: Vec<u32>,
    pub gid: [u8; 16],
}

/// `Peer`'s first word. Change it with the format.
const VERSION: u32 = 1;

/// A rank's regions and queue pairs, for its peers.
#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub(crate) struct Peer {
    /// Each region's address.
    pub regions: Vec<u64>,
    /// Per channel, per lane.
    pub channels: Vec<Vec<Lane>>,
}

impl Peer {
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut out = VERSION.to_le_bytes().to_vec();
        out.extend_from_slice(&(self.regions.len() as u32).to_le_bytes());
        for region in &self.regions {
            out.extend_from_slice(&region.to_le_bytes());
        }
        out.extend_from_slice(&(self.channels.len() as u32).to_le_bytes());
        for lanes in &self.channels {
            out.extend_from_slice(&(lanes.len() as u32).to_le_bytes());
            for l in lanes {
                let words = [l.qpn, l.mtu, l.rkeys.len() as u32]
                    .into_iter()
                    .chain(l.rkeys.iter().copied());
                for word in words {
                    out.extend_from_slice(&word.to_le_bytes());
                }
                out.extend_from_slice(&l.gid);
            }
        }
        out
    }

    /// Decodes a peer's bytes, or says why it cannot.
    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut at = Cursor(bytes);
        match at.u32() {
            Some(VERSION) => {}
            Some(v) => return Err(format!("details version {v}, expected {VERSION}")),
            None => return Err("no details version".into()),
        }
        Self::fields(at).ok_or_else(|| "details do not decode".into())
    }

    fn fields(mut at: Cursor) -> Option<Self> {
        let regions = (0..at.u32()?)
            .map(|_| at.u64())
            .collect::<Option<Vec<_>>>()?;
        let mut channels = Vec::new();
        for _ in 0..at.u32()? {
            let mut lanes = Vec::new();
            for _ in 0..at.u32()? {
                let (qpn, mtu) = (at.u32()?, at.u32()?);
                let rkeys = (0..at.u32()?)
                    .map(|_| at.u32())
                    .collect::<Option<Vec<_>>>()?;
                let gid = at.take(16)?.try_into().ok()?;
                lanes.push(Lane {
                    qpn,
                    mtu,
                    rkeys,
                    gid,
                });
            }
            channels.push(lanes);
        }
        at.0.is_empty().then_some(Self { regions, channels })
    }
}

struct Cursor<'a>(&'a [u8]);

impl Cursor<'_> {
    fn take(&mut self, n: usize) -> Option<&[u8]> {
        let (head, rest) = self.0.split_at_checked(n)?;
        self.0 = rest;
        Some(head)
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
}
