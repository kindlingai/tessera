//! A `Link` over RDMA: data, then count, then ordinal.

use crate::fabric::Fabric;
use crate::ibv::{self, SendWr, Sge, Wc};
use crate::link::{Link, Put, Span, fits};
use crate::reduce::Pinning;
use crate::verbs::{Engine, Region};
use crate::{Error, Transport};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// How long `reset` waits for completions.
const DRAIN: Duration = Duration::from_secs(10);
/// Completions per device per poll.
const BATCH: usize = 16;

/// One-sided writes into peers' regions. Needs a mesh fabric.
pub struct VerbsLink {
    rank: u32,
    ranks: u32,
    engine: Engine,
    sizes: Vec<usize>,
    /// Signaled work not yet complete.
    outstanding: u64,
}

// Each rank owns its queue pairs and regions.
unsafe impl Send for VerbsLink {}

impl VerbsLink {
    /// Rank `rank` of `ranks` over a mesh, with these regions.
    pub fn connect(
        rank: u32,
        ranks: u32,
        fabric: &Fabric,
        regions: &[usize],
        boot: &mut dyn Transport,
        pinning: Option<Arc<dyn Pinning>>,
    ) -> Result<Self, Error> {
        if !matches!(fabric, Fabric::Mesh(_)) {
            return Err(Error::Fit("one-sided puts need a mesh fabric".into()));
        }
        if regions.contains(&0) {
            return Err(Error::Fit("a region of 0 bytes".into()));
        }
        let pin = pinning.as_ref();
        let made: Result<Vec<Region>, Error> =
            regions.iter().map(|&n| Region::new(n, pin)).collect();
        let engine = Engine::connect(rank, ranks, fabric, 0, made?, boot)?;
        // Every peer must describe the same regions on every lane.
        let alike = |p: &crate::fabric::Peer| {
            let lanes = p.channels.iter().flatten();
            p.regions.len() == regions.len()
                && lanes.clone().all(|l| l.rkeys.len() == regions.len())
        };
        if let Some(r) = engine.peers.iter().position(|p| !alike(p)) {
            return Err(Error::Fit(format!("rank {r} describes other regions")));
        }
        Ok(VerbsLink {
            rank,
            ranks,
            engine,
            sizes: regions.to_vec(),
            outstanding: 0,
        })
    }

    /// Posts `chain` to `to`. The last write signals `tag`.
    fn post(&mut self, to: u32, chain: &mut [SendWr], tag: u64) -> Result<(), Error> {
        let n = chain.len();
        chain[n - 1].wr_id = tag;
        chain[n - 1].send_flags |= ibv::SEND_SIGNALED;
        for i in 0..n - 1 {
            let next: *mut SendWr = &mut chain[i + 1];
            chain[i].next = next;
        }
        let qp = self.lane(to)?.qp;
        let e = unsafe { self.engine.lib.post_send(qp, &mut chain[0]) };
        if e != 0 {
            let why = std::io::Error::from_raw_os_error(e);
            return Err(Error::Link(format!("post to rank {to}: {why}")));
        }
        self.outstanding += 1;
        Ok(())
    }

    /// The first lane's queue pair to `to`, on the mesh.
    fn lane(&self, to: u32) -> Result<&crate::verbs::Link, Error> {
        let links = self
            .engine
            .links
            .get(to as usize)
            .filter(|_| to != self.rank);
        let link = links.and_then(|l| l.first());
        link.ok_or_else(|| Error::Fit(format!("no queue pair to rank {to}")))
    }

    /// `at` in rank `to`'s regions: its address and remote key.
    fn remote(&self, to: u32, at: Span, align: usize) -> Result<(u64, u32), Error> {
        fits(&self.sizes, at, align)?;
        let peer = self.engine.peers.get(to as usize);
        let peer = peer.ok_or_else(|| Error::Fit(format!("no rank {to}")))?;
        let far = peer
            .channels
            .get(self.rank as usize)
            .and_then(|l| l.first());
        let far = far.ok_or_else(|| Error::Fit(format!("rank {to} has no lane here")))?;
        Ok((peer.regions[at.region] + at.at as u64, far.rkeys[at.region]))
    }

    fn word_at(&self, at: Span) -> Result<&AtomicU64, Error> {
        fits(&self.sizes, Span::new(at.region, at.at, 8), 8)?;
        let word = self.engine.regions[at.region].at(at.at);
        // Checked: in bounds and 8-byte aligned.
        Ok(unsafe { &*(word as *const AtomicU64) })
    }
}

impl Link for VerbsLink {
    fn rank(&self) -> u32 {
        self.rank
    }

    fn ranks(&self) -> u32 {
        self.ranks
    }

    fn put(&mut self, put: Put) -> Result<(), Error> {
        if put.from.len != put.into.len || put.flag.len != 16 {
            return Err(Error::Fit(format!("{put:?} has spans that differ")));
        }
        fits(&self.sizes, put.from, 1)?;
        let length = u32::try_from(put.from.len);
        let length = length.map_err(|_| Error::Fit(format!("a {}-byte put", put.from.len)))?;
        let (into, data_key) = self.remote(put.to, put.into, 1)?;
        let (flag, flag_key) = self.remote(put.to, put.flag, 8)?;
        let port = self.lane(put.to)?.port;
        let lkey = unsafe { (*self.engine.opened[port].mrs[put.from.region]).lkey };
        let from = self.engine.regions[put.from.region].at(put.from.at) as u64;
        let mut data = Sge {
            addr: from,
            length,
            lkey,
        };
        let [ordinal, count] = put.words;
        let mut count_sge = Sge {
            addr: &count as *const u64 as u64,
            length: 8,
            lkey: 0,
        };
        let mut ordinal_sge = Sge {
            addr: &ordinal as *const u64 as u64,
            length: 8,
            lkey: 0,
        };
        let mut chain = Vec::with_capacity(3);
        if put.from.len > 0 {
            chain.push(SendWr::write(&mut data, into, data_key, 0));
        }
        chain.push(SendWr::write(
            &mut count_sge,
            flag + 8,
            flag_key,
            ibv::SEND_INLINE,
        ));
        chain.push(SendWr::write(
            &mut ordinal_sge,
            flag,
            flag_key,
            ibv::SEND_INLINE,
        ));
        self.post(put.to, &mut chain, put.tag)
    }

    fn word(&mut self, to: u32, at: Span, value: u64, tag: u64) -> Result<(), Error> {
        let (addr, rkey) = self.remote(to, Span::new(at.region, at.at, 8), 8)?;
        let mut sge = Sge {
            addr: &value as *const u64 as u64,
            length: 8,
            lkey: 0,
        };
        let mut chain = [SendWr::write(&mut sge, addr, rkey, ibv::SEND_INLINE)];
        self.post(to, &mut chain, tag)
    }

    fn poll(&mut self, done: &mut Vec<u64>) -> Result<(), Error> {
        let mut wc = [unsafe { std::mem::zeroed::<Wc>() }; BATCH];
        let lib = self.engine.lib;
        for opened in &self.engine.opened {
            let n = unsafe { lib.poll_cq(opened.cq, &mut wc) };
            let n = usize::try_from(n).map_err(|_| Error::Link(format!("poll_cq: {n}")))?;
            for c in wc.iter().take(n) {
                if c.status != ibv::WC_SUCCESS {
                    return Err(Error::Link(format!("completion: {}", lib.status(c.status))));
                }
                done.push(c.wr_id);
                self.outstanding = self.outstanding.saturating_sub(1);
            }
        }
        Ok(())
    }

    fn reset(&mut self) -> Result<(), Error> {
        let start = Instant::now();
        let mut done = Vec::new();
        while self.outstanding > 0 {
            if start.elapsed() > DRAIN {
                return Err(Error::Link(format!(
                    "{} writes never completed",
                    self.outstanding
                )));
            }
            self.poll(&mut done)?;
        }
        Ok(())
    }

    fn memory(&self, region: usize) -> Option<(*mut u8, usize)> {
        let size = *self.sizes.get(region)?;
        Some((self.engine.regions[region].base, size))
    }

    fn load(&self, at: Span) -> Result<u64, Error> {
        Ok(self.word_at(at)?.load(Ordering::Acquire))
    }

    fn read(&self, at: Span) -> Result<Vec<u8>, Error> {
        fits(&self.sizes, at, 1)?;
        let mut out = vec![0u8; at.len];
        let from = self.engine.regions[at.region].at(at.at);
        unsafe { std::ptr::copy_nonoverlapping(from, out.as_mut_ptr(), at.len) };
        Ok(out)
    }

    fn write(&mut self, at: Span, bytes: &[u8]) -> Result<(), Error> {
        fits(&self.sizes, Span::new(at.region, at.at, bytes.len()), 1)?;
        let into = self.engine.regions[at.region].at(at.at);
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), into, bytes.len()) };
        Ok(())
    }
}
