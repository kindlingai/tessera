//! A `Link` over RDMA: data, then count, then ordinal.

use crate::fabric::Fabric;
use crate::ibv::{self, SendWr, Sge, Wc};
use crate::link::{Link, Put, Span, fits};
use crate::reduce::Pinning;
use crate::verbs::{Engine, Region};
use crate::{Error, Transport};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// The longest `reset`, or wait for a relay slot.
const DRAIN: Duration = Duration::from_secs(10);
/// Completions per device per poll.
const BATCH: usize = 16;
/// Relay slots per rank, and bytes per relay header.
const SLOTS: usize = 8;
const HEAD: usize = 128;

/// Relay header words, after the sequence word at 0.
const KIND: usize = 1;
const DEST: usize = 2;
const INTO: usize = 3;
const AT: usize = 4;
const LEN: usize = 5;
const FLAG: usize = 6;
const FLAG_AT: usize = 7;
const WORDS: usize = 8;
/// A relayed put, or a relayed word.
const PUT: u64 = 0;
const WORD: u64 = 1;

/// The meaning of one completion.
#[derive(Clone, Copy, Debug)]
enum Work {
    /// The caller's tag.
    Tag(u64),
    /// This rank forwarded relay slot `slot` for sequence `seq`.
    Forwarded { slot: usize, seq: u64 },
    /// Nothing to report.
    Quiet,
}

/// Relay state: the hidden regions and each slot's progress.
struct Relay {
    /// Region indexes: data, headers, acks, staged headers.
    data: usize,
    head: usize,
    ack: usize,
    out: usize,
    bytes: usize,
    /// As origin: the last sequence, and each slot's last.
    next: u64,
    sent: [u64; SLOTS],
    /// As relayer: each slot's last forwarded sequence.
    forwarded: [u64; SLOTS],
}

/// One-sided writes, relayed across a ring of four.
pub struct VerbsLink {
    rank: u32,
    ranks: u32,
    engine: Engine,
    /// The caller's regions. The relay's follow them.
    sizes: Vec<usize>,
    relay: Relay,
    work: HashMap<u64, Work>,
    ids: u64,
    /// Tags that finished while a put waited for a slot.
    pending: Vec<u64>,
}

// Each rank owns its queue pairs and regions.
unsafe impl Send for VerbsLink {}

/// An inline Sge over one local word.
fn inline(value: &u64) -> Sge {
    Sge {
        addr: value as *const u64 as u64,
        length: 8,
        lkey: 0,
    }
}

fn unreachable(to: u32) -> Error {
    Error::Fit(format!("no queue pair to rank {to}"))
}

impl VerbsLink {
    /// Rank `rank` of `ranks`. `relay` caps a relayed put's bytes.
    pub fn connect(
        rank: u32,
        ranks: u32,
        fabric: &Fabric,
        regions: &[usize],
        relay: usize,
        boot: &mut dyn Transport,
        pinning: Option<Arc<dyn Pinning>>,
    ) -> Result<Self, Error> {
        if regions.contains(&0) {
            return Err(Error::Fit("a region of 0 bytes".into()));
        }
        let bytes = relay.max(16).next_multiple_of(16);
        let hidden = [SLOTS * bytes, SLOTS * HEAD, SLOTS * 8, SLOTS * HEAD];
        let all: Vec<usize> = regions.iter().copied().chain(hidden).collect();
        let pin = pinning.as_ref();
        let made: Result<Vec<Region>, Error> = all.iter().map(|&n| Region::new(n, pin)).collect();
        let engine = Engine::connect(rank, ranks, fabric, 0, made?, boot)?;
        // Every peer must describe the same regions on every lane.
        let alike = |p: &crate::fabric::Peer| {
            let lanes = p.channels.iter().flatten();
            p.regions.len() == all.len() && lanes.clone().all(|l| l.rkeys.len() == all.len())
        };
        if let Some(r) = engine.peers.iter().position(|p| !alike(p)) {
            return Err(Error::Fit(format!("rank {r} describes other regions")));
        }
        let n = regions.len();
        Ok(VerbsLink {
            rank,
            ranks,
            engine,
            sizes: regions.to_vec(),
            relay: Relay {
                data: n,
                head: n + 1,
                ack: n + 2,
                out: n + 3,
                bytes,
                next: 0,
                sent: [0; SLOTS],
                forwarded: [0; SLOTS],
            },
            work: HashMap::new(),
            ids: 0,
            pending: Vec::new(),
        })
    }

    /// Every region's size, the caller's and the relay's.
    fn all(&self) -> Vec<usize> {
        self.engine.regions.iter().map(|r| r.len).collect()
    }

    /// The channel straight to `to`, if one exists.
    fn channel(&self, to: u32) -> Option<usize> {
        let direct = |c: &crate::fabric::Channel| c.peer == to && !c.ports.is_empty();
        let found = self.engine.wiring.channels.iter().position(direct);
        found.filter(|_| to != self.rank)
    }

    /// The neighbour a relayed write goes through, toward `to`.
    fn via(&self, to: u32) -> Result<u32, Error> {
        let next = (self.rank + 1) % self.ranks;
        match self.ranks == 4 && (next + 1) % self.ranks == to {
            true => Ok(next),
            false => Err(unreachable(to)),
        }
    }

    /// Posts `chain` to `to`, the last write signalling `work`.
    fn post(&mut self, to: u32, chain: &mut [SendWr], work: Work) -> Result<(), Error> {
        let c = self.channel(to).ok_or_else(|| unreachable(to))?;
        let id = self.ids;
        self.ids += 1;
        let n = chain.len();
        chain[n - 1].wr_id = id;
        chain[n - 1].send_flags |= ibv::SEND_SIGNALED;
        for i in 0..n - 1 {
            let next: *mut SendWr = &mut chain[i + 1];
            chain[i].next = next;
        }
        let qp = self.engine.links[c][0].qp;
        let e = unsafe { self.engine.lib.post_send(qp, &mut chain[0]) };
        if e != 0 {
            let why = std::io::Error::from_raw_os_error(e);
            return Err(Error::Link(format!("post to rank {to}: {why}")));
        }
        self.work.insert(id, work);
        Ok(())
    }

    /// `at` in the regions of `to`, a direct neighbour.
    fn remote(&self, to: u32, at: Span, align: usize) -> Result<(u64, u32), Error> {
        fits(&self.all(), at, align)?;
        let c = self.channel(to).ok_or_else(|| unreachable(to))?;
        let meets = self.engine.wiring.channels[c].meets;
        let peer = &self.engine.peers[to as usize];
        let far = peer.channels.get(meets).and_then(|l| l.first());
        let far = far.ok_or_else(|| Error::Fit(format!("rank {to} has no lane here")))?;
        Ok((peer.regions[at.region] + at.at as u64, far.rkeys[at.region]))
    }

    /// A local span as an Sge toward `to`.
    fn local(&self, to: u32, at: Span) -> Result<Sge, Error> {
        fits(&self.all(), at, 1)?;
        let length = u32::try_from(at.len).map_err(|_| Error::Fit(format!("{} bytes", at.len)))?;
        let c = self.channel(to).ok_or_else(|| unreachable(to))?;
        let port = self.engine.links[c][0].port;
        let lkey = unsafe { (*self.engine.opened[port].mrs[at.region]).lkey };
        let addr = self.engine.regions[at.region].at(at.at) as u64;
        Ok(Sge { addr, length, lkey })
    }

    fn word_at(&self, at: Span) -> Result<&AtomicU64, Error> {
        fits(&self.all(), Span::new(at.region, at.at, 8), 8)?;
        let word = self.engine.regions[at.region].at(at.at);
        // Checked: in bounds and 8-byte aligned.
        Ok(unsafe { &*(word as *const AtomicU64) })
    }

    /// Copies `bytes` into any of this rank's regions.
    fn place(&mut self, at: Span, bytes: &[u8]) -> Result<(), Error> {
        fits(&self.all(), Span::new(at.region, at.at, bytes.len()), 1)?;
        let into = self.engine.regions[at.region].at(at.at);
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), into, bytes.len()) };
        Ok(())
    }

    /// Bytes, then the count and the ordinal, straight to `to`.
    fn direct(
        &mut self,
        to: u32,
        spans: [Span; 3],
        words: [u64; 2],
        work: Work,
    ) -> Result<(), Error> {
        let [from, into, flag] = spans;
        let (into, data_key) = self.remote(to, into, 1)?;
        let (flag, flag_key) = self.remote(to, flag, 8)?;
        let mut data = self.local(to, from)?;
        let [ordinal, count] = words;
        let mut count_sge = inline(&count);
        let mut ordinal_sge = inline(&ordinal);
        let mut chain = Vec::with_capacity(3);
        if from.len > 0 {
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
        self.post(to, &mut chain, work)
    }

    /// One word straight to `to`.
    fn inline_word(&mut self, to: u32, at: Span, value: u64, work: Work) -> Result<(), Error> {
        let (addr, rkey) = self.remote(to, Span::new(at.region, at.at, 8), 8)?;
        let mut sge = inline(&value);
        let mut chain = [SendWr::write(&mut sge, addr, rkey, ibv::SEND_INLINE)];
        self.post(to, &mut chain, work)
    }

    /// Waits until the next relay slot's last use was acked.
    fn free_slot(&mut self) -> Result<usize, Error> {
        let slot = (self.relay.next % SLOTS as u64) as usize;
        let ack = Span::new(self.relay.ack, slot * 8, 8);
        let start = Instant::now();
        let mut done = Vec::new();
        while self.word_at(ack)?.load(Ordering::Acquire) < self.relay.sent[slot] {
            if start.elapsed() > DRAIN {
                return Err(Error::Link(format!("relay slot {slot} was never acked")));
            }
            self.advance(&mut done)?;
        }
        self.pending.append(&mut done);
        Ok(slot)
    }

    /// Hands a write for `to` to the neighbour between.
    fn relayed(&mut self, to: u32, from: Span, header: [u64; 10], tag: u64) -> Result<(), Error> {
        let via = self.via(to)?;
        if from.len > self.relay.bytes {
            return Err(Error::Fit(format!("a {}-byte relayed put", from.len)));
        }
        let slot = self.free_slot()?;
        let seq = self.relay.next + 1;
        let mut head = header;
        head[0] = seq;
        let bytes: Vec<u8> = head.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.place(Span::new(self.relay.out, slot * HEAD, HEAD), &bytes)?;
        let data = Span::new(self.relay.data, slot * self.relay.bytes, from.len);
        let (data_at, data_key) = self.remote(via, data, 1)?;
        let heads = Span::new(self.relay.head, slot * HEAD, HEAD);
        let (head_at, head_key) = self.remote(via, heads, 8)?;
        let mut data = self.local(via, from)?;
        let mut rest = self.local(via, Span::new(self.relay.out, slot * HEAD + 8, HEAD - 8))?;
        let mut seq_sge = inline(&seq);
        let mut chain = Vec::with_capacity(3);
        if from.len > 0 {
            chain.push(SendWr::write(&mut data, data_at, data_key, 0));
        }
        chain.push(SendWr::write(&mut rest, head_at + 8, head_key, 0));
        chain.push(SendWr::write(
            &mut seq_sge,
            head_at,
            head_key,
            ibv::SEND_INLINE,
        ));
        self.post(via, &mut chain, Work::Tag(tag))?;
        self.relay.next = seq;
        self.relay.sent[slot] = seq;
        Ok(())
    }

    /// Forwards each relay slot whose sequence moved on.
    fn forward(&mut self) -> Result<(), Error> {
        for slot in 0..SLOTS {
            let head = |w: usize| Span::new(self.relay.head, slot * HEAD + 8 * w, 8);
            let seq = self.word_at(head(0))?.load(Ordering::Acquire);
            if seq <= self.relay.forwarded[slot] {
                continue;
            }
            self.relay.forwarded[slot] = seq;
            let word = |w: usize| self.word_at(head(w)).map(|a| a.load(Ordering::Relaxed));
            let (kind, dest) = (word(KIND)?, word(DEST)? as u32);
            let into = Span::new(
                word(INTO)? as usize,
                word(AT)? as usize,
                word(LEN)? as usize,
            );
            let words = [word(WORDS)?, word(WORDS + 1)?];
            let work = Work::Forwarded { slot, seq };
            if kind == WORD {
                self.inline_word(dest, into, words[0], work)?;
                continue;
            }
            let flag = Span::new(word(FLAG)? as usize, word(FLAG_AT)? as usize, 16);
            let from = Span::new(self.relay.data, slot * self.relay.bytes, into.len);
            self.direct(dest, [from, into, flag], words, work)?;
        }
        Ok(())
    }

    /// Forwards relays and polls completions. Tags go to `done`.
    fn advance(&mut self, done: &mut Vec<u64>) -> Result<(), Error> {
        self.forward()?;
        let mut wc = [unsafe { std::mem::zeroed::<Wc>() }; BATCH];
        let lib = self.engine.lib;
        let mut finished = Vec::new();
        for opened in &self.engine.opened {
            let n = unsafe { lib.poll_cq(opened.cq, &mut wc) };
            let n = usize::try_from(n).map_err(|_| Error::Link(format!("poll_cq: {n}")))?;
            for c in wc.iter().take(n) {
                if c.status != ibv::WC_SUCCESS {
                    return Err(Error::Link(format!("completion: {}", lib.status(c.status))));
                }
                finished.push(c.wr_id);
            }
        }
        let prev = (self.rank + self.ranks - 1) % self.ranks;
        for id in finished {
            match self.work.remove(&id) {
                Some(Work::Tag(tag)) => done.push(tag),
                Some(Work::Forwarded { slot, seq }) => {
                    let ack = Span::new(self.relay.ack, slot * 8, 8);
                    self.inline_word(prev, ack, seq, Work::Quiet)?;
                }
                Some(Work::Quiet) => {}
                None => return Err(Error::Link(format!("a completion no work awaits: {id}"))),
            }
        }
        Ok(())
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
        for at in [put.from, put.into, put.flag] {
            fits(&self.sizes, at, 1)?;
        }
        if self.channel(put.to).is_some() {
            let spans = [put.from, put.into, put.flag];
            return self.direct(put.to, spans, put.words, Work::Tag(put.tag));
        }
        let (into, flag) = (put.into, put.flag);
        let header = [
            0,
            PUT,
            put.to as u64,
            into.region as u64,
            into.at as u64,
            into.len as u64,
            flag.region as u64,
            flag.at as u64,
            put.words[0],
            put.words[1],
        ];
        self.relayed(put.to, put.from, header, put.tag)
    }

    fn word(&mut self, to: u32, at: Span, value: u64, tag: u64) -> Result<(), Error> {
        let at = Span::new(at.region, at.at, 8);
        fits(&self.sizes, at, 8)?;
        if self.channel(to).is_some() {
            return self.inline_word(to, at, value, Work::Tag(tag));
        }
        let header = [
            0,
            WORD,
            to as u64,
            at.region as u64,
            at.at as u64,
            8,
            0,
            0,
            value,
            0,
        ];
        self.relayed(to, Span::new(self.relay.data, 0, 0), header, tag)
    }

    fn poll(&mut self, done: &mut Vec<u64>) -> Result<(), Error> {
        done.append(&mut self.pending);
        self.advance(done)
    }

    fn reset(&mut self) -> Result<(), Error> {
        let start = Instant::now();
        let mut done = Vec::new();
        while !self.work.is_empty() {
            if start.elapsed() > DRAIN {
                return Err(Error::Link(format!(
                    "{} writes never completed",
                    self.work.len()
                )));
            }
            self.advance(&mut done)?;
        }
        self.pending.clear();
        Ok(())
    }

    fn memory(&self, region: usize) -> Option<(*mut u8, Option<u64>, usize)> {
        let size = *self.sizes.get(region)?;
        let region = &self.engine.regions[region];
        Some((region.base, region.device(), size))
    }

    fn load(&self, at: Span) -> Result<u64, Error> {
        fits(&self.sizes, Span::new(at.region, at.at, 8), 8)?;
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
        self.place(at, bytes)
    }
}
