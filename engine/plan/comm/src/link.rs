//! One-sided writes into regions every rank lays out alike.

use crate::{Error, Transport};
use std::time::{Duration, Instant};

/// Bytes in one of a rank's regions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub region: usize,
    pub at: usize,
    pub len: usize,
}

impl Span {
    pub fn new(region: usize, at: usize, len: usize) -> Self {
        Span { region, at, len }
    }
}

/// One put: bytes, then a 16-byte flag, at rank `to`.
#[derive(Clone, Copy, Debug)]
pub struct Put {
    pub to: u32,
    pub from: Span,
    pub into: Span,
    /// Where the flag goes at `to`, and its two words.
    pub flag: Span,
    pub words: [u64; 2],
    pub tag: u64,
}

/// Moves bytes between ranks one-sided. Work finishes by tag.
pub trait Link: Send {
    fn rank(&self) -> u32;
    fn ranks(&self) -> u32;
    /// Writes the bytes, then the flag after them.
    fn put(&mut self, put: Put) -> Result<(), Error>;
    /// Writes one word at rank `to`.
    fn word(&mut self, to: u32, at: Span, value: u64, tag: u64) -> Result<(), Error>;
    /// Adds finished tags to `done`. It also moves work along.
    fn poll(&mut self, done: &mut Vec<u64>) -> Result<(), Error>;
    /// Drops or finishes all work in flight.
    fn reset(&mut self) -> Result<(), Error>;
    /// This rank's region as host memory, if fixed.
    fn memory(&self, region: usize) -> Option<(*mut u8, usize)> {
        let _ = region;
        None
    }
    /// This rank's regions: an acquire load, then bytes.
    fn load(&self, at: Span) -> Result<u64, Error>;
    fn read(&self, at: Span) -> Result<Vec<u8>, Error>;
    fn write(&mut self, at: Span, bytes: &[u8]) -> Result<(), Error>;
}

/// Checks `at` lies in its region, aligned to `align`.
pub(crate) fn fits(sizes: &[usize], at: Span, align: usize) -> Result<(), Error> {
    let size = sizes.get(at.region).copied();
    let inside = size.is_some_and(|s| at.at.checked_add(at.len).is_some_and(|end| end <= s));
    if !inside || !at.at.is_multiple_of(align) {
        return Err(Error::Fit(format!("{at:?} lies outside its region")));
    }
    Ok(())
}

/// The regions `Blocking` needs, for `ranks` and `capacity`.
pub fn exchange_regions(ranks: u32, capacity: usize) -> [usize; 3] {
    let ranks = ranks as usize;
    [2 * capacity, 2 * ranks * capacity, 2 * ranks * 16]
}

const SEND: usize = 0;
const RECV: usize = 1;
const FLAGS: usize = 2;

/// An exchange over any link, two slots by parity.
pub struct Blocking<L> {
    pub link: L,
    capacity: usize,
    round: u64,
    wait: Duration,
    /// An exchange failed. Ranks must start again.
    broken: bool,
}

impl<L: Link> Blocking<L> {
    /// `link` holds `exchange_regions(ranks, capacity)`.
    pub fn new(link: L, capacity: usize, wait: Duration) -> Self {
        Blocking {
            link,
            capacity,
            round: 0,
            wait,
            broken: false,
        }
    }

    fn flag(&self, parity: usize, rank: u32) -> Span {
        let ranks = self.link.ranks() as usize;
        Span::new(FLAGS, (parity * ranks + rank as usize) * 16, 16)
    }

    fn recv(&self, parity: usize, rank: u32, len: usize) -> Span {
        let ranks = self.link.ranks() as usize;
        Span::new(RECV, (parity * ranks + rank as usize) * self.capacity, len)
    }

    fn round(&mut self, mine: &[u8]) -> Result<Vec<Vec<u8>>, Error> {
        let round = self.round + 1;
        let parity = (round % 2) as usize;
        let (me, ranks) = (self.link.rank(), self.link.ranks());
        let from = Span::new(SEND, parity * self.capacity, mine.len());
        self.link.write(from, mine)?;
        let words = [round, mine.len() as u64];
        // Farthest first: a relay then lands before a neighbour's flag.
        for to in (1..ranks).rev().map(|k| (me + k) % ranks) {
            let (into, flag) = (self.recv(parity, me, mine.len()), self.flag(parity, me));
            let put = Put {
                to,
                from,
                into,
                flag,
                words,
                tag: round,
            };
            self.link.put(put)?;
        }
        let start = Instant::now();
        let late = |what: String| Error::Link(format!("{what} by round {round}"));
        let mut out = Vec::with_capacity(ranks as usize);
        let mut done = Vec::new();
        for r in 0..ranks {
            if r == me {
                out.push(mine.to_vec());
                continue;
            }
            let flag = self.flag(parity, r);
            while self.link.load(flag)? < round {
                if start.elapsed() > self.wait {
                    return Err(late(format!("rank {r} sent nothing")));
                }
                self.link.poll(&mut done)?;
                std::thread::yield_now();
            }
            let len = self.link.load(Span::new(FLAGS, flag.at + 8, 8))? as usize;
            if len > self.capacity {
                return Err(Error::Fit(format!("rank {r} sent {len} bytes")));
            }
            out.push(self.link.read(self.recv(parity, r, len))?);
        }
        // Once more, so a relaying link forwards its relays.
        self.link.poll(&mut done)?;
        // The sends must land before this slot is written again.
        while done.iter().filter(|&&t| t == round).count() < ranks as usize - 1 {
            if start.elapsed() > self.wait {
                return Err(late("this rank's sends never landed".into()));
            }
            self.link.poll(&mut done)?;
        }
        self.round = round;
        Ok(out)
    }
}

impl<L: Link> Transport for Blocking<L> {
    fn rank(&self) -> u32 {
        self.link.rank()
    }

    fn ranks(&self) -> u32 {
        self.link.ranks()
    }

    fn exchange(&mut self, mine: &[u8]) -> Result<Vec<Vec<u8>>, Error> {
        if self.broken {
            return Err(Error::Link("an earlier exchange failed".into()));
        }
        if mine.len() > self.capacity {
            let what = format!("{} bytes over a {} byte slot", mine.len(), self.capacity);
            return Err(Error::Fit(what));
        }
        let out = self.round(mine);
        self.broken = out.is_err();
        out
    }
}
