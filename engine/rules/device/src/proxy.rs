//! The host proxy: reads the doorbell, moves parts, frees slots.

use crate::layout::MAX_GROUP;
use sensai_engine_comm::Error as CommError;
use std::collections::{BTreeMap, BTreeSet};

#[cfg(test)]
pub(crate) mod sim;
#[cfg(test)]
pub(crate) mod wired;

/// A lane's operation, numbered as in `collectives.h`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Sum,
    Gather,
    AllToAll,
    Send,
    Recv,
    Fetch,
}

/// One word of host memory the proxy reads or writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Word {
    Epoch,
    Trigger {
        pool: u32,
        slot: u32,
    },
    Done {
        pool: u32,
        slot: u32,
    },
    Abort,
    Sent {
        pool: u32,
    },
    /// A flag's ordinal, then its count.
    Flag {
        pool: u32,
        slot: u32,
        rank: u32,
    },
    Count {
        pool: u32,
        slot: u32,
        rank: u32,
    },
    Credit {
        pool: u32,
        rank: u32,
    },
    /// All-to-all: elements for group rank `rank`.
    Size {
        pool: u32,
        slot: u32,
        rank: u32,
    },
}

/// Host memory: acquire loads and release stores.
pub trait Host {
    fn load(&self, word: Word) -> u64;
    fn store(&mut self, word: Word, value: u64);
}

/// A pool's shape, from the engine's plan.
#[derive(Clone, Copy, Debug)]
pub struct Pool {
    pub depth: u32,
    pub one_way: bool,
    pub group: u32,
    /// This rank's place in the group.
    pub me: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct Lane {
    pub op: Op,
    pub pool: u32,
    /// Bytes per element, and all-to-all elements per destination.
    pub width: u32,
    pub block: u32,
}

/// One put the proxy asks a link for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Part {
    pub pool: u32,
    pub slot: u32,
    pub to: u32,
    pub ordinal: u64,
    pub count: u32,
    /// The bytes to send: offset in the slot, and length.
    pub from: u32,
    pub bytes: u32,
}

/// A put that reached group rank `to`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sent {
    pub pool: u32,
    pub ordinal: u64,
    pub to: u32,
}

/// Moves bytes. Each call starts work that `sent` reports.
pub trait Link {
    /// Sends the part, then a flag, to `part.to`.
    fn put(&mut self, part: Part) -> Result<(), CommError>;
    /// Tells group rank `to` this rank read up to `ordinal`.
    fn credit(&mut self, pool: u32, to: u32, ordinal: u64) -> Result<(), CommError>;
    fn sent(&mut self) -> Result<Vec<Sent>, CommError>;
    /// Drops or finishes every put and credit in flight.
    fn reset(&mut self) -> Result<(), CommError>;
}

/// Rows a fetch wrote into its slot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fetched {
    pub pool: u32,
    pub slot: u32,
    pub ordinal: u64,
    pub rows: u32,
}

/// Reads kept rows from other tiers into a fetch slot.
pub trait Fetcher {
    fn fetch(&mut self, pool: u32, slot: u32, ordinal: u64, requests: u32) -> Result<(), String>;
    fn fetched(&mut self) -> Vec<Fetched>;
    fn reset(&mut self);
}

#[derive(Debug, PartialEq, thiserror::Error)]
pub enum Error {
    #[error("link: {0}")]
    Link(String),
    #[error("pool {pool} slot word for ordinal {ordinal} is out of order")]
    Order { pool: u32, ordinal: u64 },
    #[error("no lane {0}")]
    Lane(u32),
    #[error("pool {pool}: an unexpected event for ordinal {ordinal}")]
    Event { pool: u32, ordinal: u64 },
    #[error("bad plan: {0}")]
    Plan(&'static str),
}

fn linked(e: CommError) -> Error {
    Error::Link(e.to_string())
}

/// The bit every trigger word sets, and the lane count.
const VALID: u64 = 1 << 63;
const LANES: usize = 1 << 15;

/// One pool's progress.
#[derive(Clone, Debug)]
struct Track {
    next: u64,
    /// The trigger word last seen, per slot.
    trigger: Vec<u64>,
    /// The last credit sent.
    credit: u64,
    /// Destinations still due, by ordinal.
    open: BTreeMap<u64, BTreeSet<u32>>,
}

impl Track {
    fn new(depth: u32, base: u64) -> Self {
        Track {
            next: base,
            trigger: vec![0; depth as usize],
            credit: base - 1,
            open: BTreeMap::new(),
        }
    }
}

pub struct Proxy {
    pools: Vec<Pool>,
    lanes: Vec<Lane>,
    epoch: u64,
    tracks: Vec<Track>,
}

/// The group ranks one trigger's work goes to.
fn targets(op: Op, pool: Pool) -> BTreeSet<u32> {
    match op {
        Op::Recv => BTreeSet::new(),
        Op::Fetch => BTreeSet::from([pool.me]),
        Op::Send => BTreeSet::from([1 - pool.me]),
        _ => (0..pool.group).filter(|&k| k != pool.me).collect(),
    }
}

fn check(pools: &[Pool], lanes: &[Lane], base: u64) -> Result<(), Error> {
    let bad = |pool: &Pool| {
        !(1..=4).contains(&pool.depth)
            || pool.me >= pool.group
            || pool.group as usize > MAX_GROUP
            || (pool.one_way && pool.group != 2)
    };
    if base == 0 || pools.len() > 4 || pools.iter().any(bad) {
        return Err(Error::Plan("pool shape"));
    }
    if lanes.len() > LANES || lanes.iter().any(|l| l.pool as usize >= pools.len()) {
        return Err(Error::Plan("lane pools"));
    }
    Ok(())
}

impl Proxy {
    /// A proxy with pools at ordinal `base`. Clears the page.
    pub fn new(
        host: &mut dyn Host,
        pools: Vec<Pool>,
        lanes: Vec<Lane>,
        base: u64,
    ) -> Result<Self, Error> {
        check(&pools, &lanes, base)?;
        let mut proxy = Proxy {
            pools,
            lanes,
            epoch: 0,
            tracks: Vec::new(),
        };
        proxy.clear(host, base);
        Ok(proxy)
    }

    /// After a resync: drains the links, then restarts at `base`.
    pub fn reset(
        &mut self,
        host: &mut dyn Host,
        link: &mut dyn Link,
        fetcher: &mut dyn Fetcher,
        base: u64,
    ) -> Result<(), Error> {
        if base == 0 {
            return Err(Error::Plan("base 0"));
        }
        link.reset().map_err(linked)?;
        fetcher.reset();
        self.clear(host, base);
        Ok(())
    }

    fn clear(&mut self, host: &mut dyn Host, base: u64) {
        self.tracks.clear();
        for (p, pool) in self.pools.iter().enumerate() {
            let pool_id = p as u32;
            for slot in 0..pool.depth {
                host.store(
                    Word::Trigger {
                        pool: pool_id,
                        slot,
                    },
                    0,
                );
                host.store(
                    Word::Done {
                        pool: pool_id,
                        slot,
                    },
                    0,
                );
            }
            host.store(Word::Sent { pool: pool_id }, base - 1);
            for rank in 0..pool.group {
                host.store(
                    Word::Credit {
                        pool: pool_id,
                        rank,
                    },
                    base - 1,
                );
            }
            self.tracks.push(Track::new(pool.depth, base));
        }
        self.epoch = host.load(Word::Epoch);
    }

    /// Whether puts or fetches are still out.
    pub fn open(&self) -> bool {
        self.tracks.iter().any(|t| !t.open.is_empty())
    }

    /// One pass. True if it did work. Errors set abort.
    pub fn step(
        &mut self,
        host: &mut dyn Host,
        link: &mut dyn Link,
        fetcher: &mut dyn Fetcher,
    ) -> Result<bool, Error> {
        let done = self.pass(host, link, fetcher);
        if done.is_err() {
            host.store(Word::Abort, 1);
        }
        done
    }

    fn pass(
        &mut self,
        host: &mut dyn Host,
        link: &mut dyn Link,
        fetcher: &mut dyn Fetcher,
    ) -> Result<bool, Error> {
        let mut busy = false;
        let epoch = host.load(Word::Epoch);
        if epoch != self.epoch {
            self.epoch = epoch;
            for p in 0..self.pools.len() {
                busy |= self.triggers(p, host, link, fetcher)?;
                busy |= self.credits(p, host, link)?;
            }
        }
        for s in link.sent().map_err(linked)? {
            busy = true;
            self.finish(s.pool, s.ordinal, s.to)?;
        }
        for f in fetcher.fetched() {
            busy = true;
            let me = self.pools.get(f.pool as usize).map_or(0, |p| p.me);
            self.finish(f.pool, f.ordinal, me)?;
            let (pool, slot) = (f.pool, f.slot);
            host.store(
                Word::Count {
                    pool,
                    slot,
                    rank: 0,
                },
                u64::from(f.rows),
            );
            host.store(
                Word::Flag {
                    pool,
                    slot,
                    rank: 0,
                },
                f.ordinal,
            );
        }
        for p in 0..self.pools.len() {
            self.release(p, host);
        }
        Ok(busy)
    }

    /// Handles triggers in ordinal order, while slot words change.
    fn triggers(
        &mut self,
        p: usize,
        host: &mut dyn Host,
        link: &mut dyn Link,
        fetcher: &mut dyn Fetcher,
    ) -> Result<bool, Error> {
        let (pool, pool_id) = (self.pools[p], p as u32);
        let mut busy = false;
        loop {
            let track = &mut self.tracks[p];
            let ordinal = track.next;
            let slot = (ordinal % u64::from(pool.depth)) as u32;
            let word = host.load(Word::Trigger {
                pool: pool_id,
                slot,
            });
            if word == track.trigger[slot as usize] {
                return Ok(busy);
            }
            if word & VALID == 0 || (word >> 32) & 0xFFFF != ordinal & 0xFFFF {
                return Err(Error::Order {
                    pool: pool_id,
                    ordinal,
                });
            }
            track.trigger[slot as usize] = word;
            track.next += 1;
            let lane = ((word >> 48) as usize) & (LANES - 1);
            let lane = *self.lanes.get(lane).ok_or(Error::Lane(lane as u32))?;
            let op = lane.op;
            let job = Job {
                lane,
                pool,
                pool_id,
                slot,
                ordinal,
                count: word as u32,
            };
            job.start(host, link, fetcher)?;
            self.tracks[p].open.insert(ordinal, targets(op, pool));
            busy = true;
        }
    }

    /// One-way pools: credits the largest new done word.
    fn credits(
        &mut self,
        p: usize,
        host: &mut dyn Host,
        link: &mut dyn Link,
    ) -> Result<bool, Error> {
        let (pool, pool_id) = (self.pools[p], p as u32);
        if !pool.one_way {
            return Ok(false);
        }
        let track = &mut self.tracks[p];
        let words = (0..pool.depth).map(|slot| {
            host.load(Word::Done {
                pool: pool_id,
                slot,
            })
        });
        let most = words.fold(track.credit, u64::max);
        if most == track.credit {
            return Ok(false);
        }
        track.credit = most;
        link.credit(pool_id, 1 - pool.me, most).map_err(linked)?;
        Ok(true)
    }

    fn finish(&mut self, pool: u32, ordinal: u64, to: u32) -> Result<(), Error> {
        let open = self.tracks.get_mut(pool as usize).map(|t| &mut t.open);
        let due = open.and_then(|o| o.get_mut(&ordinal));
        if !due.is_some_and(|d| d.remove(&to)) {
            return Err(Error::Event { pool, ordinal });
        }
        Ok(())
    }

    /// Advances `sent` over the finished ordinals at the front.
    fn release(&mut self, p: usize, host: &mut dyn Host) {
        let open = &mut self.tracks[p].open;
        let mut last = None;
        while open
            .first_key_value()
            .is_some_and(|(_, due)| due.is_empty())
        {
            last = open.pop_first().map(|(ordinal, _)| ordinal);
        }
        if let Some(ordinal) = last {
            host.store(Word::Sent { pool: p as u32 }, ordinal);
        }
    }
}

/// One trigger the proxy read.
#[derive(Clone, Copy)]
struct Job {
    lane: Lane,
    pool: Pool,
    pool_id: u32,
    slot: u32,
    ordinal: u64,
    count: u32,
}

impl Job {
    fn start(
        &self,
        host: &dyn Host,
        link: &mut dyn Link,
        fetcher: &mut dyn Fetcher,
    ) -> Result<(), Error> {
        let Job {
            lane,
            pool,
            pool_id,
            slot,
            ordinal,
            count,
        } = *self;
        let op = lane.op;
        if op == Op::Fetch {
            return fetcher
                .fetch(pool_id, slot, ordinal, count)
                .map_err(Error::Link);
        }
        for to in targets(op, pool) {
            let size = Word::Size {
                pool: pool_id,
                slot,
                rank: to,
            };
            let split = op == Op::AllToAll;
            let count = if split { host.load(size) as u32 } else { count };
            let from = if split {
                to * lane.block * lane.width
            } else {
                0
            };
            let bytes = count * lane.width;
            let part = Part {
                pool: pool_id,
                slot,
                to,
                ordinal,
                count,
                from,
                bytes,
            };
            link.put(part).map_err(linked)?;
        }
        Ok(())
    }
}
