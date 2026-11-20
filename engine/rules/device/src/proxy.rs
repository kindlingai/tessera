//! The host proxy: reads the doorbell, moves parts, frees slots.

use std::collections::BTreeMap;

#[cfg(test)]
pub(crate) mod sim;

/// A lane's operation, as `collectives.h` numbers it.
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
}

/// Work a link finished.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Event {
    /// One put of `ordinal` reached its peer.
    Sent { pool: u32, ordinal: u64 },
    /// A fetch wrote `rows` rows into the slot.
    Fetched {
        pool: u32,
        slot: u32,
        ordinal: u64,
        rows: u32,
    },
}

/// Moves bytes. Each call starts work that `events` reports.
pub trait Link {
    /// Sends the slot's part, then a flag, to `to`.
    fn put(
        &mut self,
        pool: u32,
        slot: u32,
        to: u32,
        ordinal: u64,
        count: u32,
    ) -> Result<(), String>;
    /// Tells group rank `to` this rank read `ordinal`.
    fn credit(&mut self, pool: u32, to: u32, ordinal: u64) -> Result<(), String>;
    /// Reads `requests` from `slot` and writes the rows.
    fn fetch(&mut self, pool: u32, slot: u32, ordinal: u64, requests: u32) -> Result<(), String>;
    fn events(&mut self) -> Vec<Event>;
}

#[derive(Debug, PartialEq, thiserror::Error)]
pub enum Error {
    #[error("link: {0}")]
    Link(String),
    #[error("pool {pool} slot word for ordinal {ordinal} is out of order")]
    Order { pool: u32, ordinal: u64 },
    #[error("no lane {0}")]
    Lane(u32),
}

/// One pool's progress.
#[derive(Clone, Debug)]
struct Track {
    next: u64,
    /// The trigger and done words last seen, per slot.
    trigger: Vec<u64>,
    done: Vec<u64>,
    /// Puts or fetches still out, by ordinal.
    open: BTreeMap<u64, u32>,
}

impl Track {
    fn new(depth: u32, base: u64) -> Self {
        Track {
            next: base,
            trigger: vec![0; depth as usize],
            done: vec![0; depth as usize],
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

impl Proxy {
    /// A proxy whose pools all start at ordinal `base`.
    pub fn new(pools: Vec<Pool>, lanes: Vec<Lane>, base: u64) -> Self {
        let tracks = pools.iter().map(|p| Track::new(p.depth, base)).collect();
        Proxy {
            pools,
            lanes,
            epoch: 0,
            tracks,
        }
    }

    /// After a resync: every pool restarts at `base`.
    pub fn reset(&mut self, host: &mut dyn Host, base: u64) {
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
            self.tracks[p] = Track::new(pool.depth, base);
        }
        self.epoch = host.load(Word::Epoch);
    }

    /// One pass: true if it did work. Errors set abort.
    pub fn step(&mut self, host: &mut dyn Host, link: &mut dyn Link) -> Result<bool, Error> {
        let done = self.pass(host, link);
        if done.is_err() {
            host.store(Word::Abort, 1);
        }
        done
    }

    fn pass(&mut self, host: &mut dyn Host, link: &mut dyn Link) -> Result<bool, Error> {
        let mut busy = false;
        let epoch = host.load(Word::Epoch);
        if epoch != self.epoch {
            self.epoch = epoch;
            for p in 0..self.pools.len() {
                busy |= self.triggers(p, host, link)?;
                busy |= self.credits(p, host, link)?;
            }
        }
        for event in link.events() {
            busy = true;
            self.finish(event, host);
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
            if (word >> 32) & 0xFFFF != ordinal & 0xFFFF {
                return Err(Error::Order {
                    pool: pool_id,
                    ordinal,
                });
            }
            track.trigger[slot as usize] = word;
            track.next += 1;
            let lane = (word >> 48) as u32;
            let op = self.lanes.get(lane as usize).ok_or(Error::Lane(lane))?.op;
            let job = Job {
                op,
                pool,
                pool_id,
                slot,
                ordinal,
                count: word as u32,
            };
            let out = job.start(host, link)?;
            self.tracks[p].open.insert(ordinal, out);
            busy = true;
        }
    }

    /// One-way pools: sends each new done word as a credit.
    fn credits(
        &mut self,
        p: usize,
        host: &mut dyn Host,
        link: &mut dyn Link,
    ) -> Result<bool, Error> {
        let (pool, pool_id) = (self.pools[p], p as u32);
        let mut busy = false;
        for slot in 0..pool.depth {
            let word = host.load(Word::Done {
                pool: pool_id,
                slot,
            });
            let seen = &mut self.tracks[p].done[slot as usize];
            if pool.one_way && word != *seen && word != 0 {
                link.credit(pool_id, 1 - pool.me, word)
                    .map_err(Error::Link)?;
                busy = true;
            }
            *seen = word;
        }
        Ok(busy)
    }

    fn finish(&mut self, event: Event, host: &mut dyn Host) {
        let (pool, ordinal) = match event {
            Event::Sent { pool, ordinal } => (pool, ordinal),
            Event::Fetched {
                pool,
                slot,
                ordinal,
                rows,
            } => {
                host.store(
                    Word::Count {
                        pool,
                        slot,
                        rank: 0,
                    },
                    u64::from(rows),
                );
                host.store(
                    Word::Flag {
                        pool,
                        slot,
                        rank: 0,
                    },
                    ordinal,
                );
                (pool, ordinal)
            }
        };
        if let Some(left) = self.tracks[pool as usize].open.get_mut(&ordinal) {
            *left = left.saturating_sub(1);
        }
    }

    /// Advances `sent` over the finished ordinals at the front.
    fn release(&mut self, p: usize, host: &mut dyn Host) {
        let open = &mut self.tracks[p].open;
        let mut last = None;
        while let Some((&ordinal, &0)) = open.first_key_value() {
            open.pop_first();
            last = Some(ordinal);
        }
        if let Some(ordinal) = last {
            host.store(Word::Sent { pool: p as u32 }, ordinal);
        }
    }
}

/// One trigger the proxy read.
#[derive(Clone, Copy)]
struct Job {
    op: Op,
    pool: Pool,
    pool_id: u32,
    slot: u32,
    ordinal: u64,
    count: u32,
}

impl Job {
    /// Starts the work. Returns how many events to await.
    fn start(&self, host: &dyn Host, link: &mut dyn Link) -> Result<u32, Error> {
        let Job {
            op,
            pool,
            pool_id,
            slot,
            ordinal,
            count,
        } = *self;
        let mut out = 0;
        match op {
            Op::Recv => {}
            Op::Fetch => {
                link.fetch(pool_id, slot, ordinal, count)
                    .map_err(Error::Link)?;
                out = 1;
            }
            Op::Send => {
                link.put(pool_id, slot, 1 - pool.me, ordinal, count)
                    .map_err(Error::Link)?;
                out = 1;
            }
            Op::Sum | Op::Gather | Op::AllToAll => {
                for to in (0..pool.group).filter(|&k| k != pool.me) {
                    let size = Word::Size {
                        pool: pool_id,
                        slot,
                        rank: to,
                    };
                    let n = if op == Op::AllToAll {
                        host.load(size) as u32
                    } else {
                        count
                    };
                    link.put(pool_id, slot, to, ordinal, n)
                        .map_err(Error::Link)?;
                    out += 1;
                }
            }
        }
        Ok(out)
    }
}
