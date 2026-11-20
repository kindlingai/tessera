//! Ranks, proxies and links under a seeded random schedule.

use crate::proxy::{Event, Host, Lane, Link, Op, Pool, Proxy, Word};
use std::collections::{HashMap, VecDeque};

/// Words in one part, and in one all-to-all block.
const PART: usize = 4;
const BLOCK: usize = 2;

/// A small seeded generator.
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n as u64) as usize
    }
}

/// The value word `j` of `rank`'s part for `ordinal` holds.
fn word(rank: u32, ordinal: u64, j: usize) -> u64 {
    (u64::from(rank) << 48) ^ (ordinal << 16) ^ j as u64
}

#[derive(Default)]
struct Mem(HashMap<Word, u64>);

impl Host for Mem {
    fn load(&self, word: Word) -> u64 {
        self.0.get(&word).copied().unwrap_or(0)
    }

    fn store(&mut self, word: Word, value: u64) {
        self.0.insert(word, value);
    }
}

enum Msg {
    Put {
        pool: u32,
        slot: u32,
        from: u32,
        ordinal: u64,
        block: Option<u32>,
    },
    Credit {
        pool: u32,
        from: u32,
        ordinal: u64,
    },
    Fetch {
        pool: u32,
        slot: u32,
        ordinal: u64,
        requests: u32,
    },
}

/// A link that queues messages for the network.
#[derive(Default)]
struct SimLink {
    out: Vec<(u32, Msg)>,
    events: Vec<Event>,
    me: u32,
    /// Group rank to global rank, per pool.
    groups: Vec<Vec<u32>>,
    lanes: Vec<Lane>,
}

impl Link for SimLink {
    fn put(&mut self, pool: u32, slot: u32, to: u32, ordinal: u64, _: u32) -> Result<(), String> {
        let group = &self.groups[pool as usize];
        let from = group.iter().position(|&r| r == self.me).unwrap() as u32;
        let all = self
            .lanes
            .iter()
            .any(|l| l.pool == pool && l.op == Op::AllToAll);
        let block = all.then_some(to);
        self.out.push((
            group[to as usize],
            Msg::Put {
                pool,
                slot,
                from,
                ordinal,
                block,
            },
        ));
        Ok(())
    }

    fn credit(&mut self, pool: u32, to: u32, ordinal: u64) -> Result<(), String> {
        let group = &self.groups[pool as usize];
        let from = 1 - to;
        self.out.push((
            group[to as usize],
            Msg::Credit {
                pool,
                from,
                ordinal,
            },
        ));
        Ok(())
    }

    fn fetch(&mut self, pool: u32, slot: u32, ordinal: u64, requests: u32) -> Result<(), String> {
        self.out.push((
            self.me,
            Msg::Fetch {
                pool,
                slot,
                ordinal,
                requests,
            },
        ));
        Ok(())
    }

    fn events(&mut self) -> Vec<Event> {
        std::mem::take(&mut self.events)
    }
}

/// Work a rank put off: wait, read, then done.
#[derive(Clone, Copy)]
struct Later {
    lane: usize,
    slot: u32,
    ordinal: u64,
}

/// One rank's kernel, as the rules allow it to run.
struct Device {
    program: Vec<usize>,
    at: usize,
    /// The collective in flight before its trigger.
    open: Option<(Later, bool)>,
    later: VecDeque<Later>,
    next: Vec<u64>,
    epoch: u64,
}

/// A rank's send and receive buffers.
#[derive(Default)]
struct Data {
    send: HashMap<(u32, u32), Vec<u64>>,
    recv: HashMap<(u32, u32, u32), Vec<u64>>,
    /// Receive slots written but not yet read, by ordinal.
    unread: HashMap<(u32, u32, u32), u64>,
}

/// A whole run's setup.
pub(crate) struct Plan {
    pub pools: Vec<Pool>,
    /// Global ranks of each pool's group.
    pub groups: Vec<Vec<u32>>,
    pub lanes: Vec<Lane>,
    /// Lanes each rank runs, in order.
    pub programs: Vec<Vec<usize>>,
    /// How many ordinals a rank may defer reads, per pool.
    pub lag: Vec<u64>,
}

struct World<'a> {
    plan: &'a Plan,
    rank_pools: Vec<Vec<Option<u32>>>,
    devices: Vec<Device>,
    proxies: Vec<Proxy>,
    mems: Vec<Mem>,
    links: Vec<SimLink>,
    data: Vec<Data>,
    /// Ordered queues, by sender and receiver.
    wires: HashMap<(u32, u32), VecDeque<Msg>>,
}

/// Runs `plan` under `seed`. Errs on a broken rule.
pub(crate) fn run(plan: &Plan, seed: u64) -> Result<(), String> {
    let ranks = plan.programs.len();
    let mut world = World::new(plan, ranks);
    let mut rng = Rng(seed | 1);
    let mut idle = 0;
    while idle < 20_000 {
        if world.finished() {
            return Ok(());
        }
        let moved = match rng.below(3) {
            0 => world.device(rng.below(ranks), rng.below(4) == 0)?,
            1 => world.proxy(rng.below(ranks))?,
            _ => world.deliver(rng.below(ranks), rng.below(ranks))?,
        };
        idle = if moved { 0 } else { idle + 1 };
    }
    Err(format!("seed {seed}: stalled: {}", world.state()))
}

impl<'a> World<'a> {
    fn new(plan: &'a Plan, ranks: usize) -> Self {
        let rank_pools = (0..ranks as u32)
            .map(|r| {
                plan.groups
                    .iter()
                    .map(|g| g.iter().position(|&x| x == r).map(|i| i as u32))
                    .collect()
            })
            .collect::<Vec<Vec<_>>>();
        let mut proxies = Vec::new();
        let mut links = Vec::new();
        for (r, mine) in rank_pools.iter().enumerate() {
            let pools = plan.pools.iter().zip(mine).map(|(p, me)| Pool {
                me: me.unwrap_or(0),
                ..*p
            });
            proxies.push(Proxy::new(pools.collect(), plan.lanes.clone(), 1));
            let groups = plan.groups.clone();
            links.push(SimLink {
                me: r as u32,
                groups,
                lanes: plan.lanes.clone(),
                ..SimLink::default()
            });
        }
        let device = |program: &Vec<usize>| Device {
            program: program.clone(),
            at: 0,
            open: None,
            later: VecDeque::new(),
            next: vec![1; plan.pools.len()],
            epoch: 0,
        };
        World {
            plan,
            rank_pools,
            devices: plan.programs.iter().map(device).collect(),
            proxies,
            mems: (0..ranks).map(|_| Mem::default()).collect(),
            links,
            data: (0..ranks).map(|_| Data::default()).collect(),
            wires: HashMap::new(),
        }
    }

    fn finished(&self) -> bool {
        let quiet = self
            .devices
            .iter()
            .all(|d| d.at == d.program.len() && d.open.is_none() && d.later.is_empty());
        quiet && self.wires.values().all(VecDeque::is_empty)
    }

    fn state(&self) -> String {
        let at: Vec<_> = self
            .devices
            .iter()
            .map(|d| (d.at, d.later.len(), d.open.map(|o| o.0.ordinal)))
            .collect();
        let wires: usize = self.wires.values().map(VecDeque::len).sum();
        format!("devices {at:?}, {wires} messages in flight")
    }

    fn bump(&mut self, r: usize) {
        self.devices[r].epoch += 1;
        self.mems[r].store(Word::Epoch, self.devices[r].epoch);
    }

    /// One step of rank `r`. `early` runs deferred work first.
    fn device(&mut self, r: usize, early: bool) -> Result<bool, String> {
        if early && !self.devices[r].later.is_empty() {
            return self.settle(r);
        }
        if let Some((job, buffered)) = self.devices[r].open {
            return if buffered {
                self.trigger(r, job)
            } else {
                self.buffer(r, job)
            };
        }
        let d = &mut self.devices[r];
        let Some(&lane) = d.program.get(d.at) else {
            return self.settle(r);
        };
        let pool = self.plan.lanes[lane].pool as usize;
        let ordinal = d.next[pool];
        d.next[pool] += 1;
        d.at += 1;
        let slot = (ordinal % u64::from(self.plan.pools[pool].depth)) as u32;
        d.open = Some((
            Later {
                lane,
                slot,
                ordinal,
            },
            false,
        ));
        self.bump(r);
        Ok(true)
    }

    /// Waits for a free slot, then writes this rank's part.
    fn buffer(&mut self, r: usize, job: Later) -> Result<bool, String> {
        let lane = self.plan.lanes[job.lane];
        let (p, pool) = (lane.pool, self.plan.pools[lane.pool as usize]);
        let need = job.ordinal.saturating_sub(u64::from(pool.depth));
        let mem = &self.mems[r];
        let mut free = mem.load(Word::Sent { pool: p }) >= need;
        if pool.one_way && lane.op == Op::Send {
            let me = self.rank_pools[r][p as usize].unwrap();
            free &= mem.load(Word::Credit {
                pool: p,
                rank: 1 - me,
            }) >= need;
        }
        if !free {
            return Ok(false);
        }
        let group = self.plan.groups[p as usize].len();
        let words = if lane.op == Op::AllToAll {
            group * BLOCK
        } else {
            PART
        };
        let part = (0..words).map(|j| word(r as u32, job.ordinal, j)).collect();
        self.data[r].send.insert((p, job.slot), part);
        if lane.op == Op::AllToAll {
            for k in 0..group as u32 {
                self.mems[r].store(
                    Word::Size {
                        pool: p,
                        slot: job.slot,
                        rank: k,
                    },
                    BLOCK as u64,
                );
            }
        }
        self.devices[r].open = Some((job, true));
        Ok(true)
    }

    /// Settles deferred work the rules require, then triggers.
    fn trigger(&mut self, r: usize, job: Later) -> Result<bool, String> {
        let p = self.plan.lanes[job.lane].pool;
        let lag = self.plan.lag[p as usize];
        let due = self.devices[r]
            .later
            .iter()
            .any(|l| self.plan.lanes[l.lane].pool == p && l.ordinal + lag <= job.ordinal);
        if due {
            return self.settle(r);
        }
        let count = PART as u64;
        let value = (job.lane as u64) << 48 | (job.ordinal & 0xFFFF) << 32 | count;
        self.mems[r].store(
            Word::Trigger {
                pool: p,
                slot: job.slot,
            },
            value,
        );
        self.bump(r);
        self.devices[r].open = None;
        if self.plan.lanes[job.lane].op != Op::Send {
            self.devices[r].later.push_back(job);
        }
        Ok(true)
    }

    /// The oldest deferred collective: wait, check the data, done.
    fn settle(&mut self, r: usize) -> Result<bool, String> {
        let Some(&job) = self.devices[r].later.front() else {
            return Ok(false);
        };
        let lane = self.plan.lanes[job.lane];
        let p = lane.pool;
        let group = &self.plan.groups[p as usize];
        let me = self.rank_pools[r][p as usize].unwrap();
        let from: Vec<u32> = match lane.op {
            Op::Fetch => vec![0],
            Op::Recv => vec![1 - me],
            _ => (0..group.len() as u32).filter(|&k| k != me).collect(),
        };
        let flag = |k| {
            self.mems[r].load(Word::Flag {
                pool: p,
                slot: job.slot,
                rank: k,
            })
        };
        if from.iter().any(|&k| flag(k) != job.ordinal) {
            return Ok(false);
        }
        for &k in &from {
            let key = (p, job.slot, k);
            let got = self.data[r].recv.get(&key).cloned().unwrap_or_default();
            let owner = if lane.op == Op::Fetch {
                u32::MAX
            } else {
                group[k as usize]
            };
            let (first, len) = match lane.op {
                Op::AllToAll => (me as usize * BLOCK, BLOCK),
                _ => (0, PART),
            };
            let want: Vec<u64> = (first..first + len)
                .map(|j| word(owner, job.ordinal, j))
                .collect();
            if got != want {
                return Err(format!(
                    "rank {r} pool {p} ordinal {}: read {got:x?}, want {want:x?}",
                    job.ordinal
                ));
            }
            self.data[r].unread.remove(&key);
        }
        if lane.op == Op::Recv {
            self.mems[r].store(
                Word::Done {
                    pool: p,
                    slot: job.slot,
                },
                job.ordinal,
            );
            self.bump(r);
        }
        self.devices[r].later.pop_front();
        Ok(true)
    }

    fn proxy(&mut self, r: usize) -> Result<bool, String> {
        let moved = self.proxies[r]
            .step(&mut self.mems[r], &mut self.links[r])
            .map_err(|e| e.to_string())?;
        for (to, msg) in self.links[r].out.drain(..) {
            self.wires.entry((r as u32, to)).or_default().push_back(msg);
        }
        Ok(moved)
    }

    /// Delivers the oldest message from `from` to `to`.
    fn deliver(&mut self, from: usize, to: usize) -> Result<bool, String> {
        let Some(msg) = self
            .wires
            .get_mut(&(from as u32, to as u32))
            .and_then(VecDeque::pop_front)
        else {
            return Ok(false);
        };
        match msg {
            Msg::Put {
                pool,
                slot,
                from: k,
                ordinal,
                block,
            } => {
                let part = self.data[from]
                    .send
                    .get(&(pool, slot))
                    .cloned()
                    .unwrap_or_default();
                let part = match block {
                    Some(b) => part[b as usize * BLOCK..][..BLOCK].to_vec(),
                    None => part,
                };
                self.land(to, (pool, slot, k), ordinal, part)?;
                self.links[from].events.push(Event::Sent { pool, ordinal });
            }
            Msg::Credit {
                pool,
                from: k,
                ordinal,
            } => {
                self.mems[to].store(Word::Credit { pool, rank: k }, ordinal);
            }
            Msg::Fetch {
                pool,
                slot,
                ordinal,
                requests,
            } => {
                let rows = (0..PART).map(|j| word(u32::MAX, ordinal, j)).collect();
                self.land(to, (pool, slot, 0), ordinal, rows)?;
                self.links[to].events.push(Event::Fetched {
                    pool,
                    slot,
                    ordinal,
                    rows: requests,
                });
            }
        }
        Ok(true)
    }

    /// Writes a receive slot and its flag. Unread data errs.
    fn land(
        &mut self,
        r: usize,
        key: (u32, u32, u32),
        ordinal: u64,
        part: Vec<u64>,
    ) -> Result<(), String> {
        let data = &mut self.data[r];
        if let Some(old) = data.unread.insert(key, ordinal) {
            return Err(format!(
                "rank {r} slot {key:?}: ordinal {ordinal} overwrote unread {old}"
            ));
        }
        data.recv.insert(key, part);
        let (pool, slot, rank) = key;
        if self
            .plan
            .lanes
            .iter()
            .any(|l| l.pool == pool && l.op == Op::Fetch)
        {
            return Ok(());
        }
        self.mems[r].store(Word::Count { pool, slot, rank }, PART as u64);
        self.mems[r].store(Word::Flag { pool, slot, rank }, ordinal);
        Ok(())
    }
}
