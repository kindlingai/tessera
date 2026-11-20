//! Ranks, proxies and links under a seeded random schedule.

use crate::proxy::{Fetched, Fetcher, Host, Lane, Link, Op, Part, Pool, Proxy, Sent, Word};
use std::collections::{HashMap, VecDeque};

/// Words in one part, and in one all-to-all block.
const PART: usize = 4;
const BLOCK: usize = 2;
/// How far a resync moves every ordinal forward.
const JUMP: u64 = 70_000;

/// A small seeded generator.
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n as u64) as usize
    }

    /// A speed: mostly fast, some 10 or 100 times slower.
    fn speed(&mut self) -> usize {
        [100, 100, 10, 1][self.below(4)]
    }
}

/// The value word `j` of `rank`'s part for `ordinal` holds.
fn word(rank: u32, ordinal: u64, j: usize) -> u64 {
    (u64::from(rank) << 48) ^ (ordinal << 16) ^ j as u64
}

/// Host memory: counts stores, notes a word that went back.
#[derive(Default)]
struct Mem {
    words: HashMap<Word, u64>,
    back: Option<String>,
    stores: u64,
}

impl Host for Mem {
    fn load(&self, word: Word) -> u64 {
        self.words.get(&word).copied().unwrap_or(0)
    }

    fn store(&mut self, word: Word, value: u64) {
        self.stores += 1;
        let rising = matches!(word, Word::Sent { .. } | Word::Credit { .. });
        let old = self.words.insert(word, value).unwrap_or(0);
        if rising && value < old && self.back.is_none() {
            self.back = Some(format!("{word:?} went from {old} to {value}"));
        }
    }
}

enum Msg {
    Put {
        slot: u32,
        from: u32,
        ordinal: u64,
        count: u32,
        block: Option<u32>,
    },
    Credit {
        from: u32,
        ordinal: u64,
    },
    Fetch {
        slot: u32,
        ordinal: u64,
        requests: u32,
    },
}

/// Messages for the network, and finished work.
#[derive(Default)]
struct SimLink {
    out: Vec<(u32, u32, Msg)>,
    sent: Vec<Sent>,
    me: u32,
    groups: Vec<Vec<u32>>,
    split: Vec<bool>,
}

impl SimLink {
    fn place(&self, pool: u32) -> u32 {
        let group = &self.groups[pool as usize];
        group.iter().position(|&r| r == self.me).unwrap() as u32
    }
}

impl Link for SimLink {
    fn put(&mut self, part: Part) -> Result<(), String> {
        let Part {
            pool,
            slot,
            to,
            ordinal,
            count,
            ..
        } = part;
        let from = self.place(pool);
        let block = self.split[pool as usize].then_some(to);
        let msg = Msg::Put {
            slot,
            from,
            ordinal,
            count,
            block,
        };
        self.out
            .push((self.groups[pool as usize][to as usize], pool, msg));
        Ok(())
    }

    fn credit(&mut self, pool: u32, to: u32, ordinal: u64) -> Result<(), String> {
        let from = self.place(pool);
        let msg = Msg::Credit { from, ordinal };
        self.out
            .push((self.groups[pool as usize][to as usize], pool, msg));
        Ok(())
    }

    fn sent(&mut self) -> Result<Vec<Sent>, String> {
        Ok(std::mem::take(&mut self.sent))
    }

    fn reset(&mut self) {
        self.out.clear();
        self.sent.clear();
    }
}

/// Fetch requests and results, beside a rank's link.
#[derive(Default)]
struct SimFetch {
    me: u32,
    out: Vec<(u32, u32, Msg)>,
    fetched: Vec<Fetched>,
}

impl Fetcher for SimFetch {
    fn fetch(&mut self, pool: u32, slot: u32, ordinal: u64, requests: u32) -> Result<(), String> {
        self.out.push((
            self.me,
            pool,
            Msg::Fetch {
                slot,
                ordinal,
                requests,
            },
        ));
        Ok(())
    }

    fn fetched(&mut self) -> Vec<Fetched> {
        std::mem::take(&mut self.fetched)
    }

    fn reset(&mut self) {
        self.out.clear();
        self.fetched.clear();
    }
}

/// One collective a rank started.
#[derive(Clone, Copy)]
struct Job {
    lane: usize,
    slot: u32,
    ordinal: u64,
}

/// Where a rank's open collective stands.
#[derive(Clone, Copy, PartialEq)]
enum Phase {
    Prepared,
    Buffered,
    Stored,
}

/// One rank's kernel, as the rules allow it to run.
struct Device {
    program: Vec<usize>,
    at: usize,
    open: Option<(Job, Phase)>,
    /// Deferred waits and reads, per pool.
    later: Vec<VecDeque<Job>>,
    next: Vec<u64>,
    epoch: u64,
    /// A word was stored and its epoch bump is due.
    bump: bool,
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
    /// The first ordinal, and the step every rank resyncs at.
    pub base: u64,
    pub resync: Option<usize>,
}

/// Actor speeds for one seed.
struct Speeds {
    devices: Vec<usize>,
    proxies: Vec<usize>,
    wires: HashMap<(u32, u32, u32), usize>,
}

#[derive(Clone, Copy)]
enum Act {
    Device(usize, bool),
    Proxy(usize),
    Wire((u32, u32, u32)),
}

struct World<'a> {
    plan: &'a Plan,
    places: Vec<Vec<Option<u32>>>,
    devices: Vec<Device>,
    proxies: Vec<Proxy>,
    mems: Vec<Mem>,
    links: Vec<SimLink>,
    fetchers: Vec<SimFetch>,
    data: Vec<Data>,
    /// Ordered queues, by sender, receiver and pool.
    wires: HashMap<(u32, u32, u32), VecDeque<Msg>>,
    resynced: bool,
    /// A blocked device's store count, and proxies with news.
    stuck: Vec<Option<u64>>,
    news: Vec<bool>,
}

/// Runs `plan` under `seed`. Errs on a broken rule.
pub(crate) fn run(plan: &Plan, seed: u64) -> Result<(), String> {
    let ranks = plan.programs.len();
    let mut world = World::new(plan)?;
    let mut rng = Rng(seed | 1);
    let mut speeds = Speeds {
        devices: (0..ranks).map(|_| rng.speed()).collect(),
        proxies: (0..ranks).map(|_| rng.speed()).collect(),
        wires: HashMap::new(),
    };
    while !world.finished() {
        world.resync()?;
        let Some(act) = world.pick(&mut speeds, &mut rng) else {
            return Err(format!("seed {seed}: stalled: {}", world.state()));
        };
        world.act(act)?;
        if let Some(back) = world.mems.iter().find_map(|m| m.back.as_ref()) {
            return Err(format!("seed {seed}: {back}"));
        }
    }
    Ok(())
}

impl<'a> World<'a> {
    fn new(plan: &'a Plan) -> Result<Self, String> {
        let ranks = plan.programs.len() as u32;
        let place = |r: u32| {
            let found = |g: &Vec<u32>| g.iter().position(|&x| x == r).map(|i| i as u32);
            plan.groups.iter().map(found).collect()
        };
        let pools = 0..plan.pools.len() as u32;
        let split = pools.map(|p| {
            plan.lanes
                .iter()
                .any(|l| l.pool == p && l.op == Op::AllToAll)
        });
        let split: Vec<bool> = split.collect();
        let mut world = World {
            plan,
            places: (0..ranks).map(place).collect(),
            devices: Vec::new(),
            proxies: Vec::new(),
            mems: (0..ranks).map(|_| Mem::default()).collect(),
            links: Vec::new(),
            fetchers: Vec::new(),
            data: (0..ranks).map(|_| Data::default()).collect(),
            wires: HashMap::new(),
            resynced: false,
            stuck: vec![None; ranks as usize],
            news: vec![true; ranks as usize],
        };
        for r in 0..ranks {
            let mine = world.places[r as usize].iter().zip(&plan.pools);
            let pools = mine
                .map(|(me, p)| Pool {
                    me: me.unwrap_or(0),
                    ..*p
                })
                .collect();
            let proxy = Proxy::new(
                &mut world.mems[r as usize],
                pools,
                plan.lanes.clone(),
                plan.base,
            );
            world.proxies.push(proxy.map_err(|e| e.to_string())?);
            let groups = plan.groups.clone();
            world.links.push(SimLink {
                me: r,
                groups,
                split: split.clone(),
                ..SimLink::default()
            });
            world.fetchers.push(SimFetch {
                me: r,
                ..SimFetch::default()
            });
            world.devices.push(Device {
                program: plan.programs[r as usize].clone(),
                at: 0,
                open: None,
                later: vec![VecDeque::new(); plan.pools.len()],
                next: vec![plan.base; plan.pools.len()],
                epoch: 0,
                bump: false,
            });
        }
        Ok(world)
    }

    fn idle(d: &Device) -> bool {
        let done = d.later.iter().all(VecDeque::is_empty) && d.open.is_none() && !d.bump;
        d.at == d.program.len() && done
    }

    fn finished(&self) -> bool {
        self.devices.iter().all(World::idle) && self.wires.values().all(VecDeque::is_empty)
    }

    fn state(&self) -> String {
        let at: Vec<_> = self
            .devices
            .iter()
            .map(|d| (d.at, d.open.map(|o| o.0.ordinal)))
            .collect();
        let wires: usize = self.wires.values().map(VecDeque::len).sum();
        format!("devices {at:?}, {wires} messages in flight")
    }

    /// A weighted pick among the actors that may move.
    fn pick(&self, speeds: &mut Speeds, rng: &mut Rng) -> Option<Act> {
        let mut acts = Vec::new();
        for r in 0..self.devices.len() {
            let blocked = self.stuck[r] == Some(self.mems[r].stores);
            if !blocked && !World::idle(&self.devices[r]) {
                acts.push((Act::Device(r, rng.below(4) == 0), speeds.devices[r]));
            }
            if self.news[r] {
                acts.push((Act::Proxy(r), speeds.proxies[r]));
            }
        }
        for (key, queue) in &self.wires {
            if !queue.is_empty() {
                let speed = *speeds.wires.entry(*key).or_insert_with(|| rng.speed());
                acts.push((Act::Wire(*key), speed));
            }
        }
        let total: usize = acts.iter().map(|a| a.1).sum();
        let mut at = rng.below(total.max(1));
        for (act, speed) in &acts {
            if at < *speed {
                return Some(*act);
            }
            at -= speed;
        }
        None
    }

    fn act(&mut self, act: Act) -> Result<(), String> {
        match act {
            Act::Device(r, early) => {
                let moved = self.device(r, early)?;
                self.stuck[r] = (!moved).then_some(self.mems[r].stores);
                self.news[r] |= moved;
            }
            Act::Proxy(r) => {
                self.news[r] = false;
                self.proxy(r)?;
            }
            Act::Wire(key) => {
                self.deliver(key)?;
                self.news[key.0 as usize] = true;
                self.news[key.1 as usize] = true;
            }
        }
        Ok(())
    }

    /// When all ranks reach the resync step: drop work, jump.
    fn resync(&mut self) -> Result<(), String> {
        let Some(step) = self.plan.resync else {
            return Ok(());
        };
        if self.resynced || self.devices.iter().any(|d| d.at < step) {
            return Ok(());
        }
        self.resynced = true;
        let base = self.plan.base + JUMP;
        self.wires.clear();
        for r in 0..self.devices.len() {
            let d = &mut self.devices[r];
            d.at = step;
            d.open = None;
            d.bump = false;
            d.later.iter_mut().for_each(VecDeque::clear);
            d.next.iter_mut().for_each(|n| *n = base);
            self.data[r].unread.clear();
            let (link, fetch) = (&mut self.links[r], &mut self.fetchers[r]);
            let reset = self.proxies[r].reset(&mut self.mems[r], link, fetch, base);
            reset.map_err(|e| e.to_string())?;
            self.stuck[r] = None;
            self.news[r] = true;
        }
        Ok(())
    }

    fn bump(&mut self, r: usize) {
        self.devices[r].epoch += 1;
        self.devices[r].bump = false;
        self.mems[r].store(Word::Epoch, self.devices[r].epoch);
    }

    /// One step of rank `r`. `early` runs deferred work first.
    fn device(&mut self, r: usize, early: bool) -> Result<bool, String> {
        if self.devices[r].bump {
            self.bump(r);
            return Ok(true);
        }
        let waiting = self.devices[r].later.iter().position(|l| !l.is_empty());
        if let (true, Some(p)) = (early, waiting)
            && self.settle(r, p)?
        {
            return Ok(true);
        }
        if let Some((job, phase)) = self.devices[r].open {
            return match phase {
                Phase::Prepared => self.buffer(r, job),
                Phase::Buffered => self.trigger(r, job),
                Phase::Stored => {
                    self.devices[r].open = None;
                    self.bump(r);
                    Ok(true)
                }
            };
        }
        let d = &mut self.devices[r];
        let Some(&lane) = d.program.get(d.at) else {
            return waiting.map_or(Ok(false), |p| self.settle(r, p));
        };
        let pool = self.plan.lanes[lane].pool as usize;
        let ordinal = d.next[pool];
        d.next[pool] += 1;
        d.at += 1;
        let slot = (ordinal % u64::from(self.plan.pools[pool].depth)) as u32;
        d.open = Some((
            Job {
                lane,
                slot,
                ordinal,
            },
            Phase::Prepared,
        ));
        self.bump(r);
        Ok(true)
    }

    /// Waits for a free slot, then writes this rank's part.
    fn buffer(&mut self, r: usize, job: Job) -> Result<bool, String> {
        let lane = self.plan.lanes[job.lane];
        let (p, pool) = (lane.pool, self.plan.pools[lane.pool as usize]);
        let need = job.ordinal.saturating_sub(u64::from(pool.depth));
        let mem = &self.mems[r];
        let mut free = mem.load(Word::Sent { pool: p }) >= need;
        if pool.one_way && lane.op == Op::Send {
            let me = self.places[r][p as usize].unwrap();
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
                let size = Word::Size {
                    pool: p,
                    slot: job.slot,
                    rank: k,
                };
                self.mems[r].store(size, BLOCK as u64);
            }
        }
        self.devices[r].open = Some((job, Phase::Buffered));
        Ok(true)
    }

    /// Settles deferred work the rules require, then stores the word.
    fn trigger(&mut self, r: usize, job: Job) -> Result<bool, String> {
        let p = self.plan.lanes[job.lane].pool as usize;
        let lag = self.plan.lag[p];
        let front = self.devices[r].later[p].front();
        if front.is_some_and(|l| l.ordinal + lag <= job.ordinal) {
            return self.settle(r, p);
        }
        let lane = (job.lane as u64) << 48 | (job.ordinal & 0xFFFF) << 32;
        let value = 1 << 63 | lane | PART as u64;
        self.mems[r].store(
            Word::Trigger {
                pool: p as u32,
                slot: job.slot,
            },
            value,
        );
        self.devices[r].open = Some((job, Phase::Stored));
        if self.plan.lanes[job.lane].op != Op::Send {
            self.devices[r].later[p].push_back(job);
        }
        Ok(true)
    }

    /// Pool `p`'s oldest deferred collective: wait, check, done.
    fn settle(&mut self, r: usize, p: usize) -> Result<bool, String> {
        let Some(&job) = self.devices[r].later[p].front() else {
            return Ok(false);
        };
        let (op, pool, slot) = (self.plan.lanes[job.lane].op, p as u32, job.slot);
        let group = &self.plan.groups[p];
        let me = self.places[r][p].unwrap();
        let from: Vec<u32> = match op {
            Op::Fetch => vec![0],
            Op::Recv => vec![1 - me],
            _ => (0..group.len() as u32).filter(|&k| k != me).collect(),
        };
        let flag = |k| {
            self.mems[r].load(Word::Flag {
                pool,
                slot,
                rank: k,
            })
        };
        if from.iter().any(|&k| flag(k) != job.ordinal) {
            return Ok(false);
        }
        let split = op == Op::AllToAll;
        let (first, len) = if split {
            (me as usize * BLOCK, BLOCK)
        } else {
            (0, PART)
        };
        for &k in &from {
            let key = (pool, slot, k);
            let got = self.data[r].recv.get(&key).cloned().unwrap_or_default();
            let owner = if op == Op::Fetch {
                u32::MAX
            } else {
                group[k as usize]
            };
            let want: Vec<u64> = (first..first + len)
                .map(|j| word(owner, job.ordinal, j))
                .collect();
            let count = self.mems[r].load(Word::Count {
                pool,
                slot,
                rank: k,
            });
            if got != want || count != len as u64 {
                let ordinal = job.ordinal;
                return Err(format!(
                    "rank {r} pool {p} ordinal {ordinal}: read {got:x?} ({count})"
                ));
            }
            self.data[r].unread.remove(&key);
        }
        self.devices[r].later[p].pop_front();
        if op == Op::Recv {
            self.mems[r].store(Word::Done { pool, slot }, job.ordinal);
            self.devices[r].bump = true;
        }
        Ok(true)
    }

    fn proxy(&mut self, r: usize) -> Result<bool, String> {
        let (link, fetch) = (&mut self.links[r], &mut self.fetchers[r]);
        let moved = self.proxies[r].step(&mut self.mems[r], link, fetch);
        let out = link.out.drain(..).chain(fetch.out.drain(..));
        for (to, pool, msg) in out {
            self.wires
                .entry((r as u32, to, pool))
                .or_default()
                .push_back(msg);
        }
        moved.map_err(|e| e.to_string())
    }

    /// Delivers the oldest message on one queue.
    fn deliver(&mut self, key: (u32, u32, u32)) -> Result<bool, String> {
        let Some(msg) = self.wires.get_mut(&key).and_then(VecDeque::pop_front) else {
            return Ok(false);
        };
        let (from, to, pool) = (key.0 as usize, key.1 as usize, key.2);
        match msg {
            Msg::Put {
                slot,
                from: k,
                ordinal,
                count,
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
                self.land(to, (pool, slot, k), ordinal, part, Some(count))?;
                let place = self.places[to][pool as usize].unwrap();
                self.links[from].sent.push(Sent {
                    pool,
                    ordinal,
                    to: place,
                });
            }
            Msg::Credit { from: k, ordinal } => {
                self.mems[to].store(Word::Credit { pool, rank: k }, ordinal);
            }
            Msg::Fetch {
                slot,
                ordinal,
                requests,
            } => {
                let rows = (0..PART).map(|j| word(u32::MAX, ordinal, j)).collect();
                self.land(to, (pool, slot, 0), ordinal, rows, None)?;
                let rows = requests;
                self.fetchers[to].fetched.push(Fetched {
                    pool,
                    slot,
                    ordinal,
                    rows,
                });
            }
        }
        Ok(true)
    }

    /// Writes a receive slot, then any flag. Unread data errs.
    fn land(
        &mut self,
        r: usize,
        key: (u32, u32, u32),
        ordinal: u64,
        part: Vec<u64>,
        count: Option<u32>,
    ) -> Result<(), String> {
        let data = &mut self.data[r];
        if let Some(old) = data.unread.insert(key, ordinal) {
            return Err(format!(
                "rank {r} slot {key:?}: ordinal {ordinal} overwrote unread {old}"
            ));
        }
        data.recv.insert(key, part);
        let Some(count) = count else {
            return Ok(());
        };
        let (pool, slot, rank) = key;
        self.mems[r].store(Word::Count { pool, slot, rank }, u64::from(count));
        self.mems[r].store(Word::Flag { pool, slot, rank }, ordinal);
        Ok(())
    }
}
