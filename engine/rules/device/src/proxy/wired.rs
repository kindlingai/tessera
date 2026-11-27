//! The proxy's link, over a comm link's regions.

use super::{Link, Part, Pool, Sent};
use crate::layout::MAX_GROUP;
use tessera_engine_comm::{Error, Link as Comm, Put, Span};

/// The regions, in order: send, receive, flags, credits.
pub const SEND: usize = 0;
pub const RECV: usize = 1;
pub const FLAGS: usize = 2;
pub(crate) const CREDITS: usize = 3;
pub const REGIONS: usize = 4;

/// A tag: pool, then destination, then the ordinal's low bits.
const ORDINAL: u64 = (1 << 48) - 1;
const CREDIT: u64 = 0xFF;
/// Pools a tag's eight pool bits hold.
const POOLS: usize = 0xFF;

/// One pool's offset in each region, and its slot sizes.
#[derive(Clone, Copy, Debug)]
pub struct Place {
    pub send: usize,
    pub recv: usize,
    pub flags: usize,
    pub credits: usize,
    pub send_bytes: usize,
    pub recv_bytes: usize,
    depth: usize,
}

/// Each pool's place, and every region's size.
fn layout(pools: &[Pool], bytes: &[(usize, usize)]) -> (Vec<Place>, [usize; 4]) {
    let mut ends = [0; 4];
    let mut places = Vec::with_capacity(pools.len());
    for (pool, &(send_bytes, recv_bytes)) in pools.iter().zip(bytes) {
        let depth = pool.depth as usize;
        let sizes = [
            depth * send_bytes,
            depth * MAX_GROUP * recv_bytes,
            depth * MAX_GROUP * 16,
            MAX_GROUP * 8,
        ];
        let [send, recv, flags, credits] = ends;
        places.push(Place {
            send,
            recv,
            flags,
            credits,
            send_bytes,
            recv_bytes,
            depth,
        });
        for (end, size) in ends.iter_mut().zip(sizes) {
            *end += size;
        }
    }
    (places, ends)
}

/// The proxy's puts and credits, as comm puts and words.
pub struct Wired<L> {
    link: L,
    places: Vec<Place>,
    /// Each pool's group, as global ranks, and this rank's place.
    groups: Vec<Vec<u32>>,
    me: Vec<Option<usize>>,
    done: Vec<u64>,
}

fn fit(what: String) -> Error {
    Error::Fit(what)
}

/// Region sizes for pools with these slot sizes.
pub fn regions(pools: &[Pool], bytes: &[(usize, usize)]) -> [usize; REGIONS] {
    layout(pools, bytes).1
}

impl<L: Comm> Wired<L> {
    /// `link` holds `regions(pools, bytes)`.
    pub fn new(
        link: L,
        pools: &[Pool],
        bytes: &[(usize, usize)],
        groups: Vec<Vec<u32>>,
    ) -> Result<Self, Error> {
        let fits = |g: &Vec<u32>| g.len() <= MAX_GROUP;
        if pools.len() >= POOLS || pools.len() != groups.len() || !groups.iter().all(fits) {
            return Err(fit(format!("{} pools do not fit a tag", pools.len())));
        }
        let rank = link.rank();
        let me = groups
            .iter()
            .map(|g| g.iter().position(|&r| r == rank))
            .collect();
        Ok(Wired {
            link,
            places: layout(pools, bytes).0,
            groups,
            me,
            done: Vec::new(),
        })
    }

    pub fn link(&mut self) -> &mut L {
        &mut self.link
    }

    pub fn into_link(self) -> L {
        self.link
    }

    pub(crate) fn place(&self, pool: usize) -> Option<Place> {
        self.places.get(pool).copied()
    }

    /// Pool `pool`'s place, this rank's group rank, and `to`'s rank.
    fn find(&self, pool: u32, slot: u32, to: u32) -> Result<(Place, usize, u32), Error> {
        let p = pool as usize;
        let place = *self
            .places
            .get(p)
            .ok_or_else(|| fit(format!("no pool {p}")))?;
        let me = self.me[p].ok_or_else(|| fit(format!("not in pool {p}")))?;
        let rank = self.groups[p].get(to as usize).copied();
        let rank = rank.ok_or_else(|| fit(format!("pool {p} has no rank {to}")))?;
        if slot as usize >= place.depth {
            return Err(fit(format!("pool {p} has no slot {slot}")));
        }
        Ok((place, me, rank))
    }
}

fn tag(pool: u32, to: u64, ordinal: u64) -> u64 {
    u64::from(pool) << 56 | to << 48 | ordinal & ORDINAL
}

impl<L: Comm> Link for Wired<L> {
    fn put(&mut self, part: Part) -> Result<(), Error> {
        let (place, me, to) = self.find(part.pool, part.slot, part.to)?;
        let (slot, bytes) = (part.slot as usize, part.bytes as usize);
        if part.from as usize + bytes > place.send_bytes || bytes > place.recv_bytes {
            let p = part.pool;
            return Err(fit(format!(
                "pool {p}: a {bytes}-byte part does not fit its slot"
            )));
        }
        let sent = place.send + slot * place.send_bytes + part.from as usize;
        let into = place.recv + (slot * MAX_GROUP + me) * place.recv_bytes;
        let flag = place.flags + (slot * MAX_GROUP + me) * 16;
        let put = Put {
            to,
            from: Span::new(SEND, sent, bytes),
            into: Span::new(RECV, into, bytes),
            flag: Span::new(FLAGS, flag, 16),
            words: [part.ordinal, u64::from(part.count)],
            tag: tag(part.pool, u64::from(part.to), part.ordinal),
        };
        self.link.put(put)
    }

    fn credit(&mut self, pool: u32, to: u32, ordinal: u64) -> Result<(), Error> {
        let (place, me, to) = self.find(pool, 0, to)?;
        let at = Span::new(CREDITS, place.credits + me * 8, 8);
        self.link.word(to, at, ordinal, tag(pool, CREDIT, ordinal))
    }

    fn sent(&mut self) -> Result<Vec<Sent>, Error> {
        self.link.poll(&mut self.done)?;
        let mut sent = Vec::new();
        for tag in self.done.drain(..) {
            let to = (tag >> 48) & 0xFF;
            if to != CREDIT {
                let (pool, ordinal) = ((tag >> 56) as u32, tag & ORDINAL);
                sent.push(Sent {
                    pool,
                    ordinal,
                    to: to as u32,
                });
            }
        }
        Ok(sent)
    }

    fn reset(&mut self) -> Result<(), Error> {
        self.done.clear();
        self.link.reset()
    }
}
