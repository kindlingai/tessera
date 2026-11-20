//! The proxy's link, over a comm link's regions.

use super::{Link, Part, Pool, Sent};
use sensai_engine_comm::{Link as Comm, Put, Span};

/// The regions, in order: send, receive, flags, credits.
pub(crate) const SEND: usize = 0;
pub(crate) const RECV: usize = 1;
pub(crate) const FLAGS: usize = 2;
pub(crate) const CREDITS: usize = 3;

const MAX_GROUP: usize = 8;
/// A tag: pool, then destination, then the ordinal's low bits.
const ORDINAL: u64 = (1 << 48) - 1;
const CREDIT: u64 = 0xFF;

/// One pool's offset in each region, and its slot sizes.
#[derive(Clone, Copy, Debug)]
struct Place {
    send: usize,
    recv: usize,
    flags: usize,
    credits: usize,
    send_bytes: usize,
    recv_bytes: usize,
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
        });
        for (end, size) in ends.iter_mut().zip(sizes) {
            *end += size;
        }
    }
    (places, ends)
}

/// The proxy's puts and credits, as comm puts and words.
pub(crate) struct Wired<L> {
    link: L,
    places: Vec<Place>,
    /// Each pool's group, as global ranks, and this rank's place.
    groups: Vec<Vec<u32>>,
    me: Vec<u32>,
    done: Vec<u64>,
}

impl<L: Comm> Wired<L> {
    /// Region sizes for pools with these slot sizes.
    pub(crate) fn regions(pools: &[Pool], bytes: &[(usize, usize)]) -> [usize; 4] {
        layout(pools, bytes).1
    }

    /// `link` holds `regions(pools, bytes)`.
    pub(crate) fn new(
        link: L,
        pools: &[Pool],
        bytes: &[(usize, usize)],
        groups: Vec<Vec<u32>>,
    ) -> Self {
        let rank = link.rank();
        let place = |g: &Vec<u32>| g.iter().position(|&r| r == rank).unwrap_or(0) as u32;
        let me = groups.iter().map(place).collect();
        Wired {
            link,
            places: layout(pools, bytes).0,
            groups,
            me,
            done: Vec::new(),
        }
    }

    pub(crate) fn link(&mut self) -> &mut L {
        &mut self.link
    }
}

fn tag(pool: u32, to: u64, ordinal: u64) -> u64 {
    u64::from(pool) << 56 | to << 48 | ordinal & ORDINAL
}

impl<L: Comm> Link for Wired<L> {
    fn put(&mut self, part: Part) -> Result<(), String> {
        let p = part.pool as usize;
        let (place, me) = (self.places[p], self.me[p] as usize);
        let (slot, bytes) = (part.slot as usize, part.bytes as usize);
        if part.from as usize + bytes > place.send_bytes || bytes > place.recv_bytes {
            return Err(format!(
                "pool {p}: a {bytes}-byte part does not fit its slot"
            ));
        }
        let sent = place.send + slot * place.send_bytes + part.from as usize;
        let into = place.recv + (slot * MAX_GROUP + me) * place.recv_bytes;
        let flag = place.flags + (slot * MAX_GROUP + me) * 16;
        let put = Put {
            to: self.groups[p][part.to as usize],
            from: Span::new(SEND, sent, bytes),
            into: Span::new(RECV, into, bytes),
            flag: Span::new(FLAGS, flag, 16),
            words: [part.ordinal, u64::from(part.count)],
            tag: tag(part.pool, u64::from(part.to), part.ordinal),
        };
        self.link.put(put).map_err(|e| e.to_string())
    }

    fn credit(&mut self, pool: u32, to: u32, ordinal: u64) -> Result<(), String> {
        let p = pool as usize;
        let at = Span::new(CREDITS, self.places[p].credits + self.me[p] as usize * 8, 8);
        let to = self.groups[p][to as usize];
        let tag = tag(pool, CREDIT, ordinal);
        self.link
            .word(to, at, ordinal, tag)
            .map_err(|e| e.to_string())
    }

    fn sent(&mut self) -> Result<Vec<Sent>, String> {
        self.link.poll(&mut self.done).map_err(|e| e.to_string())?;
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

    fn reset(&mut self) {
        self.link.reset();
        self.done.clear();
    }
}
