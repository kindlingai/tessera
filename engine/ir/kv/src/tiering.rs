//! Which KV pages stay in VRAM.

use std::collections::VecDeque;

/// Eviction priority levels for `TIER_HOST` (lowest demotes first).
pub const EVICT_COLD: u8 = 0;
pub(crate) const EVICT_MIDDLE: u8 = 1;
pub(crate) const EVICT_HOT: u8 = 2;
pub(crate) const EVICT_SHARED: u8 = 3;
pub const EVICT_PINNED: u8 = 4;

/// A page's place in its slot's lane.
#[derive(Clone, Copy, Debug)]
pub struct Spot {
    pub at: usize,
    /// Pages in the lane.
    pub len: usize,
    /// Live runs that hold the page.
    pub holders: u32,
}

/// The VRAM pages the live slots share.
#[derive(Clone, Copy, Debug)]
pub struct Load {
    pub pages: usize,
    pub slots: usize,
}

/// Free pages, and the pages a step must place.
#[derive(Clone, Copy, Debug)]
pub struct Room {
    pub need: usize,
    pub vram_free: usize,
    pub host_free: usize,
}

/// A VRAM page's level, or `None` when it must stay.
pub type Level<'a> = &'a dyn Fn(u32) -> Option<u8>;

/// Chooses the tier of each page.
pub trait Tiering: Send {
    /// The level of a lane's page.
    fn class(&self, spot: Spot, load: Load) -> u8;
    /// `page` entered VRAM.
    fn placed(&mut self, page: u32);
    /// The pool freed `pages`.
    fn freed(&mut self, pages: &[u32]);
    /// No page is placed.
    fn clear(&mut self);
    /// VRAM pages to demote when `short` pages lack room.
    fn victims(&mut self, short: usize, host_free: usize, level: Level) -> Vec<u32>;
    /// Host pages to promote, from `host` page and level pairs.
    fn promotes(&self, room: Room, level: Level, host: &[(u32, u8)]) -> Vec<u32>;
}

/// Demotes by level, then the page longest in VRAM.
pub struct Aged {
    /// Early pages per slot kept hot.
    pub prefix: usize,
    /// Recent pages per slot kept hot.
    pub tail: usize,
    /// Fewest pages demoted at once.
    pub demote: usize,
    /// Most hot pages promoted per step.
    pub promote: usize,
    order: VecDeque<u32>,
}

impl Default for Aged {
    fn default() -> Self {
        Self {
            prefix: 32,
            tail: 192,
            demote: 256,
            promote: 256,
            order: VecDeque::new(),
        }
    }
}

impl Aged {
    fn drop_pages(&mut self, pages: &[u32]) {
        if pages.is_empty() {
            return;
        }
        let mut gone = pages.to_vec();
        gone.sort_unstable();
        self.order.retain(|p| gone.binary_search(p).is_err());
    }
}

impl Tiering for Aged {
    fn class(&self, spot: Spot, load: Load) -> u8 {
        let share = load.pages / (load.slots.max(1) * 2);
        let prefix = self.prefix.min(share / 4).max(1);
        let tail = self.tail.min(share.saturating_sub(prefix)).max(1);
        if spot.at + 1 == spot.len {
            EVICT_PINNED
        } else if spot.holders > 1 {
            EVICT_SHARED
        } else if spot.at < prefix || spot.at >= spot.len.saturating_sub(tail) {
            EVICT_HOT
        } else {
            EVICT_MIDDLE
        }
    }

    fn placed(&mut self, page: u32) {
        self.order.push_back(page);
    }

    fn freed(&mut self, pages: &[u32]) {
        self.drop_pages(pages);
    }

    fn clear(&mut self) {
        self.order.clear();
    }

    fn victims(&mut self, short: usize, host_free: usize, level: Level) -> Vec<u32> {
        let want = short.max(self.demote).min(host_free);
        let hot_cap = short.min(host_free);
        let mut out = Vec::with_capacity(want);
        for class in [EVICT_COLD, EVICT_MIDDLE, EVICT_HOT, EVICT_SHARED] {
            if class > EVICT_COLD && out.len() >= hot_cap {
                break;
            }
            let target = if class < EVICT_HOT { want } else { hot_cap };
            for &p in &self.order {
                if out.len() >= target {
                    break;
                }
                if level(p) == Some(class) {
                    out.push(p);
                }
            }
        }
        self.drop_pages(&out);
        out
    }

    fn promotes(&self, room: Room, level: Level, host: &[(u32, u8)]) -> Vec<u32> {
        let (mut cold, mut hot) = (0usize, 0usize);
        for &p in &self.order {
            match level(p) {
                Some(EVICT_HOT) => hot += 1,
                Some(class) if class < EVICT_HOT => cold += 1,
                _ => {}
            }
        }
        let Room {
            need,
            vram_free,
            host_free,
        } = room;
        let mut spare_hot = (vram_free + cold).saturating_sub(need);
        let mut spare_shared = (vram_free + cold + hot).saturating_sub(need);
        let evicts = host_free.saturating_sub(need.saturating_sub(vram_free));
        let mut budget = (vram_free.saturating_sub(need) + evicts).min(self.promote);
        let mut out = Vec::new();
        for target in [EVICT_SHARED, EVICT_HOT] {
            for &(p, class) in host {
                if budget == 0 {
                    return out;
                }
                if class != target {
                    continue;
                }
                if target == EVICT_SHARED {
                    if spare_shared == 0 {
                        break;
                    }
                    spare_shared -= 1;
                    spare_hot = spare_hot.saturating_sub(1);
                } else {
                    if spare_hot == 0 {
                        return out;
                    }
                    spare_hot -= 1;
                    spare_shared = spare_shared.saturating_sub(1);
                }
                budget -= 1;
                out.push(p);
            }
        }
        out
    }
}
