//! The proxy's words, in host memory the device maps.

use super::wired::{CREDITS, FLAGS, Wired};
use super::{Host, Word};
use crate::layout::{DOORBELL_ABORT, DOORBELL_SENT, DOORBELL_SLOTS, MAX_DEPTH, MAX_GROUP, SIZES};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use tessera_engine_comm::{Error, Link as Comm};

/// The doorbell page, and each pool's flags and credits.
pub struct Page {
    page: *mut u64,
    flags: Vec<*mut u64>,
    credits: Vec<*mut u64>,
}

// The device and this proxy share the words, through atomics.
unsafe impl Send for Page {}

impl Page {
    /// # Safety
    /// `page` holds `layout::PAGE` bytes, 8-aligned, while this lives.
    pub unsafe fn new<L: Comm>(page: *mut u64, wired: &mut Wired<L>) -> Result<Self, Error> {
        let mut bases = [std::ptr::null_mut(); 2];
        for (base, region) in bases.iter_mut().zip([FLAGS, CREDITS]) {
            let memory = wired.link().memory(region);
            let what = format!("the link has no fixed region {region}");
            *base = memory.ok_or(Error::Fit(what))?.0;
        }
        let (mut flags, mut credits) = (Vec::new(), Vec::new());
        for pool in 0.. {
            let Some(place) = wired.place(pool) else {
                break;
            };
            flags.push(unsafe { bases[0].add(place.flags) }.cast());
            credits.push(unsafe { bases[1].add(place.credits) }.cast());
        }
        Ok(Page {
            page,
            flags,
            credits,
        })
    }

    fn word(&self, word: Word) -> &AtomicU64 {
        let slot = |pool: u32, slot: u32| {
            DOORBELL_SLOTS / 8 + (pool as usize * MAX_DEPTH + slot as usize) * 2
        };
        let flag = |pool: u32, slot: u32, rank: u32| {
            let at = (slot as usize * MAX_GROUP + rank as usize) * 2;
            unsafe { self.flags[pool as usize].add(at) }
        };
        let at = match word {
            Word::Epoch => self.page,
            Word::Trigger { pool, slot: s } => unsafe { self.page.add(slot(pool, s)) },
            Word::Done { pool, slot: s } => unsafe { self.page.add(slot(pool, s) + 1) },
            Word::Abort => unsafe { self.page.add(DOORBELL_ABORT / 8) },
            Word::Sent { pool } => unsafe { self.page.add(DOORBELL_SENT / 8 + pool as usize) },
            Word::Flag { pool, slot, rank } => flag(pool, slot, rank),
            Word::Count { pool, slot, rank } => unsafe { flag(pool, slot, rank).add(1) },
            Word::Credit { pool, rank } => unsafe {
                self.credits[pool as usize].add(rank as usize)
            },
            Word::Size { .. } => unreachable!("sizes are 32-bit"),
        };
        // In bounds for the pool shapes `Proxy` accepts.
        unsafe { AtomicU64::from_ptr(at) }
    }

    /// The word the device bumps after each trigger.
    pub(super) fn epoch(&self) -> &AtomicU64 {
        self.word(Word::Epoch)
    }

    fn size(&self, pool: u32, slot: u32, rank: u32) -> u64 {
        let index = (pool as usize * MAX_DEPTH + slot as usize) * MAX_GROUP + rank as usize;
        let at = unsafe { self.page.cast::<u8>().add(SIZES + index * 4) };
        u64::from(unsafe { AtomicU32::from_ptr(at.cast()) }.load(Ordering::Acquire))
    }
}

impl Host for Page {
    fn load(&self, word: Word) -> u64 {
        match word {
            Word::Size { pool, slot, rank } => self.size(pool, slot, rank),
            _ => self.word(word).load(Ordering::Acquire),
        }
    }

    fn store(&mut self, word: Word, value: u64) {
        self.word(word).store(value, Ordering::Release);
    }
}
