//! Ranks on one box, writing into a mapped file.

use crate::Error;
use crate::link::{Blocking, Link, Put, Span, exchange_regions, fits};
use std::ffi::{c_int, c_void};
use std::fs::File;
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

unsafe extern "C" {
    fn mmap(
        addr: *mut c_void,
        len: usize,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        offset: i64,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> c_int;
}

const PROT_READ_WRITE: c_int = 3;
const MAP_SHARED: c_int = 1;
/// How long a rank waits at a round.
pub(crate) const ROUND: Duration = Duration::from_secs(300);

/// Each rank's block starts on its own 64 KiB page.
pub(crate) const BLOCK: usize = 1 << 16;

/// A file's first `len` bytes, mapped and shared.
pub(crate) struct Mapped {
    base: *mut u8,
    len: usize,
    _file: File,
}

// Shared memory. Each user states who writes which bytes.
unsafe impl Send for Mapped {}

impl Mapped {
    pub(crate) fn over(file: File, len: usize) -> Result<Self, Error> {
        let (prot, fd) = (PROT_READ_WRITE, file.as_raw_fd());
        let base = unsafe { mmap(std::ptr::null_mut(), len, prot, MAP_SHARED, fd, 0) };
        if base as isize == -1 {
            return Err(Error::Io(std::io::Error::last_os_error()));
        }
        Ok(Mapped {
            base: base as *mut u8,
            len,
            _file: file,
        })
    }

    pub(crate) fn base(&self) -> *mut u8 {
        self.base
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// The word at byte `at`, which is 8-byte aligned.
    pub(crate) fn word(&self, at: usize) -> &AtomicU64 {
        assert!(at.is_multiple_of(8) && at + 8 <= self.len);
        // SAFETY: checked in bounds and aligned in a page-aligned map.
        unsafe { &*(self.base.add(at) as *const AtomicU64) }
    }
}

impl Drop for Mapped {
    fn drop(&mut self) {
        unsafe { munmap(self.base as *mut c_void, self.len) };
    }
}

fn padded(n: usize) -> usize {
    n.next_multiple_of(16)
}

fn block(regions: &[usize]) -> usize {
    regions
        .iter()
        .map(|&n| padded(n))
        .sum::<usize>()
        .next_multiple_of(BLOCK)
}

/// A mapped file: each rank's regions, one rank after another.
pub struct ShmLink {
    rank: u32,
    ranks: u32,
    sizes: Vec<usize>,
    /// Region offsets in a rank's block, then its size.
    offsets: Vec<usize>,
    block: usize,
    map: Mapped,
    done: Vec<u64>,
}

impl ShmLink {
    /// The file size `ranks` need for these region sizes.
    pub fn room(ranks: u32, regions: &[usize]) -> usize {
        ranks as usize * block(regions)
    }

    /// Rank `rank` over `file`, zeroed and at least `room` long.
    pub fn over(rank: u32, ranks: u32, file: File, regions: &[usize]) -> Result<Self, Error> {
        let (have, len) = (file.metadata()?.len() as usize, Self::room(ranks, regions));
        if rank >= ranks || have < len {
            let what = format!("{have} bytes for rank {rank} of {ranks}");
            return Err(Error::Fit(what));
        }
        let mut offsets = Vec::with_capacity(regions.len());
        let mut at = 0;
        for &n in regions {
            offsets.push(at);
            at += padded(n);
        }
        Ok(ShmLink {
            rank,
            ranks,
            sizes: regions.to_vec(),
            offsets,
            block: block(regions),
            map: Mapped::over(file, len)?,
            done: Vec::new(),
        })
    }

    /// The address of `at` in `rank`'s regions, checked.
    fn at(&self, rank: u32, at: Span, align: usize) -> Result<*mut u8, Error> {
        if rank >= self.ranks {
            return Err(Error::Fit(format!("no rank {rank} of {}", self.ranks)));
        }
        fits(&self.sizes, at, align)?;
        let offset = rank as usize * self.block + self.offsets[at.region] + at.at;
        Ok(unsafe { self.map.base().add(offset) })
    }

    fn atomic(&self, rank: u32, at: Span) -> Result<&AtomicU64, Error> {
        let word = self.at(rank, Span::new(at.region, at.at, 8), 8)?;
        // Checked: in bounds and 8-byte aligned.
        Ok(unsafe { &*(word as *const AtomicU64) })
    }
}

impl Link for ShmLink {
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
        let from = self.at(self.rank, put.from, 1)?;
        let into = self.at(put.to, put.into, 1)?;
        let ordinal = self.atomic(put.to, put.flag)?;
        let count = self.atomic(put.to, Span::new(put.flag.region, put.flag.at + 8, 8))?;
        unsafe { std::ptr::copy_nonoverlapping(from, into, put.from.len) };
        count.store(put.words[1], Ordering::Relaxed);
        ordinal.store(put.words[0], Ordering::Release);
        self.done.push(put.tag);
        Ok(())
    }

    fn word(&mut self, to: u32, at: Span, value: u64, tag: u64) -> Result<(), Error> {
        self.atomic(to, at)?.store(value, Ordering::Release);
        self.done.push(tag);
        Ok(())
    }

    fn poll(&mut self, done: &mut Vec<u64>) -> Result<(), Error> {
        done.append(&mut self.done);
        Ok(())
    }

    fn reset(&mut self) -> Result<(), Error> {
        self.done.clear();
        Ok(())
    }

    fn memory(&self, region: usize) -> Option<(*mut u8, Option<u64>, usize)> {
        let size = *self.sizes.get(region)?;
        let at = self.at(self.rank, Span::new(region, 0, size), 1).ok()?;
        Some((at, None, size))
    }

    fn load(&self, at: Span) -> Result<u64, Error> {
        Ok(self.atomic(self.rank, at)?.load(Ordering::Acquire))
    }

    fn read(&self, at: Span) -> Result<Vec<u8>, Error> {
        let from = self.at(self.rank, at, 1)?;
        let mut out = vec![0u8; at.len];
        unsafe { std::ptr::copy_nonoverlapping(from, out.as_mut_ptr(), at.len) };
        Ok(out)
    }

    fn write(&mut self, at: Span, bytes: &[u8]) -> Result<(), Error> {
        let into = self.at(self.rank, Span::new(at.region, at.at, bytes.len()), 1)?;
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), into, bytes.len()) };
        Ok(())
    }
}

/// The exchange over a mapped file.
pub type Shm = Blocking<ShmLink>;

impl Blocking<ShmLink> {
    /// The file size `ranks` need for `capacity` byte slots.
    pub fn room(ranks: u32, capacity: usize) -> usize {
        ShmLink::room(ranks, &exchange_regions(ranks, padded(capacity)))
    }

    /// Rank `rank` over `file`, sized by `room` for `capacity`.
    pub fn over(rank: u32, ranks: u32, file: File, capacity: usize) -> Result<Self, Error> {
        let capacity = padded(capacity);
        let link = ShmLink::over(rank, ranks, file, &exchange_regions(ranks, capacity))?;
        Ok(Blocking::new(link, capacity, ROUND))
    }
}
