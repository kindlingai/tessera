//! Ranks on one box, writing into a mapped file.

use crate::link::{Blocking, Link, Put, Span, exchange_regions};
use crate::{Error, Transport};
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
const ROUND: Duration = Duration::from_secs(300);

fn padded(n: usize) -> usize {
    n.next_multiple_of(16)
}

/// A mapped file: each rank's regions, one rank after another.
pub struct ShmLink {
    rank: u32,
    ranks: u32,
    /// Region offsets in a rank's block, then its size.
    offsets: Vec<usize>,
    block: usize,
    base: *mut u8,
    len: usize,
    done: Vec<u64>,
    _file: File,
}

// Shared memory. A rank writes others' regions only by put.
unsafe impl Send for ShmLink {}

impl ShmLink {
    /// The file size `ranks` need for these region sizes.
    pub fn room(ranks: u32, regions: &[usize]) -> usize {
        ranks as usize * regions.iter().map(|&n| padded(n)).sum::<usize>()
    }

    /// Rank `rank` over `file`, zeroed and sized by `room`.
    pub fn over(rank: u32, ranks: u32, file: File, regions: &[usize]) -> Result<Self, Error> {
        let len = file.metadata()?.len() as usize;
        if rank >= ranks || len != Self::room(ranks, regions) {
            let what = format!("{len} bytes for rank {rank} of {ranks}");
            return Err(Error::Fit(what));
        }
        let mut offsets = Vec::with_capacity(regions.len());
        let mut block = 0;
        for &n in regions {
            offsets.push(block);
            block += padded(n);
        }
        let (prot, fd) = (PROT_READ_WRITE, file.as_raw_fd());
        let base = unsafe { mmap(std::ptr::null_mut(), len, prot, MAP_SHARED, fd, 0) };
        if base as isize == -1 {
            return Err(Error::Io(std::io::Error::last_os_error()));
        }
        Ok(ShmLink {
            rank,
            ranks,
            offsets,
            block,
            base: base as *mut u8,
            len,
            done: Vec::new(),
            _file: file,
        })
    }

    fn at(&self, rank: u32, at: Span) -> *mut u8 {
        let offset = rank as usize * self.block + self.offsets[at.region] + at.at;
        unsafe { self.base.add(offset) }
    }

    fn atomic(&self, rank: u32, at: Span) -> &AtomicU64 {
        // Flags and words sit at multiples of 8.
        unsafe { &*(self.at(rank, at) as *const AtomicU64) }
    }
}

impl Drop for ShmLink {
    fn drop(&mut self) {
        unsafe { munmap(self.base as *mut c_void, self.len) };
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
        let (from, into) = (self.at(self.rank, put.from), self.at(put.to, put.into));
        unsafe { std::ptr::copy_nonoverlapping(from, into, put.from.len) };
        let count = Span::new(put.flag.region, put.flag.at + 8, 8);
        self.atomic(put.to, count)
            .store(put.words[1], Ordering::Relaxed);
        self.atomic(put.to, put.flag)
            .store(put.words[0], Ordering::Release);
        self.done.push(put.tag);
        Ok(())
    }

    fn word(&mut self, to: u32, at: Span, value: u64, tag: u64) -> Result<(), Error> {
        self.atomic(to, at).store(value, Ordering::Release);
        self.done.push(tag);
        Ok(())
    }

    at_once!();

    fn load(&self, at: Span) -> u64 {
        self.atomic(self.rank, at).load(Ordering::Acquire)
    }

    fn read(&self, at: Span) -> Vec<u8> {
        unsafe { std::slice::from_raw_parts(self.at(self.rank, at), at.len) }.to_vec()
    }

    fn write(&mut self, at: Span, bytes: &[u8]) {
        let into = self.at(self.rank, at);
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), into, bytes.len()) };
    }
}

/// The exchange over a mapped file.
pub struct Shm(Blocking<ShmLink>);

impl Shm {
    /// The file size `ranks` need for `capacity` byte slots.
    pub fn room(ranks: u32, capacity: usize) -> usize {
        ShmLink::room(ranks, &exchange_regions(ranks, padded(capacity)))
    }

    /// Rank `rank` over `file`, zeroed and sized by `room`.
    pub fn over(rank: u32, ranks: u32, file: File) -> Result<Self, Error> {
        let len = file.metadata()?.len() as usize;
        let n = ranks.max(1) as usize;
        // Per rank: 2c send, 2nc receive, then 32n of flags.
        let capacity = (len / n).saturating_sub(padded(32 * n)) / (2 + 2 * n);
        let capacity = capacity / 16 * 16;
        if ranks == 0 || Self::room(ranks, capacity) != len {
            return Err(Error::Fit(format!(
                "{len} bytes for rank {rank} of {ranks}"
            )));
        }
        let link = ShmLink::over(rank, ranks, file, &exchange_regions(ranks, capacity))?;
        Ok(Shm(Blocking::new(link, capacity, ROUND)))
    }
}

impl Transport for Shm {
    fn rank(&self) -> u32 {
        self.0.rank()
    }

    fn ranks(&self) -> u32 {
        self.0.ranks()
    }

    fn exchange(&mut self, mine: &[u8]) -> Result<Vec<Vec<u8>>, Error> {
        self.0.exchange(mine)
    }
}
