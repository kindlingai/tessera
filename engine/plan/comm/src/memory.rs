//! Ranks in one process over shared buffers, for tests.

use crate::Error;
use crate::link::{Link, Put, Span, fits};
use std::sync::{Arc, Mutex, MutexGuard};

type Regions = Vec<Vec<Mutex<Vec<u8>>>>;

/// Writes that land at a rank, then their tag.
struct Landing {
    to: u32,
    writes: Vec<(Span, Vec<u8>)>,
    tag: u64,
}

/// One rank's view of buffers every rank shares.
pub struct Memory {
    rank: u32,
    all: Arc<Regions>,
    sizes: Vec<usize>,
    /// Writes land when this rank next polls, not at once.
    deferred: bool,
    queue: Vec<Landing>,
    done: Vec<u64>,
}

impl Memory {
    /// `ranks` links, each rank holding `regions` of these sizes.
    pub fn ranks(ranks: u32, regions: &[usize]) -> Vec<Memory> {
        Self::make(ranks, regions, false)
    }

    /// The same, but puts land when their sender polls.
    pub fn deferred(ranks: u32, regions: &[usize]) -> Vec<Memory> {
        Self::make(ranks, regions, true)
    }

    fn make(ranks: u32, regions: &[usize], deferred: bool) -> Vec<Memory> {
        let rank = |_| regions.iter().map(|&n| Mutex::new(vec![0u8; n])).collect();
        let all = Arc::new((0..ranks).map(rank).collect::<Regions>());
        let link = |rank| Memory {
            rank,
            all: Arc::clone(&all),
            sizes: regions.to_vec(),
            deferred,
            queue: Vec::new(),
            done: Vec::new(),
        };
        (0..ranks).map(link).collect()
    }

    fn region(&self, rank: u32, region: usize) -> MutexGuard<'_, Vec<u8>> {
        let lock = self.all[rank as usize][region].lock();
        lock.unwrap_or_else(|e| e.into_inner())
    }

    fn check(&self, rank: u32, at: Span, align: usize) -> Result<(), Error> {
        if rank as usize >= self.all.len() {
            return Err(Error::Fit(format!("no rank {rank}")));
        }
        fits(&self.sizes, at, align)
    }

    fn land(&mut self, landing: Landing) {
        for (at, bytes) in &landing.writes {
            self.region(landing.to, at.region)[at.at..at.at + bytes.len()].copy_from_slice(bytes);
        }
        self.done.push(landing.tag);
    }

    fn send(&mut self, landing: Landing) {
        match self.deferred {
            true => self.queue.push(landing),
            false => self.land(landing),
        }
    }
}

impl Link for Memory {
    fn rank(&self) -> u32 {
        self.rank
    }

    fn ranks(&self) -> u32 {
        self.all.len() as u32
    }

    fn put(&mut self, put: Put) -> Result<(), Error> {
        if put.from.len != put.into.len || put.flag.len != 16 {
            return Err(Error::Fit(format!("{put:?} has spans that differ")));
        }
        self.check(put.to, put.into, 1)?;
        self.check(put.to, put.flag, 8)?;
        let bytes = self.read(put.from)?;
        let flag: Vec<u8> = put.words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let writes = vec![(put.into, bytes), (put.flag, flag)];
        self.send(Landing {
            to: put.to,
            writes,
            tag: put.tag,
        });
        Ok(())
    }

    fn word(&mut self, to: u32, at: Span, value: u64, tag: u64) -> Result<(), Error> {
        let at = Span::new(at.region, at.at, 8);
        self.check(to, at, 8)?;
        let writes = vec![(at, value.to_le_bytes().to_vec())];
        self.send(Landing { to, writes, tag });
        Ok(())
    }

    fn poll(&mut self, done: &mut Vec<u64>) -> Result<(), Error> {
        for landing in std::mem::take(&mut self.queue) {
            self.land(landing);
        }
        done.append(&mut self.done);
        Ok(())
    }

    fn reset(&mut self) -> Result<(), Error> {
        self.queue.clear();
        self.done.clear();
        Ok(())
    }

    fn load(&self, at: Span) -> Result<u64, Error> {
        let at = Span::new(at.region, at.at, 8);
        self.check(self.rank, at, 8)?;
        let region = self.region(self.rank, at.region);
        Ok(u64::from_le_bytes(
            region[at.at..at.at + 8].try_into().expect("8 bytes"),
        ))
    }

    fn read(&self, at: Span) -> Result<Vec<u8>, Error> {
        self.check(self.rank, at, 1)?;
        Ok(self.region(self.rank, at.region)[at.at..at.at + at.len].to_vec())
    }

    fn write(&mut self, at: Span, bytes: &[u8]) -> Result<(), Error> {
        let at = Span::new(at.region, at.at, bytes.len());
        self.check(self.rank, at, 1)?;
        self.region(self.rank, at.region)[at.at..at.at + at.len].copy_from_slice(bytes);
        Ok(())
    }
}
