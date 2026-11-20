//! Ranks in one process over shared buffers, for tests.

use crate::Error;
use crate::link::{Link, Put, Span};
use std::sync::{Arc, Mutex};

type Regions = Vec<Vec<Mutex<Vec<u8>>>>;

/// One rank's view of buffers every rank shares.
pub struct Memory {
    rank: u32,
    all: Arc<Regions>,
    done: Vec<u64>,
}

impl Memory {
    /// `ranks` links, each rank holding `regions` of these sizes.
    pub fn ranks(ranks: u32, regions: &[usize]) -> Vec<Memory> {
        let rank = |_| regions.iter().map(|&n| Mutex::new(vec![0u8; n])).collect();
        let all = Arc::new((0..ranks).map(rank).collect::<Regions>());
        let link = |rank| Memory {
            rank,
            all: Arc::clone(&all),
            done: Vec::new(),
        };
        (0..ranks).map(link).collect()
    }

    fn region(&self, rank: u32, region: usize) -> std::sync::MutexGuard<'_, Vec<u8>> {
        self.all[rank as usize][region]
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    fn place(&self, rank: u32, at: Span, bytes: &[u8]) {
        self.region(rank, at.region)[at.at..at.at + bytes.len()].copy_from_slice(bytes);
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
        let bytes = self.read(put.from);
        self.place(put.to, put.into, &bytes);
        let flag: Vec<u8> = put.words.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.place(put.to, put.flag, &flag);
        self.done.push(put.tag);
        Ok(())
    }

    fn word(&mut self, to: u32, at: Span, value: u64, tag: u64) -> Result<(), Error> {
        self.place(to, at, &value.to_le_bytes());
        self.done.push(tag);
        Ok(())
    }

    at_once!();

    fn load(&self, at: Span) -> u64 {
        let region = self.region(self.rank, at.region);
        u64::from_le_bytes(region[at.at..at.at + 8].try_into().expect("8 bytes"))
    }

    fn read(&self, at: Span) -> Vec<u8> {
        self.region(self.rank, at.region)[at.at..at.at + at.len].to_vec()
    }

    fn write(&mut self, at: Span, bytes: &[u8]) {
        self.place(self.rank, at, bytes);
    }
}
