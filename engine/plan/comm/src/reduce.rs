//! The sum on the device: pinned slots, the kernel's view.

use crate::Error;
use std::sync::{Arc, Mutex};

/// The kernel that copies, publishes, waits and sums.
pub const CUDA: &str = include_str!("reduce.cu");
pub const ENTRY: &str = "all_reduce";

/// Control words at the start of the flag region.
pub(crate) const SEQ: usize = 0;
pub(crate) const BYTES: usize = 1;
pub(crate) const ERR: usize = 3;
/// Words before the first flag.
pub(crate) const CTL_WORDS: usize = 8;

/// Host memory a device addresses too, registered as is.
pub struct Pinned {
    pub host: *mut u8,
    pub device: u64,
    pub len: usize,
}

/// Allocates host memory the device maps.
pub trait Pinning: Send + Sync {
    fn pin(&self, len: usize) -> Result<Pinned, Error>;
    fn unpin(&self, pinned: Pinned);
}

/// The kernel's arguments: device addresses and the round.
pub struct Reduce {
    pub send: u64,
    pub recv: u64,
    pub flags: u64,
    pub ctl: u64,
    /// Bytes per slot, header included.
    pub capacity: usize,
    pub lanes: u32,
    pub rank: u32,
    pub ranks: u32,
    seq: u64,
    pub(crate) failed: Arc<Mutex<Option<String>>>,
}

impl Reduce {
    /// The next round's number.
    pub fn round(&mut self) -> u64 {
        self.seq += 1;
        self.seq
    }

    /// Why the proxy stopped, if it did.
    pub fn failure(&self) -> Option<String> {
        self.failed
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    pub(crate) fn new(fields: [u64; 4], capacity: usize, sizes: [u32; 3]) -> Self {
        Self {
            send: fields[0],
            recv: fields[1],
            flags: fields[2],
            ctl: fields[3],
            capacity,
            lanes: sizes[0],
            rank: sizes[1],
            ranks: sizes[2],
            seq: 0,
            failed: Arc::new(Mutex::new(None)),
        }
    }
}
