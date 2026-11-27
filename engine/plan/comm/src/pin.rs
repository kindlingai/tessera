//! Host memory a device addresses too.

use crate::Error;

/// Host memory a device addresses too, registered as is.
pub struct Pinned {
    pub host: *mut u8,
    pub device: u64,
}

/// Allocates host memory the device maps.
pub trait Pinning: Send + Sync {
    fn pin(&self, len: usize) -> Result<Pinned, Error>;
    fn unpin(&self, pinned: Pinned);
}
