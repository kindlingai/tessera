//! Steps one persistent grid runs, with its sums inside.

use crate::Error;
use tessera_lower::KernelRef;

/// One table item: a device address and its bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Item {
    pub at: u64,
    pub bytes: usize,
}

/// Runs resident steps. One launch per step, ended each time.
pub trait Grid {
    /// Readies `kernel` for launches of `items` items.
    fn open(&mut self, kernel: &KernelRef, items: usize) -> Result<(), Error>;
    /// One launch of `kernel`. It returns once the grid ends.
    fn launch(&mut self, kernel: &KernelRef, items: &[Item], values: &[u32]) -> Result<(), Error>;
}
