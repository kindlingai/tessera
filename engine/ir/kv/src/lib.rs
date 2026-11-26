//! Pages of kept rows, shared between runs by refcount.

mod pool;
mod prefix;
mod reuse;
mod spec;
mod store;
mod tier;

pub use pool::{Error, PageCopy, Pool, RunId, Stats, Tag};
pub use prefix::Prefixes;
pub use reuse::{Begin, Why, begin};
pub use spec::{GroupSpec, Keep, KvConfig, StoreKind, TierSpec};
pub use store::{DiskKv, DiskMsg, PageKey, PageStore, StoreStats, decode_msg, page_keys};
pub use tier::{RamStore, TieredStore};

#[cfg(test)]
mod pool_test;
#[cfg(test)]
mod reuse_test;
