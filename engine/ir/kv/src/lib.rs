//! Pages of kept rows, shared between runs by refcount.

mod evict;
mod flight;
mod lanes;
mod pool;
mod prefix;
mod reuse;
mod spec;
mod store;
mod tier;
mod tiering;

pub use evict::{Evict, Oldest};
pub use flight::{Move, pair_runs};
pub use lanes::Lanes;
pub use pool::{Error, PageCopy, Pool, RunId, Stats, Tag};
pub use prefix::Prefixes;
pub use reuse::{Begin, Why, begin};
pub use spec::{Keep, KvConfig, StoreKind, TierSpec};
pub use store::{DiskKv, DiskMsg, PageStore, StoreStats, decode_msg, page_keys};
pub use tier::{RamStore, TieredStore};
pub use tiering::{Aged, EVICT_COLD, EVICT_PINNED, Room, Spot, Tiering};

#[cfg(test)]
mod flight_test;
#[cfg(test)]
mod lanes_test;
#[cfg(test)]
mod pool_test;
#[cfg(test)]
mod reuse_test;
#[cfg(test)]
mod tiering_test;
