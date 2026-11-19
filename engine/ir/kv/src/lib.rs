//! Pages of kept rows, shared between runs by refcount.

mod pool;
mod prefix;
mod reuse;

pub use pool::{Error, PageCopy, Pool, RunId, Tag};
pub use prefix::Prefixes;
pub use reuse::{Begin, Why, begin};

#[cfg(test)]
mod pool_test;
#[cfg(test)]
mod reuse_test;
