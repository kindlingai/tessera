//! Moves bytes between ranks over handles the front end opened.

mod fabric;
mod ibv;
mod link;
mod memory;
pub mod reduce;
mod shm;
mod sockets;
mod verbs;

pub use fabric::{Fabric, Port};
pub use link::{Blocking, Link, Put, Span};
pub use memory::Memory;
pub use reduce::{Pinned, Pinning, Reduce};
pub use shm::{Shm, ShmLink};
pub use sockets::Sockets;
pub use verbs::Verbs;

use std::io;

/// Moves bytes between ranks. One round trip per call.
pub trait Transport {
    fn rank(&self) -> u32;
    fn ranks(&self) -> u32;
    /// Every rank's bytes, in rank order, `mine` among them.
    fn exchange(&mut self, mine: &[u8]) -> Result<Vec<Vec<u8>>, Error>;
    /// A device's way to run the rounds itself, if any.
    fn device(&mut self) -> Option<Reduce> {
        None
    }
}

impl<T: Transport + ?Sized> Transport for Box<T> {
    fn rank(&self) -> u32 {
        (**self).rank()
    }

    fn ranks(&self) -> u32 {
        (**self).ranks()
    }

    fn exchange(&mut self, mine: &[u8]) -> Result<Vec<Vec<u8>>, Error> {
        (**self).exchange(mine)
    }

    fn device(&mut self) -> Option<Reduce> {
        (**self).device()
    }
}

/// The parts summed in rank order. Every rank agrees.
pub fn summed<'a>(parts: impl Iterator<Item = &'a [f32]>) -> Vec<f32> {
    let mut sum: Option<Vec<f32>> = None;
    for part in parts {
        sum = Some(match sum {
            None => part.to_vec(),
            Some(s) => s.iter().zip(part).map(|(a, b)| a + b).collect(),
        });
    }
    sum.expect("a rank")
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] io::Error),
    /// A handle, slot or round the ranks do not fit.
    #[error("fit: {0}")]
    Fit(String),
    /// The RDMA devices or a peer failed.
    #[error("link: {0}")]
    Link(String),
}

#[cfg(test)]
mod comm_test;
#[cfg(test)]
mod fabric_test;
#[cfg(test)]
mod ibv_test;
