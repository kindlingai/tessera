//! Moves bytes between ranks over handles the front end opened.

mod arx;
mod fabric;
mod ibv;
mod link;
mod memory;
mod pace;
mod pin;
mod shared;
mod shm;
#[cfg(test)]
mod socket_link;
mod sockets;
mod verbs;
mod verbs_link;
mod words;

pub use arx::{ArxLink, ArxMem};
pub use fabric::{Fabric, Port};
pub use link::{Blocking, Link, Put, Span, exchange_regions};
pub use memory::Memory;
pub use pin::{Pinned, Pinning};
pub use shared::{Backed, HostPool, SharedPool};
pub use shm::{Shm, ShmLink};
pub use sockets::Sockets;
pub use verbs::Verbs;
pub use verbs_link::VerbsLink;
pub use words::{Reader, Writer};

use std::io;

/// Moves bytes between ranks. One round trip per call.
pub trait Transport {
    fn rank(&self) -> u32;
    fn ranks(&self) -> u32;
    /// Every rank's bytes, in rank order, `mine` among them.
    fn exchange(&mut self, mine: &[u8]) -> Result<Vec<Vec<u8>>, Error>;
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
}

/// The parts summed in rank order. Every rank agrees.
pub fn summed<'a>(parts: impl Iterator<Item = &'a [f32]>, wire: impl Fn(f32) -> f32) -> Vec<f32> {
    let mut sum: Option<Vec<f32>> = None;
    for part in parts {
        sum = Some(match sum {
            None => part.iter().map(|&b| wire(b)).collect(),
            Some(s) => s.iter().zip(part).map(|(a, &b)| a + wire(b)).collect(),
        });
    }
    sum.expect("a rank")
}

/// The slice `range` of parts summed in rank order.
pub fn summed_slice<'a>(
    parts: impl Iterator<Item = &'a [f32]>,
    range: std::ops::Range<usize>,
    wire: impl Fn(f32) -> f32,
) -> Vec<f32> {
    summed(parts.map(|p| &p[range.clone()]), wire)
}

/// The parts' `rows` rows, joined in rank order.
pub fn gathered<P: AsRef<[u8]>>(parts: &[P], rows: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(parts.iter().map(|p| p.as_ref().len()).sum());
    for r in 0..rows {
        for p in parts {
            let p = p.as_ref();
            let width = p.len() / rows;
            out.extend_from_slice(&p[r * width..(r + 1) * width]);
        }
    }
    out
}

/// Joins each rank's contiguous rows in rank order.
pub fn gathered_rows<P: AsRef<[u8]>>(parts: &[P]) -> Vec<u8> {
    let mut out = Vec::with_capacity(parts.iter().map(|p| p.as_ref().len()).sum());
    for p in parts {
        out.extend_from_slice(p.as_ref());
    }
    out
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
#[cfg(test)]
mod pace_test;
#[cfg(test)]
mod shared_test;
