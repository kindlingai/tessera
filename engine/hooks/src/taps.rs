//! Tapped states on device or host.

use crate::Error;
use tessera_lower::Slot;
use tessera_run::{Error as RunError, Run};

/// Tapped states on device or host, row-major f32.
pub enum TapsBuf {
    Device { slot: Slot, addr: u64, rows: usize },
    Host(Vec<f32>),
}

/// Borrowed tap rows on device or host.
#[derive(Clone, Copy)]
pub enum TapsIn<'a> {
    Device {
        addr: u64,
        rows: usize,
        stride: usize,
    },
    Host(&'a [f32]),
}

impl TapsBuf {
    pub fn as_in(&self) -> TapsIn<'_> {
        self.as_in_from(0, 0)
    }

    pub fn as_in_from(&self, skip: usize, width: usize) -> TapsIn<'_> {
        match self {
            TapsBuf::Device { addr, rows, .. } => {
                let (s, off) = (skip.min(*rows), (skip.min(*rows) * width * 4) as u64);
                TapsIn::Device {
                    addr: *addr + off,
                    rows: *rows - s,
                    stride: 0,
                }
            }
            TapsBuf::Host(v) => TapsIn::Host(&v[(skip * width).min(v.len())..]),
        }
    }

    pub fn keep_rows(&mut self, keep: usize, width: usize) {
        match self {
            TapsBuf::Device { rows, .. } => *rows = keep,
            TapsBuf::Host(v) => v.truncate(keep * width),
        }
    }

    pub fn into_host(self, runner: &dyn Run, width: usize) -> Result<Vec<f32>, Error> {
        match self {
            TapsBuf::Host(v) => Ok(v),
            TapsBuf::Device { slot, rows, .. } => {
                let raw = runner.read(slot)?;
                let floats = raw.floats().map_err(RunError::from)?;
                Ok(floats[..rows * width].to_vec())
            }
        }
    }
}

/// Batched verify outputs: device ids or host logits.
pub enum VerifyOut {
    Ids(Vec<u32>),
    Logits(Vec<f32>, usize),
}
