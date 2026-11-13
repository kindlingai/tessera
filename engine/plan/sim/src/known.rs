//! Tables for the machines these models were fitted on.

use crate::{Chip, Fit, Pool, gb10};

/// An Apple M3 Pro, through Metal. Its step varies twofold.
pub fn m3_pro() -> Pool {
    let fit = Fit {
        floor: 3e-6,
        spawn: 9e-11,
        step: 3.2e-8,
        curve: 1.4e-8,
        rate: 3.8e11,
        wait: 3.1e-7,
        near: 6.5e-8,
        cache: 4e6,
        cached: 2.54e11,
        read: 1.28e11,
        scan: 8.5e10,
        write: 1.15e11,
        tile: 3.9e12,
    };
    fit.pool("metal Apple M3 Pro")
}

/// The Radeon 8060S of a Ryzen AI Max, through ROCm.
pub fn strix_halo() -> Pool {
    let fit = Fit {
        floor: 7.4e-6,
        spawn: 6e-12,
        step: 1.0e-8,
        curve: 2.4e-8,
        rate: 2.0e12,
        wait: 3.35e-7,
        near: 4.3e-8,
        cache: 4e6,
        cached: 3.24e11,
        read: 2.23e11,
        scan: 1.19e11,
        write: 2.04e11,
        tile: 1.5e13,
    };
    fit.pool("rocm Radeon 8060S")
}

/// The table for a runner's device name. ROCm names none.
pub fn known(device: &str) -> Option<Box<dyn Chip>> {
    let chips: [Box<dyn Chip>; 2] = [Box::new(gb10()), Box::new(m3_pro())];
    chips.into_iter().find(|c| c.name() == device)
}
