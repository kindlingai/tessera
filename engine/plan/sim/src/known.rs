//! Tables for the machines these models were fitted on.

use crate::nvidia::gb10;
use crate::{Chip, Fit, Pool};

/// The name an Apple M3 Pro reports through Metal.
const M3_PRO: &str = "metal Apple M3 Pro";

/// An Apple M3 Pro, through Metal. Its step varies twofold.
pub fn m3_pro() -> Pool {
    m3_pro_fit().pool(M3_PRO)
}

/// Registers and group memory share one store per core.
fn m3_pro_fit() -> Fit {
    Fit {
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
        cores: 14.0,
        store: 320e3,
        resident: 1024.0,
        held: 128.0,
        shared: 3e-8,
        banked: 1.8e11,
        mma: 5e-10,
        spill: 3.2e-7,
    }
}

/// The table's constants for a pool device, if any.
pub(crate) fn fit(device: &str) -> Option<Fit> {
    (device == M3_PRO).then(m3_pro_fit)
}

/// The table for a runner's device name. ROCm names none.
pub fn known(device: &str) -> Option<Box<dyn Chip>> {
    let chips: [Box<dyn Chip>; 2] = [Box::new(gb10()), Box::new(m3_pro())];
    chips.into_iter().find(|c| c.name() == device)
}
