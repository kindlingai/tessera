//! Tables for the machines these models were fitted on.

use crate::{Chip, Const, Pool, gb10};

/// What `tessera-calibrate` fitted on one machine.
struct Fit {
    floor: f64,
    spawn: f64,
    step: f64,
    curve: f64,
    rate: f64,
    wait: f64,
    near: f64,
    cached: f64,
    read: f64,
    scan: f64,
    write: f64,
}

impl Fit {
    /// Pairs each value with its probe.
    fn pool(&self, name: &str, step: &'static str) -> Pool {
        let m = Const::measured;
        Pool {
            name: name.into(),
            floor: m(self.floor, "256 threads that do nothing"),
            spawn: m(self.spawn, "65,536 idle threads against 256"),
            step: m(self.step, step),
            curve: m(self.curve, "an exponential loop in 256 threads"),
            rate: m(self.rate, "the multiply-add loop at its best thread count"),
            wait: m(self.wait, "256 threads, each load on a new line of 512 MB"),
            near: m(self.near, "256 threads, each along its own 16 KB"),
            cache: Const::inferred(4e6, "4 MB read at cache speed; larger sizes unprobed"),
            cached: m(self.cached, "4 MB read again at the best thread count"),
            read: m(self.read, "512 MB read at the best thread count"),
            scan: m(self.scan, "512 MB read in separate stretches"),
            write: m(self.write, "256 MB written at the best thread count"),
        }
    }
}

/// The GPU of an Apple M3 Pro, through Metal.
pub fn m3_pro() -> Pool {
    let fit = Fit {
        floor: 3e-6,
        spawn: 9e-11,
        step: 3.2e-8,
        curve: 1.4e-8,
        rate: 3.8e11,
        wait: 3.1e-7,
        near: 6.5e-8,
        cached: 2.54e11,
        read: 1.28e11,
        scan: 8.5e10,
        write: 1.15e11,
    };
    let step = "a multiply-add loop in 256 threads; 14 to 32 ns between runs";
    fit.pool("metal Apple M3 Pro", step)
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
        cached: 3.24e11,
        read: 2.23e11,
        scan: 1.19e11,
        write: 2.04e11,
    };
    fit.pool("rocm Radeon 8060S", "a multiply-add loop in 256 threads")
}

/// The table for a runner's device name. ROCm names none.
pub fn known(device: &str) -> Option<Box<dyn Chip>> {
    let chips: [Box<dyn Chip>; 2] = [Box::new(gb10()), Box::new(m3_pro())];
    chips.into_iter().find(|c| c.name() == device)
}
