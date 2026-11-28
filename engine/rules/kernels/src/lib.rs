//! Kernels and their lowering rules. Linking this crate registers them.

mod allow;
mod argmax;
pub mod batched;
mod by_sum;
mod cast;
mod causal;
mod gather;
mod hadamard;
mod l2_clip;
mod log_scale;
mod mask;
mod mean;
mod neighbours;
mod norm;
mod radix;
mod rope;
mod softmax;
mod softmax_lanes;
mod take;
mod view;

pub use hadamard::{BLOCK, spin, spin_inline};
pub use norm::rms_norm::RmsNorm;

use tessera_codegen::prelude::LANES;
pub(crate) use tessera_lower::knob as __knob;
use tessera_work::{Load, Ops};

/// Elements one wide load brings a lane.
pub(crate) const WIDE: u32 = 4;

/// Trips unrolled, so a lane keeps several loads in flight.
pub(crate) const AHEAD: u32 = 4;

/// A SIMD group per row. It reads each row twice.
pub(crate) fn per_lane_row(rows: usize, width: usize, trip: Ops, shared: f64) -> Load {
    let all = 4.0 * (rows * width) as f64;
    Load {
        threads: rows * LANES,
        group: 256,
        trips: (width / (LANES * WIDE as usize)) as f64,
        trip,
        reads: vec![2.0 * all, shared],
        written: all,
        flight: (4 * WIDE * AHEAD) as f64,
        ..Load::default()
    }
}

/// One thread per row of `width`, reading each row twice.
pub(crate) fn per_row(rows: usize, width: usize, trip: Ops, shared: f64) -> Load {
    let all = 4.0 * (rows * width) as f64;
    Load {
        threads: rows,
        group: 256,
        trips: width as f64,
        trip,
        reads: vec![2.0 * all, shared],
        written: all,
        flight: 4.0,
        ..Load::default()
    }
}
