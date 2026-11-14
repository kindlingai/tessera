//! Kernels and their lowering rules. Linking this crate registers them.

/// Name, order and sizes of a kernel over one `Attn`.
macro_rules! over_attn {
    ($name:literal) => {
        fn name(&self) -> &'static str {
            $name
        }

        fn order(&self) -> Order {
            Order::PerConfig
        }

        fn sizes(&self) -> Vec<usize> {
            self.0.sizes()
        }
    };
}

mod argmax;
mod attend;
mod dense;
mod flash;
mod gather;
mod hadamard;
mod join;
mod lane_norm;
mod mean;
mod rms_norm;
mod rope;
mod softmax;
mod thin;

pub use argmax::{Argmax, Peaks};
pub use attend::Attend;
pub use dense::{Causal, DENSE_SCORES, Part, PerHead};
pub use gather::Gather;
pub use hadamard::Hadamard;
pub use join::Join;
pub use lane_norm::LaneNorm;
pub use mean::Mean;
pub use rms_norm::RmsNorm;
pub use rope::{Angles, Rope};
pub use softmax::Softmax;
pub use thin::{THIN_ROWS, Thin};

use sim::{Load, Ops};

/// One thread per row of `width`, along that row alone.
pub(crate) fn per_row(rows: usize, width: usize, trip: Ops, shared: f64) -> Load {
    let all = 4.0 * (rows * width) as f64;
    Load {
        threads: rows,
        group: 256,
        trips: width as f64,
        trip,
        reads: vec![all, shared],
        written: all,
        flight: 64.0,
    }
}
