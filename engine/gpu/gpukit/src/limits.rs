//! What one group of threads gets on a device.

use serde::{Deserialize, Serialize};

/// The device's bounds on a kernel's schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Limits {
    /// Threads that execute together and can exchange values.
    pub lanes: usize,
    /// Bytes of group memory one group may hold.
    pub group_bytes: usize,
    /// The most threads in one group.
    pub group_threads: usize,
    /// Whether hand-written kernel text can be compiled here.
    #[serde(default = "yes")]
    pub compiler: bool,
    /// Whether a launch may overlap the one before it.
    #[serde(default)]
    pub pdl: bool,
}

fn yes() -> bool {
    true
}

impl Limits {
    /// The floor across supported devices, until one is asked.
    pub const fn assumed() -> Self {
        Self {
            lanes: 32,
            group_bytes: 32 * 1024,
            group_threads: 1024,
            compiler: true,
            pdl: false,
        }
    }

    /// A host group's limits, `lanes` to a SIMD group.
    pub const fn host(lanes: usize) -> Self {
        Self {
            lanes,
            group_bytes: 64 * 1024,
            group_threads: 1024,
            compiler: false,
            pdl: false,
        }
    }
}

impl Default for Limits {
    fn default() -> Self {
        Self::assumed()
    }
}
