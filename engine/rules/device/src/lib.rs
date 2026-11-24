//! Device headers for recipe kernels, and the import scanner.

pub mod abi;
mod flatten;
#[cfg(test)]
mod flatten_test;
pub mod layout;
#[cfg(test)]
mod layout_test;
mod linear;
#[cfg(test)]
mod linear_test;
mod moe;
pub mod p2p;
pub mod proxy;
#[cfg(test)]
mod proxy_test;
pub mod resident;
pub mod sums;
pub mod whole;
#[cfg(test)]
mod whole_test;

pub use flatten::{Error, Problem};
pub use flatten::{Flat, Options, flatten};
pub use tessera_gpukit::backend::Backend;

pub(crate) use layout::EDITION;

/// A name a kernel may import after `<tessera/1/`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Import {
    /// Types and device primitives only.
    Core,
    Grid,
    AllReduce,
    AllGather,
    AllToAll,
    SendRecv,
    Kept,
    State,
    Llm,
    Image,
    Trellis,
    Whole,
}

impl Import {
    const ALL: [Import; 12] = [
        Import::Core,
        Import::Grid,
        Import::AllReduce,
        Import::AllGather,
        Import::AllToAll,
        Import::SendRecv,
        Import::Kept,
        Import::State,
        Import::Llm,
        Import::Image,
        Import::Trellis,
        Import::Whole,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Import::Core => "core",
            Import::Grid => "grid",
            Import::AllReduce => "all_reduce",
            Import::AllGather => "all_gather",
            Import::AllToAll => "all_to_all",
            Import::SendRecv => "send_recv",
            Import::Kept => "kept",
            Import::State => "state",
            Import::Llm => "llm",
            Import::Image => "image",
            Import::Trellis => "trellis",
            Import::Whole => "whole",
        }
    }

    pub fn parse(name: &str) -> Option<Import> {
        Import::ALL.into_iter().find(|i| i.name() == name)
    }

    /// Whether it needs collectives, and so a persistent grid.
    pub fn collective(self) -> bool {
        matches!(
            self,
            Import::AllReduce | Import::AllGather | Import::AllToAll | Import::SendRecv
        )
    }

    /// Whether it needs a grid barrier. Metal has none.
    pub fn persistent(self) -> bool {
        matches!(self, Import::Grid | Import::Whole) || self.collective()
    }
}

/// One header: the name `#line` gives it, and its text.
#[derive(Debug)]
pub(crate) struct Header {
    pub name: &'static str,
    pub text: fn() -> &'static str,
    /// The macro the flattener defines before it.
    cap: Option<&'static str>,
    /// Whether the kernel's imports and backend need it.
    needed: fn(&[Import], Backend) -> bool,
    /// This header's declarations in `abi.h`.
    pub abi: fn() -> Vec<abi::Decl>,
}

macro_rules! header {
    ($file:literal, $cap:expr, $needed:expr) => {
        header!($file, $cap, $needed, Vec::new)
    };
    ($file:literal, $cap:expr, $needed:expr, $abi:expr) => {
        Header {
            name: concat!("tessera/1/", $file),
            text: || include_str!(concat!("../include/tessera/", $file)),
            cap: $cap,
            needed: $needed,
            abi: $abi,
        }
    };
}

fn has(imports: &[Import], want: Import) -> bool {
    imports.contains(&want)
}

fn device(_: &[Import], backend: Backend) -> bool {
    backend != Backend::Metal
}

/// Every header, in the order a flattened kernel holds them.
pub(crate) static HEADERS: [Header; 13] = [
    header!("types.h", None, |_, _| true),
    Header {
        name: "tessera/1/abi.h",
        text: abi::text,
        cap: None,
        needed: |_, _| true,
        abi: Vec::new,
    },
    header!("core.h", None, |_, _| true, layout::core),
    header!("backend.h", None, device),
    header!("status.h", None, device),
    header!(
        "grid.h",
        Some("TESSERA_CAP_GRID"),
        |i, _| { i.iter().any(|i| i.persistent()) },
        layout::grid
    ),
    header!(
        "collectives.h",
        Some("TESSERA_CAP_COLLECTIVES"),
        |i, _| { i.iter().any(|i| i.collective()) },
        layout::collectives
    ),
    header!(
        "kept.h",
        Some("TESSERA_CAP_KEPT"),
        |i, _| has(i, Import::Kept),
        layout::kept
    ),
    header!(
        "state.h",
        Some("TESSERA_CAP_STATE"),
        |i, _| has(i, Import::State),
        layout::state
    ),
    header!(
        "llm.h",
        Some("TESSERA_CAP_LLM"),
        |i, _| has(i, Import::Llm),
        layout::llm
    ),
    header!(
        "image.h",
        Some("TESSERA_CAP_IMAGE"),
        |i, _| has(i, Import::Image),
        layout::image
    ),
    header!("trellis.h", Some("TESSERA_CAP_TRELLIS"), |i, b| {
        has(i, Import::Trellis) && b != Backend::Metal
    }),
    header!("whole.h", Some("TESSERA_CAP_WHOLE"), |i, b| {
        has(i, Import::Whole) && b != Backend::Metal
    }),
];
