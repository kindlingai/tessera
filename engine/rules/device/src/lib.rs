//! Device headers for recipe kernels, and the import scanner.

mod flatten;
pub mod layout;
pub mod proxy;
pub mod resident;
mod scan;

#[cfg(test)]
mod flatten_test;
#[cfg(test)]
mod layout_test;
#[cfg(test)]
mod proxy_test;
#[cfg(test)]
mod scan_test;

pub use flatten::{Flat, Options, flatten};
pub use scan::{Error, Problem};

/// The edition every import path names.
pub(crate) const EDITION: u32 = 1;

/// A compile target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cuda,
    Hip,
    Metal,
}

impl Backend {
    /// The backends a kernel file's extension allows.
    pub fn of(path: &str) -> &'static [Backend] {
        match path.rsplit_once('.').map(|(_, e)| e) {
            Some("cu") => &[Backend::Cuda],
            Some("hip") => &[Backend::Hip],
            Some("metal") => &[Backend::Metal],
            Some("kernel") => &[Backend::Cuda, Backend::Hip, Backend::Metal],
            _ => &[],
        }
    }

    fn define(self) -> &'static str {
        match self {
            Backend::Cuda => "TESSERA_CUDA",
            Backend::Hip => "TESSERA_HIP",
            Backend::Metal => "TESSERA_METAL",
        }
    }
}

/// A name a kernel may import after `<tessera/1/`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Import {
    Grid,
    AllReduce,
    AllGather,
    AllToAll,
    SendRecv,
    Kept,
    State,
    Llm,
    Image,
}

impl Import {
    const ALL: [Import; 9] = [
        Import::Grid,
        Import::AllReduce,
        Import::AllGather,
        Import::AllToAll,
        Import::SendRecv,
        Import::Kept,
        Import::State,
        Import::Llm,
        Import::Image,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Import::Grid => "grid",
            Import::AllReduce => "all_reduce",
            Import::AllGather => "all_gather",
            Import::AllToAll => "all_to_all",
            Import::SendRecv => "send_recv",
            Import::Kept => "kept",
            Import::State => "state",
            Import::Llm => "llm",
            Import::Image => "image",
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
        self == Import::Grid || self.collective()
    }
}

/// One header: the name `#line` gives it, and its text.
#[derive(Debug)]
pub(crate) struct Header {
    pub name: &'static str,
    pub text: &'static str,
    /// The macro the flattener defines before it.
    cap: Option<&'static str>,
    /// Whether the kernel's imports and backend need it.
    needed: fn(&[Import], Backend) -> bool,
}

macro_rules! header {
    ($file:literal, $cap:expr, $needed:expr) => {
        Header {
            name: concat!("tessera/1/", $file),
            text: include_str!(concat!("../include/tessera/", $file)),
            cap: $cap,
            needed: $needed,
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
pub(crate) static HEADERS: [Header; 9] = [
    header!("core.h", None, |_, _| true),
    header!("backend.h", None, device),
    header!("status.h", None, device),
    header!("grid.h", Some("TESSERA_CAP_GRID"), |i, _| {
        i.iter().any(|i| i.persistent())
    }),
    header!("collectives.h", Some("TESSERA_CAP_COLLECTIVES"), |i, _| {
        i.iter().any(|i| i.collective())
    }),
    header!("kept.h", Some("TESSERA_CAP_KEPT"), |i, _| has(
        i,
        Import::Kept
    )),
    header!("state.h", Some("TESSERA_CAP_STATE"), |i, _| has(
        i,
        Import::State
    )),
    header!("llm.h", Some("TESSERA_CAP_LLM"), |i, _| has(i, Import::Llm)),
    header!("image.h", Some("TESSERA_CAP_IMAGE"), |i, _| has(
        i,
        Import::Image
    )),
];
