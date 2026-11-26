//! Device headers for recipe kernels, and the import scanner.

pub mod abi;
mod attend;
pub mod calib;
#[cfg(test)]
mod calib_test;
mod flatten;
#[cfg(test)]
mod flatten_test;
#[cfg(test)]
mod forms_test;
pub mod layout;
#[cfg(test)]
mod layout_test;
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

/// Maximum rows handled by built-in fast CUDA prefill kernels.
pub const FAST_MAX_ROWS: usize = 2048;

/// Non-CUDA backends skipped by CUDA-only rules.
pub const CUDA_LACKS: &[tessera_lower::Backend] =
    &[tessera_lower::Backend::Metal, tessera_lower::Backend::Rocm];

#[doc(hidden)]
pub use tessera_lower as __lower;

/// Registers a CUDA rule, from a `check` or a proposer.
#[macro_export]
macro_rules! cuda_rule {
    ($name:literal, $propose:path, why = $why:path) => {
        $crate::__lower::rule!($name, $propose, why = $why, lacks = $crate::CUDA_LACKS);
    };
    ($name:literal, $propose:path) => {
        const _: () = {
            fn why(_: &$crate::__lower::Tree, _: $crate::__lower::T) -> Option<String> {
                None
            }
            $crate::cuda_rule!($name, $propose, why = why);
        };
    };
    ($name:literal, check = $check:path) => {
        const _: () = {
            use $crate::__lower::{Draft, T, Tree};
            fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
                $check(tree, at)?.ok()
            }
            fn why(tree: &Tree, at: T) -> Option<String> {
                $check(tree, at)?.err()
            }
            $crate::cuda_rule!($name, propose, why = why);
        };
    };
}

/// Accepts only f32 outputs.
pub fn f32_only<K>(_: &K, form: &dyn tessera_lower::form::Load) -> bool {
    form.name() == "f32"
}

/// Parses `trellis_k{2,3,4}[_{mul1,pmcgx}]` into `(bits, book)`.
pub fn parse_trellis(name: &str) -> Option<(u32, u32)> {
    let rest = name.strip_prefix("trellis_k")?;
    let (k, book) = if let Some(k) = rest.strip_suffix("_pmcgx") {
        (k, 2u32)
    } else if let Some(k) = rest.strip_suffix("_mul1") {
        (k, 1u32)
    } else {
        (rest, 0u32)
    };
    let bits: u32 = k.parse().ok()?;
    (2..=4).contains(&bits).then_some((bits, book))
}

/// Flattens one built-in kernel source into a static string.
pub fn flatten_for(backend: Backend, name: &str, kernel: &str) -> &'static str {
    let flat = flatten(name, kernel, &|_| None, &Options::plain(backend));
    Box::leak(
        flat.expect("built-in kernel flattens")
            .text
            .into_boxed_str(),
    )
}

pub(crate) fn flatten_cuda(name: &str, kernel: &str) -> &'static str {
    flatten_for(Backend::Cuda, name, kernel)
}

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
