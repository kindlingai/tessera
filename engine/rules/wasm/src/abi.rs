//! The JSON a module and the host exchange.

use sensai_graph::DimName;
pub(crate) use sensai_guest::site::{Ask, Input, Kept, Kind, Place, Shape, Site};
use serde::Deserialize;

/// The version of this exchange the host reads.
pub(crate) use sensai_guest::ABI;

/// What `manifest()` returns.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Manifest {
    pub abi: u32,
    #[serde(default)]
    pub kernels: Vec<Text>,
    #[serde(default)]
    pub rules: Vec<Declared>,
    /// Whether it exports `graph` and `tables`.
    #[serde(default)]
    pub graph: bool,
}

/// One kernel's source, the same for every launch.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Text {
    pub name: String,
    pub entry: String,
    /// Empty when the kernel has no Metal text.
    #[serde(default)]
    pub metal: String,
    pub cuda: String,
}

/// A rule, and the site kind it is asked at.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Declared {
    pub name: String,
    pub site: Kind,
}

/// A u32 argument, fixed or read at launch.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum Count {
    Fixed(u32),
    Dim {
        dim: DimName,
        #[serde(default = "one")]
        mul: usize,
    },
}

fn one() -> usize {
    1
}

/// One entry `propose` offers. `null` offers none.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum Entry {
    Launch(Offer),
    Sum(Sum),
}

/// Sums draft `all_reduce` across ranks, as a new draft.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Sum {
    pub all_reduce: usize,
    #[serde(default)]
    pub writes: Option<usize>,
}

/// One launch, in the host's form.
pub(crate) type Offer = sensai_guest::site::Offer<String, Count>;
