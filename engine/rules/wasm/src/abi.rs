//! The JSON a module and the host exchange.

use serde::Deserialize;
use tessera_graph::DimName;
pub(crate) use tessera_guest::site::{Ask, Input, Kept, Kind, Lane, Place, Scalar, Shape, Site};

/// The version of this exchange the host reads.
pub(crate) use tessera_guest::ABI;

/// The return value of `manifest()`.
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
    #[serde(default)]
    pub files: Vec<File>,
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
    #[serde(default)]
    pub cuda: String,
    /// A recipe file to flatten, in place of the texts.
    #[serde(default)]
    pub file: String,
}

/// One recipe file, by its path inside the recipe.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct File {
    pub path: String,
    pub text: String,
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
pub(crate) type Offer = tessera_guest::site::Offer<String, Count>;
