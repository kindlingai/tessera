//! The JSON a module and the host exchange.

use sensai_graph::DimName;
use serde::{Deserialize, Serialize};

/// The version of this exchange the host reads.
pub(crate) const ABI: u32 = 3;

/// What `manifest()` returns.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Manifest {
    pub abi: u32,
    pub kernels: Vec<Text>,
    pub rules: Vec<Declared>,
}

/// One kernel's source, the same for every launch.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Text {
    pub name: String,
    pub entry: String,
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

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Kind {
    Matmul,
    Mlp,
    Moe,
    Attention,
}

/// What `propose` is given.
#[derive(Debug, Serialize)]
pub(crate) struct Ask<'a> {
    pub rule: &'a str,
    pub site: Site,
}

/// A site's sizes. Offers name reads by place.
#[derive(Debug, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Site {
    /// `x[m, k]` times `w[n, k]` transposed. Reads `[x, w]`.
    Matmul {
        m: usize,
        n: usize,
        k: usize,
        form: &'static str,
    },
    /// A whole region. Reads are its inputs, in order.
    Region {
        kind: Kind,
        inputs: Vec<Input>,
        len: usize,
        kept: Vec<Kept>,
    },
}

/// Kept rows a region appends to. Its launches write them.
#[derive(Debug, Serialize)]
pub(crate) struct Kept {
    pub index: u32,
    /// Slots, rows per slot, and elements per row.
    pub slots: usize,
    pub context: usize,
    pub width: usize,
    /// The row layout, its rows per page, and its scalars.
    pub layout: &'static str,
    pub page: Option<usize>,
    pub params: &'static [&'static str],
    pub values: Vec<u32>,
    /// Input places of the slot and position ids.
    pub by: [Option<usize>; 2],
}

/// One value a region reads.
#[derive(Debug, Serialize)]
pub(crate) struct Input {
    pub shape: Vec<usize>,
    /// Per axis: the dimension read at launch, if any.
    pub dims: Vec<Option<DimName>>,
    pub form: &'static str,
    /// A weight's dotted path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight: Option<String>,
    /// This rank's part: axis, first, count, whole.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub share: Option<[usize; 4]>,
}

/// A site read, or an earlier offer's output.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum Place {
    Read(usize),
    Draft { draft: usize },
    Kept { kept: usize },
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

/// One launch `propose` offers. `null` offers none.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Offer {
    pub kernel: String,
    /// In kernel argument order.
    pub reads: Vec<Place>,
    /// Elements written.
    pub len: usize,
    /// u32 arguments after the output.
    #[serde(default)]
    pub counts: Vec<Count>,
    pub shape: Shape,
    /// Work units, and seconds per unit until measured.
    pub work: f64,
    pub guess: f64,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "lowercase", deny_unknown_fields)]
pub(crate) enum Shape {
    Line(usize),
    Grid {
        across: usize,
        down: usize,
        threads: usize,
    },
}
