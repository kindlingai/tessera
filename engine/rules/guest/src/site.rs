//! The module's side of its JSON exchange with the host.

use serde::{Deserialize, Serialize};

/// The output of `manifest()`.
#[derive(Debug, Serialize)]
pub struct Manifest {
    pub abi: u32,
    pub kernels: Vec<Kernel>,
    pub rules: Vec<Rule>,
    /// Whether the module exports `graph` and `tables`.
    pub graph: bool,
}

/// One kernel's source, the same for every launch.
#[derive(Debug, Serialize)]
pub struct Kernel {
    pub name: &'static str,
    pub entry: &'static str,
    pub metal: &'static str,
    pub cuda: &'static str,
}

/// A kernel with CUDA text alone.
pub fn kernel(name: &'static str, entry: &'static str, cuda: &'static str) -> Kernel {
    Kernel {
        name,
        entry,
        metal: "",
        cuda,
    }
}

/// A rule, and the site kind it is asked at.
#[derive(Debug, Serialize)]
pub struct Rule {
    pub name: &'static str,
    pub site: Kind,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Matmul,
    Mlp,
    Moe,
    Attention,
    /// One decoder layer.
    Layer,
    /// Embedding to logits.
    Model,
}

/// The input of `propose`.
#[derive(Debug, Deserialize, Serialize)]
pub struct Ask {
    pub rule: String,
    pub site: Site,
}

/// A site's sizes. Offers name reads by place.
#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Site {
    /// `x[m, k]` times `w[n, k]` transposed. Reads `[x, w]`.
    Matmul {
        m: usize,
        n: usize,
        k: usize,
        form: String,
    },
    /// A whole region. Reads are its inputs, in order.
    Region {
        kind: Kind,
        inputs: Vec<Input>,
        /// Elements per output. Entries name the output they write.
        outputs: Vec<usize>,
        kept: Vec<Kept>,
        /// Elements per sum across ranks, in order. Offers place each.
        sums: Vec<usize>,
    },
}

/// Kept rows a region's launches append to.
#[derive(Debug, Deserialize, Serialize)]
pub struct Kept {
    pub index: u32,
    /// Slots, rows per slot, and elements per row.
    pub slots: usize,
    pub context: usize,
    pub width: usize,
    /// The row layout, its rows per page, and its scalars.
    pub layout: String,
    pub page: Option<usize>,
    pub params: Vec<String>,
    pub values: Vec<u32>,
    /// Input places of the slot and position ids.
    pub by: [Option<usize>; 2],
}

/// One value a region reads.
#[derive(Debug, Deserialize, Serialize)]
pub struct Input {
    pub shape: Vec<usize>,
    /// Per axis: the dimension read at launch, if any.
    pub dims: Vec<Option<String>>,
    pub form: String,
    /// A weight's dotted path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight: Option<String>,
    /// This rank's part: axis, first, count, whole.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub share: Option<[usize; 4]>,
    /// A rotated weight: the lengths of `suh` and `svh`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parts: Option<[usize; 2]>,
}

/// A site read, or an earlier offer's output.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum Place {
    Read(usize),
    Draft {
        draft: usize,
    },
    Kept {
        kept: usize,
    },
    /// An input, then 0, 1 or 2: codes, `suh`, `svh`.
    Part {
        part: [usize; 2],
    },
}

/// A u32 argument, fixed or read at launch.
#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
pub enum Count {
    Fixed(u32),
    Dim { dim: String, mul: usize },
}

/// One entry `propose` offers. `null` offers none.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum Entry {
    Launch(Offer),
    /// Sums an earlier draft across ranks. Later entries read it.
    AllReduce {
        all_reduce: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        writes: Option<usize>,
    },
}

impl From<Offer> for Entry {
    fn from(offer: Offer) -> Self {
        Entry::Launch(offer)
    }
}

/// One launch. Its reads are all it depends on.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Offer<K = &'static str, C = Count> {
    pub kernel: K,
    /// In kernel argument order.
    pub reads: Vec<Place>,
    /// Elements written.
    pub len: usize,
    /// u32 arguments after the output.
    #[serde(default = "Vec::new")]
    pub counts: Vec<C>,
    pub shape: Shape,
    /// Work units, and seconds per unit until measured.
    pub work: f64,
    pub guess: f64,
    /// Output index it writes. With one output, the last entry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub writes: Option<usize>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase", deny_unknown_fields)]
pub enum Shape {
    Line(usize),
    Grid {
        across: usize,
        down: usize,
        threads: usize,
    },
}
