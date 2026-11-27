//! The module's side of its JSON exchange with the host.

pub use crate::kit::{Found, Weights, agreed, fixed, layer, same, trellis};
use serde::{Deserialize, Serialize};

/// The output of `manifest()`.
#[derive(Debug, Serialize)]
pub struct Manifest {
    pub abi: u32,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub kernels: Vec<Kernel>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub rules: Vec<Rule>,
    /// Whether the module exports `graph` and `tables`.
    pub graph: bool,
    /// Recipe files that kernels from files read.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub files: Vec<File>,
}

/// One kernel's source, the same for every launch.
#[derive(Debug, Serialize)]
pub struct Kernel {
    pub name: &'static str,
    pub entry: &'static str,
    pub metal: &'static str,
    pub cuda: &'static str,
    /// A recipe file the host flattens into the texts.
    pub file: &'static str,
}

const BLANK: Kernel = Kernel {
    name: "",
    entry: "",
    metal: "",
    cuda: "",
    file: "",
};

impl Kernel {
    /// A kernel with CUDA text alone.
    pub fn cuda(name: &'static str, entry: &'static str, text: &'static str) -> Self {
        Kernel {
            name,
            entry,
            cuda: text,
            ..BLANK
        }
    }

    /// A kernel from recipe file `path`, backend by extension.
    pub fn file(name: &'static str, entry: &'static str, path: &'static str) -> Self {
        Kernel {
            name,
            entry,
            file: path,
            ..BLANK
        }
    }
}

/// As `Kernel::cuda`.
pub fn kernel(name: &'static str, entry: &'static str, cuda: &'static str) -> Kernel {
    Kernel::cuda(name, entry, cuda)
}

/// As `Kernel::file`.
pub fn imports(name: &'static str, entry: &'static str, file: &'static str) -> Kernel {
    Kernel::file(name, entry, file)
}

/// One recipe file, by its path inside the recipe.
#[derive(Debug, Serialize)]
pub struct File {
    pub path: &'static str,
    pub text: &'static str,
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

/// A `propose` answer when the site JSON does not parse.
#[derive(Debug, Deserialize, Serialize)]
pub struct Unread {
    pub unread: String,
}

/// The input of `propose`.
#[derive(Debug, Deserialize, Serialize)]
pub struct Ask {
    pub rule: String,
    pub site: Site,
    /// Model config JSON, when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config: Option<serde_json::Value>,
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
        /// Each distinct op constant inside, in first-use order.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        scalars: Vec<Scalar>,
    },
}

/// One op constant: an eps, a theta or a scale.
#[derive(Debug, Deserialize, Serialize, PartialEq)]
pub struct Scalar {
    /// `rms_norm`, `rope_adjacent`, `rope_halves`, `scale`, `attend` or `top_k`.
    pub op: String,
    pub value: f32,
}

/// The value of `op`, if present and unique.
pub fn scalar(scalars: &[Scalar], op: &str) -> Option<f32> {
    let mut values = scalars.iter().filter(|s| s.op == op).map(|s| s.value);
    let value = values.next()?;
    values.all(|v| v == value).then_some(value)
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

impl Kept {
    /// The resident layout code, and rows per page.
    pub fn code(&self) -> Option<[u32; 2]> {
        match (self.layout.as_str(), self.page) {
            ("dense", None) => Some([0, 0]),
            ("interleaved", None) => Some([1, 0]),
            (_, Some(page)) if page.is_power_of_two() => Some([2, page as u32]),
            _ => None,
        }
    }
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
    /// Set when the launch is one persistent grid.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resident: Option<Resident>,
}

/// One grid per launch, over a table, with sums inside.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Resident {
    pub threads: u32,
    /// Bytes of dynamic shared memory per block.
    #[serde(default)]
    pub shared: u32,
    /// Bytes of scratch, the last table item.
    #[serde(default)]
    pub scratch: usize,
    /// The collective lanes, in kernel lane order.
    pub lanes: Vec<Lane>,
}

/// One collective lane's operation and element type.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Lane {
    Sum,
    SumBf16,
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
