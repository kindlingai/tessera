//! Graph nodes and the ops they run.

use crate::dims::{Dim, DimExpr, Shape};
use crate::json::VERSION;
use crate::numerics::Numerics;
pub use crate::ops::Op;
use crate::stages::Stage;
use crate::types::TensorType;
use serde::{Deserialize, Serialize};

/// A traced program: weights, nodes in order, entry points.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Graph {
    pub version: u32,
    pub dims: Vec<Dim>,
    pub weights: Vec<Weight>,
    /// Rows kept between launches, as `[Seqs, Kept, ..]`.
    pub kept: Vec<TensorType>,
    pub nodes: Vec<Node>,
    pub regions: Vec<Region>,
    pub entries: Vec<Entry>,
}

impl Graph {
    pub fn new(dims: Vec<Dim>) -> Self {
        Self {
            version: VERSION,
            dims,
            weights: Vec::new(),
            kept: Vec::new(),
            nodes: Vec::new(),
            regions: Vec::new(),
            entries: Vec::new(),
        }
    }

    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.0 as usize]
    }

    /// Outside arguments of nodes `start..end`, by first use.
    pub fn reads(&self, start: NodeId, end: NodeId) -> Vec<NodeId> {
        let mut out = Vec::new();
        for n in &self.nodes[start.0 as usize..end.0 as usize] {
            for &a in &n.args {
                if a < start && !out.contains(&a) {
                    out.push(a);
                }
            }
        }
        out
    }
}

/// A weight the loader binds by path.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Weight {
    pub path: Vec<PathSeg>,
    pub ty: TensorType,
}

/// One step of a weight's path in the description.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PathSeg {
    Field(String),
    Index(u32),
}

impl PathSeg {
    /// The path as text, steps joined by dots.
    pub fn dotted(path: &[PathSeg]) -> String {
        let mut steps = Vec::new();
        for seg in path {
            steps.push(match seg {
                PathSeg::Field(f) => f.clone(),
                PathSeg::Index(i) => i.to_string(),
            });
        }
        steps.join(".")
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(transparent)]
pub struct NodeId(pub u32);

/// An op on earlier nodes, with its inferred type.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Node {
    pub op: Op,
    pub args: Vec<NodeId>,
    pub ty: TensorType,
    /// Present exactly on ops that reduce.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub numerics: Option<Numerics>,
}

/// Over `[Seqs, Context, Heads, HeadDim]`. Key heads serve contiguous groups.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Attend {
    pub mask: Mask,
    pub scale: f32,
    /// Argument four lists each query's keys. Out-of-range ids are skipped.
    pub picked: bool,
    /// Query rows carry a slot and position: arguments four, five.
    #[serde(default)]
    pub ragged: bool,
    /// Sink logits follow: `[Heads]`, or one per query head.
    #[serde(default)]
    pub sinks: bool,
    /// Final argument biases each query's nearest `RelWin` keys.
    #[serde(default)]
    pub rel_bias: bool,
}

impl Attend {
    /// Whether queries and keys pack variable-length sequences.
    pub fn packed(&self) -> bool {
        matches!(self.mask, Mask::Packed { .. })
    }

    /// Expected argument count.
    pub fn arity(&self) -> usize {
        self.listed() + usize::from(self.sinks) + usize::from(self.rel_bias)
    }

    /// Arguments before any sinks or bias.
    fn listed(&self) -> usize {
        3 + usize::from(self.packed()) + 2 * usize::from(self.ragged) + usize::from(self.picked)
    }

    /// Argument index of the listed key ids, if `picked`.
    pub fn ids_arg(&self) -> Option<usize> {
        self.picked.then_some(if self.ragged { 5 } else { 3 })
    }

    /// Argument index of the per-head sink logits, if `sinks`.
    pub fn sinks_arg(&self) -> Option<usize> {
        self.sinks.then_some(self.listed())
    }

    /// Argument index of the relative bias, if `rel_bias`.
    pub fn rel_bias_arg(&self) -> Option<usize> {
        self.rel_bias
            .then_some(self.listed() + usize::from(self.sinks))
    }
}

fn one() -> u32 {
    1
}

/// Causal depthwise filter along the sequence axis.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Conv {
    /// Steps between adjacent taps.
    #[serde(default = "one")]
    pub dilation: u32,
    /// Kept state `[Seqs, (K - 1) * dilation, C]`, for ragged rows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<u32>,
}

/// The gated delta rule. Each value head keeps a state.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Delta {
    /// On the queries.
    pub scale: f32,
    /// Kept state `[Seqs, Heads, KeyDim, ValueDim]`, for ragged rows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<u32>,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mask {
    /// Query and key ends align, so queries see earlier keys.
    Causal,
    Bidirectional,
    /// Both ways, keys at most this far from the query.
    Within(u64),
    /// Causal, keys at most this far back.
    Back(u64),
    /// Per-row `[start, end)` segment bounds, causal or bidirectional.
    Packed {
        causal: bool,
    },
}

impl Mask {
    /// Keys a query sees before and after. `far` is all.
    pub fn window(self, far: usize) -> (usize, usize) {
        match self {
            Mask::Causal | Mask::Packed { causal: true } => (far, 0),
            Mask::Bidirectional | Mask::Packed { causal: false } => (far, far),
            Mask::Within(n) => (n as usize, n as usize),
            Mask::Back(n) => (n as usize, 0),
        }
    }

    /// Whether query and key ends align.
    pub fn causal(self) -> bool {
        matches!(
            self,
            Mask::Causal | Mask::Back(_) | Mask::Packed { causal: true }
        )
    }
}

/// Stages fused into one pass over each row.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Row(pub Vec<Stage>);

/// Which two elements of a row rotate together.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pairs {
    /// Elements `2j` and `2j + 1`.
    Adjacent,
    /// Elements `j` and `j + width / 2`.
    Halves,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    /// `0.5 * x * (1 + erf(x / sqrt(2)))`.
    Gelu,
    GeluTanh,
    Silu,
    Sigmoid,
    Relu,
    /// `ln(1 + e^x)`.
    Softplus,
    Exp,
    Tanh,
    /// `sign(x) * sqrt(max(|x|, 1e-6))`.
    SignedSqrt,
}

impl Act {
    /// Whether zero maps to zero.
    pub fn keeps_zero(self) -> bool {
        match self {
            Act::Gelu | Act::GeluTanh | Act::Silu | Act::Relu | Act::Tanh | Act::SignedSqrt => true,
            Act::Sigmoid | Act::Softplus | Act::Exp => false,
        }
    }
}

/// Error function via Abramowitz and Stegun 7.1.26.
pub fn erf(x: f32) -> f32 {
    let a = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * a);
    let p = ((((1.061_405_4 * t - 1.453_152) * t + 1.421_413_7) * t - 0.284_496_74) * t
        + 0.254_829_6)
        * t;
    let m = 1.0 - p * (-(a * a)).exp();
    if x < 0.0 { -m } else { m }
}

/// Data-dependent picks.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum Select {
    /// The first's rows at the second's indices. Zeros past them.
    Gather,
    /// Row entries at the second argument's indices. Out-of-range gives zero.
    Take,
    /// Indices of each row's largest entries. Spare slots hold width.
    TopK {
        /// Best first, NaN and -inf skipped, ties to lower index.
        k: DimExpr,
    },
}

/// Layout changes. They move data but compute nothing.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum View {
    Reshape(Shape),
    Slice {
        axis: u32,
        start: DimExpr,
        len: DimExpr,
    },
    Permute(Vec<u32>),
    /// Repeats a size-one axis.
    Expand {
        axis: u32,
        to: DimExpr,
    },
    /// Joins two arguments along the last axis.
    Concat,
}

/// A matcher may replace these contiguous nodes as one unit.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Region {
    pub kind: Composite,
    pub start: NodeId,
    /// One past the last node.
    pub end: NodeId,
    /// Equals `Graph::reads(start, end)`.
    pub inputs: Vec<NodeId>,
    /// Later nodes may read these and no other inside values.
    pub outputs: Vec<NodeId>,
}

/// Matchers find regions by kind. Inside nodes are the fallback.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Composite {
    Attention,
    Mlp,
    Moe,
    /// One block a model repeats, such as a decoder layer.
    Layer,
    /// Embedding to logits.
    Model,
}

impl Composite {
    /// Whether it spans one block at most, through any trim.
    pub fn block(self) -> bool {
        !matches!(self, Composite::Layer | Composite::Model)
    }
}

/// A way into the graph, with typed ports.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    pub kind: EntryKind,
    pub inputs: Vec<Port>,
    pub outputs: Vec<Output>,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntryKind {
    /// Whole sequences in, logits for every position out.
    Forward,
    /// New rows in. Kept rows carry what came before.
    Step,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Port {
    pub role: Role,
    pub ty: TensorType,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Output {
    pub role: Role,
    pub node: NodeId,
}

/// Activation or routing tensor tapped for Hessian calibration.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum CalibTap {
    AttnIn,
    QResid,
    KvLatent,
    OIn,
    MlpIn,
    DenseDownIn,
    SharedDownIn,
    RouterIds,
    RouterGates,
    RoutedInner,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Tokens,
    /// Each token's 0-based position in its sequence.
    Positions,
    /// Each row's kept-row lane.
    Slots,
    /// The keys each query reads, by index.
    Keys,
    Logits,
    /// The rows a step returns, by index.
    Picks,
    /// One bit per vocabulary row. Clear bits forbid the token.
    Allowed,
    /// The token ids each pick chooses from.
    Candidates,
    /// Residual rows tapped after chosen layers.
    Taps,
    /// Per-layer n-gram embedding rows from the host.
    Ple,
    /// Per-layer Hessian calibration tap.
    Calib(CalibTap),
}
