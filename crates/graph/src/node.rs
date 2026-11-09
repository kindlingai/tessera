//! Graph nodes and the ops they run.

use crate::dims::{Dim, DimExpr, Shape};
use crate::json::VERSION;
use crate::numerics::Numerics;
use crate::types::{DType, TensorType};
use serde::{Deserialize, Serialize};

/// A traced program: weights, nodes in order, entry points.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Graph {
    pub version: u32,
    pub dims: Vec<Dim>,
    pub weights: Vec<Weight>,
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

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum Op {
    /// An entry's input port.
    Input {
        entry: u32,
        port: u32,
    },
    Weight(u32),
    /// `x[.., K]` times `w[N, K]` transposed.
    Matmul,
    Attend(Attend),
    Row(Row),
    Select(Select),
    View(View),
    Cast(DType),
}

impl Op {
    /// Whether this op sums, so it carries `Numerics`.
    pub fn reduces(&self) -> bool {
        match self {
            Op::Matmul | Op::Attend(_) => true,
            Op::Row(r) => r.0.iter().any(Stage::reduces),
            _ => false,
        }
    }
}

/// Over `[Seqs, Context, Heads, HeadDim]`; key heads serve contiguous groups.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Attend {
    pub mask: Mask,
    pub scale: f32,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mask {
    /// Query and key ends align, so queries see earlier keys.
    Causal,
    Bidirectional,
}

/// Stages fused into one pass over each row.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Row(pub Vec<Stage>);

/// One step over last-axis rows; some take an argument.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum Stage {
    Add,
    Mul,
    Scale(f32),
    RmsNorm { eps: f32 },
    Act(Act),
    Softmax,
    AllReduce,
}

impl Stage {
    /// Arguments this stage reads besides the row itself.
    pub fn operands(&self) -> usize {
        match self {
            Stage::Add | Stage::Mul | Stage::RmsNorm { .. } => 1,
            _ => 0,
        }
    }

    pub fn reduces(&self) -> bool {
        matches!(
            self,
            Stage::RmsNorm { .. } | Stage::Softmax | Stage::AllReduce
        )
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    GeluTanh,
    Silu,
    Sigmoid,
}

/// Data-dependent picks.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum Select {
    /// Rows of the first argument at the second's indices.
    Gather,
}

/// Layout changes; they move data but compute nothing.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum View {
    Reshape(Shape),
    Slice {
        axis: u32,
        start: DimExpr,
        len: DimExpr,
    },
    Permute(Vec<u32>),
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

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Tokens,
    /// Each token's 0-based position in its sequence.
    Positions,
    Logits,
}
