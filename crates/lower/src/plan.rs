//! A plan: kernel launches over buffers that stay on device.

use graph::{Act, NodeId};

/// Leading elements of one buffer. Every element is four bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Slot {
    pub buffer: u32,
    pub len: usize,
}

/// One elementwise step of a generated kernel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MapStage {
    /// Adds the next operand.
    Add,
    /// Multiplies by the next operand.
    Mul,
    Scale(f32),
    Act(Act),
}

/// The sizes one attention works over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Attn {
    pub seqs: usize,
    pub queries: usize,
    pub keys: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub width: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Kernel {
    /// `x[m, k]` times `w[n, k]` transposed.
    Matmul {
        m: usize,
        n: usize,
        k: usize,
    },
    /// The same product, one thread per output column.
    Thin {
        m: usize,
        n: usize,
        k: usize,
    },
    RmsNorm {
        rows: usize,
        width: usize,
        eps: f32,
    },
    Softmax {
        rows: usize,
        width: usize,
    },
    /// Each position's cosine and sine, per pair.
    Angles {
        width: usize,
        theta: f32,
    },
    /// Rotates halves in place, by the second input's angles.
    Rope {
        positions: usize,
        heads: usize,
        width: usize,
    },
    /// Causal attention, key by key; keys and values arrive joined.
    Attend(Attn),
    /// Every query-key product, as `[Seqs, Heads, Queries, Keys]`.
    Scores(Attn),
    /// Scaled causal softmax over each score row, in place.
    Causal {
        queries: usize,
        keys: usize,
        scale: f32,
    },
    /// Each head's probabilities times its values.
    Values(Attn),
    Gather {
        rows: usize,
        width: usize,
    },
    /// A generated kernel; operands follow the first input.
    Map(Vec<MapStage>),
    /// Rows of two inputs, joined along the last axis.
    Join {
        left: usize,
        right: usize,
    },
}

impl Kernel {
    /// Whether it writes over its first input.
    pub fn overwrites(&self) -> bool {
        matches!(self, Kernel::Rope { .. } | Kernel::Causal { .. })
    }

    pub fn name(&self) -> &'static str {
        match self {
            Kernel::Matmul { .. } => "matmul",
            Kernel::Thin { .. } => "thin",
            Kernel::RmsNorm { .. } => "rms_norm",
            Kernel::Softmax { .. } => "softmax",
            Kernel::Angles { .. } => "angles",
            Kernel::Rope { .. } => "rope",
            Kernel::Attend(_) => "attend",
            Kernel::Scores(_) => "scores",
            Kernel::Causal { .. } => "causal",
            Kernel::Values(_) => "values",
            Kernel::Gather { .. } => "gather",
            Kernel::Map(_) => "map",
            Kernel::Join { .. } => "join",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    pub kernel: Kernel,
    pub reads: Vec<Slot>,
    pub write: Slot,
    /// The graph nodes this launch computes, or helps compute.
    pub covers: Vec<NodeId>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Output {
    pub slot: Slot,
    pub shape: Vec<usize>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Plan {
    /// Elements per buffer. Values that never overlap share one.
    pub buffers: Vec<usize>,
    pub steps: Vec<Step>,
    /// Each used graph weight, and where it loads.
    pub weights: Vec<(u32, Slot)>,
    /// One per input port of the entry.
    pub inputs: Vec<Slot>,
    pub outputs: Vec<Output>,
}

impl Plan {
    /// Elements in buffers that hold no weight or input.
    pub fn scratch(&self) -> usize {
        let fixed = self.weights.iter().map(|(_, s)| s).chain(&self.inputs);
        let held: usize = fixed.map(|s| s.len).sum();
        self.buffers.iter().sum::<usize>() - held
    }
}
