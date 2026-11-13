//! Pieces every token model's program uses.

use trace::DimName::*;
use trace::{
    Act, DType, Dim, DimExpr, DimName, EntryKind, Linear, Model, Port, Role, Tensor, TensorType,
};

/// Sizes every decoder stack here declares.
pub struct Stack {
    pub vocab: u64,
    pub hidden: u64,
    pub layers: u64,
    pub heads: u64,
    pub head_dim: u64,
    pub ffn: u64,
    /// Most sequences one launch may carry.
    pub max_seqs: u64,
    pub max_context: u64,
}

impl Stack {
    /// These dimensions, followed by the model's `own`.
    pub(crate) fn dims(&self, own: &[(DimName, u64)]) -> Vec<Dim> {
        let bounded = Dim::bounded;
        let fixed = |&(name, n): &(DimName, u64)| Dim::fixed(name, n);
        let shared = [
            (Vocab, self.vocab),
            (Hidden, self.hidden),
            (Layers, self.layers),
            (Heads, self.heads),
            (HeadDim, self.head_dim),
            (Ffn, self.ffn),
        ];
        let mut dims = vec![
            bounded(Seqs, self.max_seqs),
            bounded(Context, self.max_context),
        ];
        dims.extend(shared.iter().chain(own).map(fixed));
        dims
    }
}

/// A forward entry's tokens and positions, `[Seqs, Context]` each.
pub(crate) fn tokens(m: &Model) -> [Tensor; 2] {
    ports(m, EntryKind::Forward, Context)
}

/// A new entry's tokens and positions, `[Seqs, rows]` each.
pub(crate) fn ports(m: &Model, kind: EntryKind, rows: DimName) -> [Tensor; 2] {
    let ty = TensorType::new([Seqs, rows], DType::U32);
    let port = |role| Port {
        role,
        ty: ty.clone(),
    };
    m.entry(kind, [port(Role::Tokens), port(Role::Positions)])
}

/// A step entry's rows: tokens, positions and slots, `[Rows]` each.
pub(crate) fn step_ports(m: &Model) -> [Tensor; 3] {
    let ty = TensorType::new([Rows], DType::U32);
    let port = |role| Port {
        role,
        ty: ty.clone(),
    };
    let ports = [port(Role::Tokens), port(Role::Positions), port(Role::Slots)];
    m.entry(EntryKind::Step, ports)
}

/// `[Seqs, Context]` followed by `rest`.
pub(crate) fn rows<const N: usize>(rest: [DimExpr; N]) -> Vec<DimExpr> {
    let lead = [Seqs.into(), Context.into()];
    lead.into_iter().chain(rest).collect()
}

pub(crate) fn swiglu(h: &Tensor, [gate, up, down]: [&Linear; 3]) -> Tensor {
    let inner = h.matmul(gate).act(Act::Silu).mul(&h.matmul(up));
    inner.matmul(down)
}
