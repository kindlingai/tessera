//! Pieces every token model's program uses.

use trace::DimName::{Context, Seqs};
use trace::{DType, Dim, DimName, EntryKind, Model, Port, Role, Size, Tensor, TensorType};

pub(crate) fn fixed(name: DimName, n: u64) -> Dim {
    let size = Size::Fixed(n);
    Dim { name, size }
}

pub(crate) fn bounded(name: DimName, max: u64) -> Dim {
    let size = Size::Bounded(max);
    Dim { name, size }
}

/// A forward entry's tokens and positions, `[Seqs, Context]` each.
pub(crate) fn tokens(m: &Model) -> [Tensor; 2] {
    let ty = TensorType::new([Seqs, Context], DType::U32);
    let port = |role| Port {
        role,
        ty: ty.clone(),
    };
    m.entry(
        EntryKind::Forward,
        [port(Role::Tokens), port(Role::Positions)],
    )
}
