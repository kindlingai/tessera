//! The ports and outputs the engine expects of an entry.

use crate::{Model, Tensor};
use tessera_graph::DimName::*;
use tessera_graph::{DType, DimExpr, DimName, EntryKind, Graph, Port, Role, TensorType};

/// A forward entry's tokens and positions, `[Seqs, Context]` each.
pub fn tokens(m: &Model) -> [Tensor; 2] {
    ports(m, EntryKind::Forward, Context)
}

/// A forward entry's tokens, positions and key lists.
pub fn listed(m: &Model) -> [Tensor; 3] {
    let port = |role, shape: Vec<DimExpr>| Port {
        role,
        ty: TensorType::new(shape, DType::U32),
    };
    let rows = vec![Seqs.into(), Context.into()];
    let lists = vec![Seqs.into(), Context.into(), Context.into()];
    let ports = [
        port(Role::Tokens, rows.clone()),
        port(Role::Positions, rows),
        port(Role::Keys, lists),
    ];
    m.entry(EntryKind::Forward, ports)
}

/// A new entry's tokens and positions, `[Seqs, rows]` each.
pub fn ports(m: &Model, kind: EntryKind, rows: DimName) -> [Tensor; 2] {
    let ty = TensorType::new([Seqs, rows], DType::U32);
    let port = |role| Port {
        role,
        ty: ty.clone(),
    };
    m.entry(kind, [port(Role::Tokens), port(Role::Positions)])
}

/// Each pick's mask words, then its packed logit edits.
fn mask_port() -> Port {
    let words = DimExpr::Add(Box::new(VocabWords.into()), Box::new(EditWords.into()));
    Port {
        role: Role::Allowed,
        ty: TensorType::new(vec![Rows.into(), words], DType::U32),
    }
}

/// A step entry's tokens, positions, slots and optional mask.
pub fn step_ports(m: &Model, masked: bool) -> ([Tensor; 3], Option<Tensor>) {
    let (rows, extra, mask) = step_ports_with(m, masked, None);
    debug_assert!(extra.is_none());
    (rows, mask)
}

/// A step entry's rows, optional extra port, optional mask.
pub fn step_ports_with(
    m: &Model,
    masked: bool,
    extra: Option<Port>,
) -> ([Tensor; 3], Option<Tensor>, Option<Tensor>) {
    let row = |role| Port {
        role,
        ty: TensorType::new([Rows], DType::U32),
    };
    let [t, p, s] = [Role::Tokens, Role::Positions, Role::Slots].map(row);
    match (extra, masked) {
        (Some(e), true) => {
            let [t, p, s, e, bits] = m.entry(EntryKind::Step, [t, p, s, e, mask_port()]);
            ([t, p, s], Some(e), Some(bits))
        }
        (Some(e), false) => {
            let [t, p, s, e] = m.entry(EntryKind::Step, [t, p, s, e]);
            ([t, p, s], Some(e), None)
        }
        (None, true) => {
            let [t, p, s, bits] = m.entry(EntryKind::Step, [t, p, s, mask_port()]);
            ([t, p, s], None, Some(bits))
        }
        (None, false) => (m.entry(EntryKind::Step, [t, p, s]), None, None),
    }
}

/// The most token ids one sparse pick chooses from.
pub const FEW: u64 = 4096;

/// A sparse step entry's rows, and each row's candidate ids.
pub fn sparse_ports(m: &Model) -> [Tensor; 4] {
    let port = |role, ty| Port { role, ty };
    let row = |role| port(role, TensorType::new([Rows], DType::U32));
    let [t, p, s] = [Role::Tokens, Role::Positions, Role::Slots].map(row);
    let ids = port(
        Role::Candidates,
        TensorType::new([Rows, Candidates], DType::U32),
    );
    m.entry(EntryKind::Step, [t, p, s, ids])
}

/// Each row's likeliest allowed token, logits and optional taps.
pub fn pick(m: &Model, logits: &Tensor, allowed: Option<&Tensor>, taps: Option<&Tensor>) -> Graph {
    let logits = allowed.map_or_else(|| logits.clone(), |bits| logits.allow(bits));
    let ids = logits.top_k(1u64).reshape([Rows]);
    match taps {
        Some(taps) => m.finish([
            (Role::Tokens, &ids),
            (Role::Logits, &logits),
            (Role::Taps, taps),
        ]),
        None => m.finish([(Role::Tokens, &ids), (Role::Logits, &logits)]),
    }
}
