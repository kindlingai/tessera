use super::Pattern;
use crate::tree::{T, Term, Tree};
use tessera_graph::{Op, Select};

/// Indices of each row's `k` largest entries.
#[derive(Clone, Copy, Debug)]
pub struct Top {
    pub x: T,
    pub rows: usize,
    pub width: usize,
    pub k: usize,
    pub first: usize,
    pub whole: usize,
}

impl Pattern for Top {
    fn find(tree: &Tree, at: T) -> Option<Self> {
        let node = tree.node(at);
        if !matches!(node.term, Term::Op(Op::Select(Select::TopK { .. }))) {
            return None;
        }
        let x = node.args[0];
        let width = *tree.node(x).shape.last()?;
        let rows = tree.elements(x) / width;
        let k = tree.elements(at) * width / tree.elements(x);
        let (first, whole) = tree.row_share(x);
        Some(Self {
            x,
            rows,
            width,
            k,
            first,
            whole,
        })
    }
}
