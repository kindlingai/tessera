use crate::Term;
use tessera_graph::{Parts, Shape};

/// Ops with no lowering rule.
#[derive(Debug, thiserror::Error)]
pub enum Unsupported {
    #[error("entry {0}")]
    Entry(usize),
    #[error("another entry's input")]
    EntryInput,
    #[error("kept rows along a computed axis")]
    KeptRows,
    #[error("node {node} reads an unsupported node")]
    Unread { node: usize },
    #[error("node {node} reads a rotated weight other than as a matmul")]
    Turned { node: usize },
    #[error("node {node} in {form} needs rows of {align}")]
    Align {
        node: usize,
        form: &'static str,
        align: usize,
    },
    #[error("node {node} sums across ranks with no wire dtype; resolve it first")]
    Wire { node: usize },
    #[error("{0}")]
    Uncovered(Box<Uncovered>),
    #[error("any node: no rule crate is linked; add `use rules as _;`")]
    Unlinked,
}

/// An uncovered node, and the root above it.
#[derive(Debug, thiserror::Error)]
#[error("{term:?} over {shape:?} from {reads:?}{}", under_text(.under))]
pub struct Uncovered {
    pub term: Term,
    pub shape: Vec<usize>,
    pub reads: Vec<Vec<usize>>,
    pub under: Option<Term>,
}

/// A size that does not fit this launch.
#[derive(Debug, thiserror::Error)]
pub enum Size {
    #[error("{0}")]
    Shape(Shape),
    #[error("axis {axis} of {shape} over {parts:?}")]
    Split {
        axis: usize,
        shape: Shape,
        parts: Parts,
    },
}

fn under_text(under: &Option<Term>) -> String {
    under
        .as_ref()
        .map_or_else(String::new, |t| format!(", under {t:?}"))
}
