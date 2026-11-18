//! Patterns rules recognize at tree nodes, one type each.

mod attend;
mod chain;
mod gather;
mod matmul;
mod near;
mod stage;
mod top;

pub use attend::Attend;
pub use chain::Chain;
pub use gather::Gather;
pub use matmul::{After, Finished, Matmul, Picked};
pub use near::Near;
pub use stage::Stage;
pub use top::Top;

use crate::rule::Read;
use crate::tree::{T, Term, Tree};
use sensai_graph::{Composite, Op, Stage as Step};

/// A pattern a rule asks for at a node.
pub trait Pattern: Sized {
    fn find(tree: &Tree, at: T) -> Option<Self>;
}

/// A traced region with one output, met at that output.
#[derive(Clone, Debug)]
pub struct Region {
    pub kind: Composite,
    /// The values it reads, weights included, in first-read order.
    pub inputs: Vec<T>,
    pub len: usize,
}

impl Pattern for Region {
    fn find(tree: &Tree, at: T) -> Option<Self> {
        let from = tree.node(at).from;
        let region = tree.g.regions.iter().find(|r| r.outputs == [from])?;
        if tree.of(from)? != at {
            return None;
        }
        let inputs: Option<Vec<T>> = region.inputs.iter().map(|i| tree.of(*i)).collect();
        let inputs = inputs?;
        // Sums, kept rows and rotations stay with the engine.
        for n in tree.inside(at, &inputs) {
            let node = tree.node(n);
            let engine = matches!(
                node.term,
                Term::Stage(Step::AllReduce) | Term::Op(Op::Append(_))
            );
            let unread = node
                .args
                .iter()
                .any(|a| tree.leaf(*a) && !inputs.contains(a));
            if engine || unread {
                return None;
            }
        }
        let len = tree.elements(at);
        Some(Self {
            kind: region.kind,
            inputs,
            len,
        })
    }
}

impl Region {
    pub fn reads(&self) -> Vec<Read> {
        self.inputs.iter().map(|t| Read::Node(*t)).collect()
    }
}
