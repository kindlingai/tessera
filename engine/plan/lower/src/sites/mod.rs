//! Patterns rules recognize at tree nodes, one type each.

mod attend;
mod chain;
mod coded;
mod gather;
mod matmul;
mod near;
mod stage;
mod top;

pub use attend::{Attend, Delta};
pub use chain::Chain;
pub use coded::Coded;
pub use gather::Gather;
pub use matmul::{After, Finished, Matmul, Picked, Rotated};
pub use near::Near;
pub use stage::Stage;
pub use top::Top;

use crate::rule::Read;
use crate::select::Home;
use crate::tree::{T, Term, Tree};
use tessera_graph::{Composite, Op, Stage as Step};

/// A pattern a rule asks for at a node.
pub trait Pattern: Sized {
    fn find(tree: &Tree, at: T) -> Option<Self>;
}

/// A traced region, met at its last output.
#[derive(Clone, Debug)]
pub struct Region {
    /// The values it reads, weights included, in first-read order.
    pub inputs: Vec<T>,
    /// Its outputs in the order the composite returns them.
    pub outputs: Vec<T>,
    /// Elements of the node met.
    pub len: usize,
    /// Kept rows its launches append to.
    pub kept: Vec<Appended>,
    /// Per rotated input: its place, then `suh` and `svh`.
    pub parts: Vec<(usize, [T; 2])>,
    /// Each sum across ranks inside it, in order.
    pub sums: Vec<T>,
}

/// Kept rows a region appends to, in `Read::Kept` numbering.
#[derive(Clone, Copy, Debug)]
pub struct Appended {
    pub index: u32,
    /// Slots, rows per slot, and elements per row.
    pub slots: usize,
    pub context: usize,
    pub width: usize,
    /// The places in `inputs` of the slot and position ids.
    pub by: [Option<usize>; 2],
}

impl Pattern for Region {
    fn find(tree: &Tree, at: T) -> Option<Self> {
        // Sums stay with the engine.
        Self::of(tree, at, None).filter(|r| r.sums.is_empty())
    }
}

impl Region {
    /// Its inputs, then its kept rows.
    pub fn reads(&self) -> Vec<Read> {
        let inputs = self.inputs.iter().map(|t| Read::Node(*t));
        let kept = self.kept.iter().map(|k| Read::Kept(k.index));
        inputs.chain(kept).collect()
    }

    /// The innermost region met at `at`, of `kind` if given.
    pub fn of(tree: &Tree, at: T, kind: Option<Composite>) -> Option<Self> {
        let met = |r: &&tessera_graph::Region| {
            kind.is_none_or(|k| r.kind == k) && tree.anchor(r) == Some(at)
        };
        let region = tree.g.regions.iter().find(met)?;
        let inputs: Option<Vec<T>> = region.inputs.iter().map(|i| tree.of(*i)).collect();
        let inputs = inputs?;
        let outputs: Option<Vec<T>> = region.outputs.iter().map(|o| tree.of(*o)).collect();
        let outputs = outputs?;
        let inside = tree.inside(at, &inputs);
        // A cover computes only what lies below its node.
        if !outputs.iter().all(|o| inside.contains(o)) {
            return None;
        }
        let (mut kept, mut parts, mut sums) = (Vec::new(), Vec::new(), Vec::new());
        for n in inside {
            let node = tree.node(n);
            if node.term == Term::Stage(Step::AllReduce) {
                sums.push(n);
            }
            // Gathers and row-ownership crossings stay with the engine.
            if matches!(
                node.term,
                Term::Stage(Step::AllGather | Step::ReduceScatter | Step::GatherRows)
            ) {
                return None;
            }
            for a in &node.args {
                if !tree.leaf(*a) || inputs.contains(a) {
                    continue;
                }
                // A rotated input brings its scales.
                let (place, part) = scale(tree, &inputs, *a)?;
                if !parts.iter().any(|(p, _)| *p == place) {
                    parts.push((place, [*a; 2]));
                }
                for (p, scales) in &mut parts {
                    if *p == place {
                        scales[part] = *a;
                    }
                }
            }
            if let Term::Op(Op::Append(index)) = node.term {
                let (_, _, context) = tree.kept_layout(index)?;
                let place = |a: &T| inputs.iter().position(|i| i == a);
                kept.push(Appended {
                    index,
                    slots: node.shape[0],
                    context,
                    width: node.shape[2..].iter().product(),
                    by: [place(&node.args[1]), place(&node.args[2])],
                });
            }
        }
        kept.sort_by_key(|k| k.index);
        let len = tree.elements(at);
        Some(Self {
            inputs,
            outputs,
            len,
            kept,
            parts,
            sums,
        })
    }
}

/// The input `t` scales, and which scale it is.
fn scale(tree: &Tree, inputs: &[T], t: T) -> Option<(usize, usize)> {
    let Term::Leaf(Home::Part(weight, part @ 1..)) = tree.node(t).term else {
        return None;
    };
    let codes = Term::Leaf(Home::Part(weight, 0));
    let place = inputs.iter().position(|i| tree.node(*i).term == codes)?;
    Some((place, part as usize - 1))
}
