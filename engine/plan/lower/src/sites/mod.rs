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
use crate::select::Home;
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
    /// Kept rows it appends to. Its launches write them.
    pub kept: Vec<Appended>,
    /// Per rotated input: its place, then `suh` and `svh`.
    pub parts: Vec<(usize, [T; 2])>,
}

/// Kept rows a region appends to, as `Read::Kept` names them.
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
        let from = tree.node(at).from;
        let region = tree.g.regions.iter().find(|r| r.outputs == [from])?;
        if tree.of(from)? != at {
            return None;
        }
        let inputs: Option<Vec<T>> = region.inputs.iter().map(|i| tree.of(*i)).collect();
        let inputs = inputs?;
        let (mut kept, mut parts) = (Vec::new(), Vec::new());
        // Sums stay with the engine.
        for n in tree.inside(at, &inputs) {
            let node = tree.node(n);
            if node.term == Term::Stage(Step::AllReduce) {
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
                let (_, context) = tree.kept_rows(index)?;
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
            kind: region.kind,
            inputs,
            len,
            kept,
            parts,
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

impl Region {
    /// Its inputs, then its kept rows.
    pub fn reads(&self) -> Vec<Read> {
        let inputs = self.inputs.iter().map(|t| Read::Node(*t));
        let kept = self.kept.iter().map(|k| Read::Kept(k.index));
        inputs.chain(kept).collect()
    }
}
