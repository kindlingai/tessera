use super::{Chain, Pattern};
use crate::form::Form;
use crate::kernels::MapStage;
use crate::rule::Read;
use crate::tree::{T, Term, Tree};
use graph::Op;

/// `x[m, k]` times `w[n, k]` transposed, `w` stored as `form`.
#[derive(Clone, Copy, Debug)]
pub struct Matmul {
    pub x: T,
    pub w: T,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub form: &'static dyn Form,
}

impl Pattern for Matmul {
    fn find(tree: &Tree, at: T) -> Option<Self> {
        let node = tree.node(at);
        if node.term != Term::Op(Op::Matmul) {
            return None;
        }
        let (x, w) = (node.args[0], node.args[1]);
        let [n, k] = tree.node(w).shape[..] else {
            return None;
        };
        let (m, form) = (tree.elements(x) / k, tree.node(w).form);
        Some(Self {
            x,
            w,
            m,
            n,
            k,
            form,
        })
    }
}

/// Elementwise stages a sum runs through before it is stored.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct After {
    pub stages: Vec<MapStage>,
    /// Which of the chain's reads the sum stands in for.
    pub sum: usize,
}

impl After {
    /// Operands the stages read besides the sum.
    pub fn others(&self) -> usize {
        crate::kernels::operands(&self.stages)
    }
}

/// A matmul with the stages that finish it, if any.
#[derive(Clone, Debug)]
pub struct Finished {
    pub product: Matmul,
    pub after: After,
    /// The stages' other operands, in their order.
    pub others: Vec<Read>,
}

impl Finished {
    /// The rows, the weights, then the stages' other operands.
    pub fn reads(&self) -> Vec<Read> {
        let mut reads = vec![Read::Node(self.product.x), Read::Node(self.product.w)];
        reads.extend(self.others.iter().copied());
        reads
    }
}

impl Pattern for Finished {
    /// The matmul at `at`, or the one its chain finishes.
    fn find(tree: &Tree, at: T) -> Option<Self> {
        if let Some(product) = tree.site::<Matmul>(at) {
            return Some(Self {
                product,
                after: After::default(),
                others: Vec::new(),
            });
        }
        let Chain {
            stages,
            reads,
            spans,
            ..
        } = tree.site(at)?;
        if spans.iter().any(Option::is_some) {
            return None;
        }
        // A product read elsewhere is stored on its own.
        let lone = |r: &T| tree.site::<Matmul>(*r).filter(|_| !tree.root(*r));
        let sum = reads.iter().position(|r| lone(r).is_some())?;
        let others = reads.iter().filter(|r| **r != reads[sum]);
        Some(Self {
            product: lone(&reads[sum])?,
            after: After { stages, sum },
            others: others.map(|r| Read::Node(*r)).collect(),
        })
    }
}
