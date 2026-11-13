use super::Pattern;
use crate::tree::{T, Term, Tree};
use graph::DType;

/// `x[m, k]` times `w[n, k]` transposed, `w` stored as `form`.
#[derive(Clone, Copy, Debug)]
pub struct Matmul {
    pub x: T,
    pub w: T,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub form: DType,
}

impl Pattern for Matmul {
    fn find(tree: &Tree, at: T) -> Option<Self> {
        let node = tree.node(at);
        if node.term != Term::Matmul {
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
