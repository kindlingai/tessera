use super::Pattern;
use crate::form::F32_FORM;
use crate::tree::{T, Term, Tree};
use sensai_graph::{Op, Select, View};

/// A matmul input of each pixel's neighbours, gathered by id.
#[derive(Clone, Copy, Debug)]
pub struct Near {
    /// The activations, `[.., pixels, cin]`.
    pub src: T,
    /// Neighbour ids, `[pixels, taps]`. Past the end reads zero.
    pub ids: T,
    pub pixels: usize,
    pub cin: usize,
    pub taps: usize,
}

/// `t`'s one argument, if `t` is the view `want`.
fn view(tree: &Tree, t: T, want: &View) -> Option<T> {
    let n = tree.node(t);
    (n.term == Term::Op(Op::View(want.clone()))).then(|| n.args[0])
}

impl Pattern for Near {
    /// `reshape(permute(gather(permute(src), ids)))`, as models trace it.
    fn find(tree: &Tree, at: T) -> Option<Self> {
        let Term::Op(Op::View(View::Reshape(_))) = &tree.node(at).term else {
            return None;
        };
        let near = view(
            tree,
            tree.node(at).args[0],
            &View::Permute(vec![2, 0, 1, 3]),
        )?;
        let g = tree.node(near);
        if g.term != Term::Op(Op::Select(Select::Gather)) {
            return None;
        }
        let (table, ids) = (g.args[0], g.args[1]);
        let src = view(tree, table, &View::Permute(vec![1, 0, 2]))?;
        let [_, pixels, cin] = tree.node(src).shape[..] else {
            return None;
        };
        let [rows, taps] = tree.node(ids).shape[..] else {
            return None;
        };
        let plain = tree.node(src).form.name() == F32_FORM.name();
        (rows == pixels && plain).then_some(Self {
            src,
            ids,
            pixels,
            cin,
            taps,
        })
    }
}
