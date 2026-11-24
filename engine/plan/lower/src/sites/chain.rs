use super::Pattern;
use crate::kernels::MapStage;
use crate::tree::{T, Term, Tree, map_stage};
use tessera_graph::Stage;

/// Elementwise stages over `reads[0]`. Other operands come after.
#[derive(Clone, Debug)]
pub struct Chain {
    pub stages: Vec<MapStage>,
    pub reads: Vec<T>,
    pub len: usize,
    /// Per operand: its length when it repeats along the rows.
    pub spans: Vec<Option<usize>>,
}

impl Pattern for Chain {
    /// The stages ending at `at`, back through joining arguments.
    fn find(tree: &Tree, at: T) -> Option<Self> {
        let Term::Stage(last) = &tree.node(at).term else {
            return None;
        };
        map_stage(last)?;
        // Each node, and the argument its running value enters by.
        let (mut nodes, mut vias) = (vec![at], Vec::new());
        while let Some(via) = through(tree, nodes[nodes.len() - 1]) {
            vias.push(via);
            nodes.push(tree.node(nodes[nodes.len() - 1]).args[via]);
        }
        vias.push(0);
        let nodes: Vec<(T, usize)> = nodes.into_iter().zip(vias).rev().collect();
        let source = tree.node(nodes[0].0).args[0];
        let len = tree.elements(source);
        let (mut stages, mut reads) = (Vec::new(), vec![source]);
        for (n, via) in nodes.iter().map(|(n, via)| (tree.node(*n), *via)) {
            let Term::Stage(stage) = &n.term else {
                return None;
            };
            stages.push(map_stage(stage)?);
            let others = n.args.iter().enumerate().filter(|(i, _)| *i != via);
            reads.extend(others.map(|(_, r)| *r));
        }
        // An operand repeats when its length divides the rows.
        let span = |r: &T| match tree.elements(*r) {
            n if n == len => Some(None),
            n if len.is_multiple_of(n) => Some(Some(n)),
            _ => None,
        };
        let spans: Option<Vec<_>> = reads[1..].iter().map(span).collect();
        Some(Self {
            stages,
            reads,
            len,
            spans: spans?,
        })
    }
}

/// The argument continuing `n`'s chain. Add and multiply commute.
fn through(tree: &Tree, n: T) -> Option<usize> {
    let node = tree.node(n);
    let first = *node.args.first()?;
    if tree.joins(first) {
        return Some(0);
    }
    let commutes = matches!(&node.term, Term::Stage(Stage::Add | Stage::Mul));
    let second = *node.args.get(1)?;
    let whole = tree.elements(second) == tree.elements(n);
    (commutes && whole && tree.joins(second)).then_some(1)
}
