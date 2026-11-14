use super::Pattern;
use crate::kernels::MapStage;
use crate::tree::{T, Term, Tree, map_stage};

/// Elementwise stages over `reads[0]`. Operands follow it.
#[derive(Clone, Debug)]
pub struct Chain {
    pub stages: Vec<MapStage>,
    pub reads: Vec<T>,
    pub len: usize,
    /// Per operand: its length when it repeats along the rows.
    pub spans: Vec<Option<usize>>,
}

impl Pattern for Chain {
    /// The stages ending at `at`, back through first arguments.
    fn find(tree: &Tree, at: T) -> Option<Self> {
        let Term::Stage(last) = &tree.node(at).term else {
            return None;
        };
        map_stage(last)?;
        let mut nodes = vec![at];
        while let Some(first) = tree.node(nodes[nodes.len() - 1]).args.first() {
            if !tree.joins(*first) {
                break;
            }
            nodes.push(*first);
        }
        nodes.reverse();
        let source = tree.node(nodes[0]).args[0];
        let len = tree.elements(source);
        let (mut stages, mut reads) = (Vec::new(), vec![source]);
        for n in nodes.iter().map(|n| tree.node(*n)) {
            let Term::Stage(stage) = &n.term else {
                return None;
            };
            stages.push(map_stage(stage)?);
            reads.extend(&n.args[1..]);
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
