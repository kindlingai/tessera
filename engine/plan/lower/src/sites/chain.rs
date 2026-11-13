use super::Pattern;
use crate::kernels::MapStage;
use crate::tree::{T, Term, Tree, map_stage};

/// Elementwise stages over `reads[0]`; operands follow it.
#[derive(Clone, Debug)]
pub struct Chain {
    pub stages: Vec<MapStage>,
    pub reads: Vec<T>,
    pub len: usize,
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
        // An operand of another size would need broadcasting.
        let fits = reads.iter().all(|r| tree.elements(*r) == len);
        fits.then_some(Self { stages, reads, len })
    }
}
