use super::Pattern;
use crate::tree::{T, Term, Tree, map_stage};

/// One row stage that is not elementwise.
#[derive(Clone, Debug)]
pub struct Stage {
    pub stage: graph::Stage,
    pub x: T,
    /// The stage's argument, and how many elements it holds.
    pub operand: Option<(T, usize)>,
    pub len: usize,
    pub width: usize,
}

impl Pattern for Stage {
    fn find(tree: &Tree, at: T) -> Option<Self> {
        let node = tree.node(at);
        let Term::Stage(stage) = &node.term else {
            return None;
        };
        if map_stage(stage).is_some() {
            return None;
        }
        let x = node.args[0];
        let operand = node.args.get(1).map(|o| (*o, tree.elements(*o)));
        let (len, width) = (tree.elements(x), *tree.node(x).shape.last()?);
        Some(Self {
            stage: stage.clone(),
            x,
            operand,
            len,
            width,
        })
    }
}
