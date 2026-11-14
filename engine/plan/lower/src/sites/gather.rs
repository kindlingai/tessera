use super::Pattern;
use crate::form::Form;
use crate::tree::{T, Term, Tree};
use graph::{Op, Select};

/// Rows of `table` picked by `ids`.
#[derive(Clone, Copy, Debug)]
pub struct Gather {
    pub table: T,
    pub ids: T,
    pub rows: usize,
    pub width: usize,
    pub form: &'static dyn Form,
}

impl Pattern for Gather {
    fn find(tree: &Tree, at: T) -> Option<Self> {
        let node = tree.node(at);
        if node.term != Term::Op(Op::Select(Select::Gather)) {
            return None;
        }
        let (table, ids) = (node.args[0], node.args[1]);
        let rows = tree.elements(ids);
        let width = tree.elements(table) / tree.node(table).shape.first()?;
        Some(Self {
            table,
            ids,
            rows,
            width,
            form: tree.node(table).form,
        })
    }
}
