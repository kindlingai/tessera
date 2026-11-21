use super::Pattern;
use crate::form::{F32_FORM, Load};
use crate::tree::{T, Term, Tree};
use tessera_graph::{Op, Select};

/// Rows of `table` picked by `ids`, along its first axis.
#[derive(Clone, Copy, Debug)]
pub struct Gather {
    pub table: T,
    pub ids: T,
    pub rows: usize,
    pub width: usize,
    pub table_rows: usize,
    pub form: &'static dyn Load,
}

impl Pattern for Gather {
    fn find(tree: &Tree, at: T) -> Option<Self> {
        let node = tree.node(at);
        if node.term != Term::Op(Op::Select(Select::Gather)) {
            return None;
        }
        let (table, ids) = (node.args[0], node.args[1]);
        let rows = tree.elements(ids);
        // Weight tables are matrices. Values may have more axes.
        let kept = matches!(tree.node(table).term, Term::Op(Op::Append(_)));
        let plain = tree.node(table).form.name() == F32_FORM.name() && !tree.weighed(table);
        let table_rows = match tree.node(table).shape[..] {
            [rows, _] => rows,
            [rows, _, ..] if kept || plain => rows,
            _ => return None,
        };
        Some(Self {
            table,
            ids,
            rows,
            width: tree.elements(table) / table_rows,
            table_rows,
            form: tree.node(table).form,
        })
    }
}
