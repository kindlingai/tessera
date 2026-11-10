//! Rows of a table at given indices.

use crate::gather_rows;
use lower::{Arg, Draft, Kernel, Launch, Shape, Site, Source, Step, T, Tree};

#[derive(Clone, Copy, Debug)]
pub struct Gather {
    pub rows: usize,
    pub width: usize,
}

impl Kernel for Gather {
    fn name(&self) -> &'static str {
        "gather"
    }

    fn sizes(&self) -> Vec<usize> {
        vec![self.rows, self.width]
    }

    fn source(&self) -> Source {
        Source::Fixed {
            metal: gather_rows::METAL,
            cuda: gather_rows::CUDA,
            entry: gather_rows::ENTRY,
        }
    }

    /// The kernel takes its output first.
    fn launches(&self, step: &Step) -> Vec<Launch> {
        let sizes = Arg::pair(self.rows, self.width);
        vec![Launch {
            args: vec![step.out(), step.read(0), step.read(1), sizes],
            shape: Shape::Rows(self.rows),
        }]
    }
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let Site::Gather {
        table,
        ids,
        rows,
        width,
    } = tree.site(at)?
    else {
        return None;
    };
    Some(vec![Draft::new(
        Gather { rows, width },
        [table, ids],
        rows * width,
    )])
}

lower::rule!("gather", propose);
