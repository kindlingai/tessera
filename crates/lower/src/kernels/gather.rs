//! Rows of a table at given indices.

use crate::abi::{Arg, Launch, Shape, Source};
use crate::kernel::Kernel;
use crate::plan::Step;
use crate::rule::{Draft, Site};
use ops_model::gather_rows;

#[derive(Clone, Copy, Debug)]
pub struct Gather {
    pub rows: usize,
    pub width: usize,
}

impl Kernel for Gather {
    fn name(&self) -> &'static str {
        "gather"
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

fn propose(site: &Site) -> Option<Vec<Draft>> {
    let Site::Gather {
        table,
        ids,
        rows,
        width,
    } = *site
    else {
        return None;
    };
    Some(vec![Draft::new(
        Gather { rows, width },
        [table, ids],
        rows * width,
    )])
}

crate::rule!("gather", propose);
