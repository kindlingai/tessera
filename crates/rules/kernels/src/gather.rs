//! Rows of a table at given indices.

use crate::gather_rows;
use lower::{Arg, Draft, Form, Kernel, Launch, Shape, Site, Source, Step, T, Tree};

#[derive(Clone, Copy, Debug)]
pub struct Gather {
    pub rows: usize,
    pub width: usize,
    /// How the table is stored.
    pub form: Form,
}

impl Kernel for Gather {
    fn name(&self) -> &'static str {
        match self.form {
            Form::F32 => "gather",
            Form::Bf16 => "gather_bf16",
        }
    }

    fn sizes(&self) -> Vec<usize> {
        vec![self.rows, self.width]
    }

    fn source(&self) -> Source {
        Source::Fixed {
            metal: gather_rows::METAL,
            cuda: gather_rows::CUDA,
            entry: match self.form {
                Form::F32 => gather_rows::ENTRY,
                Form::Bf16 => gather_rows::ENTRY_BF16,
            },
        }
    }

    fn accepts(&self, input: usize, form: Form) -> bool {
        self.form.taken(input, 0, form)
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
        form,
    } = tree.site(at)?
    else {
        return None;
    };
    let gather = Gather { rows, width, form };
    Some(vec![Draft::new(gather, [table, ids], rows * width)])
}

lower::rule!("gather", propose);
