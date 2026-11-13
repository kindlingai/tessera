//! Rows of a table at given indices.

use codegen::prelude::*;
use lower::sites;
use lower::{Arg, Draft, Form as Stored, Kernel, Launch, Shape, Source, Step, T, Tree};

/// One thread per output element, widened from the table.
#[kernel]
fn gather(
    #[form(form)] table: Buf<Dyn>,
    ids: Buf<U32>,
    mut out: BufMut<F32>,
    rows: u32,
    width: u32,
    #[build] form: &'static dyn Stored,
) {
    if gid() >= rows * width {
        return;
    }
    let row = ids[(gid() / width) as u64];
    out[gid() as u64] = table[row as u64 * width as u64 + (gid() % width) as u64];
}

#[derive(Clone, Copy, Debug)]
pub struct Gather {
    pub rows: usize,
    pub width: usize,
    /// How the table is stored.
    pub form: &'static dyn Stored,
}

impl Kernel for Gather {
    fn name(&self) -> &'static str {
        self.form.suffixed("gather")
    }

    fn sizes(&self) -> Vec<usize> {
        vec![self.rows, self.width]
    }

    fn source(&self) -> Source {
        Source::named(
            gather_ir(self.form),
            format!("plan_gather_{}", self.form.name()),
        )
    }

    fn accepts(&self, input: usize, form: &dyn Stored) -> bool {
        self.form.taken(input, 0, form)
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        let sizes = [self.rows, self.width].map(Arg::count);
        let buffers = [step.read(0), step.read(1), step.out()];
        vec![Launch {
            args: buffers.into_iter().chain(sizes).collect(),
            shape: Shape::Line(self.rows * self.width),
        }]
    }
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let sites::Gather {
        table,
        ids,
        rows,
        width,
        form,
    } = tree.site(at)?;
    let gather = Gather { rows, width, form };
    Some(vec![Draft::new(gather, [table, ids], rows * width)])
}

lower::rule!("gather", propose);
