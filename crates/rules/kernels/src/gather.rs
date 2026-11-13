//! Rows of a table at given indices.

use codegen::prelude::*;
use lower::{Arg, DType, Draft, Kernel, Launch, Shape, Site, Source, Step, Stored, T, Tree};

/// One thread per output element, widened from the table.
#[kernel]
fn gather<F: Form<Elem = f32>>(
    table: Buf<F>,
    ids: Buf<U32>,
    mut out: BufMut<F32>,
    rows: u32,
    width: u32,
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
    pub form: DType,
}

impl Kernel for Gather {
    fn name(&self) -> &'static str {
        match self.form {
            DType::Bf16 => "gather_bf16",
            _ => "gather",
        }
    }

    fn sizes(&self) -> Vec<usize> {
        vec![self.rows, self.width]
    }

    fn source(&self) -> Source {
        let body = match self.form {
            DType::Bf16 => gather_ir::<Bf16>(),
            _ => gather_ir::<F32>(),
        };
        let mut body = body;
        body.entry = format!("plan_gather_{}", self.form.name());
        Source::Made(body)
    }

    fn accepts(&self, input: usize, form: DType) -> bool {
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
