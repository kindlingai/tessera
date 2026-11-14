//! Rows of a table at given indices.

use codegen::prelude::*;
use lower::sites;
use lower::{Draft, Form as Stored, Kernel, T, Tree};

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

#[derive(Clone, Copy, Debug, Kernel)]
#[launch(gather(self.form), name = self.form.suffixed("gather"), threads = self.rows * self.width, accepts = gather_accepts)]
pub struct Gather {
    pub rows: usize,
    pub width: usize,
    /// How the table is stored.
    pub form: &'static dyn Stored,
}

fn gather_accepts(k: &Gather, input: usize, form: &dyn Stored) -> bool {
    k.form.taken(input, 0, form)
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
