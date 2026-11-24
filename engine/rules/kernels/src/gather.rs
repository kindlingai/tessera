//! Rows of a table at given indices.

use tessera_codegen::kept::{self, Layout};
use tessera_codegen::prelude::*;
use tessera_lower::form::Load as Stored;
use tessera_lower::sites;
use tessera_lower::{Arg, Draft, Kernel, Launch, Shape, Source, Step, T, Tree};

/// One thread per output element, widened from the table.
#[kernel]
fn gather(
    #[form(form)] table: Buf<Dyn>,
    ids: Buf<U32>,
    mut out: BufMut<F32>,
    rows: u32,
    width: u32,
    table_rows: u32,
    #[build] form: &'static dyn Stored,
) {
    if gid() >= rows * width {
        return;
    }
    let row = ids[(gid() / width) as u64];
    let mut v = 0.0f32;
    if row < table_rows {
        v = table[row as u64 * width as u64 + (gid() % width) as u64];
    }
    out[gid() as u64] = v;
}

/// One thread per output from a kept table.
#[kernel]
fn kept_gather(
    #[form(form)] table: Buf<Dyn>,
    ids: Buf<U32>,
    mut out: BufMut<F32>,
    rows: u32,
    keys: u32,
    feat: u32,
    slots: u32,
    #[build] form: &'static dyn Stored,
) {
    let per_row = keys * feat;
    if gid() >= rows * per_row {
        return;
    }
    let r = gid() / per_row;
    let rem = gid() % per_row;
    let pos = rem / feat;
    let j = rem % feat;
    let slot = ids[r as u64];
    let mut v = 0.0f32;
    if slot < slots {
        v = table[(kept_row(slot, pos) * feat + j) as u64];
    }
    out[gid() as u64] = v;
}

#[derive(Clone, Copy, Debug, Kernel)]
#[launch(gather(self.form), name = self.form.suffixed("gather"), threads = self.rows * self.width, accepts = gather_accepts)]
pub struct Gather {
    pub rows: usize,
    pub width: usize,
    /// Ids from here on read zeros.
    pub table_rows: usize,
    /// How the table is stored.
    pub form: &'static dyn Stored,
}

fn gather_accepts(k: &Gather, input: usize, form: &dyn Stored) -> bool {
    k.form.taken(input, 0, form)
}

#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "kept_gather",
    sizes = [],
    work = self.rows * self.keys * self.feat,
    accepts = kept_accepts,
    source = kept_source,
    launches = kept_launches,
)]
pub struct KeptGather {
    pub rows: usize,
    pub keys: usize,
    pub feat: usize,
    pub slots: usize,
    pub context: usize,
    pub kept: &'static dyn Layout,
    pub form: &'static dyn Stored,
}

fn kept_accepts(k: &KeptGather, input: usize, form: &dyn Stored) -> bool {
    k.form.taken(input, 0, form)
}

fn kept_source(k: &KeptGather) -> Source {
    let body = kept::attach(kept_gather_ir(k.form), k.kept);
    let entry = match k.form.name() {
        "f32" => format!("plan_kept_gather_{}", k.kept.name()),
        f => format!("plan_kept_gather_{}_{f}", k.kept.name()),
    };
    Source::named(body, entry)
}

fn kept_launches(k: &KeptGather, step: &Step) -> Vec<Launch> {
    let total = k.rows * k.keys * k.feat;
    let sizes = [k.rows, k.keys, k.feat, k.slots].map(Arg::count);
    let kept = k
        .kept
        .values(k.slots, k.context)
        .into_iter()
        .map(|v| Arg::count(v as usize));
    let args = [step.read(0), step.read(1), step.out()]
        .into_iter()
        .chain(step.kept(k.kept))
        .chain(sizes)
        .chain(kept)
        .collect();
    vec![Launch {
        args,
        shape: Shape::Line(total),
    }]
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let sites::Gather {
        table,
        ids,
        rows,
        width,
        table_rows,
        form,
        kept,
    } = tree.site(at)?;
    if let Some(kg) = kept {
        let feat = width / kg.keys.max(1);
        let kernel = KeptGather {
            rows,
            keys: kg.keys,
            feat,
            slots: table_rows,
            context: kg.context,
            kept: kg.layout,
            form,
        };
        return Some(vec![Draft::new(kernel, [table, ids], rows * width)]);
    }
    let gather = Gather {
        rows,
        width,
        table_rows,
        form,
    };
    Some(vec![Draft::new(gather, [table, ids], rows * width)])
}

tessera_lower::rule!("gather", propose);
