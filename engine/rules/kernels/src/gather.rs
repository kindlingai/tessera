//! Rows of a table at given indices.

use tessera_codegen::kept::{self, Layout};
use tessera_codegen::prelude::*;
use tessera_lower::form::Load as Stored;
use tessera_lower::sites;
use tessera_lower::{Arg, Draft, Kernel, Launch, Shape, Source, Step, T, Tree};

/// Elements handled per thread on the vectorized path.
const WIDE: usize = 4;

/// One thread per element or quad, widened from the table.
#[kernel]
fn gather(
    #[form(form)] table: Buf<Dyn>,
    ids: Buf<U32>,
    mut out: BufMut<F32>,
    rows: u32,
    width: u32,
    table_rows: u32,
    #[build] form: &'static dyn Stored,
    #[build] wide: bool,
) {
    if wide {
        let width4 = width / 4;
        if gid() >= rows * width4 {
            return;
        }
        let row = ids[(gid() / width4) as u64];
        let mut o = F32x4::zero();
        if row < table_rows {
            let col = (gid() % width4) * 4;
            let v = table.load4(row as u64 * width as u64 + col as u64);
            o.x = v.x;
            o.y = v.y;
            o.z = v.z;
            o.w = v.w;
        }
        out.store4((gid() * 4) as u64, o);
    } else {
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
}

/// One thread per element or quad from a kept table.
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
    #[build] wide: bool,
) {
    if wide {
        let feat4 = feat / 4;
        let per_row = keys * feat4;
        if gid() >= rows * per_row {
            return;
        }
        let r = gid() / per_row;
        let rem = gid() % per_row;
        let pos = rem / feat4;
        let j = (rem % feat4) * 4;
        let slot = ids[r as u64];
        let mut o = F32x4::zero();
        if slot < slots {
            let v = table.load4((kept_row(slot, pos) * feat + j) as u64);
            o.x = v.x;
            o.y = v.y;
            o.z = v.z;
            o.w = v.w;
        }
        out.store4((gid() * 4) as u64, o);
    } else {
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
}

#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    gather(self.form, self.width.is_multiple_of(WIDE)),
    name = self.form.suffixed("gather"),
    threads = if self.width.is_multiple_of(WIDE) {
        self.rows * (self.width / WIDE)
    } else {
        self.rows * self.width
    },
    accepts = gather_accepts,
    source = gather_source,
)]
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

fn gather_source(k: &Gather) -> Source {
    let wide = k.width.is_multiple_of(WIDE);
    let suffix = if wide { "_x4" } else { "" };
    let entry = match k.form.name() {
        "f32" => format!("plan_gather{suffix}"),
        f => format!("plan_gather{suffix}_{f}"),
    };
    Source::named(gather_ir(k.form, wide), entry)
}

#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "kept_gather",
    sizes = [],
    work = self.rows * self.keys * self.feat,
    accepts = kept_accepts,
    source = kept_source,
    launches = kept_launches,
    shape = Shape::Line(if self.feat.is_multiple_of(WIDE) {
        self.rows * self.keys * (self.feat / WIDE)
    } else {
        self.rows * self.keys * self.feat
    }),
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
    let wide = k.feat.is_multiple_of(WIDE);
    let body = kept::attach(kept_gather_ir(k.form, wide), k.kept);
    let suffix = if wide { "_x4" } else { "" };
    let entry = match k.form.name() {
        "f32" => format!("plan_kept_gather{suffix}_{}", k.kept.name()),
        f => format!("plan_kept_gather{suffix}_{}_{f}", k.kept.name()),
    };
    Source::named(body, entry)
}

fn kept_launches(k: &KeptGather, step: &Step) -> Vec<Launch> {
    let sizes = [k.rows, k.keys, k.feat, k.slots].map(Arg::count);
    let kept = k
        .kept
        .values(k.slots, k.context)
        .into_iter()
        .map(|v| Arg::count(v as usize));
    let args = [step.read(0), step.read(1), step.out()]
        .into_iter()
        .chain(step.kept_for(k.kept, true))
        .chain(sizes)
        .chain(kept)
        .collect();
    let shape = k.shape().expect("a line");
    vec![Launch { args, shape }]
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
