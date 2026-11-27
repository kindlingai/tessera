//! Tiled flash attention over packed variable-length sequences.

use crate::flash::{
    BQ, CW, THREADS, WAYS, check_width, chunk, flash_body, flash_body_inline, flash_needs,
    form_suffix, head_at, head_at_inline, kv_accepts, padded, stream, stream_inline, way_name,
};
use std::cell::RefCell;
use std::collections::HashMap;
use tessera_codegen::ir::Body;
use tessera_codegen::prelude::*;
use tessera_graph::Order;
use tessera_lower::form::Load as Stored;
use tessera_lower::kernels::Map;
use tessera_lower::sites;
use tessera_lower::{Arg, Attn, Draft, Kernel, Launch, Load, Read, Shape, Source, Step, T, Tree};

type Key = (
    u32,
    bool,
    u32,
    bool,
    &'static str,
    &'static str,
    &'static str,
);

thread_local! {
    /// Cached packed flash bodies by width, causality, block and forms.
    static TREES: RefCell<HashMap<Key, Body>> = RefCell::new(HashMap::new());
}

/// Packed tiled attention: each row reads `[start, end)` bounds.
#[kernel]
fn flash_packed<const CAUSAL: bool>(
    q: Buf<F32>,
    #[form(k_form)] k: Buf<Dyn>,
    #[form(v_form)] values: Buf<Dyn>,
    #[form(out_form)] mut out: BufMut<Dyn>,
    bounds: Buf<U32>,
    rows: u32,
    heads: u32,
    kv_heads: u32,
    #[build] cw: u32,
    #[build] w: u32,
    #[build] bk: u32,
    #[build] held_q: bool,
    #[build] k_form: &'static dyn Stored,
    #[build] v_form: &'static dyn Stored,
    #[build] out_form: &'static dyn Stored,
) {
    let slices = (((w + 15) / 16) * 16) / cw;
    let group = gid() / THREADS;
    let slice = group % slices;
    let hd = head_at(group / slices, rows, heads, kv_heads);
    let row = lid();
    let mut st = stream(cw, bk, held_q);
    if lid() < BQ {
        let r = hd.q0 + row;
        let mut from = 0u32;
        let mut to = 0u32;
        if r < rows {
            from = bounds[(r * 2) as u64];
            let end = bounds[(r * 2 + 1) as u64];
            to = if CAUSAL && r + 1 < end { r + 1 } else { end };
            if to > rows {
                to = rows;
            }
            if from > to {
                from = to;
            }
        }
        st.from_s[row] = from;
        st.to_s[row] = to;
    }
    barrier();
    flash_body(
        &mut st, q, k, values, out, &hd, rows, slice, slices, cw, w, bk, held_q, k_form, v_form,
        out_form,
    );
}

/// Tiled attention over packed variable-length rows.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = packed_name(self.1, self.2),
    order = Order::PerConfig,
    sizes = self.0.sizes(),
    work = self.0.heads * self.0.queries * self.0.seen() * self.0.width,
    needs = flash_needs(&self.0, self.1, self.2),
    accepts = packed_accepts,
    emits = packed_emits,
    source = packed_source,
    launches = packed_launches,
    load = packed_load,
)]
pub(crate) struct FlashPacked(pub Attn, pub u32, pub bool);

fn packed_name(bk: u32, held: bool) -> &'static str {
    match (bk, held) {
        (16, false) => "flash_packed",
        (16, true) => "flash_packed_held",
        (_, false) => "flash_packed_k32",
        (_, true) => "flash_packed_k32_held",
    }
}

fn packed_accepts(k: &FlashPacked, input: usize, form: &dyn Stored) -> bool {
    kv_accepts(&k.0, input, form)
}

fn packed_emits(k: &FlashPacked, form: &dyn Stored) -> bool {
    form.name() == k.0.out_form.name()
}

impl FlashPacked {
    fn tree(&self) -> Body {
        let a = &self.0;
        let (w, bk, held, causal) = (a.width, self.1, self.2, a.mask.causal());
        let key = (
            w as u32,
            causal,
            bk,
            held,
            a.k_form.name(),
            a.v_form.name(),
            a.out_form.name(),
        );
        TREES.with(|t| {
            t.borrow_mut()
                .entry(key)
                .or_insert_with(|| {
                    let cw = chunk(w) as u32;
                    if causal {
                        flash_packed_ir::<true>(
                            cw, w as u32, bk, held, a.k_form, a.v_form, a.out_form,
                        )
                    } else {
                        flash_packed_ir::<false>(
                            cw, w as u32, bk, held, a.k_form, a.v_form, a.out_form,
                        )
                    }
                })
                .clone()
        })
    }

    fn groups(&self) -> usize {
        let slices = padded(self.0.width) / chunk(self.0.width);
        self.0.heads * self.0.queries.div_ceil(BQ as usize) * slices
    }
}

fn packed_load(k: &FlashPacked, len: usize) -> Option<Load> {
    let a = &k.0;
    let (keys, side) = (a.seen() as f64, (a.kv_heads * a.width) as f64);
    let slices = (padded(a.width) / chunk(a.width)) as f64;
    let blocks = a.queries.div_ceil(BQ as usize) as f64 * slices;
    let k_stream = a.k_form.element_bytes() * keys * side * blocks;
    let v_stream = a.v_form.element_bytes() * keys * side * blocks;
    Some(Load {
        threads: k.groups() * THREADS as usize,
        group: THREADS as usize,
        trips: keys / k.1 as f64 * slices,
        trip: k.tree().trip(),
        reads: vec![4.0 * len as f64, k_stream, v_stream],
        written: a.out_form.bytes(len) as f64,
        flight: 16.0,
        ..Load::default()
    })
}

fn packed_source(k: &FlashPacked) -> Source {
    let way = &way_name(k.1, k.2)["flash".len()..];
    let kind = if k.0.mask.causal() { "causal" } else { "full" };
    let suffix = form_suffix(&k.0);
    Source::named(
        k.tree(),
        format!("flash_packed_{kind}_{}{way}{suffix}", k.0.width),
    )
}

fn packed_launches(k: &FlashPacked, step: &Step) -> Vec<Launch> {
    let a = &k.0;
    let args = vec![
        step.read(0),
        step.read(1),
        step.read(2),
        step.out(),
        step.read(3),
        Arg::count(a.queries),
        Arg::count(a.heads),
        Arg::count(a.kv_heads),
    ];
    vec![Launch {
        args,
        shape: Shape::Grid {
            across: k.groups(),
            down: 1,
            threads: THREADS as usize,
        },
    }]
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let site: sites::Attend = tree.site(at)?;
    let bounds = site.packed?;
    let w = site.sizes.width;
    check_width(w).ok()?;
    let out = site.out();
    let mut drafts = Vec::new();
    let ways = WAYS
        .into_iter()
        .filter(|&(_, held)| !held || w <= CW as usize);
    for (way, (bk, held)) in ways.enumerate() {
        let scale = Map::scaled(site.scale, site.q_form);
        let reads = [
            Read::Draft(0),
            Read::Node(site.k),
            Read::Node(site.v),
            bounds.into(),
        ];
        drafts.push(Draft::new(scale, [site.q], out).choice(way));
        drafts.push(Draft::new(FlashPacked(site.sizes, bk, held), reads, out).choice(way));
    }
    Some(drafts)
}

fn why_not(tree: &Tree, at: T) -> Option<String> {
    let site: sites::Attend = tree.site(at)?;
    site.packed?;
    check_width(site.sizes.width).err()
}

tessera_lower::rule!("flash_packed", propose, why = why_not);
