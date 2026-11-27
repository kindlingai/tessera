//! Ragged flash attention kernel wrapper.

use crate::flash::{BQ, THREADS, WAYS, flash_rows_ir, form_suffix, kv_accepts, stream_floats};
use std::cell::RefCell;
use std::collections::HashMap;
use tessera_codegen::ir::Body;
use tessera_codegen::kept;
use tessera_codegen::prelude::*;
use tessera_graph::Order;
use tessera_lower::form::Load as Stored;
use tessera_lower::{Attn, Kernel, Launch, Load, Needs, Shape, Source, Step};

type Key = (u32, &'static str, &'static str, &'static str);

thread_local! {
    static TREES: RefCell<HashMap<Key, Body>> = RefCell::new(HashMap::new());
}

/// Scaled query rows over kept keys, by slot and position.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "flash_rows",
    order = Order::PerConfig,
    sizes = self.0.sizes(),
    work = self.0.heads * self.0.queries * self.0.keys.most() * self.0.width * self.passes(),
    needs = rows_needs(&self.0),
    accepts = rows_accepts,
    emits = rows_emits,
    source = rows_source,
    launches = rows_launches,
    load = rows_load,
)]
pub(crate) struct FlashRows(pub Attn);

fn rows_accepts(k: &FlashRows, input: usize, form: &dyn Stored) -> bool {
    kv_accepts(&k.0, input, form)
}

fn rows_emits(k: &FlashRows, form: &dyn Stored) -> bool {
    form.name() == k.0.out_form.name()
}

impl FlashRows {
    fn tree(&self) -> Body {
        let a = &self.0;
        let w = a.width as u32;
        let key = (w, a.k_form.name(), a.v_form.name(), a.out_form.name());
        TREES.with(|t| {
            t.borrow_mut()
                .entry(key)
                .or_insert_with(|| {
                    flash_rows_ir(w, WAYS[0].0, false, a.k_form, a.v_form, a.out_form)
                })
                .clone()
        })
    }

    fn groups(&self) -> usize {
        self.0.heads * self.0.queries.div_ceil(BQ as usize)
    }

    /// Key passes: one a segment, up to a block.
    fn passes(&self) -> usize {
        self.0.segments.min(BQ as usize)
    }
}

/// One launch. A block streams keys once per slot.
fn rows_load(k: &FlashRows, len: usize) -> Option<Load> {
    let a = &k.0;
    let (keys, side) = (a.keys.most() as f64, (a.kv_heads * a.width) as f64);
    let blocks = a.queries.div_ceil(BQ as usize) as f64;
    let passes = k.passes() as f64;
    let k_stream = a.k_form.element_bytes() * keys * side * blocks * passes;
    let v_stream = a.v_form.element_bytes() * keys * side * blocks * passes;
    Some(Load {
        threads: k.groups() * THREADS as usize,
        group: THREADS as usize,
        trips: passes * keys / WAYS[0].0 as f64,
        trip: k.tree().trip(),
        reads: vec![4.0 * len as f64, k_stream, v_stream],
        written: a.out_form.bytes(len) as f64,
        flight: 16.0,
        ..Load::default()
    })
}

fn rows_source(k: &FlashRows) -> Source {
    let body = kept::attach(k.tree(), k.0.kept);
    let suffix = form_suffix(&k.0);
    let entry = format!("flash_rows_{}_{}{suffix}", k.0.width, k.0.kept.name());
    Source::named(body, entry)
}

fn rows_launches(k: &FlashRows, step: &Step) -> Vec<Launch> {
    let a = &k.0;
    let mut args = a.rows_args(step);
    args.extend(a.kept_args());
    vec![Launch {
        args,
        shape: Shape::Grid {
            across: k.groups(),
            down: 1,
            threads: THREADS as usize,
        },
    }]
}

/// Q, the K and V slot, P, sums, seen counts.
fn rows_needs(a: &Attn) -> Needs {
    let floats = stream_floats(a.width, WAYS[0].0, false);
    Needs {
        lanes: LANES,
        group_bytes: 4 * floats,
        group_threads: THREADS as usize,
        tile: true,
        ..Needs::default()
    }
}
