//! Products with blocks of a stored matrix, read in place.

use tessera_codegen::prelude::*;
use tessera_graph::{Op, Order, View};
use tessera_lower::form::{Load as Stored, plain, writable};
use tessera_lower::{Draft, Kernel, Load, Needs, Source, T, Term, Tree, knob};

/// Elements one wide load brings per lane.
const WIDE: u32 = 4;

/// Trips unrolled so each lane keeps loads in flight.
const AHEAD: u32 = 4;

knob!("stacked", "ahead", 1, 16);

/// Kernel mode: 0 scalar, 1 warp-laned, 2 wide-x.
const MODE_SCALAR: u32 = 0;
const MODE_LANES: u32 = 1;
const MODE_WIDE: u32 = 2;

/// Products with blocks of a stored matrix.
#[kernel]
#[allow(clippy::too_many_arguments)]
fn stacked(
    #[form(x_form)] x: Buf<Dyn>,
    #[form(form)] w: Buf<Dyn>,
    #[form(out_form)] mut out: BufMut<Dyn>,
    len: u32,
    rows: u32,
    n: u32,
    k: u32,
    block: u32,
    first: u32,
    col_step: u32,
    k_step: u32,
    #[build] x_form: &'static dyn Stored,
    #[build] form: &'static dyn Stored,
    #[build] out_form: &'static dyn Stored,
    #[build] mode: u32,
    #[build] ahead: u32,
) {
    let idx = if mode == MODE_LANES {
        gid() / LANES as u32
    } else {
        gid()
    };
    if idx >= len {
        return;
    }
    let col = idx % n;
    let row = idx / n;
    let at = (row / rows) as u64 * block as u64 + first as u64 + col as u64 * col_step as u64;
    let from = row as u64 * k as u64;
    let mut sum = 0.0f32;
    if mode == MODE_LANES {
        for i in (lane() * WIDE..k).by(LANES as u32 * WIDE).unroll(ahead) {
            let c = w.load4(at + i as u64);
            let v = x.load4(from + i as u64);
            sum += v.x * c.x;
            sum += v.y * c.y;
            sum += v.z * c.z;
            sum += v.w * c.w;
        }
        sum = sum32(sum);
        if lane() == 0 {
            out[idx as u64] = sum;
        }
    } else {
        if mode == MODE_WIDE {
            let ks = k_step as u64;
            for i in (0..k).by(WIDE).unroll(ahead) {
                let v = x.load4(from + i as u64);
                let base = at + i as u64 * ks;
                sum += v.x * w[base];
                sum += v.y * w[base + ks];
                sum += v.z * w[base + 2 * ks];
                sum += v.w * w[base + 3 * ks];
            }
        } else {
            for i in 0..k {
                sum += x[from + i as u64] * w[at + i as u64 * k_step as u64];
            }
        }
        out[idx as u64] = sum;
    }
}

/// `x[b, rows, k]` times block `b` of a stored matrix.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    stacked(self.x_form, self.form, self.out_form, self.mode, self.ahead),
    name = self.form.suffixed(match self.mode {
        MODE_LANES => "stacked_lanes",
        MODE_WIDE => "stacked_wide",
        _ => "stacked",
    }),
    threads = if self.mode == MODE_LANES { self.len * LANES } else { self.len },
    work = self.len * self.k,
    accepts = stacked_accepts,
    emits = stacked_emits,
    source = stacked_source,
    needs = if self.mode == MODE_LANES { Needs::lanes() } else { Needs::default() },
    order = Order::PerConfig,
    load = stacked_load,
)]
pub(crate) struct Stacked {
    pub len: usize,
    pub rows: usize,
    pub n: usize,
    pub k: usize,
    /// Elements between blocks, and the first one read.
    pub block: usize,
    pub first: usize,
    /// Elements between columns, and along `k`.
    pub col_step: usize,
    pub k_step: usize,
    pub x_form: &'static dyn Stored,
    pub form: &'static dyn Stored,
    pub out_form: &'static dyn Stored,
    pub mode: u32,
    pub ahead: u32,
}

fn stacked_accepts(k: &Stacked, input: usize, form: &dyn Stored) -> bool {
    match input {
        0 => form.name() == k.x_form.name(),
        1 => form.name() == k.form.name(),
        _ => false,
    }
}

fn stacked_emits(k: &Stacked, form: &dyn Stored) -> bool {
    form.name() == k.out_form.name()
}

fn stacked_source(k: &Stacked) -> Source {
    let base = match k.mode {
        MODE_LANES => "stacked_lanes",
        MODE_WIDE => "stacked_wide",
        _ => "stacked",
    };
    let name = match (k.x_form.name(), k.out_form.name()) {
        ("f32", "f32") => format!("plan_{base}_{}", k.form.name()),
        (a, b) => format!("plan_{base}_{}_{a}_{b}", k.form.name()),
    };
    Source::named(
        stacked_ir(k.x_form, k.form, k.out_form, k.mode, k.ahead),
        name,
    )
}

fn stacked_load(k: &Stacked, _: usize) -> Option<Load> {
    let (threads, step, flight) = match k.mode {
        MODE_LANES => (k.len * LANES, LANES * WIDE as usize, k.ahead as usize),
        MODE_WIDE => (k.len, WIDE as usize, k.ahead as usize),
        _ => (k.len, 1, 1),
    };
    let width = if k.mode == MODE_SCALAR {
        1
    } else {
        WIDE as usize
    };
    let products = k.len / k.n;
    let x_elem = k.x_form.bytes(1);
    let y_elem = k.out_form.bytes(1);
    let mut trip = stacked_ir(k.x_form, k.form, k.out_form, k.mode, k.ahead).trip();
    if k.mode != MODE_LANES {
        // Threads on columns of one row share its x.
        let shared = (x_elem * width) as f64 * (1.0 - 1.0 / k.n.min(LANES) as f64);
        trip.width -= shared / trip.loads;
    }
    Some(Load {
        threads,
        group: 256,
        trips: k.k as f64 / step as f64,
        trip,
        reads: vec![
            (x_elem * products * k.k) as f64,
            k.form.bytes(products / k.rows * k.n * k.k) as f64,
        ],
        written: (y_elem * k.len) as f64,
        flight: ((x_elem + k.form.bytes(1)) * width * flight) as f64,
        ..Load::default()
    })
}

/// A stored `[b*r, c]` matrix as `[b, r, c]` blocks.
struct Blocks {
    leaf: T,
    /// Rows per block, the first row read, the row width.
    rows: usize,
    first: usize,
    width: usize,
    swapped: bool,
}

fn blocks(tree: &Tree, mut w: T) -> Option<Blocks> {
    let term = |t: T| &tree.node(t).term;
    let swapped = matches!(term(w), Term::Op(Op::View(View::Permute(p))) if p[..] == [0, 2, 1]);
    if swapped {
        w = tree.node(w).args[0];
    }
    let mut first = 0;
    if let Term::Op(Op::View(View::Slice { axis: 1, start, .. })) = term(w) {
        first = start.value(&tree.g.dims, &|n| tree.size(n))? as usize;
        w = tree.node(w).args[0];
    }
    let Term::Op(Op::View(View::Reshape(_))) = term(w) else {
        return None;
    };
    let leaf = tree.node(w).args[0];
    let (&[_, rows, width], &[_, stored]) = (&tree.node(w).shape[..], &tree.node(leaf).shape[..])
    else {
        return None;
    };
    (tree.leaf(leaf) && stored == width).then_some(Blocks {
        leaf,
        rows,
        first,
        width,
        swapped,
    })
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    if node.term != Term::Op(Op::Matmul) {
        return None;
    }
    let (x, w) = (node.args[0], node.args[1]);
    let b = blocks(tree, w)?;
    let (&[batches, n, k], xs) = (&tree.node(w).shape[..], &tree.node(x).shape[..]) else {
        return None;
    };
    if xs.len() != 3 || xs[0] != batches || xs[2] != k {
        return None;
    }
    let (col_step, k_step) = match b.swapped {
        true => (1, b.width),
        false => (b.width, 1),
    };
    let (x_form, out_form) = (plain(tree.node(x).form), writable(node.form));
    let form = tree.node(b.leaf).form;
    let mode = if !b.swapped && k.is_multiple_of(WIDE as usize) && form.pairs() == 0 {
        MODE_LANES
    } else if b.swapped && k.is_multiple_of(WIDE as usize) {
        MODE_WIDE
    } else {
        MODE_SCALAR
    };
    let ahead = tree
        .knobs
        .get("stacked", "ahead")
        .map_or(AHEAD, |v| v as u32);
    let kernel = Stacked {
        len: tree.elements(at),
        rows: xs[1],
        n,
        k,
        block: b.rows * b.width,
        first: b.first * b.width,
        col_step,
        k_step,
        x_form,
        form,
        out_form,
        mode,
        ahead,
    };
    Some(vec![Draft::new(kernel, [x, b.leaf], kernel.len)])
}

tessera_lower::rule!("stacked", propose);
