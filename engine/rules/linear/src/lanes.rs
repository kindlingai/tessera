//! A matmul for few rows, a SIMD group per column.

use crate::product::{Product, Variant};
use tessera_codegen::ir::{Body, Expr};
use tessera_codegen::prelude::*;
use tessera_graph::Order;
use tessera_lower::form::Load as Stored;
use tessera_lower::{Draft, Kernel, Launch, Load, Needs, Shape, Step, T, Tree};

/// Elements one wide load brings a lane.
const WIDE: u32 = 4;

/// Trips unrolled, so a lane keeps several loads in flight.
const AHEAD: u32 = 4;

/// The most rows this schedule is offered for.
const ROWS: usize = 32;

/// Lanes sum parts of every row, then the group reduces.
#[kernel]
fn linear<const WIDE: u32, const AHEAD: u32>(
    #[form(x_form)] x: Buf<Dyn>,
    #[form(form)] w: Buf<Dyn>,
    #[form(out_form)] out: BufMut<Dyn>,
    n: u32,
    k: u32,
    #[build] m: usize,
    #[build] x_form: &'static dyn Stored,
    #[build] form: &'static dyn Stored,
    #[build] out_form: &'static dyn Stored,
    #[build] gap: u32,
    #[build] per: u32,
    #[build] span: u32,
) {
    let col = gid() / LANES as u32;
    if col >= n {
        return;
    }
    let at = col as u64 * k as u64;
    let mut s = [0.0f32; ROWS];
    for r in 0..m as u32 {
        s[r as usize] = 0.0;
    }
    // `per` lanes share a `span`, their loads `gap` apart.
    for i in ((lane() / per) * span + (lane() % per) * WIDE..k)
        .by(LANES as u32 / per * span)
        .unroll(AHEAD)
    {
        for h in 0..span / (WIDE * per) {
            let o = i + h * gap;
            let c = w.load4(at + o as u64);
            for r in 0..m as u32 {
                let v = x.load4(r as u64 * k as u64 + o as u64);
                s[r as usize] += v.x * c.x;
                s[r as usize] += v.y * c.y;
                s[r as usize] += v.z * c.z;
                s[r as usize] += v.w * c.w;
            }
        }
    }
    for r in 0..m as u32 {
        s[r as usize] = sum32(s[r as usize]);
    }
    if lane() != 0 {
        return;
    }
}

/// `x[m, k]` times `w[n, k]` transposed, per SIMD group.
#[derive(Clone, Debug, Kernel)]
#[launch(
    name = self.product.form.suffixed("lanes"),
    sizes = self.product.sizes(),
    work = self.product.work(),
    accepts = self.product.accepts(input, form),
    emits = self.product.emits(form),
    needs = Needs::lanes(),
    // The lanes sum in one fixed tree on every device.
    order = Order::PerConfig,
    source = self.product.source(self.tree(), "linear"),
    launches = lanes_launches,
    shape = Shape::Line(self.product.n * LANES),
    load = lanes_load,
    unproven,
)]
pub(crate) struct Lanes {
    pub product: Product,
}

impl Lanes {
    /// The kernel with the chain's operands and its stores added.
    fn tree(&self) -> Body {
        let Product {
            m,
            x_form,
            form,
            out_form,
            ..
        } = self.product;
        let gap = form.pairs() as u32;
        let (per, span) = ((gap / WIDE).max(1), (2 * gap).max(WIDE));
        let build = || match (gap, form.heavy()) {
            (0, true) => linear_ir::<WIDE, 1>(m, x_form, form, out_form, gap, per, span),
            (0, false) => linear_ir::<WIDE, AHEAD>(m, x_form, form, out_form, gap, per, span),
            _ => linear_ir::<WIDE, 2>(m, x_form, form, out_form, gap, per, span),
        };
        let key: Variant = ("lanes", m, x_form.name(), form.name(), out_form.name(), 0);
        let mut body = self.product.tree(key, build);
        body.stmts
            .push(self.product.rows("r", "s", Expr::var("col")));
        body
    }
}

fn lanes_load(k: &Lanes, _: usize) -> Option<Load> {
    let Product {
        m,
        n,
        k: depth,
        form,
        out_form,
        ..
    } = k.product;
    let loads = 1 + usize::from(form.pairs() > 0);
    let bytes = form.bytes(WIDE as usize) * loads;
    Some(Load {
        threads: n * LANES,
        group: 256,
        trips: (depth / (LANES * WIDE as usize * loads)) as f64,
        trip: k.tree().trip(),
        reads: k.product.read_bytes(),
        written: out_form.bytes(m * n) as f64,
        flight: (bytes * AHEAD as usize) as f64,
        ..Load::default()
    })
}

fn lanes_launches(k: &Lanes, step: &Step) -> Vec<Launch> {
    let Product { n, k: depth, .. } = k.product;
    k.product
        .launch(step, &[n, depth], k.shape().expect("a line"))
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let (product, reads) = Product::find(tree, at)?;
    // Wide loads need whole groups of four, pairs whole spans.
    let span = (2 * product.form.pairs()).max(WIDE as usize);
    if product.picked.is_some() || product.m > ROWS || !product.k.is_multiple_of(span) {
        return None;
    }
    let len = product.outputs();
    Some(vec![Draft::new(Lanes { product }, reads, len)])
}

tessera_lower::rule!("lanes", propose);
