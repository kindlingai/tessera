//! A matmul with one thread per output column.

use crate::product::{Product, Variant};
use tessera_codegen::ir::{Body, Expr, Intrinsic};
use tessera_codegen::prelude::*;
use tessera_graph::Order;
use tessera_lower::form::Load as Stored;
use tessera_lower::{Draft, Kernel, Launch, Load, Shape, Step, T, Tree, knob};

/// Its row limit: each thread keeps one sum per row.
pub(crate) const THIN_ROWS: usize = 32;

knob!("thin", "max_rows", 1, 64);

/// The same product as `Matmul`, for few rows.
#[derive(Clone, Debug, Kernel)]
#[launch(
    name = self.product.form.suffixed("thin"),
    sizes = self.product.sizes(),
    work = self.product.work(),
    accepts = self.product.accepts(input, form),
    emits = self.product.emits(form),
    order = Order::PerConfig,
    // The text holds the stages. The name hashes the text.
    source = self.product.source(self.tree(), "thin"),
    launches = thin_launches,
    shape = Shape::Line(self.product.n),
    load = thin_load,
    unproven,
)]
pub(crate) struct Thin {
    pub product: Product,
    pub rows_cap: usize,
}

/// One thread per column with one sum per row.
#[kernel]
fn thin(
    #[form(x_form)] x: Buf<Dyn>,
    #[form(form)] w: Buf<Dyn>,
    #[form(out_form)] mut out: BufMut<Dyn>,
    n: u32,
    k: u32,
    #[build] m: usize,
    #[build] rows_cap: usize,
    #[build] x_form: &'static dyn Stored,
    #[build] form: &'static dyn Stored,
    #[build] out_form: &'static dyn Stored,
) {
    if gid() >= n {
        return;
    }
    let mut sums = local::<f32>(rows_cap);
    for row in 0..m as u32 {
        sums[row as usize] = 0.0;
    }
    for i in 0..k {
        let c = w[gid() as u64 * k as u64 + i as u64];
        for row in 0..m as u32 {
            sums[row as usize] += x[(row * k + i) as u64] * c;
        }
    }
}

impl Thin {
    /// The kernel with the chain's operands and its stores added.
    fn tree(&self) -> Body {
        let Product {
            m,
            x_form,
            form,
            out_form,
            ..
        } = self.product;
        let rows_cap = self.rows_cap;
        let key: Variant = (
            "thin",
            m,
            x_form.name(),
            form.name(),
            out_form.name(),
            0,
            rows_cap,
        );
        let mut body = self
            .product
            .tree(key, || thin_ir(m, rows_cap, x_form, form, out_form));
        let gid = Expr::Call(Intrinsic::Gid, Vec::new());
        body.stmts.push(self.product.rows("row", "sums", gid));
        body
    }
}

/// One thread per column loops over all of `k`.
fn thin_load(k: &Thin, _: usize) -> Option<Load> {
    let Product {
        m,
        n,
        k: depth,
        out_form,
        ..
    } = k.product;
    Some(Load {
        threads: n,
        group: 256,
        trips: depth as f64,
        trip: k.tree().trip(),
        reads: k.product.read_bytes(),
        written: out_form.bytes(m * n) as f64,
        flight: 64.0,
        ..Load::default()
    })
}

fn thin_launches(k: &Thin, step: &Step) -> Vec<Launch> {
    let Product { n, k: depth, .. } = k.product;
    k.product
        .launch(step, &[n, depth], k.shape().expect("a line"))
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let (product, reads) = Product::find(tree, at)?;
    let max_rows = tree.knobs.get("thin", "max_rows").unwrap_or(THIN_ROWS);
    if product.picked.is_some() || product.m > max_rows {
        return None;
    }
    let len = product.outputs();
    let rows_cap = product.m.max(THIN_ROWS);
    Some(vec![Draft::new(Thin { product, rows_cap }, reads, len)])
}

tessera_lower::rule!("thin", propose);
