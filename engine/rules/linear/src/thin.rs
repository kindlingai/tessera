//! A matmul with one thread per output column.

use crate::product::{Product, Variant};
use codegen::ir::{Body, Expr, Intrinsic};
use codegen::prelude::*;
use graph::Order;
use lower::{Draft, Form as Stored, Kernel, Launch, Load, Shape, Source, Step, T, Tree, UNPROVEN};

/// Its row limit: each thread keeps one sum per row.
pub const THIN_ROWS: usize = 32;

/// The same product as `Matmul`, for few rows.
#[derive(Clone, Debug)]
pub struct Thin {
    pub product: Product,
}

/// One thread per column with one sum per row.
#[kernel]
fn thin(
    x: Buf<F32>,
    #[form(form)] w: Buf<Dyn>,
    mut out: BufMut<F32>,
    n: u32,
    k: u32,
    #[build] m: usize,
    #[build] form: &'static dyn Stored,
) {
    if gid() >= n {
        return;
    }
    let mut sums = [0.0f32; THIN_ROWS];
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
        let Product { m, form, .. } = self.product;
        let key: Variant = ("thin", m, form.name(), false);
        let mut body = self.product.tree(key, || thin_ir(m, form));
        let gid = Expr::Call(Intrinsic::Gid, Vec::new());
        body.stmts.push(self.product.rows("row", "sums", gid));
        body
    }
}

impl Kernel for Thin {
    fn name(&self) -> &'static str {
        self.product.form.suffixed("thin")
    }

    over_product!();

    /// One thread per column loops over all of `k`.
    fn load(&self, _: usize) -> Option<Load> {
        let Product { m, n, k, form, .. } = self.product;
        let trip = self.tree().trip();
        Some(Load {
            threads: n,
            group: 256,
            trips: k as f64,
            trip,
            reads: form.matmul_bytes(m, n, k),
            written: 4.0 * (m * n) as f64,
            flight: 64.0,
        })
    }

    fn order(&self) -> Order {
        Order::PerConfig
    }

    fn source(&self) -> Source {
        // The text holds the stages. The name hashes the text.
        self.product.source(self.tree(), "thin")
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        let Product { n, k, .. } = self.product;
        self.product.launch(step, &[n, k], Shape::Line(n))
    }

    fn guess(&self) -> f64 {
        UNPROVEN
    }
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let (product, reads) = Product::find(tree, at)?;
    if product.m > THIN_ROWS {
        return None;
    }
    let len = product.outputs();
    Some(vec![Draft::new(Thin { product }, reads, len)])
}

lower::rule!("thin", propose);
