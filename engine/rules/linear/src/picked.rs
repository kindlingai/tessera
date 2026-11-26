//! A matmul whose rows each pick a matrix.

use crate::product::{Product, Variant};
use tessera_codegen::ir::{Body, Expr, Intrinsic, Ty};
use tessera_codegen::prelude::*;
use tessera_lower::form::Load as Stored;
use tessera_lower::kernels::finish;
use tessera_lower::sites::Picked as Pick;
use tessera_lower::{Arg, Draft, Kernel, Launch, Load, Shape, Step, T, Tree};

/// Each row's product with the matrix `ids` names for it.
#[derive(Clone, Debug, Kernel)]
#[launch(
    name = self.product.form.suffixed("picked"),
    sizes = self.product.sizes(),
    work = self.product.work(),
    accepts = self.product.accepts(input, form),
    emits = self.product.emits(form),
    source = self.product.source(self.tree(), "picked"),
    launches = picked_launches,
    shape = Shape::Line(self.product.m * self.product.n),
    load = picked_load,
    unproven,
)]
pub(crate) struct Picked {
    pub product: Product,
}

/// One thread per output element, along its row's matrix.
#[kernel]
fn picked(
    #[form(x_form)] x: Buf<Dyn>,
    #[form(form)] w: Buf<Dyn>,
    ids: Buf<U32>,
    #[form(out_form)] mut out: BufMut<Dyn>,
    rows: u32,
    n: u32,
    k: u32,
    count: u32,
    stride: u32,
    offset: u32,
    repeat: u32,
    first: u32,
    #[build] x_form: &'static dyn Stored,
    #[build] form: &'static dyn Stored,
    #[build] out_form: &'static dyn Stored,
) {
    if gid() >= rows * n {
        return;
    }
    let row = gid() / n;
    let col = gid() % n;
    // This rank's matrices start at `first`.
    let id = ids[row as u64] - first;
    let mut sums = [0.0f32; 1];
    sums[0] = 0.0;
    if ids[row as u64] >= first && id < count {
        let at = (id as u64 * stride as u64 + offset as u64 + col as u64) * k as u64;
        let from = (row / repeat) as u64 * k as u64;
        for i in 0..k {
            sums[0] += x[from + i as u64] * w[at + i as u64];
        }
    }
}

impl Picked {
    /// This product's pick.
    fn pick(&self) -> Pick {
        self.product.picked.expect("a picked product")
    }

    /// The kernel with the chain's operands and its store added.
    fn tree(&self) -> Body {
        let Product {
            x_form,
            form,
            out_form,
            ..
        } = self.product;
        let key: Variant = ("picked", 0, x_form.name(), form.name(), out_form.name(), 0);
        let mut body = self.product.tree(key, || picked_ir(x_form, form, out_form));
        let gid = Expr::Call(Intrinsic::Gid, Vec::new());
        let at = Expr::cast(Ty::U64, gid);
        let sum = Expr::index(Expr::var("sums"), Expr::Int(0));
        let stmts = finish(&self.product.after, sum, at);
        body.stmts.extend(stmts);
        body
    }
}

/// One thread per output element loops over all of `k`.
fn picked_load(k: &Picked, _: usize) -> Option<Load> {
    let Product {
        m,
        n,
        k: depth,
        x_form,
        form,
        out_form,
        ..
    } = k.product;
    let mut reads = vec![
        form.bytes(m * n * depth) as f64,
        x_form.bytes(m * depth) as f64,
    ];
    for other in &k.product.other_forms {
        reads.push(other.bytes(m * n) as f64);
    }
    Some(Load {
        threads: m * n,
        group: 256,
        trips: depth as f64,
        trip: k.tree().trip(),
        reads,
        written: out_form.bytes(m * n) as f64,
        flight: 64.0,
        ..Load::default()
    })
}

fn picked_launches(k: &Picked, step: &Step) -> Vec<Launch> {
    let Product { m, n, k: depth, .. } = k.product;
    let pick = k.pick();
    let counts = [
        m,
        n,
        depth,
        pick.count,
        pick.stride,
        pick.offset,
        pick.repeat,
    ];
    let mut launches = k.product.launch(step, &counts, k.shape().expect("a line"));
    launches[0].args.push(Arg::count(pick.first.unwrap_or(0)));
    launches
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let (product, reads) = Product::find(tree, at)?;
    product.picked?;
    let len = product.outputs();
    Some(vec![Draft::new(Picked { product }, reads, len)])
}

tessera_lower::rule!("picked", propose);
