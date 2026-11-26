//! `x[m, k]` times `w[n, k]` transposed, dense or picked.

use tessera_codegen::ir::{BinOp, Body, Expr, Intrinsic, Stmt, Ty};
use tessera_codegen::prelude::*;
use tessera_lower::form::Load as Stored;
use tessera_lower::kernels::{MapStage, finish};
use tessera_lower::sites::{After, Picked as Pick};
use tessera_lower::{Draft, Kernel, Launch, Shape, Step, T, Tree, ZoneKind, sites};
use tessera_lower::{GUESS, source};

/// One thread per output: its row dotted with its column.
#[kernel]
fn cpu_dense(
    x: Buf<F32>,
    #[form(form)] w: Buf<Dyn>,
    out: BufMut<F32>,
    m: u32,
    n: u32,
    k: u32,
    #[build] form: &'static dyn Stored,
) {
    if gid() >= m * n {
        return;
    }
    let mut sums = [0.0f32; 1];
    sums[0] = 0.0;
    let row = (gid() / n) as u64 * k as u64;
    let col = (gid() % n) as u64 * k as u64;
    sums[0] += x.dot(row, w, col, k);
}

/// One thread per `chunk` rows by 32 columns.
#[kernel]
fn cpu_tiled(
    x: Buf<F32>,
    #[form(form)] w: Buf<Dyn>,
    mut out: BufMut<F32>,
    m: u32,
    n: u32,
    k: u32,
    chunk: u32,
    #[build] form: &'static dyn Stored,
) {
    // Threads on one row chunk sit side by side.
    let blocks = (n + 31) / 32;
    let row = gid() / blocks * chunk;
    let col = gid() % blocks * 32;
    if row >= m {
        return;
    }
    let rows = if m - row < chunk { m - row } else { chunk };
    // The last block may hold fewer columns.
    let cols = if n - col < 32 { n - col } else { 32 };
    let at = row as u64 * n as u64 + col as u64;
    out.dots(
        at,
        n,
        x,
        row as u64 * k as u64,
        w,
        col as u64 * k as u64,
        k,
        rows,
        cols,
    );
}

/// Which matrix each row reads, and where this rank's start.
#[derive(Group)]
struct Picking {
    count: u32,
    stride: u32,
    offset: u32,
    repeat: u32,
    first: u32,
}

/// One thread per output, along the matrix its row picks.
#[kernel]
fn cpu_picked(
    x: Buf<F32>,
    #[form(form)] w: Buf<Dyn>,
    ids: Buf<U32>,
    out: BufMut<F32>,
    rows: u32,
    n: u32,
    k: u32,
    pick: Picking,
    #[build] form: &'static dyn Stored,
) {
    if gid() >= rows * n {
        return;
    }
    let row = gid() / n;
    let col = gid() % n;
    let id = ids[row as u64];
    let mut sums = [0.0f32; 1];
    sums[0] = 0.0;
    if id >= pick.first && id - pick.first < pick.count {
        let matrix = (id - pick.first) as u64 * pick.stride as u64 + pick.offset as u64;
        let line = (matrix + col as u64) * k as u64;
        sums[0] += x.dot((row / pick.repeat) as u64 * k as u64, w, line, k);
    }
}

/// The stages after the sum, into the thread's output.
fn finished(mut body: Body, after: &After, operands_at: usize) -> Body {
    tessera_lower::kernels::add_operands(&mut body, operands_at, after.others());
    let at = Expr::cast(Ty::U64, Expr::Call(Intrinsic::Gid, Vec::new()));
    let sum = Expr::index(Expr::var("sums"), Expr::Int(0));
    body.stmts.extend(finish(after, sum, at));
    body
}

/// The stages after each sum in the thread's block.
fn tiled_finish(mut body: Body, after: &After) -> Body {
    if after.stages.is_empty() {
        return body;
    }
    tessera_lower::kernels::add_operands(&mut body, 2, after.others());
    // Each of the thread's rows, then its 32 columns.
    let line = Expr::bin(BinOp::Mul, Expr::var("r"), Expr::var("n"));
    let first = Expr::bin(BinOp::Add, Expr::var("at"), Expr::cast(Ty::U64, line));
    let per_row = |inner: Vec<Stmt>| {
        let mut stmts = vec![Stmt::Let {
            name: "first".into(),
            init: first.clone(),
        }];
        stmts.extend(inner);
        Stmt::For {
            var: "r".into(),
            from: Expr::lit(0),
            to: Expr::var("rows"),
            step: Expr::lit(1),
            unroll: 0,
            body: stmts,
        }
    };
    if let Some(calls) = row_calls(after) {
        body.stmts.push(per_row(calls));
        return body;
    }
    let at = Expr::bin(
        BinOp::Add,
        Expr::var("first"),
        Expr::cast(Ty::U64, Expr::var("c")),
    );
    let sum = Expr::index(Expr::var("out"), at.clone());
    let cols = Stmt::For {
        var: "c".into(),
        from: Expr::lit(0),
        to: Expr::var("cols"),
        step: Expr::lit(1),
        unroll: 0,
        body: finish(after, sum, at),
    };
    body.stmts.push(per_row(vec![cols]));
    body
}

/// One row call per stage, where the sum can lead.
fn row_calls(after: &After) -> Option<Vec<Stmt>> {
    // Add and multiply commute: a second sum may lead.
    let leads = after.sum == 0
        || (after.sum == 1 && matches!(after.stages[0], MapStage::Add | MapStage::Mul));
    if !leads {
        return None;
    }
    let (mut calls, mut taken) = (Vec::new(), 0);
    for stage in &after.stages {
        let (row, scale) = stage.row();
        // The k-th add or multiply reads operand `ak`.
        let operand = match row.reads() {
            true => {
                taken += 1;
                format!("a{}", taken - 1)
            }
            false => "out".into(),
        };
        let args = vec![
            Expr::var("out"),
            Expr::var("first"),
            Expr::var("cols"),
            Expr::lit(row.code()),
            Expr::Float(f64::from(scale)),
            Expr::var(&operand),
            Expr::var("first"),
        ];
        calls.push(Stmt::Expr(Expr::Call(Intrinsic::Stage, args)));
    }
    Some(calls)
}

/// Blocks need f32 or bf16 weights, at any width.
fn tiles(form: &dyn Stored) -> bool {
    ["f32", "bf16"].contains(&form.name())
}

/// Multiply-adds one packed weight costs, as a guess.
const PACKING: usize = 8;

/// Multiply-adds, and each row chunk's packing of the weights.
fn tiled_work(d: &Dense) -> usize {
    d.m * d.n * d.k + PACKING * d.n * d.k * d.m.div_ceil(64)
}

/// Weights read as stored. Rows and operands come as f32.
fn accepts(input: usize, form: &dyn Stored) -> bool {
    input == 1 || !form.narrow()
}

/// A matmul and its finishing stages.
#[derive(Clone, Debug, Kernel)]
#[launch(
    crate = tessera_lower,
    name = "cpu_dense",
    sizes = [self.m, self.n, self.k],
    work = self.m * self.n * self.k,
    guess = GUESS,
    runs_on = ZoneKind::Cpu,
    accepts = accepts(input, form),
    source = source("cpu_dense", finished(cpu_dense_ir(self.form), &self.after, 2)),
    launches = dense_launches,
)]
pub struct Dense {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub form: &'static dyn Stored,
    pub after: After,
}

fn dense_launches(k: &Dense, step: &Step) -> Vec<Launch> {
    vec![Launch::plain(
        step,
        &[k.m, k.n, k.k],
        Shape::Line(k.m * k.n),
    )]
}

/// A matmul in square blocks, and its stages.
#[derive(Clone, Debug, Kernel)]
#[launch(
    crate = tessera_lower,
    name = "cpu_tiled",
    sizes = [self.0.m, self.0.n, self.0.k],
    work = tiled_work(&self.0),
    guess = GUESS / 4.0,
    runs_on = ZoneKind::Cpu,
    accepts = accepts(input, form),
    source = source("cpu_tiled", tiled_finish(cpu_tiled_ir(self.0.form), &self.0.after)),
    launches = tiled_launches,
)]
pub struct Tiled(pub Dense);

fn tiled_launches(k: &Tiled, step: &Step) -> Vec<Launch> {
    let d = &k.0;
    // Even chunks of at most 64 rows share each packing.
    let chunks = d.m.div_ceil(64);
    let chunk = d.m.div_ceil(chunks);
    // One thread a group: workers split by group.
    let shape = Shape::Grid {
        across: chunks * d.n.div_ceil(32),
        down: 1,
        threads: 1,
    };
    vec![Launch::plain(step, &[d.m, d.n, d.k, chunk], shape)]
}

/// Each row against its id's matrix, then the stages.
#[derive(Clone, Debug, Kernel)]
#[launch(
    crate = tessera_lower,
    name = "cpu_picked",
    sizes = [self.m, self.n, self.k],
    work = self.m * self.n * self.k,
    guess = GUESS,
    runs_on = ZoneKind::Cpu,
    accepts = accepts(input, form),
    source = source("cpu_picked", finished(cpu_picked_ir(self.form), &self.after, 3)),
    launches = picked_launches,
)]
pub struct Picked {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub form: &'static dyn Stored,
    pub after: After,
    pub pick: Pick,
}

fn picked_launches(k: &Picked, step: &Step) -> Vec<Launch> {
    let p = k.pick;
    let counts = [
        k.m,
        k.n,
        k.k,
        p.count,
        p.stride,
        p.offset,
        p.repeat,
        p.first.unwrap_or(0),
    ];
    vec![Launch::plain(step, &counts, Shape::Line(k.m * k.n))]
}

/// Any finished matmul, dense or picked.
fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let site: sites::Finished = tree.site(at)?;
    let sites::Matmul {
        m,
        n,
        k,
        form,
        picked,
        ..
    } = site.product;
    let after = site.after.clone();
    let kernel = match picked {
        None => Draft::new(
            Dense {
                m,
                n,
                k,
                form,
                after,
            },
            site.reads(),
            m * n,
        ),
        Some(pick) => Draft::new(
            Picked {
                m,
                n,
                k,
                form,
                after,
                pick,
            },
            site.reads(),
            m * n,
        ),
    };
    Some(vec![kernel])
}

/// Packed blocks, where the shape allows. Cost decides the rest.
fn tiled(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let site: sites::Finished = tree.site(at)?;
    let sites::Matmul {
        m,
        n,
        k,
        form,
        picked,
        ..
    } = site.product;
    if picked.is_some() || !tiles(form) {
        return None;
    }
    let after = site.after.clone();
    let dense = Dense {
        m,
        n,
        k,
        form,
        after,
    };
    Some(vec![Draft::new(Tiled(dense), site.reads(), m * n)])
}

#[cfg(test)]
pub(crate) const TILED_RULE: &str = "cpu_tiled";

tessera_lower::rule!("cpu_dense", propose);
tessera_lower::rule!("cpu_tiled", tiled);
