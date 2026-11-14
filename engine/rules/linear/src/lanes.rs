//! A matmul for few rows, a SIMD group per column.

use codegen::hash;
use codegen::ir::{BinOp, Body, Expr, Stmt, Ty};
use codegen::prelude::*;
use graph::Order;
use lower::kernels::{add_operands, finish};
use lower::sites;
use lower::sites::After;
use lower::{
    Arg, Draft, Form as Stored, Kernel, Launch, Load, Needs, Shape, Source, Step, T, Tree,
};
use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    /// The kernel per row count and form, built once.
    static TREES: RefCell<HashMap<(usize, &'static str), Body>> = RefCell::new(HashMap::new());
}

/// Elements one wide load brings a lane.
const WIDE: u32 = 4;

/// Trips unrolled, so a lane keeps several loads in flight.
const AHEAD: u32 = 4;

/// The most rows this schedule is offered for.
pub(crate) const ROWS: usize = 32;

/// Lanes sum parts of every row, then the group reduces.
#[kernel]
fn linear<const WIDE: u32, const AHEAD: u32>(
    x: Buf<F32>,
    #[form(form)] w: Buf<Dyn>,
    out: BufMut<F32>,
    n: u32,
    k: u32,
    #[build] m: usize,
    #[build] form: &'static dyn Stored,
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
    for i in (lane() * WIDE..k).by(LANES as u32 * WIDE).unroll(AHEAD) {
        let c = w.load4(at + i as u64);
        for r in 0..m as u32 {
            let v = x.load4(r as u64 * k as u64 + i as u64);
            s[r as usize] += v.x * c.x;
            s[r as usize] += v.y * c.y;
            s[r as usize] += v.z * c.z;
            s[r as usize] += v.w * c.w;
        }
    }
    for r in 0..m as u32 {
        s[r as usize] = sum32(s[r as usize]);
    }
    if lane() != 0 {
        return;
    }
}

macro_rules! block_kernel {
    ($stored:ident, $step:literal, $ahead:literal, $rule:literal; #[$attr:meta] fn $name:ident($x:ident, $w:ident, $k:ident, $col:ident, $s:ident, $m:ident, $form:ident) $body:block) => {
        use codegen::prelude::*;
        use crate::lanes::{accumulate, accumulate_inline};
        #[$attr]
        fn $name(
            $x: Buf<F32>,
            #[form($form)] $w: Buf<Dyn>,
            $k: u32,
            $col: u32,
            $s: &mut [f32; crate::lanes::ROWS],
            #[build] $m: usize,
            #[build] $form: &'static dyn lower::Form,
        ) $body

        fn propose(tree: &lower::Tree, at: lower::T) -> Option<Vec<lower::Draft>> {
            crate::lanes::Native {
                form: &codegen::block::$stored,
                build: linear_ir,
                step: $step,
                ahead: $ahead,
            }.offer(tree, at)
        }
        lower::rule!($rule, propose);
    };
}

macro_rules! f16_kernel {
    ($stored:ident, $bytes:tt, $step:tt, $ahead:tt, $lanes:tt, $rule:literal;
     #[$attr:meta] fn $name:ident($x:ident, $w:ident, $k:ident, $col:ident, $s:ident, $m:ident, $form:ident)
     for $i:ident in [$($range:tt)*] => ($sub:ident, $sc:ident, $p:ident) { $($body:tt)* }) => {
        crate::lanes::block_kernel! {
            $stored, $step, $ahead, $rule;
            #[$attr]
            fn $name($x, $w, $k, $col, $s, $m, $form) {
                let $sub = lane() & ($lanes - 1);
                for $i in ($($range)*).by($step).unroll($ahead) {
                    let base = ($col as u64 * ($k / 32) as u64 + ($i / 32) as u64) * $bytes;
                    let $sc = f16_to_f32($w.u16_le(base));
                    let $p = $w.pack4_aligned(base + 2 + ($sub * 4) as u64);
                    $($body)*
                }
            }
        }
    };
}

pub(crate) use f16_kernel;

pub(crate) use block_kernel;

#[kernel]
pub(crate) fn accumulate(
    x: Buf<F32>,
    s: &mut [f32; ROWS],
    k: u32,
    at: u32,
    c0: f32,
    c1: f32,
    c2: f32,
    c3: f32,
    #[build] m: usize,
) {
    for r in 0..m as u32 {
        let v = x.load4(r as u64 * k as u64 + at as u64);
        s[r as usize] += v.x * c0;
        s[r as usize] += v.y * c1;
        s[r as usize] += v.z * c2;
        s[r as usize] += v.w * c3;
    }
}

#[kernel]
fn frame(
    x: Buf<F32>,
    #[form(form)] w: Buf<Dyn>,
    out: BufMut<F32>,
    n: u32,
    k: u32,
    #[build] m: usize,
    #[build] form: &'static dyn Stored,
) {
    let col = gid() / LANES as u32;
    if col >= n {
        return;
    }
    let mut s = [0.0f32; ROWS];
    for r in 0..m as u32 {
        s[r as usize] = 0.0;
    }
}

#[kernel]
fn reduce(s: &mut [f32; ROWS], #[build] m: usize) {
    for r in 0..m as u32 {
        s[r as usize] = sum32(s[r as usize]);
    }
    if lane() != 0 {
        return;
    }
}

/// `x[m, k]` times `w[n, k]` transposed, per SIMD group.
#[derive(Clone, Debug)]
pub struct Lanes {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub form: &'static dyn Stored,
    pub after: After,
    native: Option<Native>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Native {
    pub form: &'static dyn Stored,
    pub build: fn(usize, &'static dyn Stored) -> Body,
    pub step: usize,
    pub ahead: usize,
}

impl Native {
    pub fn offer(self, tree: &Tree, at: T) -> Option<Vec<Draft>> {
        offer(tree, at, Some(self))
    }
}

impl Lanes {
    /// The kernel with the chain's operands and its stores added.
    fn tree(&self) -> Body {
        let build = || match self.native {
            Some(native) => {
                let mut body = frame_ir(self.m, self.form);
                body.stmts.extend((native.build)(self.m, self.form).stmts);
                body.stmts.extend(reduce_ir(self.m).stmts);
                body
            }
            None => linear_ir::<WIDE, AHEAD>(self.m, self.form),
        };
        let key = (self.m, self.name());
        let mut body = TREES.with(|t| t.borrow_mut().entry(key).or_insert_with(build).clone());
        // The operands sit between the weights and the output.
        add_operands(&mut body, 2, self.after.others());
        body.stmts.push(self.stores());
        body
    }

    /// Each row's sum through the stages, into its place.
    fn stores(&self) -> Stmt {
        let u64 = |e| Expr::cast(Ty::U64, e);
        let o = Expr::bin(BinOp::Mul, u64(Expr::var("r")), u64(Expr::var("n")));
        let o = Expr::bin(BinOp::Add, o, u64(Expr::var("col")));
        let sum = Expr::index(Expr::var("s"), Expr::var("r"));
        Stmt::For {
            var: "r".into(),
            from: Expr::Int(0),
            to: Expr::Int(self.m as i64),
            step: Expr::Int(1),
            unroll: 0,
            body: finish(&self.after, sum, o),
        }
    }
}

impl Kernel for Lanes {
    fn name(&self) -> &'static str {
        let prefix = if self.native.is_none() && self.form.block() > 1 {
            "lanes_generic"
        } else {
            "lanes"
        };
        self.form.suffixed(prefix)
    }

    fn sizes(&self) -> Vec<usize> {
        vec![self.m, self.n, self.k]
    }

    /// The lanes sum in one fixed tree on every device.
    fn order(&self) -> Order {
        Order::PerConfig
    }

    fn load(&self, _: usize) -> Option<Load> {
        let (m, n, k) = (self.m, self.n, self.k);
        let (step, flight) = match self.native {
            Some(native) => (native.step, 4 * native.ahead),
            None => (
                LANES * WIDE as usize,
                self.form.bytes(WIDE as usize) * AHEAD as usize,
            ),
        };
        Some(Load {
            threads: n * LANES,
            group: 256,
            trips: (k as f64 / step as f64).ceil(),
            trip: self.tree().trip(),
            reads: self.form.matmul_bytes(m, n, k),
            written: (4 * m * n) as f64,
            flight: flight as f64,
        })
    }

    fn source(&self) -> Source {
        let mut body = self.tree();
        body.entry = format!("linear_{}_{:x}", self.form.name(), hash(&body.text().body));
        Source::Made(body)
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        let mut args: Vec<Arg> = (0..step.reads.len()).map(|i| step.read(i)).collect();
        args.push(step.out());
        args.extend([self.n, self.k].map(Arg::count));
        vec![Launch {
            args,
            shape: Shape::Line(self.n * LANES),
        }]
    }

    fn work(&self, _: usize) -> f64 {
        (self.m * self.n * self.k) as f64
    }

    /// Unproven, so it must earn its place by measurement.
    fn guess(&self) -> f64 {
        if self.native.is_some() { 5e-10 } else { 1e-9 }
    }

    fn accepts(&self, input: usize, form: &dyn Stored) -> bool {
        self.form.taken(input, 1, form)
    }

    fn needs(&self) -> Needs {
        Needs {
            lanes: LANES,
            group_threads: 256,
            ..Needs::default()
        }
    }
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    offer(tree, at, None)
}

fn offer(tree: &Tree, at: T, native: Option<Native>) -> Option<Vec<Draft>> {
    let site: sites::Finished = tree.site(at)?;
    let sites::Matmul { m, n, k, form, .. } = site.product;
    // Wide loads need whole groups of four along `k`.
    if m > ROWS || !k.is_multiple_of(WIDE as usize) {
        return None;
    }
    // Whole blocks suffice: lanes past `k` skip the tail.
    if let Some(native) = native
        && (form.name() != native.form.name() || !k.is_multiple_of(native.form.block()))
    {
        return None;
    }
    let lanes = Lanes {
        m,
        n,
        k,
        form: native.map_or(form, |n| n.form),
        after: site.after.clone(),
        native,
    };
    Some(vec![Draft::new(lanes, site.reads(), m * n)])
}

lower::rule!("lanes", propose);
