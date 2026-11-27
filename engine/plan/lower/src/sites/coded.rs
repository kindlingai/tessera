//! A picked matmul on coded matrices, between optional rotations.

use super::Rotated;
use crate::form::Load as Stored;
use crate::tree::{T, Term, Tree};
use crate::{Arg, Load, Step};
use tessera_graph::{Act, Op, Select, Stage};

/// Each output row multiplies by the coded matrix it picks.
#[derive(Clone, Copy, Debug)]
pub struct Coded {
    pub rows: usize,
    pub n: usize,
    pub k: usize,
    /// Matrices in the set.
    pub count: usize,
    /// Output rows that share one `x` row.
    pub repeat: usize,
    /// This rank's first matrix id.
    pub first: usize,
    /// Whether per-matrix scales and rotations wrap the product.
    pub turned: bool,
    /// `1` scales by a vector, `2` by gated rows.
    pub post: u32,
    pub form: &'static dyn Stored,
    /// Whether `x` is stored as bf16.
    pub x_bf16: bool,
    /// Whether the output is stored as bf16.
    pub y_bf16: bool,
    /// Whether the `post` operand is stored as bf16.
    pub o_bf16: bool,
}

fn f32_or_bf16(tree: &Tree, t: T) -> Option<bool> {
    match tree.node(t).form.name() {
        "f32" => Some(false),
        "bf16" => Some(true),
        _ => None,
    }
}

fn elem(bf16: bool) -> usize {
    if bf16 { 2 } else { 4 }
}

fn named(bf16: bool, form: &dyn Stored) -> bool {
    form.name() == if bf16 { "bf16" } else { "f32" }
}

fn gather_of(tree: &Tree, t: T) -> Option<(T, T)> {
    let node = tree.node(t);
    if node.term != Term::Op(Op::Select(Select::Gather)) || tree.root(t) {
        return None;
    }
    Some((node.args[0], node.args[1]))
}

impl Coded {
    fn pick(tree: &Tree, [x, pick]: [T; 2], turned: bool, block: usize) -> Option<(Self, [T; 2])> {
        let (codes, ids) = gather_of(tree, pick)?;
        let named = |t: T, name: &str| tree.node(t).form.name() == name;
        let x_bf16 = named(x, "bf16");
        if !tree.leaf(codes) || !(x_bf16 || named(x, "f32")) || !named(ids, "u32") {
            return None;
        }
        let &[count, n, k] = tree.node(codes).shape.as_slice() else {
            return None;
        };
        if !n.is_multiple_of(block) || !k.is_multiple_of(block) {
            return None;
        }
        let rows = tree.elements(ids);
        let x_rows = tree.elements(x) / k;
        if rows == 0 || x_rows == 0 || !rows.is_multiple_of(x_rows) {
            return None;
        }
        let share = tree.share(codes).filter(|s| s.axis == 0);
        let site = Self {
            rows,
            n,
            k,
            count,
            repeat: rows / x_rows,
            first: share.map_or(0, |s| s.first),
            turned,
            post: 0,
            form: tree.node(codes).form,
            x_bf16,
            y_bf16: false,
            o_bf16: false,
        };
        Some((site, [codes, ids]))
    }

    fn rotated(tree: &Tree, at: T, block: usize) -> Option<(Self, Vec<T>)> {
        f32_or_bf16(tree, at)?;
        let r = tree.site::<Rotated>(at)?;
        if r.block as usize != block {
            return None;
        }
        let (su, su_ids) = gather_of(tree, r.suh)?;
        let (sv, sv_ids) = gather_of(tree, r.svh)?;
        let (site, [codes, ids]) = Self::pick(tree, [r.x, r.w], true, block)?;
        let scale = |t: T, width: usize| {
            let node = tree.node(t);
            tree.leaf(t) && node.shape == [site.count, width] && node.form.name() == "f32"
        };
        let ok = su_ids == ids
            && sv_ids == ids
            && site.first == 0
            && scale(su, site.k)
            && scale(sv, site.n);
        ok.then(|| (site, vec![r.x, codes, ids, su, sv]))
    }

    /// The site at `at` and its reads, rotating `block` channels.
    pub fn find(tree: &Tree, at: T, block: usize) -> Option<(Self, Vec<T>)> {
        let y_bf16 = f32_or_bf16(tree, at)?;
        let (mut site, reads) = Self::find_any(tree, at, block)?;
        site.y_bf16 = y_bf16;
        Some((site, reads))
    }

    fn find_any(tree: &Tree, at: T, block: usize) -> Option<(Self, Vec<T>)> {
        let node = tree.node(at);
        if node.term == Term::Op(Op::Matmul) {
            let (site, [codes, ids]) =
                Self::pick(tree, [node.args[0], node.args[1]], false, block)?;
            return Some((site, vec![node.args[0], codes, ids]));
        }
        if node.term != Term::Stage(Stage::Mul) {
            return None;
        }
        if let Some(found) = Self::rotated(tree, at, block) {
            return Some(found);
        }
        let (a, b) = (node.args[0], *node.args.get(1)?);
        let silu = Term::Stage(Stage::Act(Act::Silu));
        for (cand, other) in [(a, b), (b, a)] {
            if cand == other || tree.root(cand) {
                continue;
            }
            let Some(o_bf16) = f32_or_bf16(tree, other) else {
                continue;
            };
            // A gated product scales by whole rows.
            let gated = tree.inner(cand, &silu).filter(|i| !tree.root(i[0]));
            let (inner, post) = match gated {
                Some(&[inner]) => (inner, 2),
                _ => (cand, 1),
            };
            let Some((mut site, mut reads)) = Self::rotated(tree, inner, block) else {
                continue;
            };
            let want = if post == 2 {
                site.rows * site.n
            } else {
                site.n
            };
            if tree.elements(other) == want {
                site.post = post;
                site.o_bf16 = o_bf16;
                reads.push(other);
                return Some((site, reads));
            }
        }
        None
    }

    /// Whether `input` may be stored as `form`.
    pub fn accepts(&self, input: usize, form: &dyn Stored) -> bool {
        match input {
            0 => named(self.x_bf16, form),
            1 => form.name() == self.form.name(),
            2 => form.name() == "u32",
            5 => named(self.o_bf16, form),
            _ => form.name() == "f32",
        }
    }

    /// Whether the output may be stored as `form`.
    pub fn emits(&self, form: &dyn Stored) -> bool {
        named(self.y_bf16, form)
    }

    /// Whether any activation is stored as bf16.
    pub fn any_bf16(&self) -> bool {
        self.x_bf16 || self.y_bf16 || self.o_bf16
    }

    /// The launch arguments, absent reads standing in as `x`.
    pub fn args(&self, step: &Step, bits: u32, book: u32) -> Vec<Arg> {
        let read = |on: bool, i: usize| step.read(if on { i } else { 0 });
        vec![
            step.read(0),
            step.read(1),
            step.read(2),
            read(self.turned, 3),
            read(self.turned, 4),
            read(self.post != 0, 5),
            step.out(),
            Arg::count(self.rows),
            Arg::count(self.n),
            Arg::count(self.k),
            Arg::count(self.count),
            Arg::count(self.repeat),
            Arg::count(self.first),
            Arg::value(&bits),
            Arg::value(&book),
            Arg::count(usize::from(self.turned)),
            Arg::value(&self.post),
        ]
    }

    /// The work of one group of `threads` per output block.
    pub fn load(&self, block: usize, threads: usize) -> Load {
        let out = self.rows * self.n;
        let mut reads = vec![
            self.form.bytes(out * self.k) as f64,
            (elem(self.x_bf16) * (self.rows / self.repeat) * self.k) as f64,
            (4 * self.rows) as f64,
        ];
        if self.turned {
            reads.extend([(4 * self.rows * self.k) as f64, (4 * out) as f64]);
        }
        match self.post {
            1 => reads.push((elem(self.o_bf16) * self.n) as f64),
            2 => reads.push((elem(self.o_bf16) * out) as f64),
            _ => {}
        }
        let mut load = Load {
            threads: out / block * threads,
            group: threads,
            trips: (self.k / block) as f64,
            reads,
            written: (elem(self.y_bf16) * out) as f64,
            flight: 1024.0,
            ..Load::default()
        };
        load.trip.fresh = 1.0;
        load
    }
}
