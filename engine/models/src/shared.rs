//! Pieces every token model's program uses.

use crate::keys::{At, Keys};
use std::cell::RefCell;
use tessera_trace::DimName::*;
use tessera_trace::hf::{FFN, HEAD_DIM, HEADS, HIDDEN, LAYERS, VOCAB};
use tessera_trace::{
    Act, CalibTap, Dim, DimExpr, DimName, Embed, FEW, Graph, Linear, Linears, Model, Pairs,
    PathSeg, Role, Tensor, Vector, Weights,
};

/// A head's width outside its rotary part.
pub(crate) const NOPE: DimName = DimName::named("Nope");
/// A head's rotary width.
pub(crate) const ROPE: DimName = DimName::named("Rope");

/// A matrix a row multiplies into, however it is stored.
pub trait Dense: Weights {
    fn apply(&self, x: &Tensor) -> Tensor;
}

impl Dense for Linear {
    fn apply(&self, x: &Tensor) -> Tensor {
        x.matmul(self)
    }
}

/// exlr8 rotates in blocks of this many channels.
pub(crate) const ROTATION: u32 = 128;

/// A matrix between two rotations.
pub struct Rotated {
    pub w: Linear,
    pub suh: Vector,
    pub svh: Vector,
}

impl Weights for Rotated {
    fn declare(m: &Model, path: &mut Vec<PathSeg>, shape: &[DimExpr]) -> Self {
        Self {
            w: part(m, path, "w", shape),
            suh: part(m, path, "suh", &shape[..1]),
            svh: part(m, path, "svh", &shape[1..]),
        }
    }
}

/// Concatenates `lead` axes with `tail`.
pub(crate) fn cat<const N: usize>(lead: &[DimExpr], tail: [DimExpr; N]) -> Vec<DimExpr> {
    lead.iter().cloned().chain(tail).collect()
}

/// `n` items at `name` and each index under `path`.
pub(crate) fn each_part<T>(
    path: &mut Vec<PathSeg>,
    name: &str,
    n: usize,
    mut make: impl FnMut(&mut Vec<PathSeg>) -> T,
) -> Vec<T> {
    let mut items = Vec::with_capacity(n);
    for i in 0..n {
        path.push(PathSeg::Field(name.into()));
        path.push(PathSeg::Index(i as u32));
        items.push(make(path));
        path.truncate(path.len() - 2);
    }
    items
}

/// `n` items of `T` under `path`, each declared with `shape`.
pub(crate) fn declare_parts<T: Weights>(
    m: &Model,
    path: &mut Vec<PathSeg>,
    name: &str,
    n: usize,
    shape: &[DimExpr],
) -> Vec<T> {
    each_part(path, name, n, |p| T::declare(m, p, shape))
}

/// Field `name` under `path`, declared with `shape`.
pub(crate) fn part<T: Weights>(
    m: &Model,
    path: &mut Vec<PathSeg>,
    name: &str,
    shape: &[DimExpr],
) -> T {
    path.push(PathSeg::Field(name.into()));
    let value = T::declare(m, path, shape);
    path.pop();
    value
}

impl Dense for Rotated {
    fn apply(&self, x: &Tensor) -> Tensor {
        let turned = x.mul_vector(&self.suh).hadamard(ROTATION).matmul(&self.w);
        turned.hadamard(ROTATION).mul_vector(&self.svh)
    }
}

/// Rotated matrix whose input is already Hadamard-rotated.
pub(crate) struct PreRotated(Rotated);

impl Weights for PreRotated {
    fn declare(m: &Model, path: &mut Vec<PathSeg>, shape: &[DimExpr]) -> Self {
        Self(Rotated::declare(m, path, shape))
    }
}

impl Dense for PreRotated {
    fn apply(&self, x: &Tensor) -> Tensor {
        x.mul_vector(&self.0.suh)
            .matmul(&self.0.w)
            .hadamard(ROTATION)
            .mul_vector(&self.0.svh)
    }
}

/// Which projection a head transform is applied to.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Query,
    Key,
    Value,
}

impl Side {
    /// The query's or the key's entry. Values have none.
    pub fn of<T>(self, [q, k]: [T; 2]) -> Option<T> {
        match self {
            Side::Query => Some(q),
            Side::Key => Some(k),
            Side::Value => None,
        }
    }
}

/// One attention block's head shape and output projection.
pub struct Attend<'a, D: Dense> {
    o: &'a D,
    /// Axes before the heads: `[Seqs, Context]`, or `[Rows]`.
    lead: Vec<DimExpr>,
    heads: DimExpr,
    kv_heads: DimExpr,
    /// A value's width. Keys match unless split by hand.
    head_dim: DimExpr,
    /// Positions, theta, pairing and pairs turned, if rotating.
    rope: Option<(&'a Tensor, f32, Pairs, Option<u32>)>,
    /// The leading width that rotates, and the width kept after.
    part: Option<[DimExpr; 2]>,
}

impl<'a, D: Dense> Attend<'a, D> {
    /// Projection `o` over `lead`, with `[heads, kv_heads, head_dim]`.
    pub fn new(o: &'a D, lead: Vec<DimExpr>, shape: [DimExpr; 3]) -> Self {
        let [heads, kv_heads, head_dim] = shape;
        Self {
            o,
            lead,
            heads,
            kv_heads,
            head_dim,
            rope: None,
            part: None,
        }
    }

    /// Queries and keys rotate after their `turn`.
    pub fn rotary(mut self, positions: &'a Tensor, theta: f32, pairs: Pairs) -> Self {
        self.rope = Some((positions, theta, pairs, None));
        self
    }

    /// Rotates the first `turned` pairs, when given.
    pub fn turning(mut self, turned: Option<u32>) -> Self {
        if let Some(rope) = &mut self.rope {
            rope.3 = turned;
        }
        self
    }

    /// Only each head's first `width` rotates. The `kept` rest stays.
    pub fn partial(mut self, [width, kept]: [DimExpr; 2]) -> Self {
        self.part = Some([width, kept]);
        self
    }

    /// Heads `t`, rotated as `rotary`, `turning` and `partial` set.
    pub fn rotate(&self, t: Tensor) -> Tensor {
        let Some((positions, theta, pairs, turned)) = self.rope else {
            return t;
        };
        let spin = |t: Tensor| match turned {
            Some(n) => t.rope_part(positions, theta, pairs, n),
            None => t.rope(positions, theta, pairs),
        };
        let Some([width, kept]) = &self.part else {
            return spin(t);
        };
        let last = t.ty().rank() as u32 - 1;
        let front = t.slice(last, 0u64, width.clone());
        let back = t.slice(last, width.clone(), kept.clone());
        spin(front).concat(&back)
    }

    /// `lead`, then `rest`.
    fn shape(&self, rest: impl IntoIterator<Item = DimExpr>) -> Vec<DimExpr> {
        self.lead.iter().cloned().chain(rest).collect()
    }

    /// `t` as `heads` of `width` each, over `lead`.
    pub fn split(&self, t: Tensor, heads: impl Into<DimExpr>, width: impl Into<DimExpr>) -> Tensor {
        t.reshape(self.shape([heads.into(), width.into()]))
    }

    /// `h` through `w` as `side`'s heads, `turn`ed, then rotated.
    pub fn side(
        &self,
        w: &D,
        h: &Tensor,
        side: Side,
        turn: impl Fn(Tensor, Side) -> Tensor,
    ) -> Tensor {
        let heads = match side {
            Side::Query => self.heads.clone(),
            Side::Key | Side::Value => self.kv_heads.clone(),
        };
        let t = turn(self.split(w.apply(h), heads, self.head_dim.clone()), side);
        match side {
            Side::Query | Side::Key => self.rotate(t),
            Side::Value => t,
        }
    }

    /// Shared key/value projection split into rotated keys and values.
    pub fn tied_kv(&self, k: &D, h: &Tensor, turn: impl Fn(Tensor, Side) -> Tensor) -> [Tensor; 2] {
        let raw = self.split(k.apply(h), self.kv_heads.clone(), self.head_dim.clone());
        let key = self.rotate(turn(raw.clone(), Side::Key));
        let val = turn(raw, Side::Value);
        [key, val]
    }

    /// `h` through `[q, k, v]`, each side `turn`ed, then rotated.
    pub fn heads(
        &self,
        [q, k, v]: [&D; 3],
        h: &Tensor,
        turn: impl Fn(Tensor, Side) -> Tensor,
    ) -> [Tensor; 3] {
        let sides = [(q, Side::Query), (k, Side::Key), (v, Side::Value)];
        sides.map(|(w, side)| self.side(w, h, side, &turn))
    }

    /// Heads joined back into rows before `o`.
    pub fn merge_in(&self, out: Tensor) -> Tensor {
        let width = self.heads.clone() * self.head_dim.clone();
        out.reshape(self.shape([width]))
    }

    /// Heads joined back into rows, then projected.
    pub fn merge(&self, out: Tensor) -> Tensor {
        self.o.apply(&self.merge_in(out))
    }

    /// `heads`, then `attend`, then `merge`.
    pub fn apply(
        &self,
        w: [&D; 3],
        h: &Tensor,
        turn: impl Fn(Tensor, Side) -> Tensor,
        attend: impl FnOnce([&Tensor; 3]) -> Tensor,
    ) -> Tensor {
        let [q, k, v] = self.heads(w, h, turn);
        self.merge(attend([&q, &k, &v]))
    }
}

/// The ports a step graph adds beyond its rows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Steps {
    #[default]
    Plain,
    /// A second entry scores candidate ids.
    Scored,
    /// The first entry takes masks. The second scores.
    Masked,
}

/// Sizes every decoder stack here declares.
#[derive(Clone, Debug)]
pub struct Stack {
    pub vocab: u64,
    pub hidden: u64,
    pub layers: u64,
    pub heads: u64,
    pub head_dim: u64,
    pub ffn: u64,
    /// The most sequences in one launch.
    pub max_seqs: u64,
    pub max_context: u64,
    pub steps: Steps,
    /// 1-based layers tapped during stepping.
    pub taps: Vec<u64>,
}

/// `Stack::sized`'s sizes under `at`, each required.
pub(crate) fn sizes(k: &mut Keys, at: &At) -> [u64; 6] {
    [VOCAB, HIDDEN, LAYERS, HEADS, HEAD_DIM, FFN].map(|key| k.read(at, key))
}

impl Stack {
    /// `[vocab, hidden, layers, heads, head_dim, ffn]`, then the launch bounds.
    pub fn sized(
        [vocab, hidden, layers, heads, head_dim, ffn]: [u64; 6],
        max_seqs: u64,
        max_context: u64,
    ) -> Self {
        Self {
            vocab,
            hidden,
            layers,
            heads,
            head_dim,
            ffn,
            max_seqs,
            max_context,
            steps: Steps::Plain,
            taps: Vec::new(),
        }
    }

    /// Two layers of two heads over a vocabulary of 64.
    pub fn small(hidden: u64, head_dim: u64, ffn: u64) -> Self {
        Self::sized([64, hidden, 2, 2, head_dim, ffn], 4, 16)
    }

    /// These dimensions, followed by the model's `own`.
    pub(crate) fn dims(&self, own: &[(DimName, u64)]) -> Vec<Dim> {
        let bounded = Dim::bounded;
        let fixed = |&(name, n): &(DimName, u64)| Dim::fixed(name, n);
        let shared = [
            (Vocab, self.vocab),
            (Hidden, self.hidden),
            (Layers, self.layers),
            (Heads, self.heads),
            (HeadDim, self.head_dim),
            (Ffn, self.ffn),
        ];
        let mut dims = vec![
            bounded(Seqs, self.max_seqs),
            bounded(Context, self.max_context),
        ];
        dims.extend(shared.iter().chain(own).map(fixed));
        if self.steps == Steps::Masked {
            dims.push(Dim::fixed(VocabWords, self.vocab.div_ceil(32)));
            dims.push(bounded(EditWords, 3 * self.vocab));
        }
        if self.steps != Steps::Plain {
            dims.push(bounded(Candidates, FEW));
        }
        dims
    }
}

/// Declares `W`, plus an untied output head when needed.
pub(crate) fn with_head<W: Weights>(m: &Model, tied: bool) -> (W, Option<Embed>) {
    let w = tessera_trace::declare(m);
    let mut path = vec![PathSeg::Field("head".into())];
    let shape = [Vocab.into(), Hidden.into()];
    let head = (!tied).then(|| Embed::declare(m, &mut path, &shape));
    (w, head)
}

/// `[Seqs, n, ..]` from `a` then `b`, joined on axis 1.
pub(crate) fn join(a: &Tensor, b: &Tensor) -> Tensor {
    let swap = [0, 2, 1];
    a.permute(&swap).concat(&b.permute(&swap)).permute(&swap)
}

/// `[a, width]` then `[b, width]`, joined on axis 0.
pub(crate) fn join_rows(a: &Tensor, b: &Tensor) -> Tensor {
    let swap = [1, 0];
    a.permute(&swap).concat(&b.permute(&swap)).permute(&swap)
}

/// Captures 1-based layer outputs in requested order.
pub(crate) struct TapSink<'a> {
    layers: &'a [u64],
    taken: Vec<Option<Tensor>>,
}

impl<'a> TapSink<'a> {
    pub fn new(layers: &'a [u64]) -> Self {
        Self {
            layers,
            taken: vec![None; layers.len()],
        }
    }

    pub fn record(&mut self, at: u64, x: &Tensor) {
        for (slot, &want) in self.taken.iter_mut().zip(self.layers) {
            if want == at {
                *slot = Some(x.clone());
            }
        }
    }

    pub fn join(self) -> Option<Tensor> {
        if self.layers.is_empty() {
            return None;
        }
        let mut iter = self
            .taken
            .into_iter()
            .map(|t| t.unwrap_or_else(|| panic!("taps {:?} past the layers", self.layers)));
        let first = iter.next()?;
        Some(iter.fold(first, |all, t| all.concat(&t)))
    }
}

/// `[Seqs, Hidden]` as `[Seqs, rows, Hidden]`.
pub(crate) fn spread(v: &Tensor, rows: impl Into<DimExpr>) -> Tensor {
    v.reshape(vec![Seqs.into(), 1u64.into(), Hidden.into()])
        .expand(1, rows)
}

/// `[Seqs, Context]` followed by `rest`.
pub(crate) fn rows<const N: usize>(rest: [DimExpr; N]) -> Vec<DimExpr> {
    let lead = [Seqs.into(), Context.into()];
    lead.into_iter().chain(rest).collect()
}

/// A SwiGLU MLP with gate, up, and down projections.
pub struct Swiglu<D: Dense = Linear> {
    pub gate: D,
    pub up: D,
    pub down: D,
}

impl<D: Dense> Weights for Swiglu<D> {
    fn declare(m: &Model, path: &mut Vec<PathSeg>, shape: &[DimExpr]) -> Self {
        let [inp, inner]: [DimExpr; 2] = match shape {
            [] => [Hidden.into(), Ffn.into()],
            [inp, inner] => [inp.clone(), inner.clone()],
            _ => panic!("{path:?}: Swiglu expects empty shape or [in, inner]"),
        };
        let wide = [inp.clone(), inner.clone()];
        let narrow = [inner, inp];
        Self {
            gate: part(m, path, "gate", &wide),
            up: part(m, path, "up", &wide),
            down: part(m, path, "down", &narrow),
        }
    }
}

impl<D: Dense> Swiglu<D> {
    pub(crate) fn eval(&self, h: &Tensor) -> Tensor {
        let [out, _] = self.eval_tapped(h);
        out
    }

    pub(crate) fn eval_tapped(&self, h: &Tensor) -> [Tensor; 2] {
        swiglu_tapped(h, [&self.gate, &self.up, &self.down])
    }
}

pub(crate) fn swiglu_tapped<D: Dense>(h: &Tensor, [gate, up, down]: [&D; 3]) -> [Tensor; 2] {
    let inner = gate.apply(h).act(Act::Silu).mul(&up.apply(h));
    [down.apply(&inner), inner]
}

/// Calibration taps a traced pass records, when asked to.
#[derive(Default)]
pub struct CalibTaps(Option<RefCell<Vec<(Role, Tensor)>>>);

impl CalibTaps {
    /// A recorder that keeps every tap.
    pub fn on() -> Self {
        Self(Some(RefCell::default()))
    }

    pub(crate) fn note(&self, tap: CalibTap, t: &Tensor) {
        if let Some(taps) = &self.0 {
            taps.borrow_mut().push((Role::Calib(tap), t.clone()));
        }
    }

    /// Finishes `m` with `logits` first, then each tap in order.
    pub(crate) fn finish(self, m: &Model, logits: &Tensor) -> Graph {
        let taps = self.0.map(RefCell::into_inner).unwrap_or_default();
        let mut outs = vec![(Role::Logits, logits)];
        for (role, t) in &taps {
            outs.push((*role, t));
        }
        m.finish_many(&outs)
    }
}

/// Routed experts, then the shared expert, with calibration taps.
pub(crate) fn routed_calib<E: ExpertSet>(
    m: &Model,
    experts: &E,
    h: &Tensor,
    [ids, gates]: [Tensor; 2],
    shared: impl FnOnce() -> [Tensor; 2],
    taps: &CalibTaps,
) -> Tensor {
    taps.note(CalibTap::RouterIds, &ids);
    taps.note(CalibTap::RouterGates, &gates);
    let (routed, inner) = experts.apply_calib(m, h, [&ids, &gates]);
    if let Some(inner) = inner {
        taps.note(CalibTap::RoutedInner, &inner);
    }
    let [shared, sh_in] = shared();
    taps.note(CalibTap::SharedDownIn, &sh_in);
    routed.add(&shared)
}

/// Routed experts, however they are stored.
pub trait ExpertSet: Weights {
    /// Each row's picked experts `ids`, mixed by `gates`.
    fn apply(&self, m: &Model, h: &Tensor, picks: [&Tensor; 2]) -> Tensor;
    /// Same as `apply`, also exposing per-pick down input when plain.
    fn apply_calib(&self, m: &Model, h: &Tensor, picks: [&Tensor; 2]) -> (Tensor, Option<Tensor>) {
        (self.apply(m, h, picks), None)
    }
    /// Rotates a tap into the experts' input gauge, when present.
    fn tap_gauge(&self, h: &Tensor) -> Tensor {
        h.clone()
    }
}

/// Experts as plain matrices.
#[derive(Weights)]
pub struct PlainExperts {
    #[shape(Experts, Hidden, Ffn)]
    pub gate: Linears,
    #[shape(Experts, Hidden, Ffn)]
    pub up: Linears,
    #[shape(Experts, Ffn, Hidden)]
    pub down: Linears,
}

impl ExpertSet for PlainExperts {
    fn apply(&self, m: &Model, h: &Tensor, picks: [&Tensor; 2]) -> Tensor {
        self.apply_calib(m, h, picks).0
    }

    fn apply_calib(
        &self,
        m: &Model,
        h: &Tensor,
        [ids, gates]: [&Tensor; 2],
    ) -> (Tensor, Option<Tensor>) {
        // One output row per pick, each on its own expert.
        let on = |set: &Linears, x: &Tensor| x.matmul_rows(&m.pick(set, ids));
        let inner = on(&self.gate, h).act(Act::Silu).mul(&on(&self.up, h));
        let picks = h.ty().rank() as u32 - 1;
        let out = on(&self.down, &inner).mix(picks, gates);
        (out, Some(inner))
    }
}

/// Experts as codes between 128-channel rotations.
#[derive(Weights)]
pub struct CodedExperts {
    /// Signs on the stream into the experts, and out.
    #[shape(Hidden)]
    pub signs_in: Vector,
    #[shape(Hidden)]
    pub signs_out: Vector,
    #[shape(Experts, Hidden, Ffn)]
    pub gate: Linears,
    #[shape(Experts, Hidden)]
    pub gate_su: Embed,
    #[shape(Experts, Ffn)]
    pub gate_sv: Embed,
    #[shape(Experts, Hidden, Ffn)]
    pub up: Linears,
    #[shape(Experts, Hidden)]
    pub up_su: Embed,
    #[shape(Experts, Ffn)]
    pub up_sv: Embed,
    #[shape(Experts, Ffn, Hidden)]
    pub down: Linears,
    #[shape(Experts, Ffn)]
    pub down_su: Embed,
    #[shape(Experts, Hidden)]
    pub down_sv: Embed,
}

impl ExpertSet for CodedExperts {
    fn apply(&self, m: &Model, h: &Tensor, [ids, gates]: [&Tensor; 2]) -> Tensor {
        let picks = h.ty().rank() as u32 - 1;
        // One copy of each signed row per picked expert.
        let lead = h.ty().shape.0[..picks as usize].to_vec();
        let one = lead.into_iter().chain([DimExpr::Const(1), Hidden.into()]);
        let slots = h
            .mul_vector(&self.signs_in)
            .reshape(one.collect::<Vec<_>>())
            .expand(picks, TopK);
        let on = |x: &Tensor, codes: &Linears, [su, sv]: [&Embed; 2]| {
            let turned = x.mul(&m.embed(ids, su)).hadamard(ROTATION);
            let product = turned.matmul_rows(&m.pick(codes, ids));
            product.hadamard(ROTATION).mul(&m.embed(ids, sv))
        };
        let gate = on(&slots, &self.gate, [&self.gate_su, &self.gate_sv]);
        let up = on(&slots, &self.up, [&self.up_su, &self.up_sv]);
        let inner = gate.act(Act::Silu).mul(&up);
        let out = on(&inner, &self.down, [&self.down_su, &self.down_sv]);
        out.mul_vector(&self.signs_out).mix(picks, gates)
    }

    fn tap_gauge(&self, h: &Tensor) -> Tensor {
        h.mul_vector(&self.signs_in).hadamard(ROTATION)
    }
}
