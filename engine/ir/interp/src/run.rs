//! Walks the graph in order, binding dimensions from the inputs.

use crate::cpu;
use crate::tensor::Error;
use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use tessera_graph::{
    DType, DimExpr, DimName, Graph, Mask, Node, NodeId, Numerics, Op, Parts, PathSeg, Placement,
    Row, Select, Shape, Size, Stage, View,
};
use tessera_mem::Tensor;

/// Dimension sizes for one launch.
struct Sizes<'a> {
    g: &'a Graph,
    bound: BTreeMap<DimName, u64>,
}

impl<'a> Sizes<'a> {
    fn new(g: &'a Graph) -> Self {
        Self {
            g,
            bound: BTreeMap::new(),
        }
    }

    fn of(&self, name: DimName) -> Option<u64> {
        match self.g.dims.iter().find(|d| d.name == name)?.size {
            Size::Fixed(n) => Some(n),
            // A bounded axis no port names is its bound.
            Size::Bounded(max) => Some(self.bound.get(&name).copied().unwrap_or(max)),
        }
    }

    fn expr(&self, e: &DimExpr) -> Result<usize, Error> {
        let unknown = || Error::Mismatch(format!("{e} has no size in this launch"));
        let poly = e.poly(&self.g.dims).map_err(|_| unknown())?;
        let n = poly.eval(&|name| self.of(name)).ok_or_else(unknown)?;
        Ok(n as usize)
    }

    fn shape(&self, s: &Shape) -> Result<Vec<usize>, Error> {
        s.0.iter().map(|e| self.expr(e)).collect()
    }

    /// Binds bounded names that appear bare in `s`.
    fn learn(&mut self, s: &Shape, actual: &[usize]) -> Result<(), Error> {
        if s.rank() != actual.len() {
            return Err(Error::Mismatch(format!("{actual:?} does not fit {s}")));
        }
        for (e, &n) in s.0.iter().zip(actual) {
            let DimExpr::Dim(name) = e else { continue };
            let Some(dim) = self.g.dims.iter().find(|d| d.name == *name) else {
                continue;
            };
            if let Size::Bounded(max) = dim.size {
                let n = n as u64;
                let seen = *self.bound.entry(*name).or_insert(n);
                if n > max || seen != n {
                    let msg = format!("{name:?} = {n} (bound {max}, seen {seen})");
                    return Err(Error::Mismatch(msg));
                }
            }
        }
        Ok(())
    }
}

/// Weights in graph order, found by path and shaped here.
pub fn bind(g: &Graph, find: impl Fn(&[PathSeg]) -> Option<Tensor>) -> Result<Vec<Tensor>, Error> {
    let sizes = Sizes::new(g);
    g.weights
        .iter()
        .map(|w| {
            let shape = sizes.shape(&w.ty.shape)?;
            let missing = || Error::Mismatch(format!("no weight at {:?}", w.path));
            let data = find(&w.path).ok_or_else(missing)?;
            let want: usize = shape.iter().product();
            if data.len() != want || data.dtype() != w.ty.dtype {
                let got = (data.len(), data.dtype());
                let msg = format!("{:?}: got {got:?}, want {want} {:?}", w.path, w.ty.dtype);
                return Err(Error::Mismatch(msg));
            }
            Ok(data.reshaped(shape))
        })
        .collect()
}

/// Kept rows, held between launches, per rank.
#[derive(Clone, Debug, Default)]
pub struct Store {
    ranks: Vec<Held>,
}

/// One rank's kept rows.
#[derive(Clone, Debug, Default)]
struct Held {
    kept: Vec<Option<Tensor>>,
    /// Which slot and position each kept tensor has had written.
    written: Vec<Vec<bool>>,
}

impl Store {
    /// Positions kept tensor `i` has room for in each slot.
    pub fn len(&self, i: usize) -> usize {
        let kept = self.ranks.first().and_then(|h| h.kept.get(i));
        kept.and_then(Option::as_ref).map_or(0, |t| t.shape[1])
    }

    /// Rank `r`'s kept rows.
    fn rank(&mut self, r: usize) -> &mut Held {
        if self.ranks.len() <= r {
            self.ranks.resize(r + 1, Held::default());
        }
        &mut self.ranks[r]
    }
}

impl Held {
    /// Writes rows by slot and position. Returns `ctx` rows each.
    fn append(
        &mut self,
        i: usize,
        [rows, slots, positions]: [&Tensor; 3],
        seqs: usize,
        ctx: usize,
    ) -> Result<Tensor, Error> {
        self.kept.resize(self.kept.len().max(i + 1), None);
        self.written.resize(self.kept.len(), Vec::new());
        let (new, slots, positions) = (rows.floats()?, slots.indices()?, positions.indices()?);
        let width: usize = rows.shape[1..].iter().product();
        let old = self.kept[i].as_ref();
        let had = old.map_or(0, |t| t.shape[1]);
        let len = had.max(ctx);
        // Rows stay where they were. The store only grows.
        let mut all = vec![0.0f32; seqs * len * width];
        let mut written = vec![false; seqs * len];
        if let Some(old) = old {
            for (s, from) in old.floats()?.chunks(had * width).enumerate() {
                all[s * len * width..][..from.len()].copy_from_slice(from);
                written[s * len..][..had].copy_from_slice(&self.written[i][s * had..][..had]);
            }
        }
        for (r, row) in new.chunks(width).enumerate() {
            let (s, p) = (slots[r], positions[r]);
            if s >= seqs || p >= len {
                let msg = format!("row {r} at slot {s}, position {p} is outside the kept rows");
                return Err(Error::Mismatch(msg));
            }
            all[(s * len + p) * width..][..width].copy_from_slice(row);
            written[s * len + p] = true;
        }
        // Rows read their slot's earlier positions.
        for (r, (&s, &p)) in slots.iter().zip(&positions).enumerate() {
            if let Some(gap) = (0..p).find(|&q| !written[s * len + q]) {
                let msg = format!("row {r} at slot {s} reads unwritten position {gap}");
                return Err(Error::Mismatch(msg));
            }
        }
        self.written[i] = written;
        let mut shape = vec![seqs, len];
        shape.extend_from_slice(&rows.shape[1..]);
        let grown = Tensor::f32(shape, all);
        self.kept[i] = Some(grown.clone());
        if len == ctx {
            return Ok(grown);
        }
        let mut seen = Vec::with_capacity(seqs * ctx * width);
        for s in 0..seqs {
            seen.extend_from_slice(&grown.floats()?[s * len * width..][..ctx * width]);
        }
        let mut shape = vec![seqs, ctx];
        shape.extend_from_slice(&rows.shape[1..]);
        Ok(Tensor::f32(shape, seen))
    }
}

/// Runs `entry` with nothing kept from earlier launches.
pub fn run(
    g: &Graph,
    weights: &[Tensor],
    entry: usize,
    inputs: &[Tensor],
) -> Result<Vec<Tensor>, Error> {
    run_kept(g, weights, &mut Store::default(), entry, inputs)
}

/// Runs `entry` and returns its outputs in order.
pub fn run_kept(
    g: &Graph,
    weights: &[Tensor],
    store: &mut Store,
    entry: usize,
    inputs: &[Tensor],
) -> Result<Vec<Tensor>, Error> {
    run_on(g, weights, store, entry, inputs, &Parts::even(1))
}

/// Runs `entry` on `ranks` equal ranks in lockstep.
pub fn run_ranks(
    g: &Graph,
    weights: &[Tensor],
    entry: usize,
    inputs: &[Tensor],
    ranks: usize,
) -> Result<Vec<Tensor>, Error> {
    run_parts(g, weights, entry, inputs, &Parts::even(ranks))
}

/// Runs `entry` on ranks weighted by `parts`, in lockstep.
pub fn run_parts(
    g: &Graph,
    weights: &[Tensor],
    entry: usize,
    inputs: &[Tensor],
    parts: &Parts,
) -> Result<Vec<Tensor>, Error> {
    let mut store = Store::default();
    run_on(g, weights, &mut store, entry, inputs, parts)
}

/// Rank `r`'s part of a `whole` axis.
fn part(parts: &Parts, axis: u32, whole: usize, r: usize) -> Result<(usize, usize), Error> {
    parts.part(whole, r).ok_or_else(|| {
        let msg = format!("axis {axis} of {whole} does not divide by {:?}", parts.0);
        Error::Mismatch(msg)
    })
}

/// A weight as f32. A bf16 weight widens here.
fn weight(w: &Tensor) -> Result<Tensor, Error> {
    match w.dtype() {
        DType::Bf16 => Ok(Tensor::f32(w.shape.clone(), w.view().to_f32()?)),
        _ => Ok(w.clone()),
    }
}

/// Rank `r`'s part of a weight. Whole ones are borrowed.
fn share<'w>(
    w: &'w Tensor,
    placement: Placement,
    r: usize,
    parts: &Parts,
) -> Result<Cow<'w, Tensor>, Error> {
    let Placement::Split { axis } = placement else {
        return Ok(Cow::Borrowed(w));
    };
    let (first, count) = part(parts, axis, w.shape[axis as usize], r)?;
    Ok(Cow::Owned(slice(w, axis as usize, first, count)))
}

/// The sum over ranks, in rank order.
fn all_reduce(num: &Numerics, parts: &[Tensor]) -> Result<Tensor, Error> {
    let mut sum = parts[0].clone();
    for p in &parts[1..] {
        sum = cpu::stage(num, &Stage::Add, &sum, Some(p))?;
    }
    Ok(sum)
}

/// The parts joined on their last axis, in rank order.
fn all_gather(parts: &[Tensor]) -> Result<Tensor, Error> {
    let mut whole = parts[0].clone();
    for p in &parts[1..] {
        whole = concat(&whole, p)?;
    }
    Ok(whole)
}

/// Where a row's sum or gather lies, if any.
fn joins(row: &Row) -> Option<usize> {
    (row.0.iter()).position(|s| matches!(s, Stage::AllReduce | Stage::AllGather))
}

fn run_on(
    g: &Graph,
    weights: &[Tensor],
    store: &mut Store,
    entry: usize,
    inputs: &[Tensor],
    parts: &Parts,
) -> Result<Vec<Tensor>, Error> {
    let ranks = parts.ranks();
    let e = (g.entries.get(entry)).ok_or_else(|| Error::Mismatch(format!("no entry {entry}")))?;
    if inputs.len() != e.inputs.len() || weights.len() != g.weights.len() {
        return Err(Error::Mismatch("wrong number of inputs or weights".into()));
    }
    let mut sizes = Sizes::new(g);
    for (port, t) in e.inputs.iter().zip(inputs) {
        sizes.learn(&port.ty.shape, &t.shape)?;
        if t.dtype() != port.ty.dtype || !t.filled() {
            return Err(Error::Mismatch(format!(
                "{:?} port wants {:?} filling {:?}",
                port.role, port.ty.dtype, t.shape
            )));
        }
    }
    let needed = reachable(g, e.outputs.iter().map(|o| o.node));
    // The kept rows reach the furthest position this launch writes.
    for n in g
        .nodes
        .iter()
        .enumerate()
        .filter(|(i, _)| needed[*i])
        .map(|(_, n)| n)
    {
        if let Op::Append(i) = n.op
            && let Op::Input { port, .. } = g.node(n.args[2]).op
        {
            let furthest = inputs[port as usize].indices()?.iter().max().copied();
            let total = [furthest.map_or(0, |p| p + 1)];
            sizes.learn(&Shape(vec![g.kept[i as usize].shape.0[1].clone()]), &total)?;
        }
    }
    let mut shares = Vec::with_capacity(ranks);
    for r in 0..ranks {
        let mut own = Vec::with_capacity(weights.len());
        for (w, t) in g.weights.iter().zip(weights) {
            own.push(share(t, w.ty.placement, r, parts)?);
        }
        shares.push(own);
    }
    let ctx = |r: usize, first: usize| Context {
        sizes: &sizes,
        entry,
        inputs,
        weights: &shares[r],
        first,
    };
    let picked = picked(g, &needed, e.outputs.iter().map(|o| o.node));
    let order = order(g, &needed, &picked);
    // A value lives until its last reader has run.
    let mut last = vec![usize::MAX; g.nodes.len()];
    for (step, &i) in order.iter().enumerate() {
        for a in reads(g, &picked, &g.nodes[i]) {
            last[a.0 as usize] = step;
        }
    }
    for o in &e.outputs {
        last[o.node.0 as usize] = usize::MAX;
    }
    let mut values: Vec<Vec<Option<Tensor>>> = vec![vec![None; g.nodes.len()]; ranks];
    for (step, &i) in order.iter().enumerate() {
        let n = &g.nodes[i];
        let num = n.numerics.unwrap_or_default();
        // Each rank holds its part of a split axis.
        let whole = sizes.shape(&n.ty.shape)?;
        let mut wants = vec![whole.clone(); ranks];
        if let Placement::Split { axis } = n.ty.placement {
            for (r, want) in wants.iter_mut().enumerate() {
                want[axis as usize] = part(parts, axis, whole[axis as usize], r)?.1;
            }
        }
        let mut outs: Vec<Tensor> = Vec::with_capacity(ranks);
        for (r, held) in values.iter().enumerate() {
            let value = |a: &NodeId| held[a.0 as usize].as_ref().expect("args come first");
            if let [x, w] = n.args[..]
                && picked[w.0 as usize]
            {
                let gather = g.node(w);
                let table = [value(&gather.args[0]), value(&gather.args[1])];
                let first = first_row(g, gather, &sizes, parts, r)?;
                outs.push(cpu::matmul_picked(&num, value(&x), table, first)?);
                continue;
            }
            let args: Vec<&Tensor> = n.args.iter().map(value).collect();
            let first = first_row(g, n, &sizes, parts, r)?;
            outs.push(match &n.op {
                Op::View(View::Reshape(_)) => args[0].reshaped(wants[r].clone()),
                // Whole kept rows are the same on every rank.
                Op::Append(_) if r > 0 && n.ty.placement == Placement::Replicated => {
                    outs[0].clone()
                }
                Op::Append(i) => {
                    let kept = &g.kept[*i as usize].shape.0;
                    let (seqs, ctx) = (sizes.expr(&kept[0])?, sizes.expr(&kept[1])?);
                    let rows = [args[0], args[1], args[2]];
                    store.rank(r).append(*i as usize, rows, seqs, ctx)?
                }
                Op::Row(row) if joins(row).is_some() => {
                    let at = joins(row).expect("found");
                    ctx(r, 0).stages(&num, &row.0[..at], &args)?
                }
                op => ctx(r, first).eval(&num, op, &args)?,
            });
        }
        if let Op::Row(row) = &n.op
            && let Some(at) = joins(row)
        {
            let sum = match row.0[at] {
                Stage::AllReduce => all_reduce(&num, &outs)?,
                _ => all_gather(&outs)?,
            };
            let rest = &row.0[at + 1..];
            // Stages past the sum are operand-free.
            outs = (0..ranks)
                .map(|r| ctx(r, 0).stages(&num, rest, &[&sum]))
                .collect::<Result<_, _>>()?;
        }
        for (r, out) in outs.into_iter().enumerate() {
            if out.shape != wants[r] || out.dtype() != n.ty.dtype {
                let msg = format!(
                    "node {i} {:?} gave {:?}, graph says {:?}",
                    n.op, out.shape, wants[r]
                );
                return Err(Error::Mismatch(msg));
            }
            values[r][i] = Some(out);
        }
        for a in reads(g, &picked, n).filter(|a| last[a.0 as usize] == step) {
            for held in &mut values {
                held[a.0 as usize] = None;
            }
        }
    }
    let mut outputs = Vec::with_capacity(e.outputs.len());
    for o in &e.outputs {
        let at = o.node.0 as usize;
        let first = values[0][at].clone().expect("outputs are needed");
        if values[1..].iter().any(|v| v[at].as_ref() != Some(&first)) {
            return Err(Error::Mismatch(format!("ranks disagree on node {at}")));
        }
        outputs.push(first);
    }
    Ok(outputs)
}

/// The first table row a gather holds on rank `r`.
fn first_row(g: &Graph, n: &Node, sizes: &Sizes, parts: &Parts, r: usize) -> Result<usize, Error> {
    let table = n.args.first().map(|a| g.node(*a));
    match table {
        Some(t)
            if matches!(n.op, Op::Select(Select::Gather))
                && t.ty.placement == Placement::Split { axis: 0 } =>
        {
            Ok(part(parts, 0, sizes.shape(&t.ty.shape)?[0], r)?.0)
        }
        _ => Ok(0),
    }
}

/// Run order: a leaf waits for its first reader.
fn order(g: &Graph, needed: &[bool], picked: &[bool]) -> Vec<usize> {
    let leaf = |i: usize| g.nodes[i].args.is_empty();
    let (mut out, mut placed) = (Vec::new(), vec![false; g.nodes.len()]);
    for (i, n) in g.nodes.iter().enumerate() {
        // A picked gather runs inside its consuming matmul.
        if !needed[i] || picked[i] || leaf(i) {
            continue;
        }
        for a in reads(g, picked, n).map(|a| a.0 as usize) {
            if leaf(a) && !placed[a] {
                placed[a] = true;
                out.push(a);
            }
        }
        out.push(i);
    }
    // An unread leaf is an output.
    for i in 0..g.nodes.len() {
        if needed[i] && leaf(i) && !placed[i] {
            out.push(i);
        }
    }
    out
}

/// Gathers that only matmuls read, as their weights.
fn picked(g: &Graph, needed: &[bool], outputs: impl Iterator<Item = NodeId>) -> Vec<bool> {
    let gather = |n: &Node| matches!(n.op, Op::Select(Select::Gather));
    let mut picked: Vec<bool> = g.nodes.iter().map(gather).collect();
    for o in outputs {
        picked[o.0 as usize] = false;
    }
    for n in g
        .nodes
        .iter()
        .zip(needed)
        .filter(|(_, k)| **k)
        .map(|(n, _)| n)
    {
        for (at, a) in n.args.iter().enumerate() {
            if !(matches!(n.op, Op::Matmul) && at == 1) {
                picked[a.0 as usize] = false;
            }
        }
    }
    picked
}

/// The values `n` reads: a picked gather's own arguments too.
fn reads<'a>(g: &'a Graph, picked: &'a [bool], n: &'a Node) -> impl Iterator<Item = NodeId> + 'a {
    let inner = |a: &NodeId| match picked[a.0 as usize] {
        true => g.node(*a).args.clone(),
        false => Vec::new(),
    };
    n.args
        .iter()
        .flat_map(move |a| std::iter::once(*a).chain(inner(a)))
}

/// Which nodes the outputs depend on.
fn reachable(g: &Graph, outputs: impl Iterator<Item = NodeId>) -> Vec<bool> {
    let mut needed = vec![false; g.nodes.len()];
    for o in outputs {
        needed[o.0 as usize] = true;
    }
    for i in (0..g.nodes.len()).rev() {
        if needed[i] {
            for a in &g.nodes[i].args {
                needed[a.0 as usize] = true;
            }
        }
    }
    needed
}

struct Context<'a> {
    sizes: &'a Sizes<'a>,
    entry: usize,
    inputs: &'a [Tensor],
    weights: &'a [Cow<'a, Tensor>],
    /// This rank's first row of a routed table.
    first: usize,
}

impl Context<'_> {
    fn eval(&self, num: &Numerics, op: &Op, args: &[&Tensor]) -> Result<Tensor, Error> {
        match op {
            Op::Input { entry, port } if *entry as usize == self.entry => {
                Ok(self.inputs[*port as usize].clone())
            }
            Op::Input { .. } => Err(Error::Unsupported("another entry's input".into())),
            Op::Weight(i) => weight(&self.weights[*i as usize]),
            Op::Matmul => cpu::matmul(num, args[0], args[1]),
            Op::Attend(a) if a.ragged => {
                let keys = args.get(5).copied();
                cpu::attend_rows(
                    num,
                    a,
                    [args[0], args[1], args[2]],
                    [args[3], args[4]],
                    keys,
                )
            }
            Op::Attend(a) => cpu::attend(num, a, [args[0], args[1], args[2]], args.get(3).copied()),
            Op::Conv => cpu::conv(args[0], args[1]),
            Op::Delta(d) => cpu::delta(num, d, [args[0], args[1], args[2], args[3], args[4]]),
            Op::Mask(mask) => masked(*mask, args[0]),
            Op::Row(row) => self.stages(num, &row.0, args),
            Op::Select(Select::Gather) => cpu::gather(args[0], args[1], self.first),
            Op::Select(Select::Take) => cpu::take(args[0], args[1]),
            Op::Select(Select::TopK { k }) => top_k(args[0], self.sizes.expr(k)?),
            Op::Append(_) => lacks("kept rows outside a launch"),
            Op::View(View::Concat) => concat(args[0], args[1]),
            Op::View(v) => self.view(v, args[0]),
            Op::Cast(dtype) => Ok(args[0].cast(*dtype)?),
            Op::Mean(axis) => cpu::mean(num, args[0], *axis as usize, None),
            Op::Mix(axis) => cpu::mean(num, args[0], *axis as usize, Some(args[1])),
        }
    }

    /// `stages` over `args[0]`, operands from the rest in order.
    fn stages(&self, num: &Numerics, stages: &[Stage], args: &[&Tensor]) -> Result<Tensor, Error> {
        let mut x = args[0].clone();
        let mut rest = args[1..].iter();
        for s in stages {
            let operand = (s.operands() == 1).then(|| *rest.next().expect("verified"));
            x = cpu::stage(num, s, &x, operand)?;
        }
        Ok(x)
    }

    fn view(&self, v: &View, x: &Tensor) -> Result<Tensor, Error> {
        match v {
            View::Reshape(s) => Ok(x.reshaped(self.sizes.shape(s)?)),
            View::Slice { axis, start, len } => {
                let (start, len) = (self.sizes.expr(start)?, self.sizes.expr(len)?);
                Ok(slice(x, *axis as usize, start, len))
            }
            View::Permute(order) => Ok(permute(x, order)),
            View::Expand { axis, to } => Ok(expand(x, *axis as usize, self.sizes.expr(to)?)),
            View::Concat => lacks("concat with one argument"),
        }
    }
}

/// Each row's largest finite entries, best first.
fn top_k(x: &Tensor, k: usize) -> Result<Tensor, Error> {
    let (_, width) = x.rows();
    let mut ids = Vec::with_capacity(x.len() / width.max(1) * k);
    for row in x.floats()?.chunks(width.max(1)) {
        let mut order: Vec<usize> = (0..width).filter(|i| row[*i] > f32::NEG_INFINITY).collect();
        // A stable sort keeps the lower index. 0.0 ties -0.0.
        order.sort_by(|a, b| row[*b].partial_cmp(&row[*a]).unwrap_or(Ordering::Equal));
        // Spare slots hold the width.
        order.resize(k, width);
        ids.extend(order.into_iter().take(k).map(|i| i as u32));
    }
    let mut shape = x.shape.clone();
    *shape.last_mut().expect("verified rank") = k;
    Ok(Tensor::u32(shape, ids))
}

/// Ops that trace and verify but do not run yet.
fn lacks(what: &str) -> Result<Tensor, Error> {
    Err(Error::Unsupported(what.into()))
}

/// Elements `start..start + len` along axis `a`.
fn slice(x: &Tensor, a: usize, start: usize, len: usize) -> Tensor {
    let outer: usize = x.shape[..a].iter().product();
    let inner: usize = x.shape[a + 1..].iter().product();
    let n = x.shape[a];
    let at = (0..outer).flat_map(move |o| {
        (start..start + len).flat_map(move |j| (0..inner).map(move |i| (o * n + j) * inner + i))
    });
    let mut shape = x.shape.clone();
    shape[a] = len;
    x.pick(shape, at)
}

/// Size-one axis `a` repeated `to` times.
fn expand(x: &Tensor, a: usize, to: usize) -> Tensor {
    let inner: usize = x.shape[a + 1..].iter().product();
    let outer: usize = x.shape[..a].iter().product();
    let at = (0..outer).flat_map(move |o| (0..to).flat_map(move |_| o * inner..(o + 1) * inner));
    let mut shape = x.shape.clone();
    shape[a] = to;
    x.pick(shape, at)
}

/// Scores over `[.., queries, keys]` a query cannot see become -inf.
fn masked(mask: Mask, x: &Tensor) -> Result<Tensor, Error> {
    let rank = x.shape.len();
    let (queries, keys) = (x.shape[rank - 2], x.shape[rank - 1]);
    let past = keys.saturating_sub(queries);
    let mut out = x.floats()?.to_vec();
    for (r, row) in out.chunks_mut(keys).enumerate() {
        let (from, to) = cpu::seen(mask, past + r % queries, keys);
        for (j, v) in row.iter_mut().enumerate() {
            if !(from..to).contains(&j) {
                *v = f32::NEG_INFINITY;
            }
        }
    }
    Ok(Tensor::f32(x.shape.clone(), out))
}

/// Each row of `a`, then the row of `b`.
fn concat(a: &Tensor, b: &Tensor) -> Result<Tensor, Error> {
    let (rows, n) = a.rows();
    let m = b.rows().1;
    let (xs, ys) = (a.floats()?, b.floats()?);
    let mut out = Vec::with_capacity(rows * (n + m));
    for r in 0..rows {
        out.extend_from_slice(&xs[r * n..(r + 1) * n]);
        out.extend_from_slice(&ys[r * m..(r + 1) * m]);
    }
    let mut shape = a.shape.clone();
    *shape.last_mut().expect("verified rank") = n + m;
    Ok(Tensor::f32(shape, out))
}

fn permute(x: &Tensor, order: &[u32]) -> Tensor {
    let rank = x.shape.len();
    let mut strides = vec![1; rank];
    for d in (0..rank.saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * x.shape[d + 1];
    }
    let shape: Vec<usize> = order.iter().map(|&d| x.shape[d as usize]).collect();
    let total: usize = shape.iter().product();
    let at = (0..total).map(|mut flat| {
        let mut offset = 0;
        for d in (0..rank).rev() {
            offset += (flat % shape[d]) * strides[order[d] as usize];
            flat /= shape[d];
        }
        offset
    });
    x.pick(shape.clone(), at)
}
