//! Walks the graph in order, binding dimensions from the inputs.

use crate::cpu;
use crate::tensor::Error;
use graph::{DimExpr, DimName, Graph, NodeId, Numerics, Op, PathSeg, Select, Shape, Size, View};
use mem::Tensor;
use std::cmp::Ordering;
use std::collections::BTreeMap;

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

/// Kept rows, held between launches.
#[derive(Clone, Debug, Default)]
pub struct Store {
    kept: Vec<Option<Tensor>>,
    /// Which slot and position each kept tensor has had written.
    written: Vec<Vec<bool>>,
}

impl Store {
    /// Positions kept tensor `i` has room for in each slot.
    pub fn len(&self, i: usize) -> usize {
        let kept = self.kept.get(i).and_then(Option::as_ref);
        kept.map_or(0, |t| t.shape[1])
    }

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
                let msg = format!("row {r} at slot {s} reads position {gap}, which no row wrote");
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
    let ctx = Context {
        sizes: &sizes,
        entry,
        inputs,
        weights,
    };
    let mut values: Vec<Option<Tensor>> = vec![None; g.nodes.len()];
    for (i, n) in g.nodes.iter().enumerate() {
        if !needed[i] {
            continue;
        }
        let value = |a: &NodeId| values[a.0 as usize].as_ref().expect("args come first");
        let args: Vec<&Tensor> = n.args.iter().map(value).collect();
        let num = n.numerics.unwrap_or_default();
        let out = match n.op {
            Op::Append(i) => {
                let kept = &g.kept[i as usize].shape.0;
                let (seqs, ctx) = (sizes.expr(&kept[0])?, sizes.expr(&kept[1])?);
                store.append(i as usize, [args[0], args[1], args[2]], seqs, ctx)?
            }
            _ => ctx.eval(&num, &n.op, &args)?,
        };
        let want = sizes.shape(&n.ty.shape)?;
        if out.shape != want || out.dtype() != n.ty.dtype {
            let msg = format!(
                "node {i} {:?} gave {:?}, graph says {want:?}",
                n.op, out.shape
            );
            return Err(Error::Mismatch(msg));
        }
        values[i] = Some(out);
    }
    Ok(e.outputs
        .iter()
        .map(|o| {
            values[o.node.0 as usize]
                .clone()
                .expect("outputs are needed")
        })
        .collect())
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
    weights: &'a [Tensor],
}

impl Context<'_> {
    fn eval(&self, num: &Numerics, op: &Op, args: &[&Tensor]) -> Result<Tensor, Error> {
        match op {
            Op::Input { entry, port } if *entry as usize == self.entry => {
                Ok(self.inputs[*port as usize].clone())
            }
            Op::Input { .. } => Err(Error::Unsupported("another entry's input".into())),
            Op::Weight(i) => Ok(self.weights[*i as usize].clone()),
            Op::Matmul if args[1].shape.len() > 2 => lacks("batched matmul"),
            Op::Matmul => cpu::matmul(num, args[0], args[1]),
            Op::Attend(a) if a.picked => lacks("attention over picked keys"),
            Op::Attend(a) if a.ragged => {
                cpu::attend_rows(num, a, [args[0], args[1], args[2]], [args[3], args[4]])
            }
            Op::Attend(a) => cpu::attend(num, a, [args[0], args[1], args[2]]),
            Op::Mask(_) => lacks("masks"),
            Op::Row(row) => {
                let mut x = args[0].clone();
                let mut rest = args[1..].iter();
                for s in &row.0 {
                    let operand = (s.operands() == 1).then(|| *rest.next().expect("verified"));
                    x = cpu::stage(num, s, &x, operand)?;
                }
                Ok(x)
            }
            Op::Select(Select::Gather) => cpu::gather(args[0], args[1]),
            Op::Select(Select::TopK { k }) => top_k(args[0], self.sizes.expr(k)?),
            Op::Select(s) => lacks(&format!("{s:?}")),
            Op::Append(_) => lacks("kept rows outside a launch"),
            Op::View(View::Concat) => concat(args[0], args[1]),
            Op::View(v) => self.view(v, args[0]),
            Op::Cast(dtype) => Ok(args[0].cast(*dtype)?),
            Op::Mean(axis) => cpu::mean(num, args[0], *axis as usize),
        }
    }

    fn view(&self, v: &View, x: &Tensor) -> Result<Tensor, Error> {
        match v {
            View::Reshape(s) => Ok(x.reshaped(self.sizes.shape(s)?)),
            View::Slice { axis, start, len } => {
                let a = *axis as usize;
                let (start, len) = (self.sizes.expr(start)?, self.sizes.expr(len)?);
                let outer: usize = x.shape[..a].iter().product();
                let inner: usize = x.shape[a + 1..].iter().product();
                let n = x.shape[a];
                let at = (0..outer).flat_map(move |o| {
                    (start..start + len)
                        .flat_map(move |j| (0..inner).map(move |i| (o * n + j) * inner + i))
                });
                let mut shape = x.shape.clone();
                shape[a] = len;
                Ok(x.pick(shape, at))
            }
            View::Permute(order) => Ok(permute(x, order)),
            View::Expand { .. } | View::Concat => lacks(&format!("{v:?}")),
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
