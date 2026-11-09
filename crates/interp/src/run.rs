//! Walks the graph in order, binding dimensions from the inputs.

use crate::tensor::{Data, Error, Tensor};
use graph::{
    Attend, DimExpr, DimName, Graph, NodeId, Op, PathSeg, Select, Shape, Size, Stage, View,
};
use std::collections::BTreeMap;

/// One per backend; views and casts stay on the host.
pub trait Kernels {
    fn matmul(&mut self, x: &Tensor, w: &Tensor) -> Result<Tensor, Error>;
    fn attend(&mut self, a: &Attend, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor, Error>;
    fn stage(&mut self, s: &Stage, x: &Tensor, operand: Option<&Tensor>) -> Result<Tensor, Error>;
    fn gather(&mut self, table: &Tensor, ids: &Tensor) -> Result<Tensor, Error>;
}

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
            Size::Bounded(_) => self.bound.get(&name).copied(),
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

/// Weight tensors in graph order, found by path.
pub fn bind(g: &Graph, find: impl Fn(&[PathSeg]) -> Option<Data>) -> Result<Vec<Tensor>, Error> {
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
            Ok(Tensor { shape, data })
        })
        .collect()
}

/// Runs `entry` and returns its outputs in order.
pub fn run(
    g: &Graph,
    k: &mut dyn Kernels,
    weights: &[Tensor],
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
        let filled = t.data.len() == t.shape.iter().product::<usize>();
        if t.data.dtype() != port.ty.dtype || !filled {
            return Err(Error::Mismatch(format!(
                "{:?} port wants {:?} filling {:?}",
                port.role, port.ty.dtype, t.shape
            )));
        }
    }
    let ctx = Context {
        sizes: &sizes,
        entry,
        inputs,
        weights,
    };
    let needed = reachable(g, e.outputs.iter().map(|o| o.node));
    let mut values: Vec<Option<Tensor>> = vec![None; g.nodes.len()];
    for (i, n) in g.nodes.iter().enumerate() {
        if !needed[i] {
            continue;
        }
        let value = |a: &NodeId| values[a.0 as usize].as_ref().expect("args come first");
        let args: Vec<&Tensor> = n.args.iter().map(value).collect();
        let out = ctx.eval(k, &n.op, &args)?;
        let want = sizes.shape(&n.ty.shape)?;
        if out.shape != want || out.data.dtype() != n.ty.dtype {
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
    fn eval(&self, k: &mut dyn Kernels, op: &Op, args: &[&Tensor]) -> Result<Tensor, Error> {
        match op {
            Op::Input { entry, port } if *entry as usize == self.entry => {
                Ok(self.inputs[*port as usize].clone())
            }
            Op::Input { .. } => Err(Error::Unsupported("another entry's input".into())),
            Op::Weight(i) => Ok(self.weights[*i as usize].clone()),
            Op::Matmul => k.matmul(args[0], args[1]),
            Op::Attend(a) => k.attend(a, args[0], args[1], args[2]),
            Op::Row(row) => {
                let mut x = args[0].clone();
                let mut rest = args[1..].iter();
                for s in &row.0 {
                    let operand = (s.operands() == 1).then(|| *rest.next().expect("verified"));
                    x = k.stage(s, &x, operand)?;
                }
                Ok(x)
            }
            Op::Select(Select::Gather) => k.gather(args[0], args[1]),
            Op::View(v) => self.view(v, args[0]),
            Op::Cast(dtype) => Ok(Tensor {
                shape: args[0].shape.clone(),
                data: args[0].data.cast(*dtype)?,
            }),
        }
    }

    fn view(&self, v: &View, x: &Tensor) -> Result<Tensor, Error> {
        match v {
            View::Reshape(s) => Ok(Tensor {
                shape: self.sizes.shape(s)?,
                data: x.data.clone(),
            }),
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
                Ok(Tensor {
                    shape,
                    data: x.data.pick(at),
                })
            }
            View::Permute(order) => Ok(permute(x, order)),
        }
    }
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
    Tensor {
        data: x.data.pick(at),
        shape,
    }
}
