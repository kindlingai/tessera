//! A forward graph with its weights, launched as a whole.

use crate::Error;
use std::borrow::Cow;
use tessera_graph::{DimExpr, DimName, Graph, Shape, Size};
use tessera_interp::Tensor;
use tessera_lower::{Cache, Device, Recipe, Settings, Weight};
use tessera_run::{Run, open};

/// Plans on a runner, lowered once per launch sizes.
struct Planned {
    runner: Box<dyn Run>,
    device: Device,
    plans: Cache,
}

/// The graph's runner.
enum Executor {
    Planned(Box<Planned>),
    /// The interpreter, for graphs without a lowering rule yet.
    Interpreted(Vec<Tensor>),
}

pub struct Model {
    graph: Graph,
    executor: Executor,
    /// Dimensions each launch sizes from its inputs.
    bounded: Vec<DimName>,
    /// Dimensions planned at power-of-two buckets.
    buckets: Vec<DimName>,
}

/// Each bounded dimension's size, read off the inputs' shapes.
fn sizes_of(
    g: &Graph,
    bounded: &[DimName],
    inputs: &[Tensor],
) -> Result<Vec<(DimName, u64)>, Error> {
    let mut sizes: Vec<(DimName, u64)> = Vec::new();
    for (port, t) in g.entries[0].inputs.iter().zip(inputs) {
        for (axis, n) in port.ty.shape.0.iter().zip(&t.shape) {
            if let DimExpr::Dim(name) = axis
                && bounded.contains(name)
                && !sizes.iter().any(|(d, _)| d == name)
            {
                sizes.push((*name, *n as u64));
            }
        }
    }
    if sizes.len() != bounded.len() {
        return Err(Error::InputAxis);
    }
    Ok(sizes)
}

/// `shape` at `sizes`.
fn extents(g: &Graph, shape: &Shape, sizes: &[(DimName, u64)]) -> Vec<usize> {
    let size = |name| sizes.iter().find(|(d, _)| *d == name).map(|(_, n)| *n);
    let axis = |e: &DimExpr| e.value(&g.dims, &size).unwrap_or(1) as usize;
    shape.0.iter().map(axis).collect()
}

/// `t` at `want`, a bucketed axis cut or zero filled.
fn fitted(t: &Tensor, axes: &Shape, want: &[usize], buckets: &[DimName]) -> Result<Tensor, Error> {
    for (i, e) in axes.0.iter().enumerate() {
        let whole = matches!(e, DimExpr::Dim(d) if buckets.contains(d));
        if t.shape[i] != want[i] && !whole {
            return Err(Error::BucketAxis);
        }
    }
    let width = t.bytes().len() / t.len().max(1);
    let mut out = Tensor::zeros(want.to_vec(), t.dtype());
    copy(t.bytes(), &t.shape, out.bytes_mut(), want, width);
    Ok(out)
}

/// The rows both shapes hold, from `from` into `to`.
fn copy(from: &[u8], have: &[usize], to: &mut [u8], want: &[usize], width: usize) {
    let (Some((&n, have)), Some((&m, want))) = (have.split_first(), want.split_first()) else {
        return to.copy_from_slice(from);
    };
    let (a, b) = (
        have.iter().product::<usize>(),
        want.iter().product::<usize>(),
    );
    for i in 0..n.min(m) {
        let (src, dst) = (
            &from[i * a * width..][..a * width],
            &mut to[i * b * width..],
        );
        copy(src, have, &mut dst[..b * width], want, width);
    }
}

/// `w` as floats in the graph's shape.
fn floats(g: &Graph, at: usize, w: &Weight) -> Tensor {
    let axis = |e: &DimExpr| e.value(&g.dims, &|_| None).unwrap_or(1) as usize;
    let shape: Vec<usize> = g.weights[at].ty.shape.0.iter().map(axis).collect();
    match w {
        Weight::Floats(t) => Tensor::f32(shape, t.floats().unwrap_or_default().to_vec()),
        Weight::Packed(f, bytes) => Tensor::f32(shape, f.unpack_all(bytes)),
    }
}

impl Model {
    /// A runner for `graph`, else the interpreter.
    pub fn open(graph: Graph, recipe: &Recipe, settings: &Settings) -> Result<Self, Error> {
        let device = Device::guess(settings).with(recipe);
        let bounded: Vec<DimName> = (graph.dims.iter())
            .filter(|d| matches!(d.size, Size::Bounded(_)))
            .map(|d| d.name)
            .collect();
        let least: Vec<(DimName, u64)> = bounded.iter().map(|d| (*d, 1)).collect();
        let mut plans = Cache::default();
        let executor = match plans.lower(&graph, 0, &least, &device) {
            Err(tessera_lower::Error::Unsupported(_)) => Executor::Interpreted(Vec::new()),
            planned => {
                let mut runner = open(planned?, settings)?;
                runner.launching(recipe.launch);
                let device = Device::stored(runner.platform(), settings)
                    .map_err(tessera_lower::Error::from)?
                    .with(recipe);
                Executor::Planned(Box::new(Planned {
                    runner,
                    device,
                    plans,
                }))
            }
        };
        Ok(Self {
            graph,
            executor,
            bounded,
            buckets: Vec::new(),
        })
    }

    /// Plans per bucket of `dims`. Pads with zeros: causal only.
    pub fn bucketed(mut self, dims: &[DimName]) -> Self {
        self.buckets = dims.to_vec();
        self
    }

    /// Takes the graph's weights, in its order.
    pub fn load(&mut self, weights: &[Weight]) -> Result<(), Error> {
        for (at, w) in weights.iter().enumerate() {
            self.load_one(at as u32, w)?;
        }
        Ok(())
    }

    /// Takes the weight at `at`, so weights can stream in.
    pub fn load_one(&mut self, at: u32, w: &Weight) -> Result<(), Error> {
        match &mut self.executor {
            Executor::Planned(p) => p.runner.load_weight(at, w)?,
            Executor::Interpreted(kept) => {
                let at = at as usize;
                if kept.len() <= at {
                    kept.resize(at + 1, Tensor::f32(vec![0], Vec::new()));
                }
                kept[at] = floats(&self.graph, at, w);
            }
        }
        Ok(())
    }

    /// Whether the interpreter runs the graph.
    pub fn interpreted(&self) -> bool {
        matches!(self.executor, Executor::Interpreted(_))
    }

    /// The backend and hardware under the runner, or the interpreter.
    pub fn device(&self) -> String {
        match &self.executor {
            Executor::Planned(p) => p.runner.device(),
            Executor::Interpreted(_) => "interpreter".into(),
        }
    }

    /// One launch of the forward entry: every output, in order.
    pub fn run(&mut self, inputs: &[Tensor]) -> Result<Vec<Tensor>, Error> {
        let g = &self.graph;
        let sizes = sizes_of(g, &self.bounded, inputs)?;
        match &mut self.executor {
            Executor::Interpreted(weights) => {
                tessera_interp::run(g, weights, 0, inputs).map_err(Error::from)
            }
            Executor::Planned(p) => {
                let (plan, bound) = p.plans.upto(g, 0, &sizes, &self.buckets, &p.device)?;
                p.runner.replan(plan)?;
                p.runner.bind_all(&bound);
                if bound.is_empty() {
                    let views: Vec<_> = inputs.iter().map(Tensor::view).collect();
                    p.runner.feed(&views)?;
                    p.runner.submit(0..plan.steps.len())?;
                    let read = |o: &tessera_lower::Output| Ok(p.runner.read(o.slot)?);
                    return plan.outputs.iter().map(read).collect();
                }
                let mut planned = sizes.clone();
                for (name, n) in &bound {
                    let most = g.dims.iter().find(|d| d.name == *name).map(|d| d.size);
                    if matches!(most, Some(Size::Bounded(m)) if u64::from(*n) > m) {
                        return Err(Error::BucketBound(*name));
                    }
                    let at = planned.iter_mut().find(|(d, _)| d == name);
                    at.expect("a bounded dimension").1 = u64::from(*n);
                }
                // Inputs pad with zeros to the bucket.
                let entry = &g.entries[0];
                let mut fed = Vec::with_capacity(inputs.len());
                for (at, (t, port)) in inputs.iter().zip(&entry.inputs).enumerate() {
                    let axes = &port.ty.shape;
                    if t.shape != extents(g, axes, &sizes) {
                        return Err(tessera_run::Error::InputShape(at).into());
                    }
                    let want = extents(g, axes, &planned);
                    fed.push(match t.shape == want {
                        true => Cow::Borrowed(t),
                        false => Cow::Owned(fitted(t, axes, &want, &self.buckets)?),
                    });
                }
                let views: Vec<_> = fed.iter().map(|t| t.view()).collect();
                p.runner.feed(&views)?;
                p.runner.submit(0..plan.steps.len())?;
                let mut out = Vec::with_capacity(plan.outputs.len());
                for (o, at) in plan.outputs.iter().zip(&entry.outputs) {
                    let axes = &g.nodes[at.node.0 as usize].ty.shape;
                    let mut full = p.runner.read(o.slot)?;
                    full.shape = extents(g, axes, &planned);
                    let want = extents(g, axes, &sizes);
                    out.push(match full.shape == want {
                        true => full,
                        false => fitted(&full, axes, &want, &self.buckets)?,
                    });
                }
                Ok(out)
            }
        }
    }
}
