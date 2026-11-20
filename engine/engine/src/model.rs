//! A forward graph with its weights, launched as a whole.

use crate::Error;
use sensai_graph::{DimExpr, DimName, Graph, Size};
use sensai_interp::Tensor;
use sensai_lower::{Cache, Device, Recipe, Settings, Weight};
use sensai_run::{Run, open};

/// Plans on a runner, lowered once per launch sizes.
struct Planned {
    runner: Box<dyn Run>,
    device: Device,
    plans: Cache,
}

/// The graph's runner.
enum Executor {
    Planned(Box<Planned>),
    /// The interpreter, for graphs no rule lowers yet.
    Interpreted(Vec<Tensor>),
}

pub struct Model {
    graph: Graph,
    executor: Executor,
    /// Dimensions each launch sizes from its inputs.
    bounded: Vec<DimName>,
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
            Err(sensai_lower::Error::Unsupported(_)) => Executor::Interpreted(Vec::new()),
            planned => {
                let mut runner = open(planned?, settings)?;
                runner.launching(recipe.launch);
                let device = Device::stored(runner.platform(), settings)
                    .map_err(sensai_lower::Error::from)?
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
        })
    }

    /// Takes the graph's weights, in its order.
    pub fn load(&mut self, weights: &[Weight]) -> Result<(), Error> {
        match &mut self.executor {
            Executor::Planned(p) => {
                for (at, w) in weights.iter().enumerate() {
                    p.runner.load_weight(at as u32, w)?;
                }
            }
            Executor::Interpreted(kept) => {
                let g = &self.graph;
                *kept = (weights.iter().enumerate())
                    .map(|(at, w)| floats(g, at, w))
                    .collect();
            }
        }
        Ok(())
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
                sensai_interp::run(g, weights, 0, inputs).map_err(Error::from)
            }
            Executor::Planned(p) => {
                let plan = p.plans.lower(g, 0, &sizes, &p.device)?;
                p.runner.replan(&plan)?;
                let views: Vec<_> = inputs.iter().map(Tensor::view).collect();
                p.runner.feed(&views)?;
                p.runner.submit(0..plan.steps.len())?;
                plan.outputs
                    .iter()
                    .map(|o| Ok(p.runner.read(o.slot)?))
                    .collect()
            }
        }
    }
}
