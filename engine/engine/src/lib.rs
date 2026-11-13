//! Token ids in, token ids out, over a stepper graph.

// Linking the crates registers their rules.
use kernels as _;
use linear as _;

use graph::DimName::{Context, Rows, Seqs};
use graph::Graph;
use interp::Tensor;
use lower::{Cache, Device, Error, Recipe, Run, Settings, open};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Tokens to continue, and when to stop.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Request {
    pub tokens: Vec<u32>,
    pub max_tokens: usize,
    /// Ids that end the answer, left out of it.
    pub stop: Vec<u32>,
}

/// Why an answer ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Reason {
    Stop,
    Length,
    /// The context filled before a stop.
    Context,
}

/// One decoded token, or the end.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Event {
    Token(u32),
    Done(Reason),
}

/// Rows in, each row's likeliest id out, on one runner.
pub struct Engine {
    graph: Graph,
    runner: Box<dyn Run>,
    device: Device,
    plans: Cache,
    context: usize,
    /// Kept-row lanes the graph declares.
    slots: usize,
}

/// A token at a position, in a slot's sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Row {
    pub slot: u32,
    pub token: u32,
    pub position: u32,
}

/// The sizes and inputs one launch of `rows` needs.
fn launch_of(rows: &[Row], slots: usize) -> ([(graph::DimName, u64); 3], [Tensor; 3]) {
    let furthest = rows.iter().map(|r| r.position).max().unwrap_or(0);
    let sizes = [
        (Seqs, slots as u64),
        (Rows, rows.len() as u64),
        (Context, furthest as u64 + 1),
    ];
    let shape = vec![rows.len()];
    let column = |of: fn(&Row) -> u32| Tensor::u32(shape.clone(), rows.iter().map(of).collect());
    let feeds = [
        column(|r| r.token),
        column(|r| r.position),
        column(|r| r.slot),
    ];
    (sizes, feeds)
}

impl Engine {
    /// Opens a runner for `graph` with room for `context` rows.
    pub fn open(
        graph: Graph,
        context: usize,
        recipe: &Recipe,
        settings: &Settings,
    ) -> Result<Self, Error> {
        let device = Device::guess(settings).with(recipe);
        let mut plans = Cache::default();
        let slots = graph
            .dims
            .iter()
            .find(|d| d.name == Seqs)
            .map_or(1, |d| d.size.max() as usize);
        let first = [Row {
            slot: 0,
            token: 0,
            position: 0,
        }];
        let (sizes, _) = launch_of(&first, slots);
        let plan = plans.lower(&graph, 0, &sizes, &device)?;
        let runner = open(plan, settings)?;
        let device = Device::stored(&*runner, settings)?.with(recipe);
        Ok(Self {
            graph,
            runner,
            device,
            plans,
            context,
            slots,
        })
    }

    /// Uploads the graph's weights, in its order.
    pub fn load(&mut self, weights: &[Tensor]) -> Result<(), Error> {
        for (at, w) in weights.iter().enumerate() {
            self.runner.load(at as u32, w.view())?;
        }
        Ok(())
    }

    /// The backend and hardware under the runner.
    pub fn device(&self) -> String {
        self.runner.device()
    }

    /// How often the plan in place launches each kernel.
    pub fn kernels(&self) -> BTreeMap<&'static str, usize> {
        let mut by = BTreeMap::new();
        for step in &self.runner.plan().steps {
            *by.entry(step.kernel.name()).or_default() += step.launches().len();
        }
        by
    }

    /// One launch of `rows`: the likeliest token after each.
    pub fn step(&mut self, rows: &[Row]) -> Result<Vec<u32>, Error> {
        if rows.is_empty() {
            return Err(Error::Size("the rows".into()));
        }
        if rows
            .iter()
            .any(|r| r.slot as usize >= self.slots || r.position as usize >= self.context)
        {
            return Err(Error::Size("a row's slot or position".into()));
        }
        let (sizes, feeds) = launch_of(rows, self.slots);
        let (plan, bound) = self
            .plans
            .upto(&self.graph, 0, &sizes, &[Context], &self.device)?;
        let out = plan.outputs[0].slot;
        self.runner.replan(plan)?;
        self.runner.bind_all(&bound);
        let views: Vec<_> = feeds.iter().map(Tensor::view).collect();
        self.runner.feed(&views)?;
        self.runner.submit(0..self.runner.plan().steps.len())?;
        self.runner.ids(out)
    }

    /// Decodes greedily after `request.tokens`, one event per token.
    pub fn generate(
        &mut self,
        request: &Request,
        emit: &mut dyn FnMut(Event),
    ) -> Result<(), Error> {
        if request.tokens.is_empty() || request.tokens.len() > self.context {
            return Err(Error::Size("the prompt".into()));
        }
        let mut past = 0;
        let mut next = request.tokens.clone();
        for made in 0..request.max_tokens {
            if past + next.len() >= self.context {
                emit(Event::Done(Reason::Context));
                return Ok(());
            }
            let row = |(i, token): (usize, &u32)| Row {
                slot: 0,
                token: *token,
                position: (past + i) as u32,
            };
            let rows: Vec<Row> = next.iter().enumerate().map(row).collect();
            let ids = self.step(&rows)?;
            let id = *ids.last().expect("a row per token");
            if request.stop.contains(&id) {
                emit(Event::Done(Reason::Stop));
                return Ok(());
            }
            emit(Event::Token(id));
            past += next.len();
            next = vec![id];
            if made + 1 == request.max_tokens {
                emit(Event::Done(Reason::Length));
            }
        }
        if request.max_tokens == 0 {
            emit(Event::Done(Reason::Length));
        }
        Ok(())
    }
}
