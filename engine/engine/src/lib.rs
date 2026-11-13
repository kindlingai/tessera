//! Token ids in, token ids out, over a stepper graph.

// Linking the crates registers their rules.
use kernels as _;
use linear as _;

use graph::DimName::{Context, Rows, Seqs};
use graph::Graph;
use interp::Tensor;
use lower::{Cache, Device, Error, Pins, Run, Settings, open};
use serde::{Deserialize, Serialize};

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
}

/// One launch: `tokens` after `past` rows of one sequence.
struct Span<'a> {
    past: usize,
    tokens: &'a [u32],
}

impl Span<'_> {
    fn sizes(&self) -> [(graph::DimName, u64); 3] {
        let all = (self.past + self.tokens.len()) as u64;
        [(Seqs, 1), (Rows, self.tokens.len() as u64), (Context, all)]
    }

    fn feeds(&self) -> [Tensor; 2] {
        let shape = vec![1, self.tokens.len()];
        let from = self.past as u32;
        let positions = (from..from + self.tokens.len() as u32).collect();
        let tokens = Tensor::u32(shape.clone(), self.tokens.to_vec());
        [tokens, Tensor::u32(shape, positions)]
    }
}

impl Engine {
    /// Opens a runner for `graph` with room for `context` rows.
    pub fn open(
        graph: Graph,
        context: usize,
        pins: Pins,
        settings: &Settings,
    ) -> Result<Self, Error> {
        let device = Device::guess(settings).pinned(pins.clone());
        let mut plans = Cache::default();
        let first = Span {
            past: 0,
            tokens: &[0],
        };
        let plan = plans.lower(&graph, 0, &first.sizes(), &device)?;
        let runner = open(plan, settings)?;
        let device = Device::stored(&*runner, settings)?.pinned(pins);
        Ok(Self {
            graph,
            runner,
            device,
            plans,
            context,
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

    /// Runs one launch. Returns the likeliest token after each row.
    fn launch(&mut self, span: Span) -> Result<Vec<u32>, Error> {
        let sizes = span.sizes();
        let (plan, bound) = self
            .plans
            .upto(&self.graph, 0, &sizes, &[Context], &self.device)?;
        let out = plan.outputs[0].slot;
        self.runner.replan(plan)?;
        self.runner.bind_all(&bound);
        let [tokens, positions] = span.feeds();
        self.runner.feed(&[tokens.view(), positions.view()])?;
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
            let span = Span {
                past,
                tokens: &next,
            };
            let ids = self.launch(span)?;
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
