//! Many answers in flight, one step feeding them all.

use crate::{Engine, Event, Reason, Request, Row};
use lower::Error;
use std::collections::VecDeque;

/// Takes each event. `false` ends the answer early.
pub type Sink = Box<dyn FnMut(Event) -> bool + Send>;

/// An answer in flight, in its slot.
pub struct Seq {
    pub slot: u32,
    request: Request,
    sink: Sink,
    /// Tokens fed: the prompt, then each answer token.
    fed: usize,
    /// Tokens to feed. One at a time after the prompt.
    pending: VecDeque<u32>,
    made: usize,
}

impl Seq {
    /// The row feeding pending token `i`.
    pub fn row(&self, i: usize) -> Option<Row> {
        Some(Row {
            slot: self.slot,
            token: *self.pending.get(i)?,
            position: (self.fed + i) as u32,
        })
    }

    /// Tokens still to feed.
    pub fn pending(&self) -> usize {
        self.pending.len()
    }

    /// Whether the prompt is still going in.
    pub fn prefilling(&self) -> bool {
        self.fed < self.request.tokens.len()
    }

    /// Counts `n` tokens fed. `id` follows the last one.
    fn advance(&mut self, n: usize, id: u32) -> Option<Reason> {
        self.pending.drain(..n);
        self.fed += n;
        if !self.pending.is_empty() {
            return None;
        }
        if self.request.stop.contains(&id) {
            return Some(Reason::Stop);
        }
        self.made += 1;
        if !(self.sink)(Event::Token(id)) {
            return Some(Reason::Stop);
        }
        if self.made >= self.request.max_tokens {
            return Some(Reason::Length);
        }
        self.pending.push_back(id);
        None
    }
}

/// Which rows the next step feeds, within a row budget.
pub trait Policy {
    fn pick(&mut self, seqs: &[Seq], budget: usize) -> Vec<Row>;
}

/// Decodes first, then a prompt token each in turn.
pub struct Fair;

impl Policy for Fair {
    fn pick(&mut self, seqs: &[Seq], budget: usize) -> Vec<Row> {
        let decodes = seqs.iter().filter(|s| !s.prefilling());
        let mut rows: Vec<Row> = decodes.filter_map(|s| s.row(0)).collect();
        let mut prompts: Vec<(&Seq, usize)> = seqs
            .iter()
            .filter(|s| s.prefilling())
            .map(|s| (s, 0))
            .collect();
        loop {
            let mut any = false;
            for (s, fed) in &mut prompts {
                let Some(row) = s.row(*fed).filter(|_| rows.len() < budget) else {
                    continue;
                };
                rows.push(row);
                *fed += 1;
                any = true;
            }
            if !any {
                return rows;
            }
        }
    }
}

/// Many answers over one engine, a step at a time.
pub struct Batch<'a> {
    engine: &'a mut Engine,
    policy: Box<dyn Policy>,
    waiting: VecDeque<(Request, Sink)>,
    active: Vec<Seq>,
    free: Vec<u32>,
    /// The most rows one step feeds.
    budget: usize,
}

impl<'a> Batch<'a> {
    pub fn new(engine: &'a mut Engine, policy: Box<dyn Policy>) -> Self {
        let free = (0..engine.slots as u32).rev().collect();
        let budget = engine.context;
        Self {
            engine,
            policy,
            waiting: VecDeque::new(),
            active: Vec::new(),
            free,
            budget,
        }
    }

    /// Queues an answer. An unfit prompt ends at once.
    pub fn submit(&mut self, request: Request, mut sink: Sink) {
        let long = request.tokens.len() > self.engine.context;
        if request.tokens.is_empty() || long {
            sink(Event::Done(Reason::Context));
            return;
        }
        if request.max_tokens == 0 {
            sink(Event::Done(Reason::Length));
            return;
        }
        self.waiting.push_back((request, sink));
    }

    /// Whether any answer is queued or in flight.
    pub fn busy(&self) -> bool {
        !self.waiting.is_empty() || !self.active.is_empty()
    }

    /// Admits waiting answers while slots are free.
    fn admit(&mut self) {
        while let Some(slot) = self.free.pop() {
            let Some((request, sink)) = self.waiting.pop_front() else {
                self.free.push(slot);
                return;
            };
            self.active.push(Seq {
                slot,
                pending: request.tokens.iter().copied().collect(),
                request,
                sink,
                fed: 0,
                made: 0,
            });
        }
    }

    /// The most rows one step feeds.
    pub fn budget(&mut self, rows: usize) {
        self.budget = rows;
    }

    /// Ends each answer whose next row lies past the context.
    fn fit(&mut self) {
        let context = self.engine.context;
        let mut i = 0;
        while i < self.active.len() {
            if self.active[i].fed < context {
                i += 1;
                continue;
            }
            (self.active[i].sink)(Event::Done(Reason::Context));
            self.finish(i);
        }
    }

    fn finish(&mut self, i: usize) {
        let seq = self.active.swap_remove(i);
        self.free.push(seq.slot);
    }

    /// One step of the policy's rows. False when nothing ran.
    pub fn step(&mut self) -> Result<bool, Error> {
        self.admit();
        self.fit();
        let rows = self.policy.pick(&self.active, self.budget);
        if rows.is_empty() {
            return Ok(false);
        }
        let ids = self.engine.step(&rows)?;
        // Each answer takes its rows and the last id.
        let mut i = 0;
        while i < self.active.len() {
            let slot = self.active[i].slot;
            let mine = rows.iter().zip(&ids).filter(|(r, _)| r.slot == slot);
            let (n, last) = mine.fold((0, 0), |(n, _), (_, id)| (n + 1, *id));
            match n {
                0 => i += 1,
                _ => match self.active[i].advance(n, last) {
                    Some(reason) => {
                        (self.active[i].sink)(Event::Done(reason));
                        self.finish(i);
                    }
                    None => i += 1,
                },
            }
        }
        Ok(true)
    }
}
