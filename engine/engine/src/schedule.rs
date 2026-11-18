//! Many answers in flight, one step feeding them all.

use crate::{Allowed, Engine, Error, Event, Reason, Request, Row};
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
    /// Tokens to feed: the prompt, then picked and forced tokens.
    pending: VecDeque<u32>,
    made: usize,
    /// What the next pick may choose from.
    next: Allowed,
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

    /// Whether more than one token waits, as with a prompt.
    pub fn prefilling(&self) -> bool {
        self.pending.len() > 1
    }

    /// Counts `n` tokens fed. `id` follows the last one.
    fn advance(&mut self, n: usize, id: u32) -> Option<Reason> {
        self.pending.drain(..n);
        self.fed += n;
        if !self.pending.is_empty() {
            return None;
        }
        if let Some(guide) = &mut self.request.guide
            && guide.accept(id).is_err()
        {
            return Some(Reason::Guide);
        }
        if self.request.stop.contains(&id) {
            return Some(Reason::Stop);
        }
        if let Some(reason) = self.emit(id) {
            return Some(reason);
        }
        self.pending.push_back(id);
        self.settle()
    }

    /// Sends an answer token.
    fn emit(&mut self, id: u32) -> Option<Reason> {
        self.made += 1;
        if !(self.sink)(Event::Token(id)) {
            return Some(Reason::Stop);
        }
        (self.made >= self.request.max_tokens).then_some(Reason::Length)
    }

    /// Adds forced and single-choice tokens, then reads the next set.
    fn settle(&mut self) -> Option<Reason> {
        loop {
            let guide = self.request.guide.as_mut()?;
            let forced = guide.forced();
            let done = guide.done();
            let next = match done {
                true => Ok(Allowed::Any),
                false => guide.allowed(),
            };
            let Ok(next) = next else {
                return Some(Reason::Guide);
            };
            let one = match next {
                Allowed::One(id) if guide.accept(id).is_err() => return Some(Reason::Guide),
                Allowed::One(id) => Some(id),
                _ => None,
            };
            for id in forced.into_iter().chain(one) {
                if self.request.stop.contains(&id) {
                    return Some(Reason::Stop);
                }
                if let Some(reason) = self.emit(id) {
                    return Some(reason);
                }
                self.pending.push_back(id);
            }
            if done {
                return Some(Reason::Stop);
            }
            if one.is_none() {
                self.next = next;
                return None;
            }
        }
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
            let mut seq = Seq {
                slot,
                pending: request.tokens.iter().copied().collect(),
                request,
                sink,
                fed: 0,
                made: 0,
                next: Allowed::Any,
            };
            match seq.settle() {
                Some(reason) => {
                    (seq.sink)(Event::Done(reason));
                    self.free.push(slot);
                }
                None => self.active.push(seq),
            }
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
            let seq = &self.active[i];
            if seq.fed + seq.pending() <= context {
                i += 1;
                continue;
            }
            (self.active[i].sink)(Event::Done(Reason::Context));
            self.finish(i);
        }
    }

    /// Each pick's allowed set, if any pick is limited.
    fn allowed(&self, rows: &[Row], ends: &[u32]) -> Option<Vec<Allowed>> {
        let mut out = vec![Allowed::Any; ends.len()];
        let mut limited = false;
        for seq in &self.active {
            if seq.next == Allowed::Any {
                continue;
            }
            let Some(k) = ends.iter().position(|&e| rows[e as usize].slot == seq.slot) else {
                continue;
            };
            // Only when the answer keeps this pick.
            let mine = rows.iter().filter(|r| r.slot == seq.slot).count();
            if mine == seq.pending.len() {
                out[k] = seq.next.clone();
                limited = true;
            }
        }
        limited.then_some(out)
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
        let ends = crate::ends(&rows);
        let ids = match self.allowed(&rows, &ends) {
            Some(allowed) => self.engine.last_guided(&rows, &allowed)?,
            None => self.engine.last(&rows)?,
        };
        // Each answer takes its rows and its last id.
        let mut i = 0;
        while i < self.active.len() {
            let slot = self.active[i].slot;
            let n = rows.iter().filter(|r| r.slot == slot).count();
            let mine = ends
                .iter()
                .zip(&ids)
                .find(|(e, _)| rows[**e as usize].slot == slot);
            match mine {
                None => i += 1,
                Some((_, &last)) => match self.active[i].advance(n, last) {
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
