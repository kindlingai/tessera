//! Many answers in flight, one step feeding them all.

use crate::guide::{Edit, edit};
use crate::kept::{Kept, Totals};
use crate::lanes::Lanes;
use crate::sample::Sampler;
use crate::{Allowed, Engine, Error, Event, Reason, Request, Row, ends};
use std::collections::VecDeque;

crate::metric_registry!(REQUESTS = "requests");
crate::metric!(WAITING in REQUESTS = Gauge "tessera:requests_waiting", "Answers waiting for a slot.");
crate::metric!(RUNNING in REQUESTS = Gauge "tessera:requests_running", "Answers that hold a slot.");
crate::metric!(PROMPT in REQUESTS = Counter "tessera:prompt_tokens_total", "Prompt tokens of admitted answers.");
crate::metric!(MADE in REQUESTS = Counter "tessera:generation_tokens_total", "Tokens the answers made.");

/// Takes each event. `false` ends the answer early.
pub type Sink = Box<dyn FnMut(Event) -> bool + Send>;

/// An answer in flight, in its slot.
pub struct Seq {
    pub slot: u32,
    request: Request,
    sink: Sink,
    /// Tokens fed: the prompt, then each answer token.
    fed: usize,
    /// The tokens of the fed rows.
    history: Vec<u32>,
    /// Tokens to feed: the prompt, then picked and forced tokens.
    pending: VecDeque<u32>,
    made: usize,
    /// Choices for the next pick.
    next: Allowed,
    /// Changes to the next pick's logits.
    edits: Vec<Edit>,
    /// Per-sequence token sampler.
    sampler: Sampler,
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
        self.history.extend(self.pending.drain(..n));
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
        MADE.add(1);
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
            let edits = match one.is_none() && !done {
                true => guide.edits(),
                false => Vec::new(),
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
                self.edits = edits;
                return None;
            }
        }
    }
}

/// Which rows the next step feeds, within a row budget.
pub trait Policy {
    fn pick(&mut self, seqs: &[Seq], budget: usize) -> Vec<Row>;
}

/// Decodes first, then prompts contiguously within the budget.
pub struct Fair;

impl Policy for Fair {
    fn pick(&mut self, seqs: &[Seq], budget: usize) -> Vec<Row> {
        let mut rows: Vec<Row> = seqs
            .iter()
            .filter(|s| !s.prefilling())
            .filter_map(|s| s.row(0))
            .collect();
        for s in seqs.iter().filter(|s| s.prefilling()) {
            let room = budget.saturating_sub(rows.len());
            if room == 0 {
                break;
            }
            for i in 0..room.min(s.pending()) {
                if let Some(row) = s.row(i) {
                    rows.push(row);
                }
            }
        }
        rows
    }
}

/// Many answers over one engine, a step at a time.
pub struct Batch<'a> {
    engine: &'a mut Engine,
    policy: Box<dyn Policy>,
    waiting: VecDeque<(Request, Sink)>,
    /// Active answers paused when the shared pool filled.
    paused: VecDeque<Seq>,
    active: Vec<Seq>,
    /// Admitted reads, run before the next step.
    reading: Vec<Seq>,
    free: Vec<u32>,
    /// Set when finished answers keep their rows.
    kept: Option<Kept>,
    /// The most rows one step feeds.
    budget: usize,
}

impl<'a> Batch<'a> {
    pub fn new(engine: &'a mut Engine, policy: Box<dyn Policy>) -> Self {
        let free = (0..engine.slots as u32).rev().collect();
        let shared = engine.lanes.as_ref().is_some_and(Lanes::shared);
        if let Some(lanes) = &mut engine.lanes {
            lanes.reset();
        }
        let budget = engine.context;
        let mut batch = Self {
            engine,
            policy,
            waiting: VecDeque::new(),
            paused: VecDeque::new(),
            active: Vec::new(),
            reading: Vec::new(),
            free,
            kept: None,
            budget,
        };
        if shared {
            batch.reuse();
        }
        batch
    }

    /// Finished answers keep rows. Set it before any step.
    pub fn reuse(&mut self) {
        let (slots, context) = (self.engine.slots, self.engine.context);
        self.kept = Some(match &self.engine.lanes {
            Some(lanes) => Kept::paged(slots, context, lanes.pages(), lanes.page),
            None => Kept::new(slots, context),
        });
    }

    /// The kept rows' totals. None without `reuse`.
    pub fn totals(&self) -> Option<Totals> {
        self.kept.as_ref().map(Kept::totals)
    }

    /// Queues an answer. An unfit prompt ends at once.
    pub fn submit(&mut self, request: Request, mut sink: Sink) {
        let long = request.tokens.len() > self.engine.context;
        let rows = request.reads.iter().map(|r| r.row);
        let past = rows.clone().any(|r| r as usize >= request.tokens.len());
        let sorted = rows.clone().zip(rows.skip(1)).all(|(a, b)| a < b);
        if request.tokens.is_empty() || long || past || !sorted {
            sink(Event::Done(Reason::Context));
            return;
        }
        if request.max_tokens == 0 && request.reads.is_empty() {
            sink(Event::Done(Reason::Length));
            return;
        }
        self.waiting.push_back((request, sink));
    }

    /// Whether any answer is queued or in flight.
    pub fn busy(&self) -> bool {
        !self.waiting.is_empty() || !self.paused.is_empty() || !self.active.is_empty()
    }

    /// A slot for the next answer, and its kept rows.
    fn place(&mut self) -> Option<(u32, usize)> {
        let request = &self.waiting.front()?.0;
        // A read feeds its first read row at least.
        let fed = request.reads.first().map(|r| r.row as usize);
        let limit = if self.engine.stateful() {
            0
        } else {
            fed.unwrap_or(request.tokens.len() - 1)
        };
        match &mut self.kept {
            Some(kept) => kept.admit(&request.tokens, limit),
            None => Some((self.free.pop()?, 0)),
        }
    }

    /// Resumes paused answers, then admits waiting ones.
    fn admit(&mut self) {
        while let Some(front) = self.paused.front() {
            let tokens: Vec<u32> = front.pending.iter().copied().collect();
            let limit = match self.engine.stateful() {
                true => 0,
                false => tokens.len().saturating_sub(1),
            };
            let placed = match &mut self.kept {
                Some(kept) => kept.admit(&tokens, limit),
                None => self.free.pop().map(|slot| (slot, 0)),
            };
            let Some((slot, skip)) = placed else {
                return;
            };
            let mut seq = self.paused.pop_front().expect("a paused answer");
            seq.slot = slot;
            seq.history = tokens[..skip].to_vec();
            seq.pending = tokens[skip..].iter().copied().collect();
            seq.fed = skip;
            self.active.push(seq);
        }
        while let Some((slot, skip)) = self.place() {
            let (request, sink) = self.waiting.pop_front().expect("a waiting answer");
            PROMPT.add(request.tokens.len() as u64);
            let sampler = Sampler::new(request.sampling);
            let mut seq = Seq {
                slot,
                pending: request.tokens[skip..].iter().copied().collect(),
                history: request.tokens[..skip].to_vec(),
                request,
                sink,
                fed: skip,
                made: 0,
                next: Allowed::Any,
                edits: Vec::new(),
                sampler,
            };
            if skip > 0 {
                (seq.sink)(Event::Kept(skip));
            }
            if !seq.request.reads.is_empty() {
                self.reading.push(seq);
                continue;
            }
            match seq.settle() {
                Some(reason) => {
                    (seq.sink)(Event::Done(reason));
                    self.retire(seq);
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

    /// Each pick's answer, and whether that answer keeps the pick.
    fn owners(&self, rows: &[Row], ends: &[u32]) -> Vec<(usize, bool)> {
        let mut out = Vec::with_capacity(ends.len());
        for &end in ends {
            let slot = rows[end as usize].slot;
            let at = self.active.iter().position(|s| s.slot == slot);
            let at = at.expect("an active slot");
            let mine = rows.iter().filter(|r| r.slot == slot).count();
            out.push((at, mine == self.active[at].pending.len()));
        }
        out
    }

    /// Each pick's allowed set, if any kept pick is limited.
    fn allowed(&self, owners: &[(usize, bool)]) -> Option<Vec<Allowed>> {
        let mut out = vec![Allowed::Any; owners.len()];
        let mut limited = false;
        for (k, &(at, keep)) in owners.iter().enumerate() {
            let next = &self.active[at].next;
            if keep && *next != Allowed::Any {
                out[k] = next.clone();
                limited = true;
            }
        }
        limited.then_some(out)
    }

    /// Gives each fed slot the pages its rows need.
    fn page(&mut self, rows: &[Row]) -> Result<(), Error> {
        let (Some(kept), Some(lanes)) = (&mut self.kept, &mut self.engine.lanes) else {
            return Ok(());
        };
        for r in rows {
            if let Some(pages) = kept.grow(r.slot, r.position as usize + 1)? {
                lanes.set(r.slot, pages);
            }
        }
        Ok(())
    }

    fn finish(&mut self, i: usize) {
        let seq = self.active.swap_remove(i);
        self.retire(seq);
    }

    /// Pauses the newest active answer to free shared pages.
    fn preempt(&mut self) -> bool {
        if self.active.len() <= 1 {
            return false;
        }
        let mut seq = self.active.pop().expect("an active answer");
        let mut all = std::mem::take(&mut seq.history);
        let kept_tokens = all.clone();
        all.extend(seq.pending.drain(..));
        match &mut self.kept {
            Some(kept) => kept.finish(seq.slot, kept_tokens),
            None => self.free.push(seq.slot),
        }
        seq.pending = all.into();
        seq.fed = 0;
        self.paused.push_front(seq);
        true
    }

    /// Frees an answer's slot, or keeps its rows.
    fn retire(&mut self, seq: Seq) {
        match &mut self.kept {
            Some(kept) => kept.finish(seq.slot, seq.history),
            None => self.free.push(seq.slot),
        }
    }

    /// One step of the policy's rows. False when nothing ran.
    pub fn step(&mut self) -> Result<bool, Error> {
        self.engine.settle()?;
        let ran = self.turn();
        WAITING.set(self.waiting.len() as u64);
        RUNNING.set(self.active.len() as u64);
        if let Some(kept) = &self.kept {
            kept.publish();
        }
        self.engine.settle()?;
        ran
    }

    /// Feeds each admitted read's rows, then sends its scores.
    fn read(&mut self) -> Result<bool, Error> {
        let ran = !self.reading.is_empty();
        while let Some(mut seq) = self.reading.pop() {
            let mut scores = Vec::with_capacity(seq.request.reads.len());
            while seq.pending() > 0 {
                let n = seq.pending().min(self.budget);
                let rows: Vec<Row> = (0..n).filter_map(|i| seq.row(i)).collect();
                self.page(&rows)?;
                let (mut picks, mut ids) = (Vec::new(), Vec::new());
                for r in &seq.request.reads[scores.len()..] {
                    let at = r.row as usize - seq.fed;
                    if at >= n {
                        break;
                    }
                    picks.push(at as u32);
                    ids.push(r.ids.clone());
                }
                match picks.is_empty() {
                    true => drop(self.engine.last(&rows)?),
                    false => scores.extend(self.engine.scores(&rows, &picks, &ids)?),
                }
                seq.history.extend(seq.pending.drain(..n));
                seq.fed += n;
            }
            (seq.sink)(Event::Scores(scores));
            (seq.sink)(Event::Done(Reason::Stop));
            self.retire(seq);
        }
        Ok(ran)
    }

    fn turn(&mut self) -> Result<bool, Error> {
        self.admit();
        let read = self.read()?;
        self.fit();
        let rows = loop {
            let rows = self.policy.pick(&self.active, self.budget);
            if rows.is_empty() {
                return Ok(read);
            }
            match self.page(&rows) {
                Ok(()) => break rows,
                Err(Error::Kept(tessera_kv::Error::NoRoom(_))) if self.preempt() => continue,
                Err(e) => return Err(e),
            }
        };
        let ends = ends(&rows);
        let owners = self.owners(&rows, &ends);
        let sampling = self.active.iter().any(|s| !s.request.sampling.greedy());
        let edited = owners
            .iter()
            .any(|&(at, keep)| keep && !self.active[at].edits.is_empty());
        let ids = match (sampling || edited, self.allowed(&owners)) {
            (true, allowed) => {
                let (mut logits, vocab) = self.engine.last_logits(&rows)?;
                let mut picked = Vec::with_capacity(ends.len());
                for (k, &(at, keep)) in owners.iter().enumerate() {
                    let seq = &mut self.active[at];
                    let row = &mut logits[k * vocab..(k + 1) * vocab];
                    if keep {
                        edit(row, &seq.edits);
                    }
                    let mask = allowed.as_ref().map(|a| &a[k]);
                    picked.push(seq.sampler.pick(row, mask));
                }
                picked
            }
            (false, Some(allowed)) => self.engine.last_guided(&rows, &allowed)?,
            (false, None) => self.engine.last(&rows)?,
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
