//! Many answers in flight, one step feeding them all.

use crate::guide::{Edit, settle};
use crate::kept::{Kept, Totals};
use crate::lanes::Lanes;
use crate::ranks::ParkedProfile;
use crate::sample::{Sampler, Target, Weights, accept, take};
use crate::{Allowed, Engine, Error, Event, Launch, Reason, Request, Row, ends, launch_of};
use std::collections::VecDeque;
use tessera_graph::DimName::{Context, Segments};
use tessera_graph::Role;
use tessera_lower::{Report, Wants};

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

    /// Counts `n` tokens fed. `target` gives the token after them.
    fn advance(&mut self, n: usize, target: Target) -> Option<Reason> {
        self.history.extend(self.pending.drain(..n));
        self.fed += n;
        if !self.pending.is_empty() {
            return None;
        }
        let (_, id) = accept(&[], target);
        if let Some(guide) = &mut self.request.guide
            && guide.accept(id).is_err()
        {
            return Some(Reason::Guide);
        }
        if let Some(reason) = self.take(id) {
            return Some(reason);
        }
        self.pending.push_back(id);
        self.settle()
    }

    /// Takes an answer token and sends it.
    fn take(&mut self, id: u32) -> Option<Reason> {
        take(&self.request, &mut self.made, id, || {
            MADE.add(1);
            (self.sink)(Event::Token(id))
        })
    }

    /// Adds forced and single-choice tokens, then reads the next set.
    fn settle(&mut self) -> Option<Reason> {
        let mut guide = self.request.guide.take()?;
        let out = settle(&mut *guide, |id| {
            let reason = self.take(id);
            if reason.is_none() {
                self.pending.push_back(id);
            }
            reason
        });
        self.request.guide = Some(guide);
        match out {
            Ok((next, edits)) => {
                (self.next, self.edits) = (next, edits);
                None
            }
            Err(reason) => Some(reason),
        }
    }
}

/// One launch's result for each pick.
enum Picked {
    Ids(Vec<u32>),
    Drawn(Vec<Weights>, Vec<f32>),
}

impl Picked {
    /// The target of pick `k`.
    fn target(&self, k: usize) -> Target<'_> {
        match self {
            Picked::Ids(ids) => Target::Ids(&ids[k..=k]),
            Picked::Drawn(rows, units) => Target::Drawn(&rows[k..=k], &units[k..=k]),
        }
    }
}

/// Which rows the next step feeds, within a row budget.
pub trait Policy {
    fn pick(&mut self, seqs: &[Seq], budget: usize) -> Vec<Row>;

    /// The answer to pause when the shared pool fills.
    fn victim(&mut self, seqs: &[Seq]) -> Option<usize> {
        seqs.len().checked_sub(1)
    }
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

/// Launch row limit when `graph` uses sliding KV rings.
pub(crate) fn ring_chunk(graph: &tessera_graph::Graph) -> usize {
    let any = (0..graph.kept.len() as u32).any(|i| tessera_lower::sliding_ring(graph, i).is_some());
    if !any {
        return 0;
    }
    graph
        .dims
        .iter()
        .find(|d| d.name == tessera_graph::DimName::Rows)
        .map_or(tessera_lower::RING_CHUNK, |d| {
            (d.size.max() as usize).min(tessera_lower::RING_CHUNK)
        })
}

impl Engine {
    /// Device bytes the runner holds now.
    pub fn bytes(&self) -> u64 {
        self.runner.held().total()
    }

    /// Bytes the device reports. Zero when it reports none.
    pub fn memory(&self) -> u64 {
        self.runner.platform().memory
    }
}

/// Answers a batch gave up when its engine closed.
#[derive(Default)]
pub struct Parked {
    waiting: VecDeque<(Request, Sink)>,
    paused: VecDeque<Seq>,
    profile: Option<ParkedProfile>,
}

impl Drop for Parked {
    fn drop(&mut self) {
        if let Some(mut profile) = self.profile.take() {
            profile.flush();
        }
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
    /// Waiting answers stay waiting.
    closed: bool,
}

impl Engine {
    /// Rows one prefill step feeds: `asked`, else the default.
    pub fn chunk(&self, asked: Option<usize>) -> usize {
        let mut rows = asked.unwrap_or(tessera_device::FAST_MAX_ROWS);
        if self.ring_chunk > 0 {
            rows = rows.min(self.ring_chunk);
        }
        rows.min(self.context).max(1)
    }

    fn wants_full(&self) -> Wants {
        match self.taps.is_some() {
            true => Wants::roles(&self.graph, self.full, &[Role::Tokens, Role::Taps]),
            false => Wants::role(&self.graph, self.full, Role::Tokens),
        }
    }

    /// Fast-path diagnostic report for a `rows`-row decode launch.
    pub fn lowering_report(&mut self, rows: usize) -> Result<Report, Error> {
        self.explained(Launch::Decode(rows))
    }

    /// The report for one prompt's prefill step of `rows`.
    pub fn prefill_report(&mut self, rows: usize) -> Result<Report, Error> {
        self.explained(Launch::Prefill(rows))
    }

    fn explained(&mut self, launch: Launch) -> Result<Report, Error> {
        let view = crate::probe::Lowering {
            graphs: (&self.graph, &self.owned, &self.split),
            full: self.full,
            taps: self.taps.is_some(),
            slots: self.slots,
            context: self.context,
            device: &self.device,
        };
        Ok(view.explain(&mut self.plans, launch)?.report)
    }

    /// Lowers and compiles plans for `rows` rows.
    pub fn warm(&mut self, rows: usize, prefill: bool) -> Result<(), Error> {
        let at = if prefill { rows } else { 1 };
        self.warm_at(rows, prefill, at)
    }

    /// Lowers and compiles plans for `rows` rows at `context`.
    pub fn warm_at(&mut self, rows: usize, prefill: bool, context: usize) -> Result<(), Error> {
        let (sample, picks) =
            crate::probe::sample_at(rows, prefill, self.slots, self.context, context);
        let wants = self.wants_full();
        self.replan_for(&sample, &picks, wants)?;
        if prefill
            && sample.len() > 1
            && self.steps(&sample, &[], Wants::NONE)? <= self.steps(&sample, &picks, wants)?
        {
            self.replan_for(&sample, &[], Wants::NONE)?;
        }
        Ok(())
    }

    fn replan_for(&mut self, sample: &[Row], picks: &[u32], wants: Wants) -> Result<(), Error> {
        let (mut sizes, [filled, _]) = launch_of(sample, picks, self.slots, self.context);
        let upto = &[Context, Segments];
        let graph = crate::ranks::pick_graph(
            (&self.graph, &self.owned, &self.split),
            &self.device,
            true,
            filled,
            &mut sizes,
        );
        let (plan, _) = self
            .plans
            .wanting(graph, self.full, wants, &sizes, upto, &self.device)?;
        self.runner.replan(plan)?;
        Ok(())
    }
}

impl<'a> Batch<'a> {
    pub fn new(engine: &'a mut Engine, policy: Box<dyn Policy>) -> Self {
        let free = (0..engine.slots as u32).rev().collect();
        let shared = engine.lanes.as_ref().is_some_and(Lanes::shared);
        if let Some(lanes) = &mut engine.lanes {
            lanes.reset();
        }
        let budget = engine.chunk(None);
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
            closed: false,
        };
        if shared {
            batch.reuse();
        }
        batch
    }

    /// As `new`, with the answers another batch parked.
    pub fn resumed(engine: &'a mut Engine, policy: Box<dyn Policy>, mut parked: Parked) -> Self {
        if let Some(profile) = parked.profile.take() {
            engine.resume_profile(profile);
        }
        let mut batch = Self::new(engine, policy);
        batch.waiting = std::mem::take(&mut parked.waiting);
        batch.paused = std::mem::take(&mut parked.paused);
        batch
    }

    /// Pauses every answer. Gives them up with the queue.
    pub fn park(mut self) -> Result<Parked, Error> {
        let profile = self.engine.park_profile()?;
        while let Some(seq) = self.active.pop() {
            self.pause(seq);
        }
        Ok(Parked {
            waiting: self.waiting,
            paused: self.paused,
            profile,
        })
    }

    /// Whether waiting answers may start.
    pub fn admitting(&mut self, on: bool) {
        self.closed = !on;
    }

    /// Answers that hold a slot or lost one.
    pub fn running(&self) -> usize {
        self.active.len() + self.paused.len()
    }

    /// Device bytes the engine holds now.
    pub fn bytes(&self) -> u64 {
        self.engine.bytes()
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

    /// Slot, kept rows, and limit for next answer.
    fn place(&mut self) -> Option<(u32, usize, usize)> {
        let request = &self.waiting.front()?.0;
        // A read feeds its first read row at least.
        let fed = request.reads.first().map(|r| r.row as usize);
        let limit = if self.engine.stateful() {
            0
        } else {
            fed.unwrap_or(request.tokens.len() - 1)
        };
        let (slot, skip) = match &mut self.kept {
            Some(kept) => kept.admit(&request.tokens, limit)?,
            None => (self.free.pop()?, 0),
        };
        Some((slot, skip, limit))
    }

    /// Restores extra prefix pages from the KV store past `skip`.
    fn restore_disk(&mut self, slot: u32, tokens: &[u32], skip: usize) -> Result<usize, Error> {
        self.engine
            .restore_kv(self.kept.as_mut(), slot, tokens, skip, true)
    }

    /// Resumes paused answers, then admits waiting ones.
    fn admit(&mut self) -> Result<(), Error> {
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
                return Ok(());
            };
            let skip = self.restore_disk(slot, &tokens[..limit], skip)?;
            let mut seq = self.paused.pop_front().expect("a paused answer");
            seq.slot = slot;
            seq.history = tokens[..skip].to_vec();
            seq.pending = tokens[skip..].iter().copied().collect();
            seq.fed = skip;
            self.active.push(seq);
        }
        if self.closed {
            return Ok(());
        }
        while let Some((slot, skip, limit)) = self.place() {
            let (request, sink) = self.waiting.pop_front().expect("a waiting answer");
            let skip = self.restore_disk(slot, &request.tokens[..limit], skip)?;
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
        Ok(())
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
            let seq = &self.active[at];
            let hold = seq.made < seq.request.min_tokens && !seq.request.stop.is_empty();
            if keep && (hold || seq.next != Allowed::Any) {
                out[k] = if hold {
                    seq.next.ban(&seq.request.stop, self.engine.words())
                } else {
                    seq.next.clone()
                };
                limited = true;
            }
        }
        limited.then_some(out)
    }

    /// Gives each fed slot the pages its rows need.
    fn page(&mut self, rows: &[Row]) -> Result<(), Error> {
        let Some(kept) = &mut self.kept else {
            return Ok(());
        };
        self.engine.grow_kept(kept, rows)
    }

    fn finish(&mut self, i: usize) {
        let seq = self.active.swap_remove(i);
        self.retire(seq);
    }

    /// Pauses the policy's victim to free shared pages.
    fn preempt(&mut self) -> bool {
        if self.active.len() <= 1 {
            return false;
        }
        let victim = self.policy.victim(&self.active);
        let Some(at) = victim.filter(|at| *at < self.active.len()) else {
            return false;
        };
        let seq = self.active.remove(at);
        self.pause(seq);
        true
    }

    /// Queues an answer to feed again from its first token.
    fn pause(&mut self, mut seq: Seq) {
        let mut all = std::mem::take(&mut seq.history);
        let kept_tokens = all.clone();
        all.extend(seq.pending.drain(..));
        self.engine
            .spill_kv(self.kept.as_ref(), seq.slot, &kept_tokens, true);
        match &mut self.kept {
            Some(kept) => kept.finish(seq.slot, kept_tokens),
            None => self.free.push(seq.slot),
        }
        seq.pending = all.into();
        seq.fed = 0;
        self.paused.push_front(seq);
    }

    /// Frees an answer's slot, or keeps its rows.
    fn retire(&mut self, seq: Seq) {
        self.engine
            .spill_kv(self.kept.as_ref(), seq.slot, &seq.history, true);
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
                    true => self.engine.fill(&rows)?,
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
        self.admit()?;
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
        // Edits the device head cannot hold go to the host.
        let fits = owners
            .iter()
            .all(|&(at, keep)| !keep || self.engine.edits_fit(self.active[at].edits.len()));
        // Every pick is dropped, so skip the head.
        if owners.iter().all(|&(_, keep)| !keep) {
            self.engine.fill(&rows)?;
            for &(at, _) in &owners {
                let seq = &mut self.active[at];
                if sampling {
                    // Draws stay in step with launches that pick.
                    seq.sampler.unit();
                }
                let n = rows.iter().filter(|r| r.slot == seq.slot).count();
                seq.advance(n, Target::Ids(&[]));
            }
            return Ok(true);
        }
        let picked = match (sampling || (edited && !fits), edited, self.allowed(&owners)) {
            (true, _, allowed) => {
                let (mut logits, vocab) = self.engine.last_logits(&rows)?;
                let mut weights = Vec::with_capacity(ends.len());
                let mut units = Vec::with_capacity(ends.len());
                for (k, &(at, keep)) in owners.iter().enumerate() {
                    let seq = &mut self.active[at];
                    let row = &mut logits[k * vocab..(k + 1) * vocab];
                    let edits: &[Edit] = if keep { &seq.edits } else { &[] };
                    let mask = allowed.as_ref().map(|a| &a[k]);
                    weights.push(seq.sampler.weights(row, mask, edits));
                    units.push(seq.sampler.unit());
                }
                Picked::Drawn(weights, units)
            }
            (false, true, allowed) => {
                let allowed = allowed.unwrap_or_else(|| vec![Allowed::Any; ends.len()]);
                let mut edits: Vec<&[Edit]> = Vec::with_capacity(ends.len());
                for &(at, keep) in &owners {
                    edits.push(if keep { &self.active[at].edits } else { &[] });
                }
                Picked::Ids(self.engine.last_edited(&rows, &allowed, &edits)?)
            }
            (false, false, Some(allowed)) => Picked::Ids(self.engine.last_guided(&rows, &allowed)?),
            (false, false, None) => Picked::Ids(self.engine.last(&rows)?),
        };
        // Each answer takes its rows and its last pick.
        let mut i = 0;
        while i < self.active.len() {
            let slot = self.active[i].slot;
            let n = rows.iter().filter(|r| r.slot == slot).count();
            let mine = ends.iter().position(|e| rows[*e as usize].slot == slot);
            match mine {
                None => i += 1,
                Some(k) => match self.active[i].advance(n, picked.target(k)) {
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
