//! DFlash speculative decoding: cached drafter and verify loop.

use crate::{Engine, Error, Event, Reason, Request, Row};
use tessera_graph::DimName::{Context, Embedding, Rows, Segments, Seqs, Vocab};
use tessera_graph::{Graph, Role};
use tessera_interp::Tensor;
use tessera_lower::{Cache, Device, Recipe, Settings, Slot, Wants, Weight};
use tessera_run::{Run, open};

/// Speculative decoding telemetry for one generation call.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpecStats {
    /// Target verification launches run.
    pub steps: usize,
    /// Draft tokens proposed across verify steps.
    pub drafted: usize,
    /// Draft tokens accepted by the target model.
    pub accepted: usize,
    /// Total tokens emitted to the caller.
    pub emitted: usize,
}

impl SpecStats {
    /// Average tokens produced per speculative verify step.
    pub fn mean_step(&self) -> f32 {
        if self.steps == 0 {
            return 0.0;
        }
        (self.accepted + self.steps) as f32 / self.steps as f32
    }
}

/// Counts matching draft prefix and returns the bonus token.
pub fn accept_greedy(drafts: &[u32], posterior: &[u32]) -> (usize, u32) {
    assert_eq!(posterior.len(), drafts.len() + 1);
    let mut accepted = 0;
    while accepted < drafts.len() && drafts[accepted] == posterior[accepted] {
        accepted += 1;
    }
    (accepted, posterior[accepted])
}

/// Rejection-samples a draft block and returns accepted plus bonus.
pub fn accept_sample(
    drafts: &[u32],
    draft_probs: &[&[f32]],
    target_probs: &[&[f32]],
    uniforms: &[f32],
) -> (usize, u32) {
    let n = drafts.len();
    assert_eq!(draft_probs.len(), n);
    assert_eq!(target_probs.len(), n + 1);
    assert_eq!(uniforms.len(), n + 1);
    for i in 0..n {
        let token = drafts[i] as usize;
        let p = target_probs[i][token].max(0.0);
        let q = draft_probs[i][token].max(0.0);
        if uniforms[i] * q > p {
            let bonus = sample_residual(target_probs[i], draft_probs[i], uniforms[n]);
            return (i, bonus);
        }
    }
    (n, sample_dist(target_probs[n], uniforms[n]))
}

fn sample_dist(probs: &[f32], u: f32) -> u32 {
    let mut total = 0.0f32;
    for &p in probs {
        total += p.max(0.0);
    }
    let threshold = u.clamp(0.0, 1.0) * total;
    let mut acc = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        acc += p.max(0.0);
        if acc >= threshold {
            return i as u32;
        }
    }
    probs.len().saturating_sub(1) as u32
}

fn sample_residual(target: &[f32], draft: &[f32], u: f32) -> u32 {
    let mut total = 0.0f32;
    for (&p, &q) in target.iter().zip(draft) {
        total += (p - q).max(0.0);
    }
    if total <= 0.0 {
        return sample_dist(target, u);
    }
    let threshold = u.clamp(0.0, 1.0) * total;
    let mut acc = 0.0f32;
    for (i, (&p, &q)) in target.iter().zip(draft).enumerate() {
        acc += (p - q).max(0.0);
        if acc >= threshold {
            return i as u32;
        }
    }
    target.len().saturating_sub(1) as u32
}

/// Runs a DFlash drafter graph with kept context KV.
pub struct Drafter {
    graph: Graph,
    runner: Box<dyn Run>,
    device: Device,
    plans: Cache,
    context: usize,
    slots: usize,
    block: usize,
    tap_width: usize,
    vocab: usize,
    mask_token: u32,
}

impl Drafter {
    /// Opens a compiled runner for a DFlash stepper graph.
    pub fn open(
        graph: Graph,
        context: usize,
        block: usize,
        mask_token: u32,
        recipe: &Recipe,
        settings: &Settings,
    ) -> Result<Self, Error> {
        if block < 2 || block > context {
            return Err(Error::Rows);
        }
        tessera_run::configure(settings);
        let dim = |name| {
            graph
                .dims
                .iter()
                .find(|d| d.name == name)
                .map(|d| d.size.max() as usize)
        };
        let slots = dim(Seqs).unwrap_or(1);
        let tap_width = dim(Embedding).ok_or(Error::Taps)?;
        let vocab = dim(Vocab).ok_or(Error::Scores)?;
        let device = Device::guess(settings).with(recipe);
        let mut plans = Cache::default();
        let first = [
            (Seqs, slots as u64),
            (Rows, 1),
            (Context, block as u64),
            (Segments, 1),
        ];
        let wants = Wants::role(&graph, 0, Role::Tokens);
        let (plan, _) = plans.wanting(&graph, 0, wants, &first, &[Context, Segments], &device)?;
        let mut runner = open(plan.clone(), settings)?;
        runner.launching(recipe.launch);
        runner.profiling(settings.profile.is_some());
        let device = Device::stored(runner.platform(), settings)
            .map_err(tessera_lower::Error::from)?
            .with(recipe);
        Ok(Self {
            graph,
            runner,
            device,
            plans,
            context,
            slots,
            block,
            tap_width,
            vocab,
            mask_token,
        })
    }

    /// Uploads the drafter's weights, in graph order.
    pub fn load(&mut self, weights: &[Weight]) -> Result<(), Error> {
        for (at, w) in weights.iter().enumerate() {
            self.load_weight(at as u32, w)?;
        }
        Ok(())
    }

    /// Uploads weight `at` of the drafter graph.
    pub fn load_weight(&mut self, at: u32, w: &Weight) -> Result<(), Error> {
        Ok(self.runner.load_weight(at, w)?)
    }

    pub fn block(&self) -> usize {
        self.block
    }

    pub fn context(&self) -> usize {
        self.context
    }

    pub fn tap_width(&self) -> usize {
        self.tap_width
    }

    pub fn mask_token(&self) -> u32 {
        self.mask_token
    }

    pub fn device(&self) -> String {
        self.runner.device()
    }

    /// Proposes `block - 1` greedy draft tokens after `anchor`.
    pub fn draft(
        &mut self,
        slot: u32,
        anchor: u32,
        start: usize,
        ctx_pos: usize,
        taps: &[f32],
    ) -> Result<Vec<u32>, Error> {
        let wants = Wants::role(&self.graph, 0, Role::Tokens);
        let outputs = self.run(slot, anchor, start, ctx_pos, taps, wants)?;
        let mut ids = self.runner.ids(outputs[0])?;
        ids.truncate(self.block - 1);
        Ok(ids)
    }

    /// Computes `(block - 1) * vocab` draft logits after `anchor`.
    pub fn draft_logits(
        &mut self,
        slot: u32,
        anchor: u32,
        start: usize,
        ctx_pos: usize,
        taps: &[f32],
    ) -> Result<Vec<f32>, Error> {
        let wants = Wants::role(&self.graph, 0, Role::Logits);
        let outputs = self.run(slot, anchor, start, ctx_pos, taps, wants)?;
        let raw = self.runner.read(outputs[0])?;
        let logits = raw.floats().map_err(tessera_run::Error::from)?;
        Ok(logits[..(self.block - 1) * self.vocab].to_vec())
    }

    fn run(
        &mut self,
        slot: u32,
        anchor: u32,
        start: usize,
        ctx_pos: usize,
        taps: &[f32],
        wants: Wants,
    ) -> Result<Vec<Slot>, Error> {
        let n_new = start.checked_sub(ctx_pos).ok_or(Error::RowPlace)?;
        if n_new == 0 || taps.len() != n_new * self.tap_width {
            return Err(Error::Rows);
        }
        if slot as usize >= self.slots || start + self.block > self.context {
            return Err(Error::RowPlace);
        }
        let filled = if n_new <= self.block {
            self.block
        } else {
            n_new.next_power_of_two().min(self.context)
        };
        let mut ctx_taps = Vec::with_capacity(filled * self.tap_width);
        ctx_taps.extend_from_slice(taps);
        let last = &taps[(n_new - 1) * self.tap_width..];
        for _ in n_new..filled {
            ctx_taps.extend_from_slice(last);
        }
        let kv_len = filled + self.block;
        let mut kv_pos = Vec::with_capacity(kv_len);
        for p in ctx_pos..start {
            kv_pos.push(p as u32);
        }
        for _ in n_new..filled {
            kv_pos.push((start - 1) as u32);
        }
        for p in start..start + self.block {
            kv_pos.push(p as u32);
        }
        let mut block_tokens = vec![self.mask_token; self.block];
        block_tokens[0] = anchor;
        let block_pos: Vec<u32> = (start..start + self.block).map(|p| p as u32).collect();
        let block_slots = vec![slot; self.block];
        let attend_pos = vec![(start + self.block - 1) as u32; self.block];
        let kv_slots = vec![slot; kv_len];
        let feeds = [
            Tensor::u32(vec![self.block], block_tokens),
            Tensor::u32(vec![self.block], block_pos),
            Tensor::u32(vec![self.block], block_slots),
            Tensor::u32(vec![self.block], attend_pos),
            Tensor::f32(vec![filled, self.tap_width], ctx_taps),
            Tensor::u32(vec![kv_len], kv_pos),
            Tensor::u32(vec![kv_len], kv_slots),
        ];
        let sizes = [
            (Seqs, self.slots as u64),
            (Rows, filled as u64),
            (Context, (start + self.block) as u64),
            (Segments, 1),
        ];
        let (plan, bound) = self.plans.wanting(
            &self.graph,
            0,
            wants,
            &sizes,
            &[Context, Segments],
            &self.device,
        )?;
        let outputs = plan.outputs.iter().map(|o| o.slot).collect();
        self.runner.replan(plan)?;
        self.runner.bind_all(&bound);
        let views: Vec<_> = feeds.iter().map(Tensor::view).collect();
        self.runner.feed(&views)?;
        self.runner.submit(0..self.runner.plan().steps.len())?;
        Ok(outputs)
    }
}

/// DFlash speculative decoding over a target and drafter pair.
pub struct Speculate<'a> {
    target: &'a mut Engine,
    drafter: &'a mut Drafter,
}

impl<'a> Speculate<'a> {
    /// Pairs a tapped target engine with its DFlash drafter.
    pub fn new(target: &'a mut Engine, drafter: &'a mut Drafter) -> Result<Self, Error> {
        let Some(width) = target.tap_width() else {
            return Err(Error::Taps);
        };
        if width != drafter.tap_width() {
            return Err(Error::Rows);
        }
        Ok(Self { target, drafter })
    }

    /// Greedy speculative decoding matching target output token-for-token.
    pub fn generate(
        &mut self,
        request: Request,
        emit: &mut dyn FnMut(Event) -> bool,
    ) -> Result<SpecStats, Error> {
        let mut stats = SpecStats::default();
        let prompt = &request.tokens;
        if prompt.is_empty() || prompt.len() > self.target.context() {
            let _ = emit(Event::Done(Reason::Context));
            return Ok(stats);
        }
        if request.max_tokens == 0 {
            let _ = emit(Event::Done(Reason::Length));
            return Ok(stats);
        }
        let prefill: Vec<Row> = prompt
            .iter()
            .enumerate()
            .map(|(i, &token)| Row {
                slot: 0,
                token,
                position: i as u32,
            })
            .collect();
        let (ids, mut pending_taps) = self.target.step_taps(&prefill)?;
        let mut anchor = *ids.last().expect("non-empty prompt");
        if let Some(reason) = emit_one(&request, &mut stats, anchor, emit) {
            let _ = emit(Event::Done(reason));
            return Ok(stats);
        }
        let mut start = prompt.len();
        let mut ctx_pos = 0usize;
        let block = self.drafter.block();
        let tap_width = self.drafter.tap_width();
        let max_ctx = self.target.context().min(self.drafter.context());
        while stats.emitted < request.max_tokens {
            if start >= self.target.context() {
                let _ = emit(Event::Done(Reason::Context));
                return Ok(stats);
            }
            let remaining = request.max_tokens - stats.emitted;
            if remaining == 1 || start + block > max_ctx {
                let row = [Row {
                    slot: 0,
                    token: anchor,
                    position: start as u32,
                }];
                let next = self.target.step(&row)?[0];
                start += 1;
                if let Some(reason) = emit_one(&request, &mut stats, next, emit) {
                    let _ = emit(Event::Done(reason));
                    return Ok(stats);
                }
                anchor = next;
                continue;
            }
            let drafts = self
                .drafter
                .draft(0, anchor, start, ctx_pos, &pending_taps)?;
            let mut verify = Vec::with_capacity(block);
            verify.push(Row {
                slot: 0,
                token: anchor,
                position: start as u32,
            });
            for (i, &token) in drafts.iter().enumerate() {
                verify.push(Row {
                    slot: 0,
                    token,
                    position: (start + 1 + i) as u32,
                });
            }
            let (posterior, block_taps) = self.target.step_taps(&verify)?;
            let (accepted, bonus) = accept_greedy(&drafts, &posterior);
            stats.steps += 1;
            stats.drafted += drafts.len();
            stats.accepted += accepted;
            for id in drafts[..accepted].iter().copied().chain([bonus]) {
                if let Some(reason) = emit_one(&request, &mut stats, id, emit) {
                    let _ = emit(Event::Done(reason));
                    return Ok(stats);
                }
            }
            let commit = 1 + accepted;
            ctx_pos = start;
            pending_taps = block_taps[..commit * tap_width].to_vec();
            start += commit;
            anchor = bonus;
        }
        let _ = emit(Event::Done(Reason::Length));
        Ok(stats)
    }
}

/// Emits one token, returning the stop reason if finished.
fn emit_one(
    request: &Request,
    stats: &mut SpecStats,
    id: u32,
    emit: &mut dyn FnMut(Event) -> bool,
) -> Option<Reason> {
    if request.stop.contains(&id) {
        return Some(Reason::Stop);
    }
    stats.emitted += 1;
    if !emit(Event::Token(id)) {
        return Some(Reason::Stop);
    }
    (stats.emitted >= request.max_tokens).then_some(Reason::Length)
}
