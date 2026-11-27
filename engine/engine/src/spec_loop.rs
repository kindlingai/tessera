//! Shared SPMD speculative decode loop and device tap helpers.

use crate::guide::{MinTokens, Rules, Stacked, settle, walk};
use crate::lanes::{decode_patches, encode_patches};
use crate::sample::{Sampler, Sampling, Target, accept, take};
use crate::spec_len::SpecStats;
use crate::{Allowed, Edit, Engine, Error, Event, Guide, Head, Reason, Request, Row};
use tessera_engine_comm::{Reader, Writer};
use tessera_graph::Role;
use tessera_lower::{Slot, Wants};
use tessera_run::{Error as RunError, Profile, Run};

/// Wire tag identifying an SPMD speculative request header.
const SPEC_REQ_TAG: u32 = 1;

/// Rounds between cross-rank FNV-1a determinism and stop checks.
const SYNC_EVERY: usize = 64;

/// Tapped states on device or host, row-major f32.
pub(crate) enum TapsBuf {
    Device { slot: Slot, addr: u64, rows: usize },
    Host(Vec<f32>),
}

/// Borrowed tap rows on device or host.
#[derive(Clone, Copy)]
pub(crate) enum TapsIn<'a> {
    Device {
        addr: u64,
        rows: usize,
        stride: usize,
    },
    Host(&'a [f32]),
}

/// Device-to-device copy descriptor for one drafter input span.
#[derive(Clone, Copy)]
pub(crate) struct D2dCopy {
    pub input: usize,
    pub src: u64,
    pub dst_row: usize,
    pub rows: usize,
    pub pad_to: usize,
    pub width: usize,
}

impl TapsBuf {
    pub(crate) fn as_in(&self) -> TapsIn<'_> {
        self.as_in_from(0, 0)
    }

    pub(crate) fn as_in_from(&self, skip: usize, width: usize) -> TapsIn<'_> {
        match self {
            TapsBuf::Device { addr, rows, .. } => {
                let (s, off) = (skip.min(*rows), (skip.min(*rows) * width * 4) as u64);
                TapsIn::Device {
                    addr: *addr + off,
                    rows: *rows - s,
                    stride: 0,
                }
            }
            TapsBuf::Host(v) => TapsIn::Host(&v[(skip * width).min(v.len())..]),
        }
    }

    pub(crate) fn keep_rows(&mut self, keep: usize, width: usize) {
        match self {
            TapsBuf::Device { rows, .. } => *rows = keep,
            TapsBuf::Host(v) => v.truncate(keep * width),
        }
    }

    pub(crate) fn into_host(self, runner: &dyn Run, width: usize) -> Result<Vec<f32>, Error> {
        match self {
            TapsBuf::Host(v) => Ok(v),
            TapsBuf::Device { slot, rows, .. } => {
                let raw = runner.read(slot)?;
                let floats = raw.floats().map_err(RunError::from)?;
                Ok(floats[..rows * width].to_vec())
            }
        }
    }
}

/// Copies `rows` at `dst_row` and pads tail to `pad_to`.
pub(crate) fn fill_d2d(runner: &mut dyn Run, dst: Slot, copy: D2dCopy) -> Result<(), Error> {
    let row_bytes = copy.width * 4;
    runner.copy_from(
        dst,
        copy.dst_row * row_bytes,
        copy.src,
        copy.rows * row_bytes,
    )?;
    let end = copy.dst_row + copy.rows;
    if copy.pad_to > end {
        let base = runner.address(dst).ok_or(RunError::Unaddressed(0))?;
        let tail = end - 1;
        let tail_addr = base + (tail * row_bytes) as u64;
        let mut have = 1usize;
        while tail + have < copy.pad_to {
            let chunk = have.min(copy.pad_to - (tail + have));
            runner.copy_from(dst, (tail + have) * row_bytes, tail_addr, chunk * row_bytes)?;
            have += chunk;
        }
    }
    Ok(())
}

pub(crate) fn fnv_u32(h: &mut u32, v: u32) {
    for b in v.to_le_bytes() {
        *h = (*h ^ b as u32).wrapping_mul(16777619);
    }
}

/// Checks FNV-1a determinism across ranks and propagates early stop.
fn sync_ranks(target: &mut Engine, stopped: bool, hash: u32) -> Result<bool, Error> {
    let Some(control) = &mut target.control else {
        return Ok(stopped);
    };
    let lead = control.rank() == 0;
    let mut mine = Vec::with_capacity(5);
    mine.push(u8::from(lead && stopped));
    mine.extend_from_slice(&hash.to_le_bytes());
    if !lead && let Some((_, p)) = target.held.take() {
        mine.extend_from_slice(&p.to_bytes());
    }
    let parts = control.exchange(&mine).map_err(RunError::from)?;
    let first = parts.first().ok_or(RunError::OutputsDisagree)?;
    if first.len() < 5 {
        return Err(RunError::OutputsDisagree.into());
    }
    let stop = first[0] != 0;
    let want = u32::from_le_bytes(first[1..5].try_into().unwrap());
    for p in &parts[1..] {
        if p.len() < 5 || u32::from_le_bytes(p[1..5].try_into().unwrap()) != want {
            return Err(RunError::OutputsDisagree.into());
        }
    }
    if lead {
        target.gather(parts[1..].iter().map(|p| Profile::from_bytes(&p[5..])));
    }
    Ok(stop)
}

/// Speculative request paired with an optional prefill chunk limit.
pub(crate) struct SpecReq {
    pub request: Request,
    pub chunk: Option<usize>,
}

/// Maps slot zero for a shared paged pool.
pub(crate) fn solo_cap(target: &mut Engine) -> usize {
    let ctx = target.context();
    match &mut target.lanes {
        Some(lanes) if lanes.shared() => ctx.min(lanes.map_solo()),
        Some(lanes) => ctx.min(lanes.pages() * lanes.page),
        None => ctx,
    }
}

/// Validates a request and broadcasts its header once to followers.
pub(crate) fn start_req(
    target: &mut Engine,
    spec_req: &mut SpecReq,
    emit: &mut dyn FnMut(Event) -> bool,
    extra: impl FnOnce(&mut Writer),
) -> Result<bool, Error> {
    spec_req.request.sampling = spec_req.request.sampling.seeded();
    let request = &spec_req.request;
    if (request.guide.is_some() || request.min_tokens > 0) && target.control.is_some() {
        return Err(Error::Guided);
    }
    let cap = solo_cap(target);
    if request.tokens.is_empty() || request.tokens.len() > cap {
        let _ = emit(Event::Done(Reason::Context));
        return Ok(false);
    }
    if request.max_tokens == 0 {
        let _ = emit(Event::Done(Reason::Length));
        return Ok(false);
    }
    if target.control.as_ref().is_some_and(|c| c.rank() == 0) {
        let patches = target
            .lanes
            .as_ref()
            .map(crate::lanes::Lanes::pending)
            .unwrap_or_default();
        let mut w = Writer(Vec::new());
        w.u32(SPEC_REQ_TAG)
            .u32(request.max_tokens as u32)
            .u32(spec_req.chunk.unwrap_or(0) as u32)
            .u32s(&request.tokens)
            .u32s(&request.stop);
        encode_patches(&mut w, &patches);
        request.sampling.encode(&mut w);
        extra(&mut w);
        target.told(&w.0)?;
    }
    Ok(true)
}

type DecodedReq<T> = (SpecReq, Vec<(u32, Vec<u32>)>, T);

fn decode_req<T>(
    bytes: &[u8],
    extra: impl FnOnce(&mut Reader) -> Option<T>,
) -> Option<DecodedReq<T>> {
    let mut r = Reader(bytes);
    if r.u32()? != SPEC_REQ_TAG {
        return None;
    }
    let max_tokens = r.u32()? as usize;
    let chunk = r.u32()? as usize;
    let tokens = r.u32s()?;
    let stop = r.u32s()?;
    let patches = decode_patches(&mut r)?;
    let sampling = Sampling::decode(&mut r)?;
    let policy = extra(&mut r)?;
    let req = SpecReq {
        request: Request {
            tokens,
            max_tokens,
            stop,
            sampling,
            ..Request::default()
        },
        chunk: (chunk > 0).then_some(chunk),
    };
    Some((req, patches, policy))
}

/// Shared follower outer loop receiving SPMD speculative requests.
pub(crate) fn follow_loop<T>(
    target: &mut Engine,
    mut decode: impl FnMut(&mut Reader) -> Option<T>,
    mut run: impl FnMut(&mut Engine, SpecReq, T) -> Result<(), Error>,
) -> Result<(), Error> {
    loop {
        let mine = target
            .held
            .take()
            .map_or_else(Vec::new, |(_, p)| p.to_bytes());
        let Some(told) = target.told(&mine)? else {
            return Ok(());
        };
        let Some((req, patches, policy)) = decode_req(&told, &mut decode) else {
            return Ok(());
        };
        if let Some(lanes) = &mut target.lanes
            && !patches.is_empty()
        {
            lanes.apply(&patches);
        }
        run(target, req, policy)?;
    }
}

macro_rules! spec_entry {
    (
        $ty:ty,
        $policy:ty,
        $field:ident,
        |$this:ident, $target:ident, $req:ident, $p:ident, $emit:ident| $body:expr
    ) => {
        impl<'a> $ty {
            /// Sets the maximum prompt rows per prefill step.
            pub fn with_prefill_chunk(mut self, rows: usize) -> Self {
                self.prefill_chunk = Some(rows.max(1));
                self
            }

            /// Starts or flushes dynamic profiling on the target engine.
            pub fn settle(&mut self) -> Result<(), $crate::Error> {
                self.target.settle()
            }

            /// Follows rank 0's speculative requests until dismissed.
            pub fn follow(&mut self) -> Result<(), $crate::Error> {
                let $this = self;
                $crate::spec_loop::follow_loop(
                    $this.target,
                    <$policy>::decode,
                    |$target, $req, $p| {
                        let $emit: &mut dyn FnMut($crate::Event) -> bool = &mut |_| true;
                        $body?;
                        Ok(())
                    },
                )
            }

            /// Greedy speculative decoding matching target output token-for-token.
            pub fn generate(
                &mut self,
                request: $crate::Request,
                emit: &mut dyn FnMut($crate::Event) -> bool,
            ) -> Result<$crate::SpecStats, $crate::Error> {
                let $this = self;
                let $p = $this.$field.fresh();
                let mut $req = $crate::spec_loop::SpecReq {
                    request,
                    chunk: $this.prefill_chunk,
                };
                $this.target.settle()?;
                if !$crate::spec_loop::start_req($this.target, &mut $req, emit, |w| $p.encode(w))? {
                    $this.target.settle()?;
                    return Ok($crate::SpecStats::default());
                }
                let ($target, $emit) = (&mut *$this.target, emit);
                let stats = $body;
                $target.settle()?;
                stats
            }
        }
    };
}
pub(crate) use spec_entry;

struct LoopState {
    stats: SpecStats,
    sim_emitted: usize,
    stopped: bool,
    hash: u32,
}

/// Appends rows for `tokens` at positions from `start`.
pub(crate) fn span(
    rows: &mut Vec<Row>,
    slot: u32,
    start: usize,
    tokens: impl IntoIterator<Item = u32>,
) {
    for (i, token) in tokens.into_iter().enumerate() {
        rows.push(Row {
            slot,
            token,
            position: (start + i) as u32,
        });
    }
}

fn prompt_rows(tokens: &[u32], pos: usize) -> Vec<Row> {
    let mut rows = Vec::with_capacity(tokens.len());
    span(&mut rows, 0, pos, tokens.iter().copied());
    rows
}

fn accept_token(
    request: &Request,
    st: &mut LoopState,
    id: u32,
    lead: bool,
    emit: &mut dyn FnMut(Event) -> bool,
) -> Option<Reason> {
    let LoopState { stats, stopped, .. } = st;
    take(request, &mut st.sim_emitted, id, || {
        if lead && !*stopped {
            stats.emitted += 1;
            if !emit(Event::Token(id)) {
                *stopped = true;
                let _ = emit(Event::Done(Reason::Stop));
            }
        }
        true
    })
}

/// Shared SPMD greedy speculative decode loop for all drafters.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_speculate<D, S>(
    target: &mut Engine,
    drafter: &mut D,
    spec_req: SpecReq,
    emit: &mut dyn FnMut(Event) -> bool,
    mut on_chunk: impl FnMut(&mut Engine, &mut D, usize, &[u32], usize, TapsBuf) -> Result<(), Error>,
    init: impl FnOnce(&mut Engine, &mut D, usize, &[u32], u32, TapsBuf) -> Result<S, Error>,
    mut draft: impl FnMut(
        &mut Engine,
        &mut D,
        &mut S,
        usize,
        u32,
        usize,
    ) -> Result<Option<Vec<u32>>, Error>,
    mut advance: impl FnMut(
        &mut Engine,
        &mut D,
        &mut S,
        usize,
        usize,
        usize,
        &[u32],
        TapsBuf,
        &mut SpecStats,
    ) -> Result<(), Error>,
) -> Result<SpecStats, Error> {
    let ranked = target.control.is_some();
    let lead = target.control.as_ref().is_none_or(|c| c.rank() == 0);
    let mut st = LoopState {
        stats: SpecStats::default(),
        sim_emitted: 0,
        stopped: false,
        hash: 2166136261,
    };
    let mut request = spec_req.request;
    let mut sampler = Sampler::new(request.sampling);
    let sampling = !sampler.greedy();
    let mut prompt = std::mem::take(&mut request.tokens);
    let mut guide = request.guide.take();
    if request.min_tokens > 0 && !request.stop.is_empty() {
        let hold: Box<dyn Guide> = Box::new(MinTokens {
            stop: request.stop.clone(),
            left: request.min_tokens,
        });
        guide = Stacked::of(guide.into_iter().chain([hold]).collect());
    }
    let mut guided = guide.map(|guide| Guided {
        forks: guide.fork().is_some(),
        guide,
        queue: Vec::new(),
        next: (Allowed::Any, Vec::new()),
    });
    let done = 'run: {
        if let Some(g) = &mut guided {
            // Tokens forced before the first pick join the prompt.
            if let Some(reason) = g.settle(&request, &mut st, lead, emit) {
                break 'run Some(reason);
            }
            prompt.append(&mut g.queue);
        }
        let cap = target.lanes.as_ref().map_or(target.context(), |l| {
            target.context().min(l.pages() * l.page)
        });
        if prompt.len() > cap {
            break 'run Some(Reason::Context);
        }
        let n = prompt.len();
        let step = spec_req
            .chunk
            .unwrap_or(tessera_device::FAST_MAX_ROWS)
            .min(n)
            .max(1);
        let need_taps = request.max_tokens - st.sim_emitted > 1;
        // Guided or sampled last chunks hold one row.
        let end = match guided.is_some() || sampling {
            true => n - 1,
            false => (n - 1) / step * step,
        };
        let mut pos = 0usize;
        while pos < end {
            let len = step.min(end - pos);
            let rows = prompt_rows(&prompt[pos..pos + len], pos);
            let all: Vec<u32> = (0..len as u32).collect();
            let picks = if need_taps { &all[..] } else { &all[len - 1..] };
            let (_, taps) =
                target.launch_local(&rows, picks, Head::Full(None), need_taps, false)?;
            if let Some(taps) = taps {
                on_chunk(target, drafter, pos, &prompt, len, taps)?;
            }
            pos += len;
        }
        let last_rows = prompt_rows(&prompt[pos..], pos);
        let (mut anchor, taps) = if sampling {
            let (mut logits, _, taps) = target.logits_local(&last_rows, &[0], need_taps, false)?;
            let (allowed, edits) = match &guided {
                Some(g) => (Some(&g.next.0), g.next.1.as_slice()),
                None => (None, &[][..]),
            };
            (sampler.draw_row(&mut logits, allowed, edits), taps)
        } else {
            match &guided {
                Some(g) => target.pick_last(&last_rows, &g.next, need_taps)?,
                None => {
                    let all: Vec<u32> = (0..last_rows.len() as u32).collect();
                    let picks = if need_taps {
                        &all[..]
                    } else {
                        &all[all.len() - 1..]
                    };
                    let (ids, taps) = target.launch_local(
                        &last_rows,
                        picks,
                        Head::Full(None),
                        need_taps,
                        false,
                    )?;
                    (*ids.last().expect("non-empty prompt"), taps)
                }
            }
        };
        fnv_u32(&mut st.hash, anchor);
        if let Some(reason) = take_one(&mut guided, &request, &mut st, anchor, lead, emit) {
            break 'run Some(reason);
        }
        if !ranked && st.stopped {
            return Ok(st.stats);
        }
        let taps = taps.expect("taps when more tokens follow");
        let mut state = init(target, drafter, pos, &prompt, anchor, taps)?;
        let mut start = n;
        let mut round = 0usize;
        while st.sim_emitted < request.max_tokens {
            let fixed = guided.as_ref().map_or(0, |g| g.queue.len());
            if start + fixed >= target.context() {
                break 'run Some(Reason::Context);
            }
            let remaining = request.max_tokens - st.sim_emitted;
            let speculate = guided
                .as_ref()
                .is_none_or(|g| g.forks && device(target, &g.next));
            let drafts = match speculate {
                true => draft(target, drafter, &mut state, start, anchor, remaining)?,
                false => None,
            };
            let (drafted, block, rules) = match &guided {
                Some(g) => g.block(target, drafts),
                None => (drafts.is_some(), drafts.unwrap_or_default(), Vec::new()),
            };
            let mut verify = Vec::with_capacity(block.len() + 1);
            span(
                &mut verify,
                0,
                start,
                [anchor].into_iter().chain(block.iter().copied()),
            );
            // A drafted round also reads taps.
            let (accepted, bonus, posterior, block_taps) = if sampling {
                match &guided {
                    Some(g) if !drafted => {
                        let last = [verify.len() as u32 - 1];
                        let (mut logits, _, _) =
                            target.logits_local(&verify, &last, false, false)?;
                        let id = sampler.draw_row(&mut logits, Some(&g.next.0), &g.next.1);
                        let post = block.iter().copied().chain([id]).collect();
                        (block.len(), id, post, None)
                    }
                    _ => {
                        let all: Vec<u32> = (0..verify.len() as u32).collect();
                        // A drafted verify leaves its state for `commit`.
                        let (mut logits, vocab, taps) =
                            target.logits_local(&verify, &all, drafted, drafted)?;
                        let (acc, bonus) =
                            sampler.verify(&block, &mut logits, vocab, &rules, fixed);
                        let post = block[..acc].iter().copied().chain([bonus]).collect();
                        (acc, bonus, post, taps)
                    }
                }
            } else {
                let (mut post, taps) = match &guided {
                    Some(g) if !drafted => {
                        let (id, _) = target.pick_last(&verify, &g.next, false)?;
                        (block.iter().copied().chain([id]).collect(), None)
                    }
                    _ => {
                        let all: Vec<u32> = (0..verify.len() as u32).collect();
                        let head = head(target, &rules)?;
                        // A drafted verify leaves its state for `commit`.
                        target.launch_local(&verify, &all, head, drafted, drafted)?
                    }
                };
                post[..fixed].copy_from_slice(&block[..fixed]);
                let (acc, bonus) = accept(&block, Target::Ids(&post));
                (acc, bonus, post, taps)
            };
            if drafted {
                target.commit(accepted + 1)?;
            }
            if drafted && lead && !st.stopped {
                st.stats.steps += 1;
                st.stats.drafted += block.len() - fixed;
                st.stats.accepted += accepted - fixed;
            }
            fnv_u32(&mut st.hash, start as u32);
            fnv_u32(&mut st.hash, block.len() as u32);
            for &d in &block {
                fnv_u32(&mut st.hash, d);
            }
            fnv_u32(&mut st.hash, accepted as u32);
            fnv_u32(&mut st.hash, bonus);
            if let Some(g) = &mut guided {
                g.queue.clear();
            }
            let mut ended = None;
            for id in block[fixed..accepted].iter().copied().chain([bonus]) {
                ended = take_one(&mut guided, &request, &mut st, id, lead, emit);
                if ended.is_some() {
                    break;
                }
            }
            fnv_u32(&mut st.hash, st.sim_emitted as u32);
            round += 1;
            if ended.is_some() {
                break 'run ended;
            }
            if !ranked && st.stopped {
                return Ok(st.stats);
            }
            if ranked
                && round.is_multiple_of(SYNC_EVERY)
                && sync_ranks(target, st.stopped, st.hash)?
            {
                return Ok(st.stats);
            }
            if let Some(block_taps) = block_taps {
                let mut discard = SpecStats::default();
                let stat_ref = if lead && !st.stopped {
                    &mut st.stats
                } else {
                    &mut discard
                };
                advance(
                    target,
                    drafter,
                    &mut state,
                    start,
                    block.len(),
                    accepted,
                    &posterior,
                    block_taps,
                    stat_ref,
                )?;
            }
            start += 1 + accepted;
            anchor = bonus;
        }
        None
    };
    if lead && !st.stopped {
        let _ = emit(Event::Done(done.unwrap_or(Reason::Length)));
    }
    sync_ranks(target, st.stopped, st.hash)?;
    Ok(st.stats)
}

/// A guided answer between verify rounds.
struct Guided {
    guide: Box<dyn Guide>,
    /// Whether the guide forks, so drafts can be walked.
    forks: bool,
    /// Taken tokens to feed after the anchor.
    queue: Vec<u32>,
    /// The rules of the pick after the queue.
    next: Rules,
}

impl Guided {
    /// Queues forced tokens and reads the next rules.
    fn settle(
        &mut self,
        request: &Request,
        st: &mut LoopState,
        lead: bool,
        emit: &mut dyn FnMut(Event) -> bool,
    ) -> Option<Reason> {
        let queue = &mut self.queue;
        let out = settle(&mut *self.guide, |id| {
            let reason = accept_token(request, st, id, lead, emit);
            if reason.is_none() {
                queue.push(id);
            }
            reason
        });
        match out {
            Ok(next) => {
                self.next = next;
                None
            }
            Err(reason) => Some(reason),
        }
    }

    /// Whether the round drafts, its block, and each row's rules.
    fn block(&self, target: &Engine, drafts: Option<Vec<u32>>) -> (bool, Vec<u32>, Vec<Rules>) {
        let fixed = self.queue.len();
        // A queue replaces the drafts when it fits in them.
        if fixed > 0 {
            let fits = drafts.is_some_and(|d| fixed <= d.len());
            let mut rules = vec![(Allowed::Any, Vec::new()); fixed];
            rules.push(self.next.clone());
            return (fits, self.queue.clone(), rules);
        }
        let Some(mut drafts) = drafts else {
            return (false, Vec::new(), vec![self.next.clone()]);
        };
        let rules = walk(&*self.guide, self.next.clone(), &drafts, |r| {
            device(target, r)
        });
        drafts.truncate(rules.len() - 1);
        (true, drafts, rules)
    }
}

/// Takes answer token `id`, and moves the guide past it.
fn take_one(
    guided: &mut Option<Guided>,
    request: &Request,
    st: &mut LoopState,
    id: u32,
    lead: bool,
    emit: &mut dyn FnMut(Event) -> bool,
) -> Option<Reason> {
    let Some(g) = guided else {
        return accept_token(request, st, id, lead, emit);
    };
    if g.guide.accept(id).is_err() {
        return Some(Reason::Guide);
    }
    accept_token(request, st, id, lead, emit).or_else(|| g.settle(request, st, lead, emit))
}

/// Whether the device head can apply `rules`.
fn device(target: &Engine, rules: &Rules) -> bool {
    match rules.1.is_empty() {
        true => rules.0 == Allowed::Any || target.words().is_some(),
        false => target.edits_fit(rules.1.len()),
    }
}

/// A full head with each pick's mask and edits.
fn head(target: &Engine, rules: &[Rules]) -> Result<Head, Error> {
    if rules.iter().any(|r| !r.1.is_empty()) {
        let allowed: Vec<&Allowed> = rules.iter().map(|r| &r.0).collect();
        let edits: Vec<&[Edit]> = rules.iter().map(|r| &r.1[..]).collect();
        return target.edited(&allowed, &edits);
    }
    Ok(match target.words() {
        Some(words) if rules.iter().any(|r| r.0 != Allowed::Any) => {
            Head::Full(Some(rules.iter().flat_map(|r| r.0.bits(words)).collect()))
        }
        _ => Head::Full(None),
    })
}

/// Batched verify outputs: device ids or host logits.
pub(crate) enum VerifyOut {
    Ids(Vec<u32>),
    Logits(Vec<f32>, usize),
}

impl Engine {
    /// A local launch picking after every row, with taps.
    pub(crate) fn step_taps_local(&mut self, rows: &[Row]) -> Result<(Vec<u32>, TapsBuf), Error> {
        let all: Vec<u32> = (0..rows.len() as u32).collect();
        let (ids, taps) = self.launch_local(rows, &all, Head::Full(None), true, false)?;
        Ok((ids, taps.expect("taps were asked for")))
    }

    /// Runs batched verify, reading logits when sampling.
    pub(crate) fn step_verify_local(
        &mut self,
        rows: &[Row],
        sampling: bool,
    ) -> Result<(VerifyOut, TapsBuf), Error> {
        if !sampling {
            let (ids, taps) = self.step_taps_local(rows)?;
            return Ok((VerifyOut::Ids(ids), taps));
        }
        let all: Vec<u32> = (0..rows.len() as u32).collect();
        let (logits, vocab, taps) = self.logits_local(rows, &all, true, false)?;
        Ok((
            VerifyOut::Logits(logits, vocab),
            taps.expect("taps were asked for"),
        ))
    }

    /// Picks the final prefill anchor and taps.
    pub(crate) fn prefill_pick(
        &mut self,
        rows: &[Row],
        sampler: &mut Sampler,
    ) -> Result<(u32, TapsBuf), Error> {
        if sampler.greedy() {
            let (ids, taps) = self.step_taps_local(rows)?;
            return Ok((*ids.last().expect("non-empty chunk"), taps));
        }
        let (mut logits, _, taps) = self.logits_local(rows, &[0], true, false)?;
        let anchor = sampler.draw_row(&mut logits, None, &[]);
        Ok((anchor, taps.expect("taps were asked for")))
    }

    /// Reads full logits for `picks` rows, with optional device taps.
    pub(crate) fn logits_local(
        &mut self,
        rows: &[Row],
        picks: &[u32],
        taps: bool,
        peek: bool,
    ) -> Result<(Vec<f32>, usize, Option<TapsBuf>), Error> {
        let vocab = self.vocab.ok_or(Error::Scores)?;
        let (out, taps) = self.tapped(rows, picks, (Head::Full(None), peek), Role::Logits, taps)?;
        let logits = self.read_floats(out, picks.len() * vocab)?;
        self.report();
        Ok((logits, vocab, taps))
    }

    /// The ids after `picks` rows. Taps and peek on request.
    pub(crate) fn launch_local(
        &mut self,
        rows: &[Row],
        picks: &[u32],
        head: Head,
        taps: bool,
        peek: bool,
    ) -> Result<(Vec<u32>, Option<TapsBuf>), Error> {
        let (out, taps) = self.tapped(rows, picks, (head, peek), Role::Tokens, taps)?;
        let mut ids = self.runner.ids(out)?;
        ids.truncate(picks.len());
        self.report();
        Ok((ids, taps))
    }

    /// The pick after the last row, and its taps.
    fn pick_last(
        &mut self,
        rows: &[Row],
        rules: &Rules,
        taps: bool,
    ) -> Result<(u32, Option<TapsBuf>), Error> {
        let last = [rows.len() as u32 - 1];
        if device(self, rules) {
            let head = head(self, std::slice::from_ref(rules))?;
            let (ids, taps) = self.launch_local(rows, &last, head, taps, false)?;
            return Ok((ids[0], taps));
        }
        // Rules the device head cannot apply run on the host.
        let (mut row, _, taps) = self.logits_local(rows, &last, taps, false)?;
        let mut sampler = Sampler::new(Sampling::default());
        Ok((sampler.draw_row(&mut row, Some(&rules.0), &rules.1), taps))
    }

    /// Runs `rows` locally: the `first` output, and taps if asked.
    fn tapped(
        &mut self,
        rows: &[Row],
        picks: &[u32],
        (head, peek): (Head, bool),
        first: Role,
        taps: bool,
    ) -> Result<(Slot, Option<TapsBuf>), Error> {
        let width = match taps {
            true => self.taps.ok_or(Error::Taps)?,
            false => 0,
        };
        let roles = &[first, Role::Taps][..1 + taps as usize];
        let wants = Wants::roles(&self.graph, self.full, roles);
        // Outputs come in the entry's order.
        let order = &self.graph.entries[self.full].outputs;
        let wanted = order.iter().filter(|o| roles.contains(&o.role));
        let at = |role| wanted.clone().position(|o| o.role == role).unwrap_or(0);
        let (out_at, taps_at) = (at(first), at(Role::Taps));
        let outputs = self.run_local(rows, picks, head, wants, peek)?;
        let out = outputs[out_at];
        if !taps {
            return Ok((out, None));
        }
        let slot = outputs[taps_at];
        let buf = if slot.form.name() == "f32"
            && let Some(addr) = self.runner.address(slot)
        {
            TapsBuf::Device {
                slot,
                addr,
                rows: rows.len(),
            }
        } else {
            TapsBuf::Host(self.read_floats(slot, rows.len() * width)?)
        };
        Ok((out, Some(buf)))
    }
}
