//! Shared SPMD speculative decode loop and device tap helpers.

use crate::sample::{Target, accept, take};
use crate::spec_len::SpecStats;
use crate::{Engine, Error, Event, Reason, Request, Row};
use tessera_engine_comm::{Reader, Writer};
use tessera_lower::Slot;
use tessera_run::{Error as RunError, Profile, Ranks, Run};

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
        match self {
            TapsBuf::Device { addr, rows, .. } => TapsIn::Device {
                addr: *addr,
                rows: *rows,
                stride: 0,
            },
            TapsBuf::Host(v) => TapsIn::Host(v),
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

/// Exchanges control bytes and gathers rank profiles on rank 0.
pub(crate) fn tell_ranks(target: &mut Engine, mine: &[u8]) -> Result<Option<Vec<u8>>, Error> {
    let Some(control) = &mut target.control else {
        return Ok(None);
    };
    let lead = control.rank() == 0;
    let mut parts = control.exchange(mine).map_err(RunError::from)?;
    if lead && let Some((launches, own)) = target.held.take() {
        let mut profiles = vec![Some(own)];
        profiles.extend(parts[1..].iter().map(|p| Profile::from_bytes(p)));
        let ranks = Ranks { launches, profiles };
        eprintln!("profile of {launches} launches:\n{ranks}");
        target.ranks = Some(ranks);
    }
    Ok(Some(parts.swap_remove(0)))
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
    if lead && let Some((launches, own)) = target.held.take() {
        let mut profiles = vec![Some(own)];
        profiles.extend(parts[1..].iter().map(|p| Profile::from_bytes(&p[5..])));
        let ranks = Ranks { launches, profiles };
        eprintln!("profile of {launches} launches:\n{ranks}");
        target.ranks = Some(ranks);
    }
    Ok(stop)
}

/// Speculative request paired with an optional prefill chunk limit.
pub(crate) struct SpecReq {
    pub request: Request,
    pub chunk: Option<usize>,
}

/// Validates a request and broadcasts its header once to followers.
pub(crate) fn start_req(
    target: &mut Engine,
    spec_req: &SpecReq,
    emit: &mut dyn FnMut(Event) -> bool,
    extra: impl FnOnce(&mut Writer),
) -> Result<bool, Error> {
    let request = &spec_req.request;
    if request.tokens.is_empty() || request.tokens.len() > target.context() {
        let _ = emit(Event::Done(Reason::Context));
        return Ok(false);
    }
    if request.max_tokens == 0 {
        let _ = emit(Event::Done(Reason::Length));
        return Ok(false);
    }
    if target.control.as_ref().is_some_and(|c| c.rank() == 0) {
        let table = target
            .lanes
            .as_ref()
            .and_then(crate::lanes::Lanes::pending)
            .unwrap_or(&[]);
        let mut w = Writer(Vec::new());
        w.u32(SPEC_REQ_TAG)
            .u32(request.max_tokens as u32)
            .u32(spec_req.chunk.unwrap_or(0) as u32)
            .u32s(&request.tokens)
            .u32s(&request.stop)
            .u32s(table);
        extra(&mut w);
        tell_ranks(target, &w.0)?;
    }
    Ok(true)
}

fn decode_req<T>(
    bytes: &[u8],
    extra: impl FnOnce(&mut Reader) -> Option<T>,
) -> Option<(SpecReq, Vec<u32>, T)> {
    let mut r = Reader(bytes);
    if r.u32()? != SPEC_REQ_TAG {
        return None;
    }
    let max_tokens = r.u32()? as usize;
    let chunk = r.u32()? as usize;
    let tokens = r.u32s()?;
    let stop = r.u32s()?;
    let table = r.u32s()?;
    let policy = extra(&mut r)?;
    let req = SpecReq {
        request: Request {
            tokens,
            max_tokens,
            stop,
            ..Request::default()
        },
        chunk: (chunk > 0).then_some(chunk),
    };
    Some((req, table, policy))
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
        let Some(told) = tell_ranks(target, &mine)? else {
            return Ok(());
        };
        let Some((req, table, policy)) = decode_req(&told, &mut decode) else {
            return Ok(());
        };
        if let Some(lanes) = &mut target.lanes
            && !table.is_empty()
        {
            lanes.replace(table);
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
                let $req = $crate::spec_loop::SpecReq {
                    request,
                    chunk: $this.prefill_chunk,
                };
                if !$crate::spec_loop::start_req($this.target, &$req, emit, |w| $p.encode(w))? {
                    return Ok($crate::SpecStats::default());
                }
                let ($target, $emit) = (&mut *$this.target, emit);
                $body
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
    let request = spec_req.request;
    let prompt = &request.tokens;
    let n = prompt.len();
    let step = spec_req.chunk.unwrap_or(n).max(1);
    let need_taps = request.max_tokens > 1;
    let mut pos = 0usize;
    while pos + step < n {
        let rows = prompt_rows(&prompt[pos..pos + step], pos);
        if need_taps {
            let (_, taps) = target.step_taps_local(&rows)?;
            on_chunk(target, drafter, pos, prompt, step, taps)?;
        } else {
            target.step_last_local(&rows)?;
        }
        pos += step;
    }
    let last_rows = prompt_rows(&prompt[pos..], pos);
    let (mut anchor, taps) = if need_taps {
        let (ids, taps) = target.step_taps_local(&last_rows)?;
        (*ids.last().expect("non-empty prompt"), Some(taps))
    } else {
        (target.step_last_local(&last_rows)?, None)
    };
    fnv_u32(&mut st.hash, anchor);
    if let Some(reason) = accept_token(&request, &mut st, anchor, lead, emit) {
        if lead && !st.stopped {
            let _ = emit(Event::Done(reason));
        }
        sync_ranks(target, st.stopped, st.hash)?;
        return Ok(st.stats);
    }
    if !ranked && st.stopped {
        return Ok(st.stats);
    }
    let taps = taps.expect("taps when max_tokens > 1");
    let mut state = init(target, drafter, pos, prompt, anchor, taps)?;
    let mut start = n;
    let mut round = 0usize;
    let mut done_reason: Option<Reason> = None;
    while st.sim_emitted < request.max_tokens {
        if start >= target.context() {
            done_reason = Some(Reason::Context);
            break;
        }
        let remaining = request.max_tokens - st.sim_emitted;
        let drafts = draft(target, drafter, &mut state, start, anchor, remaining)?;
        let block = drafts.as_deref().unwrap_or(&[]);
        let mut verify = Vec::with_capacity(block.len() + 1);
        span(
            &mut verify,
            0,
            start,
            [anchor].into_iter().chain(block.iter().copied()),
        );
        // A drafted round also reads taps.
        let (posterior, block_taps) = match drafts {
            Some(_) => {
                let (ids, taps) = target.step_taps_local(&verify)?;
                (ids, Some(taps))
            }
            None => (target.step_local(&verify)?, None),
        };
        let (accepted, bonus) = accept(block, Target::Ids(&posterior));
        if accepted < block.len() && target.stateful() && target.slots() > 1 {
            target.restore_recurrent_slot(accepted + 1)?;
        }
        if drafts.is_some() && lead && !st.stopped {
            st.stats.steps += 1;
            st.stats.drafted += block.len();
            st.stats.accepted += accepted;
        }
        fnv_u32(&mut st.hash, start as u32);
        fnv_u32(&mut st.hash, block.len() as u32);
        for &d in block {
            fnv_u32(&mut st.hash, d);
        }
        fnv_u32(&mut st.hash, accepted as u32);
        fnv_u32(&mut st.hash, bonus);
        for id in block[..accepted].iter().copied().chain([bonus]) {
            if let Some(reason) = accept_token(&request, &mut st, id, lead, emit) {
                done_reason = Some(reason);
                break;
            }
        }
        fnv_u32(&mut st.hash, st.sim_emitted as u32);
        round += 1;
        if done_reason.is_some() {
            break;
        }
        if !ranked && st.stopped {
            return Ok(st.stats);
        }
        if ranked && round.is_multiple_of(SYNC_EVERY) && sync_ranks(target, st.stopped, st.hash)? {
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
    if lead && !st.stopped {
        let _ = emit(Event::Done(done_reason.unwrap_or(Reason::Length)));
    }
    sync_ranks(target, st.stopped, st.hash)?;
    Ok(st.stats)
}
