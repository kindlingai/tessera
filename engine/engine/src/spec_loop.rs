//! Shared SPMD speculative decode loop and device tap helpers.

use crate::spec_len::SpecStats;
use crate::speculate::accept_greedy;
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
    Device { addr: u64, rows: usize },
    Host(&'a [f32]),
}

/// Device-to-device copy descriptor for one drafter input.
#[derive(Clone, Copy)]
pub(crate) struct D2dCopy {
    pub input: usize,
    pub src: u64,
    pub rows: usize,
    pub filled: usize,
    pub width: usize,
}

impl TapsBuf {
    pub(crate) fn as_in(&self) -> TapsIn<'_> {
        match self {
            TapsBuf::Device { addr, rows, .. } => TapsIn::Device {
                addr: *addr,
                rows: *rows,
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

/// Copies `rows` on device and pads tail rows to `filled`.
pub(crate) fn fill_d2d(runner: &mut dyn Run, dst: Slot, copy: D2dCopy) -> Result<(), Error> {
    let row_bytes = copy.width * 4;
    runner.copy_from(dst, 0, copy.src, copy.rows * row_bytes)?;
    if copy.rows < copy.filled {
        let base = runner.address(dst).ok_or(RunError::Unaddressed(0))?;
        let tail = copy.rows - 1;
        let tail_addr = base + (tail * row_bytes) as u64;
        let mut have = 1usize;
        while tail + have < copy.filled {
            let chunk = have.min(copy.filled - (tail + have));
            runner.copy_from(dst, (tail + have) * row_bytes, tail_addr, chunk * row_bytes)?;
            have += chunk;
        }
    }
    Ok(())
}

fn fnv_u32(h: &mut u32, v: u32) {
    for b in v.to_le_bytes() {
        *h = (*h ^ b as u32).wrapping_mul(16777619);
    }
}

/// Exchanges control bytes and gathers rank profiles on rank 0.
fn tell_ranks(target: &mut Engine, mine: &[u8]) -> Result<Option<Vec<u8>>, Error> {
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

/// Validates a request and broadcasts its header once to followers.
pub(crate) fn start_req(
    target: &mut Engine,
    request: &Request,
    emit: &mut dyn FnMut(Event) -> bool,
    extra: impl FnOnce(&mut Writer),
) -> Result<bool, Error> {
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
) -> Option<(Request, Vec<u32>, T)> {
    let mut r = Reader(bytes);
    if r.u32()? != SPEC_REQ_TAG {
        return None;
    }
    let max_tokens = r.u32()? as usize;
    let tokens = r.u32s()?;
    let stop = r.u32s()?;
    let table = r.u32s()?;
    let policy = extra(&mut r)?;
    let req = Request {
        tokens,
        max_tokens,
        stop,
        ..Request::default()
    };
    Some((req, table, policy))
}

/// Shared follower outer loop receiving SPMD speculative requests.
pub(crate) fn follow_loop<T>(
    target: &mut Engine,
    mut decode: impl FnMut(&mut Reader) -> Option<T>,
    mut run: impl FnMut(&mut Engine, Request, T) -> Result<(), Error>,
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
                if !$crate::spec_loop::start_req($this.target, &request, emit, |w| $p.encode(w))? {
                    return Ok($crate::SpecStats::default());
                }
                let ($target, $req, $emit) = (&mut *$this.target, request, emit);
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

fn accept_token(
    request: &Request,
    st: &mut LoopState,
    id: u32,
    lead: bool,
    emit: &mut dyn FnMut(Event) -> bool,
) -> Option<Reason> {
    if request.stop.contains(&id) {
        return Some(Reason::Stop);
    }
    st.sim_emitted += 1;
    if lead && !st.stopped {
        st.stats.emitted += 1;
        if !emit(Event::Token(id)) {
            st.stopped = true;
            let _ = emit(Event::Done(Reason::Stop));
        }
    }
    (st.sim_emitted >= request.max_tokens).then_some(Reason::Length)
}

/// Shared SPMD greedy speculative decode loop for all drafters.
pub(crate) fn run_speculate<S>(
    target: &mut Engine,
    request: Request,
    emit: &mut dyn FnMut(Event) -> bool,
    init: impl FnOnce(&mut Engine, &[u32], u32, TapsBuf) -> Result<S, Error>,
    mut draft: impl FnMut(&mut Engine, &mut S, usize, u32, usize) -> Result<Option<Vec<u32>>, Error>,
    mut advance: impl FnMut(
        &mut Engine,
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
    let prompt = &request.tokens;
    let prefill: Vec<Row> = prompt
        .iter()
        .enumerate()
        .map(|(i, &token)| Row {
            slot: 0,
            token,
            position: i as u32,
        })
        .collect();
    let (ids, taps) = target.step_taps_local(&prefill)?;
    let mut anchor = *ids.last().expect("non-empty prompt");
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
    let mut state = init(target, prompt, anchor, taps)?;
    let mut start = prompt.len();
    let mut round = 0usize;
    let mut done_reason: Option<Reason> = None;
    while st.sim_emitted < request.max_tokens {
        if start >= target.context() {
            done_reason = Some(Reason::Context);
            break;
        }
        let remaining = request.max_tokens - st.sim_emitted;
        let Some(drafts) = draft(target, &mut state, start, anchor, remaining)? else {
            let row = [Row {
                slot: 0,
                token: anchor,
                position: start as u32,
            }];
            let next = target.step_local(&row)?[0];
            start += 1;
            fnv_u32(&mut st.hash, start as u32);
            fnv_u32(&mut st.hash, next);
            round += 1;
            if let Some(reason) = accept_token(&request, &mut st, next, lead, emit) {
                done_reason = Some(reason);
                break;
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
            anchor = next;
            continue;
        };
        let mut verify = Vec::with_capacity(drafts.len() + 1);
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
        let (posterior, block_taps) = target.step_taps_local(&verify)?;
        let (accepted, bonus) = accept_greedy(&drafts, &posterior);
        if lead && !st.stopped {
            st.stats.steps += 1;
            st.stats.drafted += drafts.len();
            st.stats.accepted += accepted;
        }
        fnv_u32(&mut st.hash, start as u32);
        fnv_u32(&mut st.hash, drafts.len() as u32);
        for &d in &drafts {
            fnv_u32(&mut st.hash, d);
        }
        fnv_u32(&mut st.hash, accepted as u32);
        fnv_u32(&mut st.hash, bonus);
        for id in drafts[..accepted].iter().copied().chain([bonus]) {
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
        let commit = 1 + accepted;
        let mut discard = SpecStats::default();
        let stat_ref = if lead && !st.stopped {
            &mut st.stats
        } else {
            &mut discard
        };
        advance(
            target,
            &mut state,
            start,
            drafts.len(),
            accepted,
            &posterior,
            block_taps,
            stat_ref,
        )?;
        start += commit;
        anchor = bonus;
    }
    if lead && !st.stopped {
        let _ = emit(Event::Done(done_reason.unwrap_or(Reason::Length)));
    }
    sync_ranks(target, st.stopped, st.hash)?;
    Ok(st.stats)
}
