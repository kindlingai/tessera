//! Native MTP speculative decoding: one-layer MLA drafter and loop.

use crate::ranks::open_rank;
use crate::spec_len::{CostModel, SpecLen};
use crate::speculate::{
    DraftCore, TargetCmd, decode_target, draft_methods, run_speculate, run_target, taps_output,
    tell_ranks,
};
use crate::{Engine, Error, Event, Place, Request, SpecStats};
use tessera_engine_comm::{Reader, Transport, Writer};
use tessera_graph::DimName::{Context, Hidden, Rows, Segments, Seqs};
use tessera_graph::{Graph, Role};
use tessera_interp::Tensor;
use tessera_lower::{Recipe, Settings, Slot, Wants, Weight};
use tessera_models::mtp;
use tessera_run::{Run, open};

/// Control tag for an MTP step reading target taps.
pub(crate) const MTP_FIRST_TAG: u32 = 4;

/// Control tag for a chained MTP step reading prior hidden.
pub(crate) const MTP_CHAIN_TAG: u32 = 5;

/// Runs an MTP one-layer MLA drafter with kept KV caches.
pub struct MtpDrafter {
    core: DraftCore,
    hidden: usize,
}

draft_methods!(MtpDrafter);

impl MtpDrafter {
    /// Opens a compiled runner for an MTP stepper graph.
    pub fn open(
        graph: Graph,
        context: usize,
        recipe: &Recipe,
        settings: &Settings,
    ) -> Result<Self, Error> {
        let run = |plan| Ok(open(plan, settings)?);
        Self::placed(graph, context, recipe, settings, None, run)
    }

    /// Opens one rank of a TP-sharded MTP drafter.
    pub fn open_ranked<T: Transport + 'static>(
        graph: Graph,
        context: usize,
        (recipe, settings): (&Recipe, &Settings),
        mut place: Place,
        sums: T,
    ) -> Result<Self, Error> {
        place.rank = sums.rank();
        let grid = place.grid.take();
        let run = |plan| open_rank(plan, settings, sums, grid);
        Self::placed(graph, context, recipe, settings, Some(&place), run)
    }

    fn placed(
        graph: Graph,
        context: usize,
        recipe: &Recipe,
        settings: &Settings,
        place: Option<&Place>,
        run: impl FnOnce(tessera_lower::Plan) -> Result<Box<dyn Run>, Error>,
    ) -> Result<Self, Error> {
        if context < 2 {
            return Err(Error::Rows);
        }
        tessera_run::configure(settings);
        let graph = match place {
            Some(place) => {
                let shares = recipe.pins.shares(&graph);
                let sharded = mtp::shard(&graph, &place.parts, &shares, place.moe_tp)?;
                for w in sharded.whole.iter().filter(|_| place.rank == 0) {
                    let path = &graph.weights[w.weight as usize].path;
                    eprintln!("tessera: mtp {path:?} stays whole: {}", w.why);
                }
                sharded.graph
            }
            None => graph,
        };
        let hidden = graph
            .dims
            .iter()
            .find(|d| d.name == Hidden)
            .map(|d| d.size.max() as usize)
            .ok_or(Error::Taps)?;
        let wants = Wants::roles(&graph, 0, &[Role::Tokens, Role::Taps]);
        let core = DraftCore::open(
            graph, context, 1, wants, recipe, settings, place, "mtp", run,
        )?;
        Ok(Self { core, hidden })
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// Number of cached plan hits in the MTP drafter.
    pub fn plan_hits(&self) -> usize {
        self.core.plans.hits
    }

    pub(crate) fn wants(&self) -> Wants {
        Wants::roles(&self.core.graph, 0, &[Role::Tokens, Role::Taps])
    }

    pub(crate) fn read_last(
        &mut self,
        outputs: &[Slot],
        rows: usize,
        keep_hidden: bool,
    ) -> Result<(u32, Vec<f32>), Error> {
        let ids = self.core.runner.ids(outputs[0])?;
        let token = ids[rows - 1];
        if !keep_hidden {
            return Ok((token, Vec::new()));
        }
        let raw = self.core.runner.read(outputs[1])?;
        let floats = raw.floats().map_err(tessera_run::Error::from)?;
        let from = (rows - 1) * self.hidden;
        Ok((token, floats[from..from + self.hidden].to_vec()))
    }

    pub(crate) fn run(
        &mut self,
        slot: u32,
        start_pos: u32,
        tokens: &[u32],
        hidden: &[f32],
        wants: Wants,
    ) -> Result<Vec<Slot>, Error> {
        let m = tokens.len();
        if m == 0 || hidden.len() != m * self.hidden {
            return Err(Error::Rows);
        }
        if slot as usize >= self.core.slots || start_pos as usize + m > self.core.context {
            return Err(Error::RowPlace);
        }
        let filled = m.next_power_of_two().min(self.core.context);
        let mut tok_buf = Vec::with_capacity(filled);
        let mut pos_buf = Vec::with_capacity(filled);
        let mut scale_buf = Vec::with_capacity(filled);
        let mut hid_buf = Vec::with_capacity(filled * self.hidden);
        for (i, &t) in tokens.iter().enumerate() {
            let p = start_pos + i as u32;
            tok_buf.push(t);
            pos_buf.push(p);
            scale_buf.push(if p == 0 { 0.0 } else { 1.0 });
        }
        hid_buf.extend_from_slice(hidden);
        let last_tok = *tokens.last().expect("non-empty");
        let last_pos = start_pos + (m - 1) as u32;
        let last_scale = if last_pos == 0 { 0.0 } else { 1.0 };
        let last_hid = &hidden[(m - 1) * self.hidden..];
        for _ in m..filled {
            tok_buf.push(last_tok);
            pos_buf.push(last_pos);
            scale_buf.push(last_scale);
            hid_buf.extend_from_slice(last_hid);
        }
        let feeds = [
            Tensor::u32(vec![filled], tok_buf),
            Tensor::u32(vec![filled], pos_buf),
            Tensor::u32(vec![filled], vec![slot; filled]),
            Tensor::f32(vec![filled, self.hidden], hid_buf),
            Tensor::f32(vec![filled, 1], scale_buf),
        ];
        let sizes = [
            (Seqs, self.core.slots as u64),
            (Rows, filled as u64),
            (Context, (start_pos as usize + m) as u64),
            (Segments, 1),
        ];
        self.core.exec(&feeds, &sizes, wants)
    }
}

/// Decoded MTP launch command from rank 0.
pub(crate) struct MtpStep {
    pub slot: u32,
    pub start_pos: u32,
    pub tokens: Vec<u32>,
    pub wants: Wants,
    pub chain: bool,
    pub keep: bool,
}

enum MtpCmd {
    Target(TargetCmd),
    Step(MtpStep),
}

pub(crate) fn encode_step(
    slot: u32,
    start_pos: u32,
    tokens: &[u32],
    wants: Wants,
    chain: bool,
    keep: bool,
) -> Vec<u8> {
    let tag = if chain { MTP_CHAIN_TAG } else { MTP_FIRST_TAG };
    let mut w = Writer(Vec::with_capacity(24 + tokens.len() * 4));
    w.u32(tag)
        .u32(slot)
        .u32(start_pos)
        .u32(wants.0)
        .u32(u32::from(keep))
        .u32s(tokens);
    w.0
}

pub(crate) fn decode_step(tag: u32, r: &mut Reader) -> Option<MtpStep> {
    if tag != MTP_FIRST_TAG && tag != MTP_CHAIN_TAG {
        return None;
    }
    Some(MtpStep {
        slot: r.u32()?,
        start_pos: r.u32()?,
        wants: Wants(r.u32()?),
        keep: r.u32()? != 0,
        tokens: r.u32s()?,
        chain: tag == MTP_CHAIN_TAG,
    })
}

fn decode_mtp(bytes: &[u8]) -> Option<MtpCmd> {
    let mut r = Reader(bytes);
    let tag = r.u32()?;
    if let Some(step) = decode_step(tag, &mut r) {
        return Some(MtpCmd::Step(step));
    }
    decode_target(tag, &mut r).map(MtpCmd::Target)
}

/// Runs one decoded MTP step on a follower rank.
pub(crate) fn follow_mtp(
    drafter: &mut MtpDrafter,
    step: MtpStep,
    last_taps: &[f32],
    prev_hidden: &mut Vec<f32>,
) -> Result<(), Error> {
    let hidden = drafter.hidden();
    let src = if step.chain {
        prev_hidden.as_slice()
    } else {
        last_taps
    };
    let need = step.tokens.len() * hidden;
    if src.len() < need {
        return Err(Error::Rows);
    }
    let outputs = drafter.run(
        step.slot,
        step.start_pos,
        &step.tokens,
        &src[..need],
        step.wants,
    )?;
    if step.keep
        && let Some(at) = taps_output(&drafter.core.graph, 0, step.wants)
    {
        let raw = drafter.core.runner.read(outputs[at])?;
        let floats = raw.floats().map_err(tessera_run::Error::from)?;
        let from = (step.tokens.len() - 1) * hidden;
        prev_hidden.clear();
        prev_hidden.extend_from_slice(&floats[from..from + hidden]);
    }
    Ok(())
}

/// Runs one MTP step on rank 0, broadcasting if ranked.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_mtp(
    target: &mut Engine,
    drafter: &mut MtpDrafter,
    start_pos: u32,
    tokens: &[u32],
    hid: &[f32],
    keep: bool,
    chain: bool,
) -> Result<(u32, Vec<f32>), Error> {
    let wants = drafter.wants();
    if target.control.is_some() {
        tell_ranks(
            target,
            &encode_step(0, start_pos, tokens, wants, chain, keep),
        )?;
    }
    let outputs = drafter.run(0, start_pos, tokens, hid, wants)?;
    drafter.read_last(&outputs, tokens.len(), keep)
}

/// Runs an MTP first-form pass plus adaptive chain steps.
#[allow(clippy::too_many_arguments)]
pub(crate) fn draft_mtp(
    target: &mut Engine,
    drafter: &mut MtpDrafter,
    spec: &SpecLen,
    start_pos: u32,
    tokens: &[u32],
    taps: &[f32],
    start: usize,
    avail: usize,
) -> Result<(Vec<u32>, f64), Error> {
    let chain_n = if !spec.is_adaptive() {
        avail
    } else {
        let mut n = 1;
        while n < avail && spec.more(None, n) {
            n += 1;
        }
        n
    };
    let mut drafts = Vec::with_capacity(chain_n);
    let (first, mut prev_hid) =
        run_mtp(target, drafter, start_pos, tokens, taps, chain_n > 1, false)?;
    drafts.push(first);
    for j in 1..chain_n {
        let pos = (start - 1 + j) as u32;
        let (tok, next_hid) = run_mtp(
            target,
            drafter,
            pos,
            &[drafts[j - 1]],
            &prev_hid,
            j + 1 < chain_n,
            true,
        )?;
        drafts.push(tok);
        prev_hid = next_hid;
    }
    let (k, pred) = spec.choose(None, drafts.len(), drafts.len());
    drafts.truncate(k);
    Ok((drafts, pred))
}

/// Native MTP speculative decoding over a target and MTP pair.
pub struct MtpSpeculate<'a> {
    target: &'a mut Engine,
    drafter: &'a mut MtpDrafter,
    depth: usize,
    spec_len: SpecLen,
}

impl<'a> MtpSpeculate<'a> {
    /// Pairs a tapped target engine with its MTP drafter.
    pub fn new(
        target: &'a mut Engine,
        drafter: &'a mut MtpDrafter,
        depth: usize,
    ) -> Result<Self, Error> {
        if depth == 0 {
            return Err(Error::Rows);
        }
        let Some(width) = target.tap_width() else {
            return Err(Error::Taps);
        };
        if width != drafter.hidden() {
            return Err(Error::Rows);
        }
        let spec_len = SpecLen::fixed(depth)
            .with_cost(CostModel::mtp())
            .with_mtp_seeds();
        Ok(Self {
            target,
            drafter,
            depth,
            spec_len,
        })
    }

    /// Configures adaptive or truncated draft length selection.
    pub fn with_spec_len(mut self, spec_len: SpecLen) -> Self {
        self.spec_len = spec_len;
        self
    }

    /// Follows rank 0's target and MTP launches until dismissed.
    pub fn follow(&mut self) -> Result<(), Error> {
        let mut last_taps: Vec<f32> = Vec::new();
        let mut prev_hidden: Vec<f32> = Vec::new();
        let hidden = self.drafter.hidden();
        loop {
            let mine = self
                .target
                .held
                .take()
                .map_or_else(Vec::new, |(_, p)| p.to_bytes());
            let Some(told) = tell_ranks(self.target, &mine)? else {
                return Ok(());
            };
            match decode_mtp(&told) {
                None => return Ok(()),
                Some(MtpCmd::Target(cmd)) => run_target(self.target, cmd, hidden, &mut last_taps)?,
                Some(MtpCmd::Step(step)) => {
                    follow_mtp(self.drafter, step, &last_taps, &mut prev_hidden)?;
                }
            }
        }
    }

    /// Greedy MTP speculative decoding matching target output token-for-token.
    pub fn generate(
        &mut self,
        request: Request,
        emit: &mut dyn FnMut(Event) -> bool,
    ) -> Result<SpecStats, Error> {
        let depth = self.depth;
        let hidden = self.drafter.hidden();
        let max_ctx = self.target.context().min(self.drafter.context());
        let drafter = &mut *self.drafter;
        let spec = self.spec_len.fresh();
        run_speculate(
            self.target,
            request,
            emit,
            |prompt, anchor, taps| {
                let pending: Vec<u32> = prompt[1..].iter().copied().chain([anchor]).collect();
                (0u32, pending, taps, spec, 1.0f64)
            },
            |target,
             (pending_start, pending_tokens, pending_taps, spec, last_pred),
             start,
             _,
             remaining| {
                let avail = depth
                    .min(spec.max_k())
                    .min(remaining.saturating_sub(1))
                    .min(max_ctx.saturating_sub(start + 1));
                if avail == 0 {
                    return Ok(None);
                }
                let (drafts, pred) = draft_mtp(
                    target,
                    drafter,
                    spec,
                    *pending_start,
                    pending_tokens,
                    pending_taps,
                    start,
                    avail,
                )?;
                *last_pred = pred;
                Ok(Some(drafts))
            },
            |(pending_start, pending_tokens, pending_taps, spec, last_pred),
             start,
             drafted,
             accepted,
             posterior,
             mut taps,
             stats| {
                spec.observe(drafted, accepted, None, *last_pred);
                stats.mtp_steps += 1;
                let commit = 1 + accepted;
                *pending_start = start as u32;
                *pending_tokens = posterior[..commit].to_vec();
                taps.truncate(commit * hidden);
                *pending_taps = taps;
            },
        )
    }
}
