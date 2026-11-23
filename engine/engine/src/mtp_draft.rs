//! Native MTP speculative decoding: one-layer MLA drafter and loop.

use crate::ranks::open_rank;
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
const MTP_FIRST_TAG: u32 = 4;

/// Control tag for a chained MTP step reading prior hidden.
const MTP_CHAIN_TAG: u32 = 5;

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

    fn wants(&self, keep_hidden: bool) -> Wants {
        match keep_hidden {
            true => Wants::roles(&self.core.graph, 0, &[Role::Tokens, Role::Taps]),
            false => Wants::role(&self.core.graph, 0, Role::Tokens),
        }
    }

    fn read_last(
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

    fn run(
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

enum MtpCmd {
    Target(TargetCmd),
    Step {
        slot: u32,
        start_pos: u32,
        tokens: Vec<u32>,
        wants: Wants,
        chain: bool,
    },
}

fn encode_step(slot: u32, start_pos: u32, tokens: &[u32], wants: Wants, chain: bool) -> Vec<u8> {
    let tag = if chain { MTP_CHAIN_TAG } else { MTP_FIRST_TAG };
    let mut w = Writer(Vec::with_capacity(20 + tokens.len() * 4));
    w.u32(tag)
        .u32(slot)
        .u32(start_pos)
        .u32(wants.0)
        .u32s(tokens);
    w.0
}

fn decode_mtp(bytes: &[u8]) -> Option<MtpCmd> {
    let mut r = Reader(bytes);
    let tag = r.u32()?;
    if tag == MTP_FIRST_TAG || tag == MTP_CHAIN_TAG {
        return Some(MtpCmd::Step {
            slot: r.u32()?,
            start_pos: r.u32()?,
            wants: Wants(r.u32()?),
            tokens: r.u32s()?,
            chain: tag == MTP_CHAIN_TAG,
        });
    }
    decode_target(tag, &mut r).map(MtpCmd::Target)
}

/// Native MTP speculative decoding over a target and MTP pair.
pub struct MtpSpeculate<'a> {
    target: &'a mut Engine,
    drafter: &'a mut MtpDrafter,
    depth: usize,
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
        Ok(Self {
            target,
            drafter,
            depth,
        })
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
                Some(MtpCmd::Step {
                    slot,
                    start_pos,
                    tokens,
                    wants,
                    chain,
                }) => {
                    let src = if chain { &prev_hidden } else { &last_taps };
                    let need = tokens.len() * hidden;
                    if src.len() < need {
                        return Err(Error::Rows);
                    }
                    let outputs =
                        self.drafter
                            .run(slot, start_pos, &tokens, &src[..need], wants)?;
                    if let Some(at) = taps_output(&self.drafter.core.graph, 0, wants) {
                        let raw = self.drafter.core.runner.read(outputs[at])?;
                        let floats = raw.floats().map_err(tessera_run::Error::from)?;
                        let from = (tokens.len() - 1) * hidden;
                        prev_hidden.clear();
                        prev_hidden.extend_from_slice(&floats[from..from + hidden]);
                    }
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
        let run_step = |target: &mut Engine,
                        drafter: &mut MtpDrafter,
                        start_pos: u32,
                        tokens: &[u32],
                        hid: &[f32],
                        keep: bool,
                        chain: bool|
         -> Result<(u32, Vec<f32>), Error> {
            let wants = drafter.wants(keep);
            if target.control.is_some() {
                tell_ranks(target, &encode_step(0, start_pos, tokens, wants, chain))?;
            }
            let outputs = drafter.run(0, start_pos, tokens, hid, wants)?;
            drafter.read_last(&outputs, tokens.len(), keep)
        };
        run_speculate(
            self.target,
            request,
            emit,
            |prompt, anchor, taps| {
                let pending: Vec<u32> = prompt[1..].iter().copied().chain([anchor]).collect();
                (0u32, pending, taps)
            },
            |target, (pending_start, pending_tokens, pending_taps), start, _, remaining| {
                let k = depth
                    .min(remaining.saturating_sub(1))
                    .min(max_ctx.saturating_sub(start + 1));
                if k == 0 {
                    return Ok(None);
                }
                let mut drafts = Vec::with_capacity(k);
                let (first, mut prev_hid) = run_step(
                    target,
                    drafter,
                    *pending_start,
                    pending_tokens,
                    pending_taps,
                    k > 1,
                    false,
                )?;
                drafts.push(first);
                for j in 1..k {
                    let pos = (start - 1 + j) as u32;
                    let (tok, next_hid) = run_step(
                        target,
                        drafter,
                        pos,
                        &[drafts[j - 1]],
                        &prev_hid,
                        j + 1 < k,
                        true,
                    )?;
                    drafts.push(tok);
                    prev_hid = next_hid;
                }
                Ok(Some(drafts))
            },
            |(pending_start, pending_tokens, pending_taps), start, commit, posterior, mut taps| {
                *pending_start = start as u32;
                *pending_tokens = posterior[..commit].to_vec();
                taps.truncate(commit * hidden);
                *pending_taps = taps;
            },
        )
    }
}
