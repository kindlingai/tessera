//! DFlash speculative decoding: cached drafter and verify loop.

use crate::ranks::open_rank;
pub use crate::spec_len::SpecStats;
use crate::spec_len::{CostModel, SpecLen};
use crate::{Engine, Error, Event, Head, Place, Reason, Request, Row};
use std::cell::RefCell;
use std::rc::Rc;
use tessera_engine_comm::{Error as CommError, Reader, Transport, Writer};
use tessera_graph::DimName::{Context, Embedding, Rows, Segments, Seqs, Vocab};
use tessera_graph::{Graph, Role};
use tessera_interp::Tensor;
use tessera_lower::{Cache, Device, Recipe, Settings, Slot, Wants, Weight};
use tessera_models::dflash;
use tessera_run::{Error as RunError, Profile, Ranks, Run, Sum, open};

/// Wire tag identifying a drafter launch on the control channel.
pub(crate) const DRAFT_TAG: u32 = 3;

/// One transport shared by target and drafter in alternation.
struct SharedTransport(Rc<RefCell<Box<dyn Transport>>>);

impl Transport for SharedTransport {
    fn rank(&self) -> u32 {
        self.0.borrow().rank()
    }

    fn ranks(&self) -> u32 {
        self.0.borrow().ranks()
    }

    fn exchange(&mut self, mine: &[u8]) -> Result<Vec<Vec<u8>>, CommError> {
        self.0.borrow_mut().exchange(mine)
    }
}

/// One persistent sum grid shared by target and drafter.
struct SharedSum(Rc<RefCell<Box<dyn Sum>>>);

impl Sum for SharedSum {
    fn sum(&mut self, from: u64, to: u64, len: usize) -> Result<(), RunError> {
        self.0.borrow_mut().sum(from, to, len)
    }

    fn gather(
        &mut self,
        from: u64,
        to: u64,
        part: usize,
        total: usize,
        rows: usize,
    ) -> Result<bool, RunError> {
        self.0.borrow_mut().gather(from, to, part, total, rows)
    }

    fn check(&mut self) -> Result<(), RunError> {
        self.0.borrow_mut().check()
    }

    fn timing(&mut self, on: bool) {
        self.0.borrow_mut().timing(on);
    }

    fn spent(&mut self) -> Result<Option<[f64; 2]>, RunError> {
        self.0.borrow_mut().spent()
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

/// Greedy walk over a `[n, k, k]` candidate lattice.
fn walk_lattice(cands: &[u32], scores: &[f32], n: usize, k: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(n);
    let mut pred = 0usize;
    for slot in 0..n {
        let base = (slot * k + pred) * k;
        let mut best = 0usize;
        let mut val = f32::NEG_INFINITY;
        for c in 0..k {
            let v = scores[base + c];
            if v > val {
                val = v;
                best = c;
            }
        }
        out.push(cands[slot * k + best]);
        pred = best;
    }
    out
}

/// Shared runner, plan cache and profile state for drafters.
pub(crate) struct DraftCore {
    pub graph: Graph,
    pub runner: Box<dyn Run>,
    pub device: Device,
    pub plans: Cache,
    pub context: usize,
    pub slots: usize,
    pub profile: Option<u64>,
    pub launches: u64,
    pub label: &'static str,
}

impl DraftCore {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn open(
        graph: Graph,
        context: usize,
        first_rows: u64,
        wants: Wants,
        recipe: &Recipe,
        settings: &Settings,
        place: Option<&Place>,
        label: &'static str,
        run: impl FnOnce(tessera_lower::Plan) -> Result<Box<dyn Run>, Error>,
    ) -> Result<Self, Error> {
        let slots = graph
            .dims
            .iter()
            .find(|d| d.name == Seqs)
            .map_or(1, |d| d.size.max() as usize);
        let ranked = |device: Device| match place {
            Some(place) => place.device(device),
            None => device,
        };
        let device = ranked(Device::guess(settings).with(recipe));
        let mut plans = Cache::default();
        let first = [
            (Seqs, slots as u64),
            (Rows, first_rows),
            (Context, first_rows),
            (Segments, 1),
        ];
        let (plan, _) = plans.wanting(&graph, 0, wants, &first, &[Context, Segments], &device)?;
        let mut runner = run(plan.clone())?;
        runner.launching(recipe.launch);
        runner.profiling(settings.profile.is_some());
        let device = ranked(
            Device::stored(runner.platform(), settings)
                .map_err(tessera_lower::Error::from)?
                .with(recipe),
        );
        Ok(Self {
            graph,
            runner,
            device,
            plans,
            context,
            slots,
            profile: settings.profile,
            launches: 0,
            label,
        })
    }

    pub(crate) fn exec(
        &mut self,
        feeds: &[Tensor],
        sizes: &[(tessera_graph::DimName, u64)],
        wants: Wants,
    ) -> Result<Vec<Slot>, Error> {
        let (plan, bound) = self.plans.wanting(
            &self.graph,
            0,
            wants,
            sizes,
            &[Context, Segments],
            &self.device,
        )?;
        let outputs = plan.outputs.iter().map(|o| o.slot).collect();
        self.runner.replan(plan)?;
        self.runner.bind_all(&bound);
        let views: Vec<_> = feeds.iter().map(Tensor::view).collect();
        self.runner.feed(&views)?;
        self.runner.submit(0..self.runner.plan().steps.len())?;
        if let Some(every) = self.profile {
            self.launches += 1;
            if every > 0
                && self.launches.is_multiple_of(every)
                && let Some(p) = self.runner.take_profile()
                && self.runner.plan().rank == 0
            {
                eprintln!(
                    "{} profile of {every} launches:\n{}",
                    self.label,
                    p.report(every)
                );
            }
        }
        Ok(outputs)
    }
}

macro_rules! draft_methods {
    ($ty:ty) => {
        impl $ty {
            /// This rank's `(axis, first, count)` slice of `at`, if split.
            pub fn share(&self, at: u32) -> Option<(u32, usize, usize)> {
                self.core.runner.plan().share(at)
            }

            /// Number of weights in the drafter graph.
            pub fn weight_count(&self) -> u32 {
                self.core.graph.weights.len() as u32
            }

            /// Whether the active plan launches kernel `name`.
            pub fn has_kernel(&self, name: &str) -> bool {
                self.core
                    .runner
                    .plan()
                    .steps
                    .iter()
                    .any(|s| s.kernel.name() == name)
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
                Ok(self.core.runner.load_weight(at, w)?)
            }

            pub fn context(&self) -> usize {
                self.core.context
            }

            pub fn device(&self) -> String {
                self.core.runner.device()
            }
        }
    };
}
pub(crate) use draft_methods;

/// Runs a DFlash drafter graph with kept context KV.
pub struct Drafter {
    core: DraftCore,
    block: usize,
    tap_width: usize,
    vocab: usize,
    mask_token: u32,
    causal: bool,
    selector_k: Option<usize>,
}

draft_methods!(Drafter);

fn selector_top_k(graph: &Graph, block: usize) -> Option<usize> {
    let entry = graph.entries.first()?;
    let out = entry.outputs.iter().find(|o| o.role == Role::Candidates)?;
    let n = block.checked_sub(1)?;
    let total = graph.nodes[out.node.0 as usize]
        .ty
        .shape
        .0
        .first()?
        .value(&graph.dims, &|_| None)? as usize;
    (n > 0 && total.is_multiple_of(n)).then_some(total / n)
}

impl Drafter {
    /// Splits one transport into `N` alternating handles.
    pub fn share_transports<const N: usize>(
        transport: Box<dyn Transport>,
    ) -> [Box<dyn Transport>; N] {
        let shared = Rc::new(RefCell::new(transport));
        std::array::from_fn(|_| Box::new(SharedTransport(shared.clone())) as Box<dyn Transport>)
    }

    /// Splits one transport into two alternating handles.
    pub fn share_transport(transport: Box<dyn Transport>) -> [Box<dyn Transport>; 2] {
        Self::share_transports(transport)
    }

    /// Splits one persistent sum grid into `N` alternating handles.
    pub fn share_sums<const N: usize>(grid: Box<dyn Sum>) -> [Box<dyn Sum>; N] {
        let shared = Rc::new(RefCell::new(grid));
        std::array::from_fn(|_| Box::new(SharedSum(shared.clone())) as Box<dyn Sum>)
    }

    /// Splits one persistent sum grid into two alternating handles.
    pub fn share_sum(grid: Box<dyn Sum>) -> [Box<dyn Sum>; 2] {
        Self::share_sums(grid)
    }

    /// Opens a compiled runner for a DFlash stepper graph.
    pub fn open(
        graph: Graph,
        context: usize,
        block: usize,
        mask_token: u32,
        recipe: &Recipe,
        settings: &Settings,
    ) -> Result<Self, Error> {
        let run = |plan| Ok(open(plan, settings)?);
        Self::placed(
            graph, context, block, mask_token, recipe, settings, None, run,
        )
    }

    /// Opens one rank of a TP-sharded DFlash drafter.
    #[allow(clippy::too_many_arguments)]
    pub fn open_ranked<T: Transport + 'static>(
        graph: Graph,
        context: usize,
        block: usize,
        mask_token: u32,
        (recipe, settings): (&Recipe, &Settings),
        mut place: Place,
        sums: T,
        grid: Option<Box<dyn Sum>>,
    ) -> Result<Self, Error> {
        place.rank = sums.rank();
        let run = |plan| open_rank(plan, settings, sums, grid);
        Self::placed(
            graph,
            context,
            block,
            mask_token,
            recipe,
            settings,
            Some(&place),
            run,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn placed(
        graph: Graph,
        context: usize,
        block: usize,
        mask_token: u32,
        recipe: &Recipe,
        settings: &Settings,
        place: Option<&Place>,
        run: impl FnOnce(tessera_lower::Plan) -> Result<Box<dyn Run>, Error>,
    ) -> Result<Self, Error> {
        if block < 2 || block > context {
            return Err(Error::Rows);
        }
        tessera_run::configure(settings);
        let graph = match place {
            Some(place) => {
                let sharded = dflash::shard(&graph, &place.parts, &recipe.pins.shares(&graph))?;
                for w in sharded.whole.iter().filter(|_| place.rank == 0) {
                    let path = &graph.weights[w.weight as usize].path;
                    eprintln!("tessera: drafter {path:?} stays whole: {}", w.why);
                }
                sharded.graph
            }
            None => graph,
        };
        let dim = |name| {
            graph
                .dims
                .iter()
                .find(|d| d.name == name)
                .map(|d| d.size.max() as usize)
        };
        let tap_width = dim(Embedding).ok_or(Error::Taps)?;
        let vocab = dim(Vocab).ok_or(Error::Scores)?;
        let selector_k = selector_top_k(&graph, block);
        let wants = match selector_k {
            Some(_) => Wants::roles(&graph, 0, &[Role::Candidates, Role::Taps]),
            None => Wants::role(&graph, 0, Role::Tokens),
        };
        let core = DraftCore::open(
            graph,
            context,
            block as u64,
            wants,
            recipe,
            settings,
            place,
            "drafter",
            run,
        )?;
        Ok(Self {
            core,
            block,
            tap_width,
            vocab,
            mask_token,
            causal: false,
            selector_k,
        })
    }

    /// Sets whether draft block queries attend causally within the block.
    pub fn set_causal(&mut self, causal: bool) {
        self.causal = causal;
    }

    pub fn block(&self) -> usize {
        self.block
    }

    pub fn tap_width(&self) -> usize {
        self.tap_width
    }

    pub fn mask_token(&self) -> u32 {
        self.mask_token
    }

    pub(crate) fn draft_wants(&self) -> Wants {
        match self.selector_k {
            Some(_) => Wants::roles(&self.core.graph, 0, &[Role::Candidates, Role::Taps]),
            None => Wants::role(&self.core.graph, 0, Role::Tokens),
        }
    }

    pub(crate) fn read_draft(&mut self, outputs: &[Slot]) -> Result<Vec<u32>, Error> {
        if let Some(k) = self.selector_k {
            let cands = self.core.runner.ids(outputs[0])?;
            let raw = self.core.runner.read(outputs[1])?;
            let scores = raw.floats().map_err(tessera_run::Error::from)?;
            return Ok(walk_lattice(&cands, scores, self.block - 1, k));
        }
        let mut ids = self.core.runner.ids(outputs[0])?;
        ids.truncate(self.block - 1);
        Ok(ids)
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
        let wants = self.draft_wants();
        let outputs = self.run(slot, anchor, start, ctx_pos, taps, wants)?;
        self.read_draft(&outputs)
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
        let wants = Wants::role(&self.core.graph, 0, Role::Logits);
        let outputs = self.run(slot, anchor, start, ctx_pos, taps, wants)?;
        let raw = self.core.runner.read(outputs[0])?;
        let logits = raw.floats().map_err(tessera_run::Error::from)?;
        Ok(logits[..(self.block - 1) * self.vocab].to_vec())
    }

    pub(crate) fn run(
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
        if slot as usize >= self.core.slots || start + self.block > self.core.context {
            return Err(Error::RowPlace);
        }
        let filled = if n_new <= self.block {
            self.block
        } else {
            n_new.next_power_of_two().min(self.core.context)
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
        let attend_pos = if self.causal {
            block_pos.clone()
        } else {
            vec![(start + self.block - 1) as u32; self.block]
        };
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
            (Seqs, self.core.slots as u64),
            (Rows, filled as u64),
            (Context, (start + self.block) as u64),
            (Segments, 1),
        ];
        self.core.exec(&feeds, &sizes, wants)
    }
}

/// Target launch decoded from rank 0's control message.
pub(crate) struct TargetCmd {
    pub rows: Vec<Row>,
    pub picks: Vec<u32>,
    pub head: Head,
    pub wants: Wants,
    pub table: Vec<u32>,
}

/// DFlash drafter launch decoded from rank 0's control message.
pub(crate) struct DraftCmd {
    pub slot: u32,
    pub anchor: u32,
    pub start: usize,
    pub ctx_pos: usize,
    pub wants: Wants,
}

/// One control message received by a speculative follower rank.
enum Cmd {
    Target(TargetCmd),
    Draft(DraftCmd),
}

pub(crate) fn encode_draft(
    slot: u32,
    anchor: u32,
    start: usize,
    ctx_pos: usize,
    wants: Wants,
) -> Vec<u8> {
    let mut w = Writer(Vec::with_capacity(24));
    w.u32(DRAFT_TAG)
        .u32(slot)
        .u32(anchor)
        .u32(start as u32)
        .u32(ctx_pos as u32)
        .u32(wants.0);
    w.0
}

pub(crate) fn decode_draft(r: &mut Reader) -> Option<DraftCmd> {
    Some(DraftCmd {
        slot: r.u32()?,
        anchor: r.u32()?,
        start: r.u32()? as usize,
        ctx_pos: r.u32()? as usize,
        wants: Wants(r.u32()?),
    })
}

pub(crate) fn decode_target(tag: u32, r: &mut Reader) -> Option<TargetCmd> {
    let k = r.u32()? as usize;
    let wants = Wants(r.u32()?);
    let table = r.u32s()?;
    let n_words = r.u32()? as usize;
    let raw_bytes = r.take(n_words.checked_mul(4)?)?;
    let mut rr = Reader(raw_bytes);
    let mut rows = Vec::with_capacity(n_words / 3);
    for _ in 0..n_words / 3 {
        let slot = rr.u32()?;
        let token = rr.u32()?;
        let position = rr.u32()?;
        rows.push(Row {
            slot,
            token,
            position,
        });
    }
    let picks = r.u32s()?;
    let list = r.u32s()?;
    let head = match tag {
        0 => Head::Full(None),
        1 => Head::Full(Some(list)),
        _ => Head::Sparse(list, k),
    };
    Some(TargetCmd {
        rows,
        picks,
        head,
        wants,
        table,
    })
}

fn decode_cmd(bytes: &[u8]) -> Option<Cmd> {
    let mut r = Reader(bytes);
    let tag = r.u32()?;
    if tag == DRAFT_TAG {
        return decode_draft(&mut r).map(Cmd::Draft);
    }
    decode_target(tag, &mut r).map(Cmd::Target)
}

/// Output index of `Role::Taps` inside a `wants` subset.
pub(crate) fn taps_output(g: &Graph, entry: usize, wants: Wants) -> Option<usize> {
    g.entries
        .get(entry)?
        .outputs
        .iter()
        .enumerate()
        .filter(|(i, _)| (wants.0 >> i) & 1 == 1)
        .position(|(_, o)| o.role == Role::Taps)
}

/// Exchanges control bytes and gathers rank profiles on rank 0.
pub(crate) fn tell_ranks(target: &mut Engine, mine: &[u8]) -> Result<Option<Vec<u8>>, Error> {
    let Some(control) = &mut target.control else {
        return Ok(None);
    };
    let lead = control.rank() == 0;
    let mut parts = control.exchange(mine).map_err(tessera_run::Error::from)?;
    if lead && let Some((launches, own)) = target.held.take() {
        let mut profiles = vec![Some(own)];
        profiles.extend(parts[1..].iter().map(|p| Profile::from_bytes(p)));
        let ranks = Ranks { launches, profiles };
        eprintln!("profile of {launches} launches:\n{ranks}");
        target.ranks = Some(ranks);
    }
    Ok(Some(parts.swap_remove(0)))
}

/// Executes one decoded target launch on a follower rank.
pub(crate) fn run_target(
    target: &mut Engine,
    cmd: TargetCmd,
    tap_width: usize,
    last_taps: &mut Vec<f32>,
) -> Result<(), Error> {
    if let Some(lanes) = &mut target.lanes
        && !cmd.table.is_empty()
    {
        lanes.replace(cmd.table);
    }
    let entry = cmd.head.entry();
    let outputs = target.run(&cmd.rows, &cmd.picks, cmd.head, cmd.wants)?;
    if let Some(at) = taps_output(&target.graph, entry, cmd.wants) {
        let raw = target.runner.read(outputs[at])?;
        let floats = raw.floats().map_err(tessera_run::Error::from)?;
        let keep = cmd.rows.len() * tap_width;
        last_taps.clear();
        last_taps.extend_from_slice(&floats[..keep]);
    }
    target.report();
    Ok(())
}

/// DFlash speculative decoding over a target and drafter pair.
pub struct Speculate<'a> {
    target: &'a mut Engine,
    drafter: &'a mut Drafter,
    spec_len: SpecLen,
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
        let spec_len = SpecLen::fixed(drafter.block() - 1).with_cost(CostModel::dflash());
        Ok(Self {
            target,
            drafter,
            spec_len,
        })
    }

    /// Configures adaptive or truncated draft length selection.
    pub fn with_spec_len(mut self, spec_len: SpecLen) -> Self {
        self.spec_len = spec_len;
        self
    }

    /// Follows rank 0's target and drafter launches until dismissed.
    pub fn follow(&mut self) -> Result<(), Error> {
        let mut last_taps: Vec<f32> = Vec::new();
        let tap_width = self.drafter.tap_width();
        loop {
            let mine = self
                .target
                .held
                .take()
                .map_or_else(Vec::new, |(_, p)| p.to_bytes());
            let Some(told) = tell_ranks(self.target, &mine)? else {
                return Ok(());
            };
            match decode_cmd(&told) {
                None => return Ok(()),
                Some(Cmd::Target(cmd)) => run_target(self.target, cmd, tap_width, &mut last_taps)?,
                Some(Cmd::Draft(cmd)) => {
                    let n_new = cmd.start.checked_sub(cmd.ctx_pos).ok_or(Error::RowPlace)?;
                    let need = n_new * tap_width;
                    if last_taps.len() < need {
                        return Err(Error::Rows);
                    }
                    self.drafter.run(
                        cmd.slot,
                        cmd.anchor,
                        cmd.start,
                        cmd.ctx_pos,
                        &last_taps[..need],
                        cmd.wants,
                    )?;
                }
            }
        }
    }

    /// Greedy speculative decoding matching target output token-for-token.
    pub fn generate(
        &mut self,
        request: Request,
        emit: &mut dyn FnMut(Event) -> bool,
    ) -> Result<SpecStats, Error> {
        let block = self.drafter.block();
        let tap_width = self.drafter.tap_width();
        let max_ctx = self.target.context().min(self.drafter.context());
        let drafter = &mut *self.drafter;
        let spec = self.spec_len.fresh();
        run_speculate(
            self.target,
            request,
            emit,
            |_, _, taps| (0usize, taps, spec, 1.0f64),
            |target, (ctx_pos, pending, spec, last_pred), start, anchor, remaining| {
                if remaining == 1 || start + block > max_ctx {
                    return Ok(None);
                }
                let wants = drafter.draft_wants();
                if target.control.is_some() {
                    tell_ranks(target, &encode_draft(0, anchor, start, *ctx_pos, wants))?;
                }
                let outputs = drafter.run(0, anchor, start, *ctx_pos, pending, wants)?;
                let mut drafts = drafter.read_draft(&outputs)?;
                let avail = drafts
                    .len()
                    .min(remaining.saturating_sub(1))
                    .min(max_ctx.saturating_sub(start + 1));
                if avail == 0 {
                    return Ok(None);
                }
                let (k, pred) = spec.choose(None, avail, drafts.len());
                *last_pred = pred;
                drafts.truncate(k);
                Ok(Some(drafts))
            },
            |(ctx_pos, pending, spec, last_pred),
             start,
             drafted,
             accepted,
             _,
             mut block_taps,
             stats| {
                spec.observe(drafted, accepted, None, *last_pred);
                stats.dflash_steps += 1;
                let commit = 1 + accepted;
                *ctx_pos = start;
                block_taps.truncate(commit * tap_width);
                *pending = block_taps;
            },
        )
    }
}

/// Shared greedy speculative decode loop for DFlash and MTP.
pub(crate) fn run_speculate<S>(
    target: &mut Engine,
    request: Request,
    emit: &mut dyn FnMut(Event) -> bool,
    init: impl FnOnce(&[u32], u32, Vec<f32>) -> S,
    mut draft: impl FnMut(&mut Engine, &mut S, usize, u32, usize) -> Result<Option<Vec<u32>>, Error>,
    mut advance: impl FnMut(&mut S, usize, usize, usize, &[u32], Vec<f32>, &mut SpecStats),
) -> Result<SpecStats, Error> {
    let mut stats = SpecStats::default();
    let prompt = &request.tokens;
    if prompt.is_empty() || prompt.len() > target.context() {
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
    let (ids, taps) = target.step_taps(&prefill)?;
    let mut anchor = *ids.last().expect("non-empty prompt");
    if let Some(reason) = emit_one(&request, &mut stats, anchor, emit) {
        let _ = emit(Event::Done(reason));
        return Ok(stats);
    }
    let mut state = init(prompt, anchor, taps);
    let mut start = prompt.len();
    while stats.emitted < request.max_tokens {
        if start >= target.context() {
            let _ = emit(Event::Done(Reason::Context));
            return Ok(stats);
        }
        let remaining = request.max_tokens - stats.emitted;
        let Some(drafts) = draft(target, &mut state, start, anchor, remaining)? else {
            let row = [Row {
                slot: 0,
                token: anchor,
                position: start as u32,
            }];
            let next = target.step(&row)?[0];
            start += 1;
            if let Some(reason) = emit_one(&request, &mut stats, next, emit) {
                let _ = emit(Event::Done(reason));
                return Ok(stats);
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
        let (posterior, block_taps) = target.step_taps(&verify)?;
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
        advance(
            &mut state,
            start,
            drafts.len(),
            accepted,
            &posterior,
            block_taps,
            &mut stats,
        );
        start += commit;
        anchor = bonus;
    }
    let _ = emit(Event::Done(Reason::Length));
    Ok(stats)
}

/// Emits one token, returning the stop reason if finished.
pub(crate) fn emit_one(
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
