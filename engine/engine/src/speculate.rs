//! DFlash speculative decoding: cached drafter and verify loop.

pub use crate::spec_len::SpecStats;
use crate::spec_len::{CostModel, SpecLen};
use crate::spec_loop::{D2dCopy, SpecReq, fill_d2d, run_speculate, spec_entry};
use std::cell::RefCell;
use std::rc::Rc;
use tessera_engine_comm::{Error as CommError, Transport};
use tessera_engine_hooks::{
    Error, Event, Hooks, Lanes, Place, TapsIn, context_bucket, open_rank, stored_device,
};
use tessera_graph::DimName::{Context, Embedding, Rows, Segments, Seqs};
use tessera_graph::{DType, Graph, Role};
use tessera_interp::Tensor;
use tessera_lower::{Cache, Device, Recipe, Settings, Slot, Wants, Weight};
use tessera_models::dflash;
use tessera_run::{Error as RunError, Run, Sum, open};

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
    fn sum(&mut self, from: u64, to: u64, len: usize, wire: DType) -> Result<(), RunError> {
        self.0.borrow_mut().sum(from, to, len, wire)
    }

    fn wire(&self) -> DType {
        self.0.borrow().wire()
    }

    fn carries(&self, wire: DType) -> bool {
        self.0.borrow().carries(wire)
    }

    fn shares_bus(&self) -> bool {
        self.0.borrow().shares_bus()
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

    fn reduce_scatter(
        &mut self,
        from: u64,
        to: u64,
        start: usize,
        len: usize,
        total: usize,
        wire: DType,
    ) -> Result<bool, RunError> {
        self.0
            .borrow_mut()
            .reduce_scatter(from, to, start, len, total, wire)
    }

    fn gather_rows(
        &mut self,
        from: u64,
        to: u64,
        start: usize,
        part: usize,
        total: usize,
        wire: DType,
    ) -> Result<bool, RunError> {
        self.0
            .borrow_mut()
            .gather_rows(from, to, start, part, total, wire)
    }

    fn capturable(&self) -> bool {
        self.0.borrow().capturable()
    }

    fn pause(&mut self) -> Result<(), RunError> {
        self.0.borrow_mut().pause()
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

/// Greedy walk returning tokens and softmax log-probs.
fn walk_lattice(cands: &[u32], scores: &[f32], n: usize, k: usize) -> (Vec<u32>, Vec<f32>) {
    let mut out = Vec::with_capacity(n);
    let mut lp = Vec::with_capacity(n);
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
        let mut sum = 0.0f32;
        if val.is_finite() {
            for c in 0..k {
                sum += (scores[base + c] - val).exp();
            }
        }
        lp.push(if sum > 0.0 { -sum.ln() } else { f32::NAN });
        out.push(cands[slot * k + best]);
        pred = best;
    }
    (out, lp)
}

/// Shared runner, plan cache and profile state for drafters.
pub(crate) struct DraftCore {
    pub graph: Graph,
    pub runner: Box<dyn Run>,
    pub device: Device,
    pub plans: Cache,
    pub context: usize,
    pub slots: usize,
    pub lanes: Option<Lanes>,
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
        let mut device = Device::guess(settings).with(recipe);
        if let Some(place) = place {
            device = place.device(device);
        }
        let mut plans = Cache::default();
        let first = [
            (Seqs, slots as u64),
            (Rows, first_rows),
            (Context, first_rows),
            (Segments, 1),
        ];
        let (plan, _) = plans.wanting(&graph, 0, wants, &first, &[Context, Segments], &device)?;
        let lanes = plan.table.zip(recipe.kept.page());
        let pool = plan.pool_pages;
        let lanes = lanes.map(|(table, page)| {
            let pages = if pool > 0 { pool } else { table.len };
            Lanes::new(page, table.len, slots, pages)
        });
        let mut runner = run(plan.clone())?;
        let device = stored_device(runner.as_mut(), recipe, settings, place, true)?;
        Ok(Self {
            graph,
            runner,
            device,
            plans,
            context,
            slots,
            lanes,
            profile: settings.profile,
            launches: 0,
            label,
        })
    }

    pub(crate) fn exec_tap(
        &mut self,
        feeds: &[Option<Tensor>],
        d2d: &[D2dCopy],
        sizes: &[(tessera_graph::DimName, u64)],
        wants: Wants,
        wait: bool,
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
        let inputs = plan.inputs.clone();
        let table = plan.table;
        self.runner.replan(plan)?;
        if let (Some(table), Some(lanes)) = (table, self.lanes.as_mut()) {
            for (offset, bytes) in lanes.take_dirty() {
                self.runner.held_mut().upload_at(table, offset, &bytes)?;
            }
        }
        self.runner.bind_all(&bound);
        if feeds.len() != inputs.len() {
            return Err(RunError::InputCount.into());
        }
        // D2D copies run before host writes.
        for &copy in d2d {
            let slot = *inputs
                .get(copy.input)
                .ok_or(RunError::InputShape(copy.input))?;
            let plain4 = slot.form.block() == 1 && slot.form.bytes(1) == 4;
            let need = copy.pad_to.max(copy.dst_row + copy.rows) * copy.width;
            if !plain4 || need > slot.len {
                return Err(RunError::InputShape(copy.input).into());
            }
            fill_d2d(&mut *self.runner, slot, copy)?;
        }
        for (at, (slot, feed)) in inputs.into_iter().zip(feeds).enumerate() {
            match feed {
                Some(tensor) => {
                    if tensor.len() != slot.len || tensor.dtype() != slot.form.dtype() {
                        return Err(RunError::InputShape(at).into());
                    }
                    self.runner.write(slot, tensor.view())?;
                }
                None if d2d.iter().any(|c| c.input == at) => {}
                None => return Err(RunError::InputShape(at).into()),
            }
        }
        let range = 0..self.runner.plan().steps.len();
        if wait {
            self.runner.submit(range)?;
        } else {
            self.runner.queue(range)?;
        }
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
                if self.core.runner.base().has_extra_weight(at) {
                    return None;
                }
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

            pub fn slots(&self) -> usize {
                self.core.slots
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
    mask_token: u32,
    causal: bool,
    selector_k: Option<usize>,
    sliding_window: Option<usize>,
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
        place.settle(grid.as_deref());
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
                place.log_whole(&graph, "drafter ", &sharded.whole);
                place.resolve(sharded.graph, &recipe.sums)
            }
            None => graph,
        };
        let tap_width = graph
            .dims
            .iter()
            .find(|d| d.name == Embedding)
            .map(|d| d.size.max() as usize)
            .ok_or(Error::Taps)?;
        let selector_k = selector_top_k(&graph, block);
        let sliding_window = graph.nodes.iter().find_map(|n| {
            let tessera_graph::Op::Attend(a) = &n.op else {
                return None;
            };
            let tessera_graph::Mask::Back(w) = a.mask else {
                return None;
            };
            Some(w as usize + 1)
        });
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
            mask_token,
            causal: false,
            selector_k,
            sliding_window,
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

    /// Sliding attention window in tokens, if configured.
    pub fn sliding_window(&self) -> Option<usize> {
        self.sliding_window
    }

    pub(crate) fn draft_wants(&self) -> Wants {
        match self.selector_k {
            Some(_) => Wants::roles(&self.core.graph, 0, &[Role::Candidates, Role::Taps]),
            None => Wants::role(&self.core.graph, 0, Role::Tokens),
        }
    }

    pub(crate) fn read_draft(
        &mut self,
        outputs: &[Slot],
    ) -> Result<(Vec<u32>, Option<Vec<f32>>), Error> {
        if let Some(k) = self.selector_k {
            let cands = self.core.runner.ids(outputs[0])?;
            let raw = self.core.runner.read(outputs[1])?;
            let scores = raw.floats().map_err(tessera_run::Error::from)?;
            let (out, lp) = walk_lattice(&cands, scores, self.block - 1, k);
            return Ok((out, Some(lp)));
        }
        let mut ids = self.core.runner.ids(outputs[0])?;
        ids.truncate(self.block - 1);
        Ok((ids, None))
    }

    pub(crate) fn run_taps(
        &mut self,
        slot: u32,
        anchor: u32,
        start: usize,
        ctx_pos: usize,
        taps: TapsIn<'_>,
        wants: Wants,
    ) -> Result<Vec<Slot>, Error> {
        let n_new = start.checked_sub(ctx_pos).ok_or(Error::RowPlace)?;
        if n_new == 0 {
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
        let (tap_tensor, d2d) = match taps {
            TapsIn::Host(h) => {
                if h.len() != n_new * self.tap_width {
                    return Err(Error::Rows);
                }
                let mut ctx_taps = Vec::with_capacity(filled * self.tap_width);
                ctx_taps.extend_from_slice(h);
                let last = &h[(n_new - 1) * self.tap_width..];
                for _ in n_new..filled {
                    ctx_taps.extend_from_slice(last);
                }
                (
                    Some(Tensor::f32(vec![filled, self.tap_width], ctx_taps)),
                    None,
                )
            }
            TapsIn::Device { addr, rows, .. } => {
                if rows != n_new {
                    return Err(Error::Rows);
                }
                (
                    None,
                    Some(D2dCopy {
                        input: 4,
                        src: addr,
                        dst_row: 0,
                        rows,
                        pad_to: filled,
                        width: self.tap_width,
                    }),
                )
            }
        };
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
            Some(Tensor::u32(vec![self.block], block_tokens)),
            Some(Tensor::u32(vec![self.block], block_pos)),
            Some(Tensor::u32(vec![self.block], block_slots)),
            Some(Tensor::u32(vec![self.block], attend_pos)),
            tap_tensor,
            Some(Tensor::u32(vec![kv_len], kv_pos)),
            Some(Tensor::u32(vec![kv_len], kv_slots)),
        ];
        let sizes = [
            (Seqs, self.core.slots as u64),
            (Rows, filled as u64),
            (
                Context,
                context_bucket(start + self.block, self.core.context),
            ),
            (Segments, 1),
        ];
        self.core
            .exec_tap(&feeds, d2d.as_slice(), &sizes, wants, true)
    }
}

/// DFlash speculative decoding over a target and drafter pair.
pub struct Speculate<'a> {
    target: &'a mut dyn Hooks,
    drafter: &'a mut Drafter,
    spec_len: SpecLen,
    prefill_chunk: Option<usize>,
}

impl<'a> Speculate<'a> {
    /// Pairs a tapped target engine with its DFlash drafter.
    pub fn new(target: &'a mut dyn Hooks, drafter: &'a mut Drafter) -> Result<Self, Error> {
        let width = target.tap_width().ok_or(Error::Taps)?;
        if width != drafter.tap_width() {
            return Err(Error::Rows);
        }
        let spec_len = SpecLen::fixed(drafter.block() - 1).with_cost(CostModel::dflash());
        Ok(Self {
            target,
            drafter,
            spec_len,
            prefill_chunk: None,
        })
    }
}

spec_entry!(
    Speculate<'a>,
    SpecLen,
    spec_len => with_spec_len,
    |s, target, req, spec, emit| run_dflash_req(target, s.drafter, req, spec, emit)
);

fn run_dflash_req(
    target: &mut dyn Hooks,
    drafter: &mut Drafter,
    req: SpecReq,
    spec: SpecLen,
    emit: &mut dyn FnMut(Event) -> bool,
) -> Result<SpecStats, Error> {
    let (block, tap_width) = (drafter.block(), drafter.tap_width());
    let (max_ctx, d_ctx) = (target.context(), drafter.context());
    let win = drafter.sliding_window();
    let drop_before = win.map_or(0, |w| req.request.tokens.len().saturating_sub(w));
    run_speculate(
        target,
        drafter,
        req,
        emit,
        |_, drafter, pos, prompt, len, taps| {
            let end = pos + len;
            if end > drop_before && (prompt.len() - drop_before) + block <= d_ctx {
                let skip = drop_before.saturating_sub(pos);
                let (d_end, d_pos) = (end - drop_before, pos + skip - drop_before);
                let (in_t, w) = (taps.as_in_from(skip, tap_width), drafter.draft_wants());
                drafter.run_taps(0, prompt[end], d_end, d_pos, in_t, w)?;
            }
            Ok(())
        },
        |_, _, pos, _, _, taps| {
            let skip = drop_before.saturating_sub(pos);
            Ok((pos + skip - drop_before, skip, taps, spec, None::<Vec<f32>>))
        },
        |_, drafter, (ctx_pos, skip, pending, spec, last_lp), start, anchor, remaining| {
            let d_start = start - drop_before;
            if remaining == 1 || start + block > max_ctx || d_start + block > d_ctx {
                return Ok(None);
            }
            let (in_t, w) = (pending.as_in_from(*skip, tap_width), drafter.draft_wants());
            let out = drafter.run_taps(0, anchor, d_start, *ctx_pos, in_t, w)?;
            let (mut drafts, lp) = drafter.read_draft(&out)?;
            let avail = drafts.len().min(remaining - 1).min(max_ctx - start - 1);
            if avail == 0 {
                return Ok(None);
            }
            let k = spec.choose(lp.as_deref(), avail, drafts.len());
            *last_lp = lp;
            drafts.truncate(k);
            Ok(Some(drafts))
        },
        |_,
         _,
         (ctx_pos, skip, pending, spec, last_lp),
         start,
         drafted,
         accepted,
         _,
         mut block_taps,
         stats| {
            spec.observe(drafted, accepted, last_lp.as_deref());
            stats.dflash_steps += 1;
            let commit = 1 + accepted;
            *ctx_pos = start - drop_before;
            *skip = 0;
            block_taps.keep_rows(commit, tap_width);
            *pending = block_taps;
            Ok(())
        },
    )
}
