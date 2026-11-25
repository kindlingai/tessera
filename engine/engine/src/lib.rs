//! Graphs on runners: decode loops and single launches.

// Linking the crates registers their rules.
use tessera_rules as _;

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::mpsc;
use std::time::Instant;
use tessera_graph::DimName::{
    Candidates, Context, EditWords, Picks, Rows, Segments, Seqs, VocabWords,
};
use tessera_graph::{EntryKind, Role};
use tessera_lower::{COMMIT, Cache, Device, PEEK, Report, Slot, Wants};
use tessera_run::{Run, open};

mod batch_dual;
mod batch_spec;
mod dual;
mod guide;
mod kept;
mod lanes;
pub mod metrics;
#[cfg(test)]
mod metrics_test;
mod model;
mod mtp_draft;
mod ple;
pub mod progress;
#[cfg(test)]
mod progress_test;
mod ranks;
mod sample;
#[cfg(test)]
mod sample_test;
mod schedule;
mod spec_len;
mod spec_loop;
mod speculate;

pub use batch_spec::BatchMtp;
pub use dual::{DualSelector, DualSpeculate};
pub use guide::{
    Allowed, Edit, FEW, Fault, Guide, Move, Penalties, Prefix, Program, Scoped, Stacked, Tally,
    Term,
};
pub use kept::Totals;
pub use model::Model;
pub use mtp_draft::{MtpDrafter, MtpSpeculate};
pub use ple::{PleConfig, PleStore};
pub use ranks::Place;
pub use sample::{Sampling, Target, Weights, accept};
pub use schedule::{Batch, Fair, Policy, Seq, Sink};
pub use spec_len::{CostModel, SpecLen};
pub use speculate::{Drafter, SpecStats, Speculate};

// The front end reaches the engine crates only through these.
pub use tessera_device::{p2p, sums, whole};
pub use tessera_engine_comm as comm;
pub use tessera_engine_settings::{Ask, Asks, Backend, Memory, Reads};
pub use tessera_graph::{Bf16, DType, DimExpr, DimName, E4M3, F16, Graph, PathSeg, Shape};
pub use tessera_interp::{Tensor, bind, fold};
pub use tessera_lower::{
    Compute, Dtype, Filed, Knobs, Pick, Pins, Recipe, Settings, Site, Weight, form,
};
pub use tessera_mem::form as storage;
pub use tessera_mem::{block, trellis};
// The front end's name for `mtp`.
pub use tessera_models::mtp as mtp_model;
pub use tessera_models::*;
pub use tessera_run::{Grid, gpu, pinning, select, sizer};
pub use tessera_trace::{Linear, Norm};
pub use tessera_wasm as wasm;

/// Tokens to continue, and when to stop.
#[derive(Default, Serialize, Deserialize)]
pub struct Request {
    pub tokens: Vec<u32>,
    pub max_tokens: usize,
    /// Ids that end the answer, excluded from its text.
    pub stop: Vec<u32>,
    /// Rules on each token. Masks need a mask port.
    #[serde(skip)]
    pub guide: Option<Box<dyn Guide>>,
    /// Prompt rows to score. Any reads replace decoding.
    #[serde(default)]
    pub reads: Vec<Read>,
    /// Sampling parameters. Default is greedy argmax.
    #[serde(default)]
    pub sampling: Sampling,
}

/// The logits after prompt row `row`, over `ids`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Read {
    pub row: u32,
    pub ids: Vec<u32>,
}

/// Why an answer ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Reason {
    Stop,
    Length,
    /// The context filled before a stop.
    Context,
    /// The guide failed. Other answers go on.
    Guide,
}

/// One decoded token, or the end.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Event {
    /// Prompt rows that kept rows cover. Sent before any token.
    Kept(usize),
    Token(u32),
    /// Each read's logits, in the request's order.
    Scores(Vec<Vec<f32>>),
    Done(Reason),
}

/// A failure while planning, loading or running.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Lower(#[from] tessera_lower::Error),
    #[error(transparent)]
    Run(#[from] tessera_run::Error),
    #[error("interpreter: {0}")]
    Interp(#[from] tessera_interp::Error),
    #[error("the rows do not fit the engine")]
    Rows,
    #[error("a row's slot or position is out of range")]
    RowPlace,
    #[error("an input has no axis for a bounded dimension")]
    InputAxis,
    #[error(transparent)]
    Graph(#[from] tessera_graph::VerifyError),
    #[error("the masks do not fit the graph's mask port")]
    Mask,
    #[error("kept rows: {0}")]
    Kept(#[from] tessera_kv::Error),
    #[error("a slot's next row is in another run's page")]
    SharedPage,
    #[error("the graph has no entry that scores these ids")]
    Scores,
    #[error("a bucketed dimension sits inside a larger axis")]
    BucketAxis,
    #[error("the graph has no tap output")]
    Taps,
    #[error("entry {entry} has an input the engine does not fill: {role:?}")]
    Port { entry: usize, role: Role },
    #[error("the graph has no step entry without candidates")]
    Full,
    #[error("ple store: {0}")]
    Ple(#[from] std::io::Error),
    #[error("drafts rewind slot state that the graph cannot restore")]
    Stateful,
}

/// Row width of `entry`'s `role` output, if any.
fn width(g: &Graph, entry: usize, role: Role) -> Option<usize> {
    let out = g.entries[entry].outputs.iter().find(|o| o.role == role)?;
    let last = g.node(out.node).ty.shape.0.last()?;
    last.value(&g.dims, &|_| None).map(|n| n as usize)
}

/// Whether the engine fills input ports of `role`.
fn fillable(role: Role) -> bool {
    matches!(
        role,
        Role::Tokens
            | Role::Positions
            | Role::Slots
            | Role::Picks
            | Role::Allowed
            | Role::Candidates
            | Role::Ple
    )
}

/// The first plain and the first scoring step entries.
fn steps(g: &Graph) -> Result<(Option<usize>, Option<usize>), Error> {
    let (mut full, mut sparse) = (None, None);
    for (at, e) in g.entries.iter().enumerate() {
        if e.kind != EntryKind::Step {
            continue;
        }
        let scored = e.inputs.iter().any(|p| p.role == Role::Candidates);
        let found = if scored { &mut sparse } else { &mut full };
        if found.is_some() {
            continue;
        }
        if let Some(p) = e.inputs.iter().find(|p| !fillable(p.role)) {
            return Err(Error::Port {
                entry: at,
                role: p.role,
            });
        }
        *found = Some(at);
    }
    Ok((full, sparse))
}

/// Rows in, picked rows' likeliest ids out, on one runner.
pub struct Engine {
    graph: Graph,
    runner: Box<dyn Run>,
    device: Device,
    plans: Cache,
    context: usize,
    /// Kept-row lanes the graph declares.
    slots: usize,
    /// The page table, when kept rows are paged.
    lanes: Option<lanes::Lanes>,
    /// Mask words per pick, if the graph takes masks.
    words: Option<usize>,
    /// The most edits per pick, if the graph takes edits.
    edit_cap: Option<usize>,
    /// The step entry that reads every vocabulary row.
    full: usize,
    /// The step entry that picks among candidate ids, if any.
    sparse: Option<usize>,
    /// Floats per tapped row, if the full entry emits taps.
    taps: Option<usize>,
    /// Vocabulary width of the full entry's logits output.
    vocab: Option<usize>,
    /// Carries rank 0's launches to the other ranks.
    control: Option<Box<dyn comm::Transport>>,
    /// Launches per profile report, and launches since the last.
    profile: Option<[u64; 2]>,
    /// Requests that ask for a profile, under dynamic profiling.
    asks: Option<Asks>,
    /// A ranked profile until the next control round carries it.
    held: Option<(u64, tessera_run::Profile)>,
    /// The last profiles rank 0 gathered from all ranks.
    ranks: Option<tessera_run::Ranks>,
    /// Host n-gram embedding table for models with a PLE port.
    ple: Option<PleStore>,
    /// Whether slots carry state that a kept prefix cannot restore.
    stateful: bool,
    /// Rows a pending verify can promote, if any.
    verified: Option<usize>,
}

/// Inputs a launch feeds before its picks.
pub(crate) enum Head {
    /// The full entry. Per-pick mask words, or None for all.
    Full(Option<Vec<u32>>),
    /// The sparse entry, with `k` candidate ids per pick.
    Sparse(Vec<u32>, usize),
    /// Full entry: masks, and `k` edits per pick.
    Edited(Vec<u32>, Vec<u32>, usize),
    /// The full entry, leaving recurrent state for a commit.
    Peek,
}

/// Packs `k` edits: ids, adds, reps. Pads are neutral.
fn pack(edits: &[Edit], k: usize, out: &mut Vec<u32>) {
    let sorted = guide::merged(edits.to_vec());
    let pad = Edit {
        id: u32::MAX,
        add: 0.0,
        rep: 1.0,
    };
    let all = sorted.iter().chain(std::iter::repeat(&pad)).take(k);
    let at = out.len();
    out.extend(all.clone().map(|e| e.id));
    out.extend(all.clone().map(|e| e.add.to_bits()));
    out.extend(all.map(|e| e.rep.to_bits()));
    debug_assert_eq!(out.len(), at + 3 * k);
}

/// A token at a position, in a slot's sequence.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Row {
    pub slot: u32,
    pub token: u32,
    pub position: u32,
}

/// Each slot's last row in `rows`, in order.
pub(crate) fn ends(rows: &[Row]) -> Vec<u32> {
    let last = |(i, r): (usize, &Row)| !rows[i + 1..].iter().any(|o| o.slot == r.slot);
    let at = rows.iter().enumerate().filter(|&p| last(p));
    at.map(|(i, _)| i as u32).collect()
}

/// Rounds a causal context length up to a power-of-two bucket.
pub(crate) fn context_bucket(needed: usize, cap: usize) -> u64 {
    needed.max(256).next_power_of_two().min(cap) as u64
}

/// Rounds a launch row count up to compact buckets.
pub(crate) fn row_bucket(n: usize) -> usize {
    if n <= 24 {
        n.max(1)
    } else if n <= 64 {
        n.div_ceil(8) * 8
    } else {
        n.next_power_of_two()
    }
}

/// Sizes and padded counts for `rows` returning `picks`.
fn launch_of(
    rows: &[Row],
    picks: &[u32],
    slots: usize,
    max_context: usize,
) -> (Vec<(tessera_graph::DimName, u64)>, [usize; 2]) {
    let furthest = rows.iter().map(|r| r.position).max().unwrap_or(0);
    let mut seen: Vec<u32> = rows.iter().map(|r| r.slot).collect();
    seen.sort_unstable();
    seen.dedup();
    let filled = row_bucket(rows.len());
    let wanted = row_bucket(picks.len());
    let sizes = vec![
        (Seqs, slots as u64),
        (Rows, filled as u64),
        (Context, context_bucket(furthest as usize + 1, max_context)),
        (Segments, seen.len() as u64),
        (Picks, wanted as u64),
        // No edit words, until a launch packs edits.
        (EditWords, 0),
    ];
    (sizes, [filled, wanted])
}

impl Engine {
    /// Opens a runner for `graph` with room for `context` rows.
    pub fn open(
        graph: Graph,
        context: usize,
        recipe: &Recipe,
        settings: &Settings,
    ) -> Result<Self, Error> {
        let run = |plan| Ok(open(plan, settings)?);
        Self::placed(graph, context, recipe, settings, None, run)
    }

    /// As `open`. A `place` shards the graph for its rank.
    fn placed(
        graph: Graph,
        context: usize,
        recipe: &Recipe,
        settings: &Settings,
        place: Option<&Place>,
        run: impl FnOnce(tessera_lower::Plan) -> Result<Box<dyn Run>, Error>,
    ) -> Result<Self, Error> {
        tessera_run::configure(settings);
        tessera_wasm::configure(settings.dump_sites.clone());
        let (full, sparse) = steps(&graph)?;
        let Some(full) = full else {
            tessera_graph::trim(&graph, 0)?;
            return Err(Error::Full);
        };
        let mut graph = tessera_graph::trim(&graph, full)?;
        if let Some(at) = sparse {
            graph = tessera_graph::trim(&graph, at)?;
        }
        let ranked = |device: Device| match place {
            Some(place) => place.device(device),
            None => device,
        };
        if let Some(place) = place {
            graph = place.split(&graph, (&recipe.pins, &recipe.sums))?;
        }
        let device = ranked(Device::guess(settings).with(recipe));
        let mut plans = Cache::default();
        let most = |name| graph.dims.iter().find(|d| d.name == name);
        let slots = most(Seqs).map_or(1, |d| d.size.max() as usize);
        // Kept rows end at the graph's context.
        let context = most(Context).map_or(context, |d| context.min(d.size.max() as usize));
        let first = [Row {
            slot: 0,
            token: 0,
            position: 0,
        }];
        let (sizes, _) = launch_of(&first, &[0], slots, context);
        let plan = plans.lower(&graph, full, &sizes, &device)?;
        let lanes = plan.table.zip(recipe.kept.page());
        let pool = plan.pool_pages;
        let lanes = lanes.map(|(table, page)| {
            let pages = if pool > 0 { pool } else { table.len };
            lanes::Lanes::new(page, table.len, slots, pages)
        });
        let mut runner = run(plan)?;
        runner.launching(recipe.launch);
        runner.profiling(settings.profile.is_some());
        let device = Device::stored(runner.platform(), settings)
            .map_err(tessera_lower::Error::from)?
            .with(recipe);
        let device = ranked(device);
        let words = graph.dims.iter().find(|d| d.name == VocabWords);
        let words = words.map(|d| d.size.max() as usize);
        let edit_cap = graph.dims.iter().find(|d| d.name == EditWords);
        let edit_cap = edit_cap.map(|d| d.size.max() as usize / 3);
        let taps = width(&graph, full, Role::Taps);
        let vocab = width(&graph, full, Role::Logits);
        let stateful = graph.nodes.iter().any(|n| n.op.recurrent());
        Ok(Self {
            graph,
            runner,
            device,
            plans,
            context,
            slots,
            lanes,
            words,
            edit_cap,
            full,
            sparse,
            taps,
            vocab,
            control: None,
            profile: settings.profile.map(|every| [every, 0]),
            asks: settings.asks.clone(),
            held: None,
            ranks: None,
            ple: None,
            stateful,
            verified: None,
        })
    }

    /// Attaches a host PLE store for `Role::Ple` graphs.
    pub fn set_ple(&mut self, store: PleStore) {
        self.ple = Some(store);
        self.stateful = true;
    }

    /// Whether slots carry state that a kept prefix cannot restore.
    pub fn stateful(&self) -> bool {
        self.stateful
    }

    /// Runs resident steps through `grid` from now on.
    pub fn persist(&mut self, grid: Box<dyn Grid>) -> Result<(), Error> {
        Ok(self.runner.persist(grid)?)
    }

    /// Uploads the graph's weights, in its order.
    pub fn load(&mut self, weights: &[Weight]) -> Result<(), Error> {
        for (at, w) in weights.iter().enumerate() {
            self.load_weight(at as u32, w)?;
        }
        Ok(())
    }

    /// This rank's `(axis, first, count)` slice of `at`, if split.
    pub fn share(&self, at: u32) -> Option<(u32, usize, usize)> {
        self.runner.plan().share(at)
    }

    /// This rank's first-axis rows of weight `at`, if split.
    pub fn rows(&self, at: u32) -> Option<(usize, usize)> {
        self.runner.plan().rows(at)
    }

    /// Uploads weight `at` of the graph.
    pub fn load_weight(&mut self, at: u32, w: &Weight) -> Result<(), Error> {
        Ok(self.runner.load_weight(at, w)?)
    }

    /// The backend and hardware under the runner.
    pub fn device(&self) -> String {
        self.runner.device()
    }

    /// How often the plan in place launches each kernel.
    pub fn kernels(&self) -> BTreeMap<&'static str, usize> {
        let mut by = BTreeMap::new();
        for step in &self.runner.plan().steps {
            *by.entry(step.kernel.name()).or_default() += step.launches().len();
        }
        by
    }

    /// Floats per tapped row, if drafts may rewind slots.
    pub(crate) fn draft_taps(&self) -> Result<usize, Error> {
        if self.stateful {
            return Err(Error::Stateful);
        }
        self.taps.ok_or(Error::Taps)
    }

    /// Fast-path diagnostic report for a `rows`-row decode launch.
    pub fn lowering_report(&mut self, rows: usize) -> Result<Report, Error> {
        let rows = rows.max(1);
        let max_pos = self.context.saturating_sub(1);
        let sample: Vec<Row> = (0..rows)
            .map(|i| Row {
                slot: (i % self.slots) as u32,
                token: 0,
                position: (i / self.slots).min(max_pos) as u32,
            })
            .collect();
        let picks: Vec<u32> = (0..rows as u32).collect();
        let (sizes, [filled, _]) = launch_of(&sample, &picks, self.slots, self.context);
        let wants = match self.taps.is_some() {
            true => Wants::roles(&self.graph, self.full, &[Role::Tokens, Role::Taps]),
            false => Wants::role(&self.graph, self.full, Role::Tokens),
        };
        let mut report = self.plans.explain(
            &self.graph,
            self.full,
            wants,
            &sizes,
            &[Context, Segments],
            &self.device,
        )?;
        report.label = format!("rows={filled}");
        Ok(report)
    }

    /// Mask words per pick, if the graph takes masks.
    pub fn words(&self) -> Option<usize> {
        self.words
    }

    /// Stores state after the last verify's first `rows` rows.
    pub fn commit(&mut self, rows: usize) -> Result<(), Error> {
        // Only the verify just before it can be committed.
        let Some(whole) = self.verified.take() else {
            return Ok(());
        };
        let plan = self.runner.plan();
        let steps = match rows == whole {
            true => plan.promote(),
            false => plan.commit(),
        };
        if steps.is_empty() {
            return Ok(());
        }
        self.runner.bind(COMMIT, rows as u32);
        Ok(self.runner.submit(steps)?)
    }

    /// Every slot's recurrent state, one kept buffer after another.
    pub fn recurrent_state(&self) -> Result<Vec<f32>, Error> {
        let plan = self.runner.plan();
        let mut all = Vec::new();
        for ((_, slot), info) in plan.kept.iter().zip(&plan.kept_info) {
            if info.context == 1 {
                let raw = self.runner.read(*slot)?;
                all.extend_from_slice(raw.floats().map_err(tessera_run::Error::from)?);
            }
        }
        Ok(all)
    }

    /// Floats per tapped row, if the full entry emits taps.
    pub fn tap_width(&self) -> Option<usize> {
        self.taps
    }

    /// Maximum sequence length in tokens.
    pub fn context(&self) -> usize {
        self.context
    }

    pub fn slots(&self) -> usize {
        self.slots
    }

    /// One launch of `rows`: the likeliest token after each.
    pub fn step(&mut self, rows: &[Row]) -> Result<Vec<u32>, Error> {
        let all: Vec<u32> = (0..rows.len() as u32).collect();
        self.launch(rows, &all, Head::Full(None))
    }

    /// One launch of `rows`: likeliest tokens and tapped states.
    pub fn step_taps(&mut self, rows: &[Row]) -> Result<(Vec<u32>, Vec<f32>), Error> {
        let width = self.taps.ok_or(Error::Taps)?;
        let all: Vec<u32> = (0..rows.len() as u32).collect();
        let wants = Wants::roles(&self.graph, self.full, &[Role::Tokens, Role::Taps]);
        let outputs = self.run(rows, &all, Head::Full(None), wants)?;
        let mut ids = self.runner.ids(outputs[0])?;
        ids.truncate(rows.len());
        let raw = self.runner.read(outputs[1])?;
        let taps = raw.floats().map_err(tessera_run::Error::from)?;
        let taps = taps[..rows.len() * width].to_vec();
        self.report();
        Ok((ids, taps))
    }

    /// Local-only step without control broadcast, for SPMD loops.
    pub(crate) fn step_local(&mut self, rows: &[Row]) -> Result<Vec<u32>, Error> {
        let all: Vec<u32> = (0..rows.len() as u32).collect();
        let wants = Wants::role(&self.graph, 0, Role::Tokens);
        let outputs = self.run_local(rows, &all, Head::Full(None), wants)?;
        let mut ids = self.runner.ids(outputs[0])?;
        ids.truncate(rows.len());
        self.report();
        Ok(ids)
    }

    /// Local-only step returning the last row's predicted token.
    pub(crate) fn step_last_local(&mut self, rows: &[Row]) -> Result<u32, Error> {
        let pick = [(rows.len() - 1) as u32];
        let wants = Wants::role(&self.graph, self.full, Role::Tokens);
        let outputs = self.run_local(rows, &pick, Head::Full(None), wants)?;
        let ids = self.runner.ids(outputs[0])?;
        self.report();
        Ok(ids[0])
    }

    /// Local-only tapped step keeping device taps resident when possible.
    pub(crate) fn step_taps_local(
        &mut self,
        rows: &[Row],
    ) -> Result<(Vec<u32>, spec_loop::TapsBuf), Error> {
        self.tapped(rows, false)
    }

    /// As `step_taps_local`, leaving recurrent state for `commit`.
    pub(crate) fn verify_local(
        &mut self,
        rows: &[Row],
    ) -> Result<(Vec<u32>, spec_loop::TapsBuf), Error> {
        self.tapped(rows, self.stateful)
    }

    /// A tapped launch that leaves recurrent state for `commit`.
    pub fn verify(&mut self, rows: &[Row]) -> Result<(Vec<u32>, Vec<f32>), Error> {
        let width = self.taps.ok_or(Error::Taps)?;
        let (ids, taps) = self.verify_local(rows)?;
        Ok((ids, taps.into_host(self.runner.as_ref(), width)?))
    }

    fn tapped(
        &mut self,
        rows: &[Row],
        peek: bool,
    ) -> Result<(Vec<u32>, spec_loop::TapsBuf), Error> {
        let width = self.taps.ok_or(Error::Taps)?;
        let all: Vec<u32> = (0..rows.len() as u32).collect();
        let wants = Wants::roles(&self.graph, 0, &[Role::Tokens, Role::Taps]);
        let head = if peek { Head::Peek } else { Head::Full(None) };
        let outputs = self.run_local(rows, &all, head, wants)?;
        let mut ids = self.runner.ids(outputs[0])?;
        ids.truncate(rows.len());
        let slot = outputs[1];
        let taps = if slot.form.name() == "f32"
            && let Some(addr) = self.runner.address(slot)
        {
            spec_loop::TapsBuf::Device {
                slot,
                addr,
                rows: rows.len(),
            }
        } else {
            let raw = self.runner.read(slot)?;
            let floats = raw.floats().map_err(tessera_run::Error::from)?;
            spec_loop::TapsBuf::Host(floats[..rows.len() * width].to_vec())
        };
        self.report();
        Ok((ids, taps))
    }

    /// One launch of `rows`: full logits and tapped states.
    pub fn step_logits(&mut self, rows: &[Row]) -> Result<(Vec<f32>, Vec<f32>), Error> {
        let width = self.taps.ok_or(Error::Taps)?;
        let vocab = self.vocab.ok_or(Error::Scores)?;
        let all: Vec<u32> = (0..rows.len() as u32).collect();
        let wants = Wants::roles(&self.graph, self.full, &[Role::Logits, Role::Taps]);
        let outputs = self.run(rows, &all, Head::Full(None), wants)?;
        let raw_logits = self.runner.read(outputs[0])?;
        let logits = raw_logits.floats().map_err(tessera_run::Error::from)?;
        let logits = logits[..rows.len() * vocab].to_vec();
        let raw_taps = self.runner.read(outputs[1])?;
        let taps = raw_taps.floats().map_err(tessera_run::Error::from)?;
        let taps = taps[..rows.len() * width].to_vec();
        self.report();
        Ok((logits, taps))
    }

    /// One launch: the likeliest token after each slot's last row.
    pub fn last(&mut self, rows: &[Row]) -> Result<Vec<u32>, Error> {
        self.launch(rows, &ends(rows), Head::Full(None))
    }

    /// One launch: full logits after each slot's last row.
    pub fn last_logits(&mut self, rows: &[Row]) -> Result<(Vec<f32>, usize), Error> {
        let picks = ends(rows);
        let vocab = self.vocab.ok_or(Error::Scores)?;
        let wants = Wants::role(&self.graph, self.full, Role::Logits);
        let outputs = self.run(rows, &picks, Head::Full(None), wants)?;
        let raw = self.runner.read(outputs[0])?;
        let floats = raw.floats().map_err(tessera_run::Error::from)?;
        let logits = floats[..picks.len() * vocab].to_vec();
        self.report();
        Ok((logits, vocab))
    }

    /// As `last`, each pick from its `allowed` set.
    pub fn last_guided(&mut self, rows: &[Row], allowed: &[Allowed]) -> Result<Vec<u32>, Error> {
        let picks = ends(rows);
        if allowed.len() != picks.len() {
            return Err(Error::Mask);
        }
        let mut lists = Vec::with_capacity(allowed.len());
        for a in allowed {
            match a {
                Allowed::One(id) => lists.push(vec![*id]),
                Allowed::Few(ids) if !ids.is_empty() && ids.len() <= FEW => lists.push(ids.clone()),
                _ => break,
            }
        }
        let k = lists
            .iter()
            .map(Vec::len)
            .max()
            .unwrap_or(1)
            .next_power_of_two();
        let gathered = picks.len().next_power_of_two() * k;
        let words = self.words.ok_or(Error::Mask)?;
        // Past half the vocabulary, the full head reads less.
        if self.sparse.is_some() && lists.len() == allowed.len() && 2 * gathered <= 32 * words {
            let mut ids = Vec::with_capacity(lists.len() * k);
            for list in &lists {
                // Repeats of the first id never win over it.
                ids.extend(list.iter().chain(std::iter::repeat(&list[0])).take(k));
            }
            let at = self.launch(rows, &picks, Head::Sparse(ids, k))?;
            return Ok(lists.iter().zip(at).map(|(l, i)| l[i as usize]).collect());
        }
        let bits = allowed.iter().flat_map(|a| a.bits(words)).collect();
        self.launch(rows, &picks, Head::Full(Some(bits)))
    }

    /// Whether the device head takes `n` edits in one pick.
    pub fn edits_fit(&self, n: usize) -> bool {
        self.words.is_some() && self.edit_cap.is_some_and(|cap| n <= cap)
    }

    /// As `last_guided`, with each pick's logits changed by its edits.
    pub fn last_edited(
        &mut self,
        rows: &[Row],
        allowed: &[Allowed],
        edits: &[&[Edit]],
    ) -> Result<Vec<u32>, Error> {
        let picks = ends(rows);
        let most = edits.iter().map(|e| e.len()).max().unwrap_or(0);
        if allowed.len() != picks.len() || edits.len() != picks.len() || !self.edits_fit(most) {
            return Err(Error::Mask);
        }
        let words = self.words.ok_or(Error::Mask)?;
        let bits = allowed.iter().flat_map(|a| a.bits(words)).collect();
        let k = most
            .max(1)
            .next_power_of_two()
            .min(self.edit_cap.unwrap_or(1));
        let mut packed = Vec::with_capacity(edits.len() * 3 * k);
        for e in edits {
            pack(e, k, &mut packed);
        }
        self.launch(rows, &picks, Head::Edited(bits, packed, k))
    }

    /// Each pick's logits over its `ids`, in one launch.
    pub fn scores(
        &mut self,
        rows: &[Row],
        picks: &[u32],
        ids: &[Vec<u32>],
    ) -> Result<Vec<Vec<f32>>, Error> {
        let fits = |l: &Vec<u32>| !l.is_empty() && l.len() <= FEW;
        if self.sparse.is_none() || ids.len() != picks.len() || !ids.iter().all(fits) {
            return Err(Error::Scores);
        }
        let k = ids
            .iter()
            .map(Vec::len)
            .max()
            .unwrap_or(1)
            .next_power_of_two();
        let mut flat = Vec::with_capacity(ids.len() * k);
        for list in ids {
            flat.extend(list.iter().chain(std::iter::repeat(&list[0])).take(k));
        }
        let head = Head::Sparse(flat, k);
        let wants = Wants::role(&self.graph, self.entry(&head)?, Role::Logits);
        let outputs = self.run(rows, picks, head, wants)?;
        let all = self.runner.read(outputs[0])?;
        let all = all.floats().map_err(tessera_run::Error::from)?;
        let each = all.chunks(k).zip(ids);
        Ok(each.map(|(row, list)| row[..list.len()].to_vec()).collect())
    }

    /// The entry `head` launches.
    fn entry(&self, head: &Head) -> Result<usize, Error> {
        match head {
            Head::Full(_) | Head::Edited(..) | Head::Peek => Ok(self.full),
            Head::Sparse(..) => self.sparse.ok_or(Error::Scores),
        }
    }

    /// One launch of `rows`, the likeliest ids after `picks`.
    fn launch(&mut self, rows: &[Row], picks: &[u32], head: Head) -> Result<Vec<u32>, Error> {
        let wants = Wants::role(&self.graph, self.entry(&head)?, Role::Tokens);
        let outputs = self.run(rows, picks, head, wants)?;
        let ran = Instant::now();
        let mut ids = self.runner.ids(outputs[0])?;
        ids.truncate(picks.len());
        self.runner.note("engine: ids", ran.elapsed().as_secs_f64());
        self.report();
        Ok(ids)
    }

    fn check_rows(&self, rows: &[Row]) -> Result<(), Error> {
        if rows.is_empty() {
            return Err(Error::Rows);
        }
        if rows
            .iter()
            .any(|r| r.slot as usize >= self.slots || r.position as usize >= self.context)
        {
            return Err(Error::RowPlace);
        }
        Ok(())
    }

    /// One launch of `rows`. Returns the `wants` output slots.
    fn run(
        &mut self,
        rows: &[Row],
        picks: &[u32],
        head: Head,
        wants: Wants,
    ) -> Result<Vec<Slot>, Error> {
        self.check_rows(rows)?;
        self.announce(rows, picks, &head, wants)?;
        self.run_local(rows, picks, head, wants)
    }

    /// Runs `rows` on this rank without control broadcast.
    pub(crate) fn run_local(
        &mut self,
        rows: &[Row],
        picks: &[u32],
        head: Head,
        wants: Wants,
    ) -> Result<Vec<Slot>, Error> {
        self.check_rows(rows)?;
        let (mut sizes, [filled, wanted]) = launch_of(rows, picks, self.slots, self.context);
        // A promotion copies slot 0 alone.
        let solo = rows.iter().all(|r| r.slot == 0);
        let whole = if solo { rows.len() } else { usize::MAX };
        self.verified = matches!(head, Head::Peek).then_some(whole);
        let entry = self.entry(&head)?;
        // The last row fills out. Copies rewrite its kept row.
        let last = rows[rows.len() - 1];
        let column = |of: fn(&Row) -> u32| {
            let values = rows.iter().chain(std::iter::repeat(&last));
            Tensor::u32(vec![filled], values.take(filled).map(of).collect())
        };
        let (mut masks, mut listed, mut edits) = match head {
            Head::Full(masks) => (masks, None, None),
            Head::Sparse(ids, k) => (None, Some((ids, k)), None),
            Head::Edited(masks, packed, k) => (Some(masks), None, Some((packed, k))),
            Head::Peek => {
                sizes.push((PEEK, 1));
                (None, None, None)
            }
        };
        let mut feeds = Vec::new();
        for port in &self.graph.entries[entry].inputs {
            let feed = match port.role {
                Role::Tokens => column(|r| r.token),
                Role::Positions => column(|r| r.position),
                Role::Slots => column(|r| r.slot),
                Role::Picks => {
                    let tail = picks.last().copied().unwrap_or_default();
                    let picked = picks.iter().copied().chain(std::iter::repeat(tail));
                    Tensor::u32(vec![wanted], picked.take(wanted).collect())
                }
                Role::Allowed => {
                    let words = self.words.ok_or(Error::Mask)?;
                    // Unmasked and filler picks allow everything.
                    let mut bits = masks.take().unwrap_or_default();
                    bits.resize(wanted * words, u32::MAX);
                    let Some((packed, k)) = edits.take() else {
                        feeds.push(Tensor::u32(vec![wanted, words], bits));
                        continue;
                    };
                    // Each pick's edits follow its mask words.
                    let mut row = Vec::with_capacity(wanted * (words + 3 * k));
                    for p in 0..wanted {
                        row.extend_from_slice(&bits[p * words..(p + 1) * words]);
                        match packed.get(p * 3 * k..(p + 1) * 3 * k) {
                            Some(mine) => row.extend_from_slice(mine),
                            None => pack(&[], k, &mut row),
                        }
                    }
                    sizes.retain(|(name, _)| *name != EditWords);
                    sizes.push((EditWords, 3 * k as u64));
                    Tensor::u32(vec![wanted, words + 3 * k], row)
                }
                Role::Candidates => {
                    let (mut ids, k) = listed.take().ok_or(Error::Scores)?;
                    // Filler picks repeat the last pick's list.
                    let last = ids[ids.len() - k..].to_vec();
                    while ids.len() < wanted * k {
                        ids.extend_from_slice(&last);
                    }
                    sizes.push((Candidates, k as u64));
                    Tensor::u32(vec![wanted, k], ids)
                }
                Role::Ple => self
                    .ple
                    .as_mut()
                    .ok_or(Error::Port {
                        entry,
                        role: Role::Ple,
                    })?
                    .fill(rows, filled)?,
                role => return Err(Error::Port { entry, role }),
            };
            feeds.push(feed);
        }
        if masks.is_some() || edits.is_some() {
            return Err(Error::Mask);
        }
        let start = Instant::now();
        let (plan, bound) = self.plans.wanting(
            &self.graph,
            entry,
            wants,
            &sizes,
            &[Context, Segments],
            &self.device,
        )?;
        let outputs = plan.outputs.iter().map(|o| o.slot).collect();
        self.runner.replan(plan)?;
        let planned = Instant::now();
        let table = plan.table.zip(self.lanes.as_mut());
        if let Some((table, bytes)) = table.and_then(|(t, l)| Some((t, l.changed()?))) {
            self.runner.held_mut().upload(table, &bytes)?;
        }
        self.runner.bind_all(&bound);
        let views: Vec<_> = feeds.iter().map(Tensor::view).collect();
        self.runner.feed(&views)?;
        let fed = Instant::now();
        self.runner.submit(self.runner.plan().forward())?;
        let ran = Instant::now();
        let phases = [
            ("engine: plan", start, planned),
            ("engine: feed", planned, fed),
            ("engine: steps", fed, ran),
            ("engine: launch", start, ran),
        ];
        for (name, from, to) in phases {
            self.runner.note(name, (to - from).as_secs_f64());
        }
        Ok(outputs)
    }

    /// Seconds per kernel and phase since the last take.
    pub fn take_profile(&mut self) -> Option<tessera_run::Profile> {
        self.runner.take_profile()
    }

    /// Prints the profile once every so many launches.
    fn report(&mut self) {
        let Some([every, done]) = &mut self.profile else {
            return;
        };
        *done += 1;
        if *done < *every {
            return;
        }
        let launches = std::mem::take(done);
        let Some(profile) = self.take_profile() else {
            return;
        };
        match self.control {
            Some(_) => self.held = Some((launches, profile)),
            None => eprintln!(
                "profile of {launches} launches:\n{}",
                profile.report(launches)
            ),
        }
    }

    /// Decodes greedily after `request.tokens`, one event per token.
    pub fn generate(&mut self, request: Request, emit: &mut dyn FnMut(Event)) -> Result<(), Error> {
        let (events, taken) = mpsc::channel();
        let mut batch = Batch::new(self, Box::new(Fair));
        batch.submit(request, Box::new(move |e| events.send(e).is_ok()));
        while batch.step()? {
            taken.try_iter().for_each(&mut *emit);
        }
        taken.try_iter().for_each(emit);
        Ok(())
    }
}
