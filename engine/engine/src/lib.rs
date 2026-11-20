//! Graphs on runners: decode loops and single launches.

// Linking the crates registers their rules.
use tessera_rules as _;

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::mpsc;
use std::time::Instant;
use tessera_graph::DimName::{Candidates, Context, Picks, Rows, Segments, Seqs, VocabWords};
use tessera_lower::{Cache, Device};
use tessera_run::{Run, open};

mod guide;
mod kept;
mod lanes;
pub mod metrics;
#[cfg(test)]
mod metrics_test;
mod model;
mod ranks;
mod schedule;

pub use guide::{Allowed, FEW, Fault, Guide, Prefix};
pub use kept::Totals;
pub use model::Model;
pub use ranks::Place;
pub use schedule::{Batch, Fair, Policy, Seq, Sink};

// The front end reaches the engine crates only through these.
pub use tessera_engine_comm as comm;
pub use tessera_engine_settings::{Backend, Memory};
pub use tessera_graph::{Bf16, DimExpr, DimName, E4M3, F16, Graph, PathSeg, Shape};
pub use tessera_interp::{Tensor, bind, fold};
pub use tessera_lower::{Pick, Pins, Recipe, Settings, Weight, form};
pub use tessera_mem::form as storage;
pub use tessera_mem::{block, trellis};
pub use tessera_models::{Stack, clip, decoder, eg2, flux2, keys, mla, sprite, tagger, vae};
pub use tessera_run::{gpu, pinning};
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Event {
    /// Prompt rows that kept rows cover. Sent before any token.
    Kept(usize),
    Token(u32),
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
    /// Whether entry 1 picks among candidate ids.
    sparse: bool,
    /// Carries rank 0's launches to the other ranks.
    control: Option<Box<dyn comm::Transport>>,
    /// Launches per profile report, and launches since the last.
    profile: Option<[u64; 2]>,
}

/// Inputs a launch feeds before its picks.
enum Head {
    /// Entry 0, with each pick's mask words. None allows all.
    Full(Option<Vec<u32>>),
    /// Entry 1, with `k` candidate ids per pick.
    Sparse(Vec<u32>, usize),
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

/// Sizes and inputs for a launch of `rows` returning `picks`.
fn launch_of(
    rows: &[Row],
    picks: &[u32],
    slots: usize,
) -> (Vec<(tessera_graph::DimName, u64)>, Vec<Tensor>) {
    let furthest = rows.iter().map(|r| r.position).max().unwrap_or(0);
    let mut seen: Vec<u32> = rows.iter().map(|r| r.slot).collect();
    seen.sort_unstable();
    seen.dedup();
    // Rows and picks round up to a power of two.
    let filled = rows.len().next_power_of_two();
    let wanted = picks.len().next_power_of_two();
    let sizes = vec![
        (Seqs, slots as u64),
        (Rows, filled as u64),
        (Context, furthest as u64 + 1),
        (Segments, seen.len() as u64),
        (Picks, wanted as u64),
    ];
    // The last row fills out. Copies rewrite its kept row.
    let last = rows.last().copied().unwrap_or_default();
    let column = |of: fn(&Row) -> u32| {
        let values = rows.iter().chain(std::iter::repeat(&last));
        Tensor::u32(vec![filled], values.take(filled).map(of).collect())
    };
    let tail = picks.last().copied().unwrap_or_default();
    let picked = picks.iter().copied().chain(std::iter::repeat(tail));
    let feeds = vec![
        column(|r| r.token),
        column(|r| r.position),
        column(|r| r.slot),
        Tensor::u32(vec![wanted], picked.take(wanted).collect()),
    ];
    (sizes, feeds)
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
        let mut graph = tessera_graph::trim(&graph, 0)?;
        let sparse = graph.entries.len() > 1;
        if sparse {
            graph = tessera_graph::trim(&graph, 1)?;
        }
        let ranked = |device: Device| match place {
            Some(place) => place.device(device),
            None => device,
        };
        if let Some(place) = place {
            graph = place.split(&graph)?;
        }
        let device = ranked(Device::guess(settings).with(recipe));
        let mut plans = Cache::default();
        let slots = graph
            .dims
            .iter()
            .find(|d| d.name == Seqs)
            .map_or(1, |d| d.size.max() as usize);
        let first = [Row {
            slot: 0,
            token: 0,
            position: 0,
        }];
        let (sizes, _) = launch_of(&first, &[0], slots);
        let plan = plans.lower(&graph, 0, &sizes, &device)?;
        let lanes = plan.table.zip(recipe.kept.page());
        let lanes = lanes.map(|(table, page)| lanes::Lanes::new(page, table.len, slots));
        let mut runner = run(plan)?;
        runner.launching(recipe.launch);
        runner.profiling(settings.profile.is_some());
        let device = Device::stored(runner.platform(), settings)
            .map_err(tessera_lower::Error::from)?
            .with(recipe);
        let device = ranked(device);
        let words = graph.dims.iter().find(|d| d.name == VocabWords);
        let words = words.map(|d| d.size.max() as usize);
        Ok(Self {
            graph,
            runner,
            device,
            plans,
            context,
            slots,
            lanes,
            words,
            sparse,
            control: None,
            profile: settings.profile.map(|every| [every, 0]),
        })
    }

    /// Uploads the graph's weights, in its order.
    pub fn load(&mut self, weights: &[Weight]) -> Result<(), Error> {
        for (at, w) in weights.iter().enumerate() {
            self.load_weight(at as u32, w)?;
        }
        Ok(())
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

    /// Mask words per pick, if the graph takes masks.
    pub fn words(&self) -> Option<usize> {
        self.words
    }

    /// One launch of `rows`: the likeliest token after each.
    pub fn step(&mut self, rows: &[Row]) -> Result<Vec<u32>, Error> {
        let all: Vec<u32> = (0..rows.len() as u32).collect();
        self.launch(rows, &all, Head::Full(None))
    }

    /// One launch: the likeliest token after each slot's last row.
    pub fn last(&mut self, rows: &[Row]) -> Result<Vec<u32>, Error> {
        self.launch(rows, &ends(rows), Head::Full(None))
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
        if self.sparse && lists.len() == allowed.len() && 2 * gathered <= 32 * words {
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

    /// One launch of `rows`, the likeliest ids after `picks`.
    fn launch(&mut self, rows: &[Row], picks: &[u32], head: Head) -> Result<Vec<u32>, Error> {
        if rows.is_empty() {
            return Err(Error::Rows);
        }
        if rows
            .iter()
            .any(|r| r.slot as usize >= self.slots || r.position as usize >= self.context)
        {
            return Err(Error::RowPlace);
        }
        let table = self.lanes.as_ref().and_then(lanes::Lanes::pending);
        let table = table.map(<[u32]>::to_vec).unwrap_or_default();
        self.announce(rows, picks, &head, &table)?;
        let (mut sizes, mut feeds) = launch_of(rows, picks, self.slots);
        let wanted = feeds[3].len();
        let entry = match head {
            Head::Full(None) if self.words.is_none() => 0,
            Head::Full(masks) => {
                let words = self.words.ok_or(Error::Mask)?;
                // Unmasked and filler picks allow everything.
                let mut bits = masks.unwrap_or_default();
                bits.resize(wanted * words, u32::MAX);
                feeds.insert(3, Tensor::u32(vec![wanted, words], bits));
                0
            }
            Head::Sparse(mut ids, k) => {
                // Filler picks repeat the last pick's list.
                let last = ids[ids.len() - k..].to_vec();
                while ids.len() < wanted * k {
                    ids.extend_from_slice(&last);
                }
                feeds.insert(3, Tensor::u32(vec![wanted, k], ids));
                sizes.push((Candidates, k as u64));
                1
            }
        };
        let start = Instant::now();
        let (plan, bound) = self.plans.upto(
            &self.graph,
            entry,
            &sizes,
            &[Context, Segments],
            &self.device,
        )?;
        let out = plan.outputs[0].slot;
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
        self.runner.submit(0..self.runner.plan().steps.len())?;
        let ran = Instant::now();
        let mut ids = self.runner.ids(out)?;
        ids.truncate(picks.len());
        let phases = [
            ("engine: plan", start, planned),
            ("engine: feed", planned, fed),
            ("engine: steps", fed, ran),
            ("engine: ids", ran, Instant::now()),
            ("engine: launch", start, Instant::now()),
        ];
        for (name, from, to) in phases {
            self.runner.note(name, (to - from).as_secs_f64());
        }
        self.report();
        Ok(ids)
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
        let rank = self.runner.plan().rank;
        if let Some(profile) = self.take_profile() {
            eprintln!(
                "profile of {launches} launches, rank {rank}:\n{}",
                profile.report(launches)
            );
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
