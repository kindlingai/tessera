//! Engine calls that speculative decoders make, and shared types.

use serde::{Deserialize, Serialize};
use tessera_graph::Role;
use tessera_lower::Slot;
use tessera_run::{Profile, Run};

pub mod guide;
pub mod kept;
#[cfg(test)]
mod kept_test;
pub mod lanes;
mod place;
pub mod sample;
#[cfg(test)]
mod sample_test;
mod taps;

pub use guide::{Allowed, Edit, Guide, Rules};
pub use kept::Kept;
pub use lanes::Lanes;
pub use place::{Place, open_rank, stored_device};
pub use sample::Sampler;
pub use taps::{TapsBuf, TapsIn, VerifyOut};

/// Tokens to continue, and when to stop.
#[derive(Default, Serialize, Deserialize)]
pub struct Request {
    pub tokens: Vec<u32>,
    pub max_tokens: usize,
    /// Minimum tokens before stop ids end the answer.
    #[serde(default)]
    pub min_tokens: usize,
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
    pub sampling: sample::Sampling,
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

/// Takes each event. `false` ends the answer early.
pub type Sink = Box<dyn FnMut(Event) -> bool + Send>;

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
    #[error("kv store: {0}")]
    Store(std::io::Error),
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
    #[error("guides need a single-rank speculative loop")]
    Guided,
}

/// A token at a position, in a slot's sequence.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Row {
    pub slot: u32,
    pub token: u32,
    pub position: u32,
}

/// Inputs a launch feeds before its picks.
pub enum Head {
    /// The full entry. Per-pick mask words, or None for all.
    Full(Option<Vec<u32>>),
    /// The sparse entry, with `k` candidate ids per pick.
    Sparse(Vec<u32>, usize),
    /// Full entry: masks, and `k` edits per pick.
    Edited(Vec<u32>, Vec<u32>, usize),
}

/// Rounds a causal context length up to a power-of-two bucket.
pub fn context_bucket(needed: usize, cap: usize) -> u64 {
    needed.max(256).next_power_of_two().min(cap) as u64
}

/// Rounds a launch row count up to compact buckets.
pub fn row_bucket(n: usize) -> usize {
    if n <= 24 {
        n.max(1)
    } else if n <= 64 {
        n.div_ceil(8) * 8
    } else if n <= tessera_device::FAST_MAX_ROWS {
        n.div_ceil(64) * 64
    } else {
        n.next_power_of_two()
    }
}

/// The engine calls that speculative decoders make.
pub trait Hooks {
    /// Maximum sequence length in tokens.
    fn context(&self) -> usize;
    /// Kept-row lanes the graph declares.
    fn slots(&self) -> usize;
    /// Mask words per pick, if the graph takes masks.
    fn words(&self) -> Option<usize>;
    /// Whether the device head takes `n` edits in one pick.
    fn edits_fit(&self, n: usize) -> bool;
    /// Whether slots carry state that a kept prefix cannot restore.
    fn stateful(&self) -> bool;
    /// Floats per tapped row, if the full entry emits taps.
    fn tap_width(&self) -> Option<usize>;
    /// Floats per tapped row, if drafts may rewind slots.
    fn draft_taps(&self) -> Result<usize, Error>;
    /// This engine's rank, when launches cross ranks.
    fn rank(&self) -> Option<u32>;
    /// The page table, when kept rows are paged.
    fn lanes(&self) -> Option<&Lanes>;
    fn lanes_mut(&mut self) -> Option<&mut Lanes>;
    /// The runner, to read device buffers.
    fn runner(&self) -> &dyn Run;
    /// Copies `bytes` from device address `src` into `dst`.
    fn copy_from(&mut self, dst: Slot, offset: usize, src: u64, bytes: usize) -> Result<(), Error>;
    /// A full head with each pick's mask and edits.
    fn edited(&self, allowed: &[&Allowed], edits: &[&[Edit]]) -> Result<Head, Error>;
    /// Starts or flushes dynamic profiling.
    fn settle(&mut self) -> Result<(), Error>;
    /// Sends `mine` to the other ranks. Returns rank 0's bytes.
    fn told(&mut self, mine: &[u8]) -> Result<Option<Vec<u8>>, Error>;
    /// Takes a ranked profile that waits for a control round.
    fn take_held(&mut self) -> Option<Profile>;
    /// Checks hashes across ranks. Returns rank 0's stop.
    fn sync(&mut self, stopped: bool, hash: u32) -> Result<bool, Error>;
    /// Stores state after the last verify's first `rows` rows.
    fn commit(&mut self, rows: usize) -> Result<(), Error>;
    /// Grows paged slots for `rows`, syncing host KV pages.
    fn grow_kept(&mut self, kept: &mut Kept, rows: &[Row]) -> Result<(), Error>;
    /// Restores prefix pages from the KV store past `skip`.
    fn restore_kv(
        &mut self,
        kept: Option<&mut Kept>,
        slot: u32,
        tokens: &[u32],
        skip: usize,
        lead: bool,
    ) -> Result<usize, Error>;
    /// Writes a finished slot's pages to the KV store.
    fn spill_kv(&mut self, kept: Option<&Kept>, slot: u32, history: &[u32], lead: bool);
    /// Local prefill step appending kept state only.
    fn fill_local(&mut self, rows: &[Row]) -> Result<(), Error>;
    /// Local launch returning only taps without the head.
    fn step_taps_only(&mut self, rows: &[Row]) -> Result<TapsBuf, Error>;
    /// Runs batched verify, reading logits when sampling.
    fn step_verify_local(
        &mut self,
        rows: &[Row],
        sampling: bool,
    ) -> Result<(VerifyOut, TapsBuf), Error>;
    /// Picks the final prefill anchor and taps.
    fn prefill_pick(
        &mut self,
        rows: &[Row],
        sampler: &mut Sampler,
    ) -> Result<(u32, TapsBuf), Error>;
    /// Reads full logits for `picks` rows, with optional taps.
    fn logits_local(
        &mut self,
        rows: &[Row],
        picks: &[u32],
        taps: bool,
        peek: bool,
    ) -> Result<(Vec<f32>, usize, Option<TapsBuf>), Error>;
    /// The ids after `picks` rows. Taps and peek on request.
    fn launch_local(
        &mut self,
        rows: &[Row],
        picks: &[u32],
        head: Head,
        taps: bool,
        peek: bool,
    ) -> Result<(Vec<u32>, Option<TapsBuf>), Error>;
    /// The pick after the last row, and its taps.
    fn pick_last(
        &mut self,
        rows: &[Row],
        rules: &Rules,
        taps: bool,
    ) -> Result<(u32, Option<TapsBuf>), Error>;

    /// Whether the device head can apply `rules`.
    fn on_device(&self, rules: &Rules) -> bool {
        match rules.1.is_empty() {
            true => rules.0 == Allowed::Any || self.words().is_some(),
            false => self.edits_fit(rules.1.len()),
        }
    }

    /// A full head with each pick's mask and edits.
    fn head_of(&self, rules: &[Rules]) -> Result<Head, Error> {
        if rules.iter().any(|r| !r.1.is_empty()) {
            let allowed: Vec<&Allowed> = rules.iter().map(|r| &r.0).collect();
            let edits: Vec<&[Edit]> = rules.iter().map(|r| &r.1[..]).collect();
            return self.edited(&allowed, &edits);
        }
        Ok(match self.words() {
            Some(words) if rules.iter().any(|r| r.0 != Allowed::Any) => {
                Head::Full(Some(rules.iter().flat_map(|r| r.0.bits(words)).collect()))
            }
            _ => Head::Full(None),
        })
    }
}
