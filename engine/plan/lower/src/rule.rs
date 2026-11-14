//! Rules propose kernels for a site. Link time collects them.

use crate::Error;
use crate::abi::{Arg, Launch, Shape};
use crate::extent::Extent;
use crate::kernel::{Kernel, KernelRef};
use crate::plan::Step;
use crate::tree::{T, Tree};
use codegen::kept::Layout;
use graph::Mask;

pub use scattered_collect::declarative::{gather, scatter};

/// The sizes one attention works over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Attn {
    pub seqs: usize,
    pub queries: usize,
    /// The key count. Decode reads it at launch.
    pub keys: Extent,
    pub heads: usize,
    pub kv_heads: usize,
    pub width: usize,
    /// Rows one slot may hold in the kept keys.
    pub context: usize,
    /// Where kept rows lie.
    pub kept: &'static dyn Layout,
    /// Slots the query rows span, at most.
    pub segments: usize,
    pub mask: Mask,
}

impl Attn {
    /// What an attention kernel's time depends on.
    pub fn sizes(&self) -> Vec<usize> {
        let keys = self.keys.most();
        vec![
            self.seqs,
            self.queries,
            keys,
            self.heads,
            self.width,
            self.segments,
        ]
    }

    /// The most keys one query sees.
    pub fn seen(&self) -> usize {
        let keys = self.keys.most();
        match self.mask {
            Mask::Within(n) => keys.min(2 * n as usize + 1),
            _ => keys,
        }
    }

    /// How far back and ahead a query sees.
    pub fn reach(&self) -> [Arg; 2] {
        let far = u32::MAX as usize;
        let (back, ahead) = match self.mask {
            Mask::Causal => (far, 0),
            Mask::Bidirectional => (far, far),
            Mask::Within(n) => (n as usize, n as usize),
        };
        [Arg::count(back), Arg::count(ahead)]
    }

    /// One group of `threads` per `across`, over whole sequences.
    pub fn launch(
        &self,
        buffers: impl IntoIterator<Item = Arg>,
        tail: impl IntoIterator<Item = Arg>,
        across: usize,
        threads: usize,
    ) -> Launch {
        let args = buffers
            .into_iter()
            .chain(self.args())
            .chain(tail)
            .chain(self.reach());
        Launch {
            args: args.collect(),
            shape: Shape::Grid {
                across,
                down: 1,
                threads,
            },
        }
    }

    /// The layout's scalars, which come after a kernel's own.
    pub fn kept_args(&self) -> Vec<Arg> {
        let values = self.kept.values(self.seqs, self.context);
        values.into_iter().map(|v| Arg::count(v as usize)).collect()
    }

    /// A ragged launch: six buffers, then its four sizes.
    pub fn rows_args(&self, step: &Step) -> Vec<Arg> {
        let buffers = [
            step.read(0),
            step.read(1),
            step.read(2),
            step.out(),
            step.read(3),
            step.read(4),
        ];
        let sizes = [
            Arg::count(self.queries),
            Arg::size(self.keys),
            Arg::count(self.heads),
            Arg::count(self.kv_heads),
        ];
        buffers.into_iter().chain(sizes).collect()
    }

    /// Queries, keys, the keys before the queries, heads, kv heads.
    pub fn args(&self) -> Vec<Arg> {
        vec![
            Arg::count(self.queries),
            Arg::size(self.keys),
            Arg::size(self.keys.less(self.queries)),
            Arg::count(self.heads),
            Arg::count(self.kv_heads),
        ]
    }
}

/// A draft reads a tree node, or an earlier draft.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Read {
    Node(T),
    Draft(usize),
}

impl From<T> for Read {
    fn from(t: T) -> Self {
        Read::Node(t)
    }
}

/// One launch a rule proposes, writing `len` elements.
#[derive(Clone, Debug)]
pub struct Draft {
    pub kernel: KernelRef,
    pub reads: Vec<Read>,
    pub len: usize,
    /// Drafts sharing a choice form one way. The cheapest wins.
    pub choice: usize,
}

impl Draft {
    pub fn new<R: Into<Read>>(
        kernel: impl Kernel + 'static,
        reads: impl IntoIterator<Item = R>,
        len: usize,
    ) -> Self {
        Self {
            kernel: KernelRef::new(kernel),
            reads: reads.into_iter().map(Into::into).collect(),
            len,
            choice: 0,
        }
    }

    /// The same draft, as part of way `choice`.
    pub fn choice(mut self, choice: usize) -> Self {
        self.choice = choice;
        self
    }
}

/// A rule's drafts grouped into ways, in first-seen order.
pub fn ways(drafts: Vec<Draft>) -> Vec<Vec<Draft>> {
    let mut ways: Vec<(usize, Vec<Draft>)> = Vec::new();
    for d in drafts {
        match ways.iter_mut().find(|(c, _)| *c == d.choice) {
            Some((_, way)) => way.push(d),
            None => ways.push((d.choice, vec![d])),
        }
    }
    ways.into_iter().map(|(_, way)| way).collect()
}

/// Launches for a tree node, covering nodes above their reads.
pub struct Rule {
    pub name: &'static str,
    pub propose: fn(&Tree, T) -> Option<Vec<Draft>>,
    /// The module that registered it.
    pub module: &'static str,
}

gather! {
    #[gather]
    pub static RULES: scattered_collect::ScatteredSlice<Rule>;
}

/// Registers a rule. Selection then asks it at every node.
#[macro_export]
macro_rules! rule {
    ($name:literal, $propose:path) => {
        const _: () = {
            $crate::rule::scatter! {
                #[scatter($crate::rule::RULES)]
                static RULE: $crate::rule::Rule = $crate::rule::Rule {
                    name: $name,
                    propose: $propose,
                    module: module_path!(),
                };
            }
        };
    };
}

/// Every registered rule, in name order.
pub fn rules() -> Vec<&'static Rule> {
    let mut all: Vec<&'static Rule> = RULES.iter().collect();
    all.sort_by_key(|r| r.name);
    all
}

/// Fails when `rules` holds only this crate's own rules.
pub fn linked(rules: &[&Rule]) -> Result<(), Error> {
    let ours = Some(env!("CARGO_CRATE_NAME"));
    if rules.iter().any(|r| r.module.split("::").next() != ours) {
        return Ok(());
    }
    let hint = "add `use kernels as _; use linear as _;`";
    Err(Error::Unsupported(format!(
        "any node: no rule crate is linked; {hint}"
    )))
}
