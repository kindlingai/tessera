//! Rules propose kernels for a site; link time collects them.

use crate::abi::Arg;
use crate::extent::Extent;
use crate::kernel::{Kernel, KernelRef};
use crate::plan::Step;
use crate::tree::{T, Tree};

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
    /// Rows one slot holds in the kept keys.
    pub stride: usize,
    /// Slots the query rows span, at most.
    pub segments: usize,
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

    /// A ragged launch: six buffers, then its five sizes.
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
            Arg::count(self.stride),
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
