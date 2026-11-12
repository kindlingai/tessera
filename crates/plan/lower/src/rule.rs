//! Rules propose kernels for a site; link time collects them.

use crate::form::Form;
use crate::kernel::{Kernel, KernelRef};
use crate::kernels::MapStage;
use crate::tree::{T, Tree};
use graph::Stage;

pub use scattered_collect::declarative::{gather, scatter};

/// The sizes one attention works over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Attn {
    pub seqs: usize,
    pub queries: usize,
    pub keys: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub width: usize,
}

impl Attn {
    /// What an attention kernel's time depends on.
    pub fn sizes(&self) -> Vec<usize> {
        vec![self.seqs, self.queries, self.keys, self.heads, self.width]
    }
}

/// A pattern at a tree node, for kernels to compute.
#[derive(Clone, Debug)]
pub enum Site<'a> {
    /// `x[m, k]` times `w[n, k]` transposed, `w` stored as `form`.
    Matmul {
        x: T,
        w: T,
        m: usize,
        n: usize,
        k: usize,
        form: Form,
    },
    /// Elementwise stages over `reads[0]`; operands follow it.
    Map {
        stages: Vec<MapStage>,
        reads: Vec<T>,
        len: usize,
    },
    /// One row stage that is not elementwise.
    Stage {
        stage: &'a Stage,
        x: T,
        /// The stage's argument, and how many elements it holds.
        operand: Option<(T, usize)>,
        len: usize,
        width: usize,
    },
    /// Causal attention with `scale` on the scores.
    Attend {
        scale: f32,
        q: T,
        k: T,
        v: T,
        sizes: Attn,
    },
    Gather {
        table: T,
        ids: T,
        rows: usize,
        width: usize,
        form: Form,
    },
    /// Indices of each row's `k` largest entries.
    Top {
        x: T,
        rows: usize,
        width: usize,
        k: usize,
    },
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
        }
    }
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
