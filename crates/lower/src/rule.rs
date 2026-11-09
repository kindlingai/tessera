//! Rules propose kernels for a site; link time collects them.

use crate::kernel::{Kernel, KernelRef};
use crate::kernels::MapStage;
use crate::select::Val;
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

/// One piece of a graph that some kernels can compute.
#[derive(Clone, Copy, Debug)]
pub enum Site<'a> {
    /// `x[m, k]` times `w[n, k]` transposed.
    Matmul {
        x: Val,
        w: Val,
        m: usize,
        n: usize,
        k: usize,
    },
    /// Elementwise stages over `reads[0]`; operands follow it.
    Map {
        stages: &'a [MapStage],
        reads: &'a [Val],
        len: usize,
    },
    /// One row stage that is not elementwise.
    Stage {
        stage: &'a Stage,
        x: Val,
        /// The stage's argument, and how many elements it holds.
        operand: Option<(Val, usize)>,
        len: usize,
        width: usize,
    },
    /// Causal attention with `scale` on the scores.
    Attend {
        scale: f32,
        q: Val,
        k: Val,
        v: Val,
        sizes: Attn,
    },
    Gather {
        table: Val,
        ids: Val,
        rows: usize,
        width: usize,
    },
}

/// A draft reads a value, or an earlier draft's result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Read {
    Value(Val),
    Draft(usize),
}

impl From<Val> for Read {
    fn from(v: Val) -> Self {
        Read::Value(v)
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

/// Launches for a site, the last writing its value.
pub struct Rule {
    pub name: &'static str,
    pub propose: fn(&Site) -> Option<Vec<Draft>>,
}

gather! {
    #[gather]
    pub static RULES: scattered_collect::ScatteredSlice<Rule>;
}

/// Registers a rule. Selection then considers it at every site.
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
