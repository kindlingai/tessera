//! Rules propose kernels for a site, collected at link time.

use crate::abi::{Arg, Launch, Shape};
use crate::extent::Extent;
use crate::kernel::{Kernel, KernelRef};
use crate::plan::Step;
use crate::tree::{T, Tree};
use crate::{Error, Unsupported};
use std::fmt;
use tessera_codegen::kept::Layout;
use tessera_graph::Mask;

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
    /// Inputs to an attention kernel's time.
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
        let (back, ahead) = self.mask.window(keys);
        keys.min(back.saturating_add(ahead).saturating_add(1))
    }

    /// How far back and ahead a query sees.
    pub fn reach(&self) -> [Arg; 2] {
        let far = u32::MAX as usize;
        let (back, ahead) = self.mask.window(far);
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

    /// A ragged launch: six buffers, any page table, four sizes.
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
        let table = step.kept(self.kept);
        buffers.into_iter().chain(table).chain(sizes).collect()
    }

    /// Keys before its own that a row sees.
    pub fn back(&self) -> Arg {
        Arg::count(self.mask.window(u32::MAX as usize).0)
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

/// A tree node, an earlier draft, or kept rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, derive_more::From)]
pub enum Read {
    Node(T),
    #[from(ignore)]
    Draft(usize),
    /// Kept rows by index. The launch may also write these.
    #[from(ignore)]
    Kept(u32),
}

/// One launch a rule proposes, writing `len` elements.
#[derive(Clone, Debug)]
pub struct Draft {
    pub kernel: KernelRef,
    pub reads: Vec<Read>,
    pub len: usize,
    /// Drafts sharing a choice form one way. The cheapest wins.
    pub choice: usize,
    /// The region output it writes, for readers past the cover.
    pub writes: Option<T>,
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
            writes: None,
        }
    }

    /// The same draft, writing region output `t`.
    pub fn writes(mut self, t: T) -> Self {
        self.writes = Some(t);
        self
    }

    /// The same draft, as part of way `choice`.
    pub fn choice(mut self, choice: usize) -> Self {
        self.choice = choice;
        self
    }
}

/// A rule's drafts grouped into ways, in first-seen order.
pub(crate) fn ways(drafts: Vec<Draft>) -> Vec<Vec<Draft>> {
    let mut ways: Vec<(usize, Vec<Draft>)> = Vec::new();
    for d in drafts {
        match ways.iter_mut().find(|(c, _)| *c == d.choice) {
            Some((_, way)) => way.push(d),
            None => ways.push((d.choice, vec![d])),
        }
    }
    ways.into_iter().map(|(_, way)| way).collect()
}

/// Drafts for a tree node, covering nodes above their reads.
pub trait Propose: Send + Sync {
    fn at(&self, tree: &Tree, at: T) -> Option<Vec<Draft>>;
    /// Whether it covers whole regions. Readers stop at their outputs.
    fn regions(&self) -> bool {
        false
    }
}

/// The form `rule!` registers.
pub type ProposeFn = fn(&Tree, T) -> Option<Vec<Draft>>;

impl Propose for ProposeFn {
    fn at(&self, tree: &Tree, at: T) -> Option<Vec<Draft>> {
        self(tree, at)
    }
}

/// A named proposer.
pub struct Rule {
    pub name: &'static str,
    pub propose: &'static dyn Propose,
    /// The registering module.
    pub module: &'static str,
}

impl Rule {
    /// A rule made at run time. It is never freed.
    pub fn made(name: &str, module: &str, propose: impl Propose + 'static) -> &'static Rule {
        Box::leak(Box::new(Rule {
            name: Box::leak(name.into()),
            propose: Box::leak(Box::new(propose)),
            module: Box::leak(module.into()),
        }))
    }
}

impl fmt::Debug for Rule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} in {}", self.name, self.module)
    }
}

/// Rules with one name and module are one rule.
impl PartialEq for Rule {
    fn eq(&self, other: &Self) -> bool {
        (self.name, self.module) == (other.name, other.module)
    }
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
                    propose: &($propose as $crate::rule::ProposeFn),
                    module: module_path!(),
                };
            }
        };
    };
}

/// One row stage's rule, from a closure on its site.
#[macro_export]
macro_rules! stage_rule {
    ($name:literal, $kernel:expr) => {
        $crate::stage_rule!($name, $kernel, |site| site.reads());
    };
    // The third closure gives the reads, else the site's.
    ($name:literal, $kernel:expr, $reads:expr) => {
        fn propose(tree: &$crate::Tree, at: $crate::T) -> Option<Vec<$crate::Draft>> {
            let site: $crate::sites::Stage = tree.site(at)?;
            let kernel = $crate::rule::at_site(&site, $kernel)?;
            let reads = $crate::rule::at_site(&site, $reads);
            Some(vec![$crate::Draft::new(kernel, reads, site.len)])
        }

        $crate::rule!($name, propose);
    };
}

/// `f` at `site`, typing the closure `stage_rule!` is given.
pub fn at_site<R>(site: &crate::sites::Stage, f: impl FnOnce(&crate::sites::Stage) -> R) -> R {
    f(site)
}

/// Every registered rule, in name order.
pub fn rules() -> Vec<&'static Rule> {
    rules_with(&[])
}

/// The registered rules and `made`, in name order.
pub(crate) fn rules_with(made: &[&'static Rule]) -> Vec<&'static Rule> {
    let mut all: Vec<&'static Rule> = RULES.iter().chain(made.iter().copied()).collect();
    all.sort_by_key(|r| r.name);
    all
}

/// Fails when `rules` holds only this crate's own rules.
pub fn linked(rules: &[&Rule]) -> Result<(), Error> {
    // Module path, so a merged crate still tells rules apart.
    let ours = module_path!().trim_end_matches("::rule");
    let own = |m: &str| {
        m.strip_prefix(ours)
            .is_some_and(|r| r.is_empty() || r.starts_with("::"))
    };
    if rules.iter().any(|r| !own(r.module)) {
        return Ok(());
    }
    Err(Unsupported::Unlinked.into())
}
