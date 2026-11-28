//! An entry as trees: values with one reader join it.

mod own;

use crate::extent::Extent;
use crate::form::{F32_FORM, Load, Pins, Site};
use crate::kernels::MapStage;
use crate::plan::Share;
use crate::recipe::{Compute, Knobs};
use crate::rule::Rule;
use crate::select::Home;
use crate::sites::Pattern;
use crate::{Error, Size, Unsupported};
pub use own::sliding_ring;
use std::collections::HashMap;
use tessera_codegen::kept::Layout;
use tessera_graph::{
    Composite, DType, DimName, Graph, Lower, NodeId, Op, Parts, Pin, Placement, Region, Role,
    Shape, Stage, View,
};

/// Max rows per launch a sliding ring reserves.
pub const RING_CHUNK: usize = 2048;

/// Entry outputs a launch reads, one bit each.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Wants(pub u32);

impl Wants {
    pub const ALL: Self = Self(u32::MAX);
    /// No outputs: the launch only appends kept rows and state.
    pub const NONE: Self = Self(0);

    /// The outputs of `entry` with `role`.
    pub fn role(g: &Graph, entry: usize, role: Role) -> Self {
        Self::roles(g, entry, &[role])
    }

    /// The outputs of `entry` matching any of `roles`.
    pub fn roles(g: &Graph, entry: usize, roles: &[Role]) -> Self {
        let outputs = g.entries.get(entry).map_or(&[][..], |e| &e.outputs);
        let at = outputs
            .iter()
            .enumerate()
            .filter(|(_, o)| roles.contains(&o.role));
        Self(at.fold(0, |bits, (i, _)| bits | 1 << i))
    }

    fn has(self, output: usize) -> bool {
        self.0 >> output & 1 == 1
    }
}

/// A node of the view. Arguments come before their readers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct T(pub u32);

/// A view node's computation.
#[derive(Clone, Debug, PartialEq)]
pub enum Term {
    /// A weight or an input, there before any launch.
    Leaf(Home),
    /// One stage of a row.
    Stage(Stage),
    /// Any other op, unchanged from the graph.
    Op(Op),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub term: Term,
    pub args: Vec<T>,
    /// Each axis at the plan's largest size.
    pub shape: Vec<usize>,
    /// Each axis, with runtime dimensions kept symbolic.
    pub extents: Vec<Extent>,
    /// How its elements are stored.
    pub form: &'static dyn Load,
    /// The graph node it came from.
    pub from: NodeId,
}

impl Node {
    pub(crate) fn new(
        term: Term,
        args: Vec<T>,
        (shape, extents): (Vec<usize>, Vec<Extent>),
        form: &'static dyn Load,
        from: NodeId,
    ) -> Self {
        Self {
            term,
            args,
            shape,
            extents,
            form,
            from,
        }
    }
}

/// A dimension's size among `sizes`.
pub(crate) fn lookup(sizes: &[(DimName, u64)], name: DimName) -> Option<u64> {
    sizes.iter().find(|(n, _)| *n == name).map(|(_, v)| *v)
}

/// The nodes an entry needs, with their sizes and readers.
pub struct Tree<'a> {
    pub g: &'a Graph,
    nodes: Vec<Node>,
    /// Per node: how many arguments and outputs name it.
    readers: Vec<u32>,
    /// The entry's outputs, with each one's graph shape.
    pub outputs: Vec<(T, Vec<usize>)>,
    /// Elements, shape and type of each input port.
    pub ports: Vec<(usize, Vec<usize>, DType)>,
    /// The sizes this tree was viewed at.
    pub(crate) sizes: Vec<(DimName, u64)>,
    /// Where kept rows lie.
    pub kept: &'static dyn Layout,
    /// Shared paged KV row capacity across all slots.
    pub kept_rows: Option<usize>,
    /// Split `(kv_vram_rows, index_vram_rows)` for sparse MLA models.
    pub index_rows: Option<(usize, usize)>,
    /// Operand precision for matrix-unit kernels.
    pub compute: Compute,
    /// Spot precision per graph node, when any spot sets one.
    pub(crate) computes: Vec<Option<Compute>>,
    /// Kernel shape values the recipe set.
    pub knobs: Knobs,
    /// Model config JSON for runtime rules.
    pub config: Option<String>,
    /// How split axes divide, and which part this plan is.
    pub(crate) parts: Parts,
    pub(crate) rank: u32,
    /// Per node: the part of a split weight loaded here.
    shares: Vec<Option<Share>>,
    /// Per graph node: the tree node holding its value.
    at: Vec<Option<T>>,
    /// Per node: a region output a region rule claims.
    ends: Vec<bool>,
    /// First append node for each kept tensor.
    kept_appends: Vec<Option<T>>,
    /// Whether each kept tensor feeds only picked attention.
    sparse_kept: Vec<bool>,
    /// Sliding-window ring capacity for each kept tensor, if smaller.
    kept_rings: Vec<Option<usize>>,
}

/// Rows that lower to `Map`. Other stages lower unfused.
macro_rules! map_stages {
    ($(
        $(#[$m:meta])*
        $name:ident
        $(($tb:ident: $tty:ty = $ts:expr))?
        $({$($(#[$fm:meta])* $f:ident: $fty:ty = $fs:expr),* $(,)?})?
        $(as $method:ident)?
        => $lower:tt {$($col:tt)*}
    )*) => {
        pub(crate) fn map_stage(s: &Stage) -> Option<MapStage> {
            $(map_arm!($lower, s, $name [$(($tb))? $({$($f),*})?]);)*
            None
        }
    };
}

macro_rules! map_arm {
    (Map, $s:ident, $name:ident [$($bind:tt)*]) => {
        if let Stage::$name $($bind)* = *$s {
            return Some(MapStage::$name $($bind)*);
        }
    };
    ($lower:tt, $($skip:tt)*) => {};
}

tessera_graph::stages!(map_stages);

/// The pinned form for a float node or stage.
fn pinned_form(
    pins: &Pins,
    op: &Op,
    stage: Option<&Stage>,
    role: Option<Role>,
    in_moe: bool,
) -> &'static dyn Load {
    if matches!(op, Op::Append(_)) {
        return pins.kept;
    }
    let facts = stage.map(Stage::facts);
    if facts.is_some_and(|f| f.lower == Lower::Crossing) {
        return F32_FORM;
    }
    if let Some(role) = role {
        if role == Role::Logits
            && let Some(form) = pins.site_form(Site::Logits)
        {
            return form;
        }
        return pins.outputs;
    }
    let routed = in_moe && pins.site_form(Site::Router).is_some();
    let pin = facts.map(|f| f.pin).filter(|p| *p != Pin::Plain);
    let site = match pin.unwrap_or(op.facts().pin) {
        Pin::Norm => Some(Site::Norm),
        Pin::Prob if routed => Some(Site::Router),
        Pin::Prob => Some(Site::Softmax),
        Pin::Residual if routed => Some(Site::Router),
        Pin::Residual => Some(Site::Residual),
        Pin::Router => Some(Site::Router),
        Pin::Attention => Some(Site::Attention),
        Pin::Matmul => Some(Site::Matmul),
        Pin::Plain if in_moe => Some(Site::Router),
        Pin::Plain => None,
    };
    site.and_then(|s| pins.site_form(s))
        .unwrap_or(pins.activations)
}

/// Output role per graph node, propagated back through reshapes.
fn output_roles(g: &Graph, entry: usize) -> Vec<Option<Role>> {
    let mut roles = vec![None; g.nodes.len()];
    for o in &g.entries[entry].outputs {
        if matches!(o.role, Role::Calib(_)) {
            continue;
        }
        let mut id = o.node.0 as usize;
        loop {
            roles[id] = Some(o.role);
            let Op::View(View::Reshape(_)) = g.nodes[id].op else {
                break;
            };
            id = g.nodes[id].args[0].0 as usize;
        }
    }
    roles
}

/// Which graph nodes lie inside a MoE region.
fn moe_nodes(g: &Graph) -> Vec<bool> {
    let mut mask = vec![false; g.nodes.len()];
    for region in &g.regions {
        if region.kind == Composite::Moe {
            for slot in &mut mask[region.start.0 as usize..region.end.0 as usize] {
                *slot = true;
            }
        }
    }
    mask
}

/// Which nodes the outputs depend on.
fn needed(g: &Graph, outputs: &[usize]) -> Vec<bool> {
    let mut needed = vec![false; g.nodes.len()];
    for &o in outputs {
        needed[o] = true;
    }
    for i in (0..g.nodes.len()).rev() {
        if needed[i] {
            for a in &g.nodes[i].args {
                needed[a.0 as usize] = true;
            }
        }
    }
    needed
}

/// Whether `ids` is a causal `top_k` covering every key.
fn all_causal(tree: &Tree, args: &[T]) -> bool {
    let [_, k, _, _, positions, ids] = *args else {
        return false;
    };
    let Some(&most) = tree.node(k).shape.get(1) else {
        return false;
    };
    let top = tree.node(ids);
    if !matches!(
        top.term,
        Term::Op(Op::Select(tessera_graph::Select::TopK { .. }))
    ) {
        return false;
    }
    if !top.shape.last().is_some_and(|&picks| most <= picks) {
        return false;
    }
    let seen = tree.node(top.args[0]);
    matches!(seen.term, Term::Stage(Stage::Causal))
        && seen.shape.last() == Some(&most)
        && seen.args.get(1) == Some(&positions)
}

/// Each axis of `s`, fixed or affine in `upto`.
fn sized(
    g: &Graph,
    sizes: &[(DimName, u64)],
    upto: &[DimName],
    s: &Shape,
) -> Result<Vec<Extent>, Error> {
    // `e` with `vary` at `value`, other sizes as given.
    let at = |e: &tessera_graph::DimExpr, vary: DimName, value: u64| {
        let size = |name| match name == vary {
            true => Some(value),
            false => lookup(sizes, name),
        };
        e.value(&g.dims, &size).map(|v| v as i64)
    };
    let most = |e: &tessera_graph::DimExpr| {
        let size = |name| lookup(sizes, name);
        e.value(&g.dims, &size).map(|n| n as usize)
    };
    let dim = |e: &tessera_graph::DimExpr| -> Option<Extent> {
        let most = most(e)?;
        let mut runtime = None;
        for name in upto {
            let (v0, v1, v2) = (at(e, *name, 0)?, at(e, *name, 1)?, at(e, *name, 2)?);
            if v0 == v1 && v1 == v2 {
                continue;
            }
            // At most one runtime dimension, entering linearly.
            let (mul, add) = (v1 - v0, v0);
            if v2 - v1 != mul || mul < 0 || runtime.is_some() {
                return None;
            }
            runtime = Some(Extent::Dim {
                name: *name,
                mul: mul as usize,
                add,
                most,
            });
        }
        Some(runtime.unwrap_or(Extent::Fixed(most)))
    };
    let all: Option<Vec<Extent>> = s.0.iter().map(dim).collect();
    all.ok_or_else(|| Size::Shape(s.clone()).into())
}

impl<'a> Tree<'a> {
    /// Views `entry` at these sizes, one node per row stage.
    pub fn new(
        g: &'a Graph,
        entry: usize,
        sizes: &[(DimName, u64)],
        pins: &Pins,
        kept: &'static dyn Layout,
    ) -> Result<Self, Error> {
        let parts = &Parts::even(1);
        Self::upto(g, entry, Wants::ALL, sizes, &[], pins, kept, parts, 0)
    }

    /// The same, `upto` read at launch, as `rank` of `parts`.
    #[allow(clippy::too_many_arguments, reason = "one view, every way it varies")]
    pub fn upto(
        g: &'a Graph,
        entry: usize,
        wants: Wants,
        sizes: &[(DimName, u64)],
        upto: &[DimName],
        pins: &Pins,
        kept: &'static dyn Layout,
        parts: &Parts,
        rank: u32,
    ) -> Result<Self, Error> {
        let Some(e) = g.entries.get(entry) else {
            return Err(Unsupported::Entry(entry).into());
        };
        let wanted = e.outputs.iter().enumerate().filter(|(i, _)| wants.has(*i));
        let outputs: Vec<usize> = wanted.map(|(_, o)| o.node.0 as usize).collect();
        let mut roots = outputs.clone();
        if wants != Wants::ALL {
            // Keep every append a full launch would make.
            let all: Vec<usize> = e.outputs.iter().map(|o| o.node.0 as usize).collect();
            let full = needed(g, &all);
            roots.extend((0..g.nodes.len()).filter(|&i| full[i] && g.nodes[i].op.keeps()));
        }
        let needed = needed(g, &roots);
        let roles = output_roles(g, entry);
        let in_moe = moe_nodes(g);
        let mut tree = Tree {
            g,
            nodes: Vec::new(),
            readers: Vec::new(),
            outputs: Vec::new(),
            ports: Vec::new(),
            sizes: sizes.to_vec(),
            kept,
            kept_rows: None,
            index_rows: None,
            compute: Compute::default(),
            computes: Vec::new(),
            knobs: Knobs::default(),
            config: None,
            parts: parts.clone(),
            rank,
            shares: Vec::new(),
            at: Vec::new(),
            ends: Vec::new(),
            kept_appends: Vec::new(),
            sparse_kept: Vec::new(),
            kept_rings: Vec::new(),
        };
        let ranks = parts.ranks();
        for port in &e.inputs {
            let extents = sized(g, sizes, upto, &port.ty.shape)?;
            let shape: Vec<usize> = extents.iter().map(Extent::most).collect();
            let len = shape.iter().product();
            tree.ports.push((len, shape, port.ty.dtype));
        }
        // Each graph node's value, as a view node.
        let mut at: Vec<Option<T>> = vec![None; g.nodes.len()];
        // A rotated weight's leaves: codes, `suh`, `svh`.
        let mut turned: Vec<Option<[T; 3]>> = vec![None; g.nodes.len()];
        // Sliced views of replicated nodes for owned readers.
        let mut sliced: HashMap<(T, usize, usize), T> = HashMap::new();
        // Spot pins, built once per label.
        let base_pins = pins;
        let labels = (!pins.spots.is_empty()).then(|| crate::recipe::labels(g));
        let mut spotted: HashMap<&str, Pins> = HashMap::new();
        for i in (0..g.nodes.len()).filter(|i| needed[*i]) {
            let pins: &Pins = match &labels {
                Some(l) => spotted
                    .entry(l[i].as_str())
                    .or_insert_with(|| base_pins.at(&l[i])),
                None => base_pins,
            };
            let n = &g.nodes[i];
            let mut extents = sized(g, sizes, upto, &n.ty.shape)?;
            let mut share = None;
            if let Placement::Split { axis } = n.ty.placement
                && ranks > 1
            {
                let a = axis as usize;
                let whole = extents[a].most();
                let fixed = extents[a] == Extent::Fixed(whole);
                let Some((first, count)) = tree.part(whole).filter(|_| fixed) else {
                    let (shape, parts) = (n.ty.shape.clone(), parts.clone());
                    return Err(Size::Split {
                        axis: a,
                        shape,
                        parts,
                    }
                    .into());
                };
                extents[a] = Extent::Fixed(count);
                let inner = extents[a + 1..].iter().map(Extent::most).product();
                share = Some(Share {
                    axis,
                    first,
                    count,
                    whole,
                    inner,
                });
            }
            own::own_extents(n, &mut extents, parts, rank)?;
            let shape: Vec<usize> = extents.iter().map(Extent::most).collect();
            let from = NodeId(i as u32);
            let missing = || Error::from(Unsupported::Unread { node: i });
            let args: Result<Vec<T>, Error> =
                (n.args.iter().map(|a| at[a.0 as usize].ok_or_else(missing))).collect();
            let mut args = args?;
            own::slice_owned_args(&mut tree, g, n, &mut args, &mut sliced);
            let form = match &n.op {
                Op::Weight(w) => base_pins.form(&g.weights[*w as usize].path),
                Op::Input { .. } => tessera_codegen::form::of(n.ty.dtype),
                Op::View(View::Reshape(_)) => tree.node(args[0]).form,
                _ if !n.ty.dtype.is_float() => tessera_codegen::form::of(n.ty.dtype),
                op => pinned_form(pins, op, None, roles[i], in_moe[i]),
            };
            // Kernels read whole blocks, so a row holds whole ones.
            if !shape.last().is_none_or(|k| k.is_multiple_of(form.align())) {
                let (form, align) = (form.name(), form.align());
                return Err(Unsupported::Align {
                    node: i,
                    form,
                    align,
                }
                .into());
            }
            let rotated = |a: &NodeId| turned[a.0 as usize].is_some();
            let read = |(j, a): (usize, &NodeId)| rotated(a) && (n.op != Op::Matmul || j != 1);
            if n.args.iter().enumerate().any(read) {
                return Err(Unsupported::Turned { node: i }.into());
            }
            if let (Op::Weight(w), Some(codes)) = (&n.op, form.codes()) {
                let parts = tree.turned(*w, (&shape, &extents), share, codes, from);
                at[i] = Some(parts[0]);
                turned[i] = Some(parts);
                continue;
            }
            let weight = n.args.get(1).filter(|_| n.op == Op::Matmul);
            if let Some(parts) = weight.and_then(|w| turned[w.0 as usize]) {
                at[i] = Some(tree.turn(args[0], parts, (&shape, &extents), form, from));
                continue;
            }
            let bypass = matches!(
                &n.op,
                Op::Attend(a) if a.ragged && a.picked && a.mask == tessera_graph::Mask::Causal
            ) && all_causal(&tree, &args);
            let mut push = |term: Term, args: Vec<T>, node_form: &'static dyn Load| {
                let (shape, extents) = (shape.clone(), extents.clone());
                let node = Node::new(term, args, (shape, extents), node_form, from);
                tree.add(node, share.filter(|_| matches!(n.op, Op::Weight(_))))
            };
            at[i] = Some(match &n.op {
                Op::Input { entry: e, port } if *e as usize == entry => {
                    push(Term::Leaf(Home::Input(*port)), args, form)
                }
                Op::Weight(w) => push(Term::Leaf(Home::Weight(*w)), args, form),
                Op::Input { .. } => return Err(Unsupported::EntryInput.into()),
                Op::Row(r) => {
                    let (mut cur, mut rest) = (args[0], args[1..].iter());
                    // One rank holds the whole value already.
                    let crossing = |s: &Stage| s.facts().lower == Lower::Crossing;
                    let stages: Vec<&Stage> =
                        r.0.iter().filter(|s| ranks > 1 || !crossing(s)).collect();
                    let wired = n.numerics.and_then(|num| num.wire).is_some();
                    let needs_wire = stages
                        .iter()
                        .any(|s| matches!(s, Stage::AllReduce | Stage::ReduceScatter));
                    if needs_wire && !wired {
                        return Err(Unsupported::Wire { node: i }.into());
                    }
                    let has_cross = stages.iter().any(|s| crossing(s));
                    let mut crossed = !has_cross;
                    let (in_shape, in_extents) = (
                        tree.node(args[0]).shape.clone(),
                        tree.node(args[0]).extents.clone(),
                    );
                    let last = stages.len().saturating_sub(1);
                    for (idx, stage) in stages.into_iter().enumerate() {
                        if matches!(stage, Stage::OwnRows) {
                            crossed = true;
                            let whole = tree.node(cur).shape[0];
                            let (first, count) = parts.rows(whole, rank as usize);
                            if count != whole || first != 0 {
                                cur = own::slice_axis0(&mut tree, cur, first, count, &mut sliced);
                            }
                            continue;
                        }
                        if crossing(stage) {
                            crossed = true;
                        }
                        let operand = (stage.operands() == 1).then(|| rest.next());
                        let operand = operand.flatten().copied();
                        let stage_args = [cur].into_iter().chain(operand).collect();
                        let stage_role = (idx == last).then_some(roles[i]).flatten();
                        let stage_form = match n.ty.dtype.is_float() {
                            true => pinned_form(pins, &n.op, Some(stage), stage_role, in_moe[i]),
                            false => tessera_codegen::form::of(n.ty.dtype),
                        };
                        let (sh, ex) = match crossed {
                            true => (shape.clone(), extents.clone()),
                            false => (in_shape.clone(), in_extents.clone()),
                        };
                        let term = Term::Stage(stage.clone());
                        let node = Node::new(term, stage_args, (sh, ex), stage_form, from);
                        cur = tree.add(node, None);
                    }
                    cur
                }
                Op::Attend(a) if bypass => {
                    let mut plain = a.clone();
                    plain.picked = false;
                    let mut kept = args[..5].to_vec();
                    for i in a.sinks_arg().into_iter().chain(a.rel_bias_arg()) {
                        kept.push(args[i]);
                    }
                    push(Term::Op(Op::Attend(plain)), kept, form)
                }
                op => push(Term::Op(op.clone()), args, form),
            });
        }
        tree.at = at;
        tree.ends = vec![false; tree.nodes.len()];
        tree.readers = vec![0; tree.nodes.len()];
        for n in &tree.nodes {
            for a in &n.args {
                tree.readers[a.0 as usize] += 1;
            }
        }
        tree.index_kept();
        for o in outputs {
            let t = tree.at[o].expect("outputs are needed");
            tree.readers[t.0 as usize] += 1;
            let shape = sized(g, sizes, upto, &g.nodes[o].ty.shape)?;
            tree.outputs
                .push((t, shape.iter().map(Extent::most).collect()));
        }
        Ok(tree)
    }

    /// Matrix-unit precision at `t`, with spot overrides.
    pub fn compute_at(&self, t: T) -> Compute {
        let from = self.node(t).from.0 as usize;
        self.computes
            .get(from)
            .copied()
            .flatten()
            .unwrap_or(self.compute)
    }

    /// The same tree, with matrix-unit precision `compute`.
    pub(crate) fn with_compute(mut self, compute: Compute) -> Self {
        self.compute = compute;
        self
    }

    pub(crate) fn add(&mut self, node: Node, share: Option<Share>) -> T {
        self.nodes.push(node);
        self.shares.push(share);
        T(self.nodes.len() as u32 - 1)
    }

    /// Whether `t` reuses its input buffer without copying.
    pub(crate) fn alias(&self, t: T) -> bool {
        let n = self.node(t);
        match &n.term {
            Term::Op(Op::View(View::Reshape(_))) => true,
            Term::Op(Op::View(View::Permute(order))) => {
                let src = self.node(n.args[0]);
                if src.form.name() != n.form.name() {
                    return false;
                }
                let mut prev = None;
                for &ax in order {
                    if src.shape[ax as usize] > 1 {
                        if prev.is_some_and(|p| ax <= p) {
                            return false;
                        }
                        prev = Some(ax);
                    }
                }
                true
            }
            _ => false,
        }
    }

    /// This rank's part of a split weight.
    pub fn share(&self, t: T) -> Option<Share> {
        self.shares[t.0 as usize]
    }

    /// Global `(first, whole)` along `t`'s last axis.
    pub fn row_share(&self, t: T) -> (usize, usize) {
        let node = self.node(t);
        let width = *node.shape.last().unwrap_or(&0);
        let g = self.g.node(node.from);
        let last = g.ty.rank().saturating_sub(1) as u32;
        if g.ty.placement != (Placement::Split { axis: last }) || self.parts.ranks() <= 1 {
            return (0, width);
        }
        let Some(whole) = self.eval(&g.ty.shape.0[last as usize]).map(|v| v as usize) else {
            return (0, width);
        };
        let first = self.part(whole).map_or(0, |(f, _)| f);
        (first, whole)
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn node(&self, t: T) -> &Node {
        &self.nodes[t.0 as usize]
    }

    pub fn elements(&self, t: T) -> usize {
        self.node(t).shape.iter().product()
    }

    pub fn bytes(&self, t: T) -> usize {
        self.node(t).form.bytes(self.elements(t))
    }

    /// The wire dtype of the cross-rank sum at `t`.
    pub fn wire(&self, t: T) -> Option<DType> {
        self.g.node(self.node(t).from).numerics?.wire
    }

    pub fn leaf(&self, t: T) -> bool {
        matches!(self.node(t).term, Term::Leaf(_))
    }

    /// Whether `t` is a weight, or a view of one.
    pub fn weighed(&self, mut t: T) -> bool {
        loop {
            let node = self.node(t);
            match &node.term {
                Term::Leaf(Home::Weight(_) | Home::Part(..)) => return true,
                Term::Op(Op::View(_)) => t = node.args[0],
                _ => return false,
            }
        }
    }

    /// Bytes of every weight the tree reads.
    pub fn weight_bytes(&self) -> usize {
        let weight = |t: &T| self.leaf(*t) && self.weighed(*t);
        (0..self.len() as u32)
            .map(T)
            .filter(weight)
            .map(|t| self.bytes(t))
            .sum()
    }

    /// Whether `t` mutates kept rows or fixed state.
    pub fn appends(&self, t: T) -> bool {
        matches!(&self.node(t).term, Term::Op(op) if op.keeps())
    }

    fn output(&self, t: T) -> bool {
        self.outputs.iter().any(|(o, _)| *o == t)
    }

    /// Whether the value must have a buffer of its own.
    pub fn root(&self, t: T) -> bool {
        let kept = self.leaf(t) || self.appends(t);
        let end = self.ends[t.0 as usize];
        kept || end || self.readers[t.0 as usize] > 1 || self.output(t)
    }

    /// `t`'s arguments if it is an unshared `term`.
    pub fn inner(&self, t: T, term: &Term) -> Option<&[T]> {
        let node = self.node(t);
        (node.term == *term && !self.root(t)).then_some(&node.args[..])
    }

    /// Whether a region rule claims `t` as an output.
    pub(crate) fn end(&self, t: T) -> bool {
        self.ends[t.0 as usize]
    }

    /// The pattern `P` at `t`, if the node shows it.
    pub fn site<P: Pattern>(&self, t: T) -> Option<P> {
        P::find(self, t)
    }

    /// Whether `t` is an elementwise stage only its reader needs.
    pub(crate) fn joins(&self, t: T) -> bool {
        let map = matches!(&self.node(t).term, Term::Stage(s) if map_stage(s).is_some());
        map && !self.root(t) && !self.closes_turn(t)
    }

    /// Region outputs a rule in `rules` covers become roots.
    pub(crate) fn end_regions(&mut self, rules: &[&Rule]) {
        let mut ends = Vec::new();
        for region in &self.g.regions {
            let Some(t) = self.anchor(region) else {
                continue;
            };
            if rules.iter().any(|r| r.propose.at(self, t).is_some()) {
                ends.extend(region.outputs.iter().filter_map(|o| self.of(*o)));
            }
        }
        for t in ends {
            self.ends[t.0 as usize] = true;
        }
    }

    /// Where a cover of `region` sits: its last output.
    pub fn anchor(&self, region: &Region) -> Option<T> {
        self.of(*region.outputs.iter().max()?)
    }

    /// The tree node holding graph node `id`'s value.
    pub fn of(&self, id: NodeId) -> Option<T> {
        self.at.get(id.0 as usize).copied().flatten()
    }

    /// A dimension's size in this view.
    pub fn size(&self, name: DimName) -> Option<u64> {
        lookup(&self.sizes, name)
    }

    /// This rank's `(first, count)` of `whole` along a split axis.
    pub(crate) fn part(&self, whole: usize) -> Option<(usize, usize)> {
        self.parts.part(whole, self.rank as usize)
    }

    /// Roots inside a cover at `t`. `None` if one leaks.
    pub(crate) fn absorbs(&self, t: T, reads: &[T], writes: &[T]) -> Option<Vec<T>> {
        let inside = self.inside(t, reads);
        let roots: Vec<T> = inside
            .iter()
            .copied()
            .filter(|n| *n != t && self.root(*n))
            .collect();
        // Outside readers may read written outputs after `t`.
        for &r in &roots {
            if writes.contains(&r) {
                let early = (r.0 + 1..t.0)
                    .map(T)
                    .any(|u| !inside.contains(&u) && self.node(u).args.contains(&r));
                if early {
                    return None;
                }
            } else {
                let args = inside.iter().flat_map(|n| &self.node(*n).args);
                let seen = args.filter(|a| **a == r).count();
                if seen != self.readers[r.0 as usize] as usize {
                    return None;
                }
            }
        }
        Some(roots)
    }

    /// Tree nodes a cover at `t` computes, given its reads.
    pub fn inside(&self, t: T, reads: &[T]) -> Vec<T> {
        let (mut nodes, mut stack) = (Vec::new(), vec![t]);
        while let Some(n) = stack.pop() {
            // A root with two readers inside is reached twice.
            if nodes.contains(&n) {
                continue;
            }
            nodes.push(n);
            let inside = |a: &&T| !reads.contains(a) && !self.leaf(**a);
            stack.extend(self.node(n).args.iter().filter(inside));
        }
        nodes.sort_unstable();
        nodes
    }

    /// The graph nodes those came from.
    pub fn covered(&self, t: T, reads: &[T]) -> Vec<NodeId> {
        let nodes = self.inside(t, reads);
        let mut from: Vec<NodeId> = nodes.iter().map(|n| self.node(*n).from).collect();
        from.dedup();
        from
    }

    /// Evaluates a dimension expression at this tree's concrete sizes.
    pub fn eval(&self, expr: &tessera_graph::DimExpr) -> Option<u64> {
        expr.value(&self.g.dims, &|name| self.size(name))
    }
}
