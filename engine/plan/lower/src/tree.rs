//! An entry as trees: values with one reader join it.

use crate::extent::Extent;
use crate::form::{Load, Pins};
use crate::kernels::MapStage;
use crate::plan::Share;
use crate::rule::Rule;
use crate::select::Home;
use crate::sites::Pattern;
use crate::{Error, Size, Unsupported};
use tessera_codegen::kept::Layout;
use tessera_graph::{
    DType, DimExpr, DimName, Graph, NodeId, Op, Parts, Placement, Region, Shape, Stage,
};

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

/// The nodes an entry needs, with their sizes and readers.
pub struct Tree<'a> {
    pub g: &'a Graph,
    nodes: Vec<Node>,
    /// Per node: how many arguments and outputs name it.
    readers: Vec<u32>,
    /// The entry's outputs, with each one's graph shape.
    pub outputs: Vec<(T, Vec<usize>)>,
    /// Elements and type of each input port, used or not.
    pub ports: Vec<(usize, DType)>,
    /// The sizes this tree was viewed at.
    pub(crate) sizes: Vec<(DimName, u64)>,
    /// Where kept rows lie.
    pub kept: &'static dyn Layout,
    /// How split axes divide, and which part this plan is.
    pub(crate) parts: Parts,
    pub(crate) rank: u32,
    /// Per node: the part of a split weight loaded here.
    shares: Vec<Option<Share>>,
    /// Per graph node: the tree node holding its value.
    at: Vec<Option<T>>,
    /// Per node: a region output a region rule claims.
    ends: Vec<bool>,
}

pub(crate) fn map_stage(s: &Stage) -> Option<MapStage> {
    match s {
        Stage::Add => Some(MapStage::Add),
        Stage::Mul => Some(MapStage::Mul),
        Stage::Scale(by) => Some(MapStage::Scale(*by)),
        Stage::Act(a) => Some(MapStage::Act(*a)),
        _ => None,
    }
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
            false => sizes.iter().find(|(n, _)| *n == name).map(|(_, n)| *n),
        };
        e.value(&g.dims, &size).map(|v| v as i64)
    };
    let most = |e: &tessera_graph::DimExpr| {
        let size = |name| sizes.iter().find(|(n, _)| *n == name).map(|(_, n)| *n);
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
        Self::upto(g, entry, sizes, &[], pins, kept, &Parts::even(1), 0)
    }

    /// The same, `upto` read at launch, as `rank` of `parts`.
    #[allow(clippy::too_many_arguments, reason = "one view, every way it varies")]
    pub fn upto(
        g: &'a Graph,
        entry: usize,
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
        let outputs: Vec<usize> = e.outputs.iter().map(|o| o.node.0 as usize).collect();
        let needed = needed(g, &outputs);
        let mut tree = Tree {
            g,
            nodes: Vec::new(),
            readers: Vec::new(),
            outputs: Vec::new(),
            ports: Vec::new(),
            sizes: sizes.to_vec(),
            kept,
            parts: parts.clone(),
            rank,
            shares: Vec::new(),
            at: Vec::new(),
            ends: Vec::new(),
        };
        let ranks = parts.ranks();
        for port in &e.inputs {
            let shape = sized(g, sizes, upto, &port.ty.shape)?;
            let len = shape.iter().map(Extent::most).product();
            tree.ports.push((len, port.ty.dtype));
        }
        // Each graph node's value, as a view node.
        let mut at: Vec<Option<T>> = vec![None; g.nodes.len()];
        // A rotated weight's leaves: codes, `suh`, `svh`.
        let mut turned: Vec<Option<[T; 3]>> = vec![None; g.nodes.len()];
        for i in (0..g.nodes.len()).filter(|i| needed[*i]) {
            let n = &g.nodes[i];
            let mut extents = sized(g, sizes, upto, &n.ty.shape)?;
            let mut share = None;
            if let Placement::Split { axis } = n.ty.placement
                && ranks > 1
            {
                let a = axis as usize;
                let whole = extents[a].most();
                let fixed = extents[a] == Extent::Fixed(whole);
                let Some((first, count)) = parts.part(whole, rank as usize).filter(|_| fixed)
                else {
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
            let shape: Vec<usize> = extents.iter().map(Extent::most).collect();
            let from = NodeId(i as u32);
            let missing = || Error::from(Unsupported::Unread { node: i });
            let args: Result<Vec<T>, Error> =
                (n.args.iter().map(|a| at[a.0 as usize].ok_or_else(missing))).collect();
            let args = args?;
            let form = match &n.op {
                Op::Weight(w) => pins.form(&g.weights[*w as usize].path),
                _ => tessera_codegen::form::of(n.ty.dtype),
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
                at[i] = Some(tree.turn(args[0], parts, (&shape, &extents), from));
                continue;
            }
            let mut push = |term: Term, args: Vec<T>| {
                let (shape, extents) = (shape.clone(), extents.clone());
                tree.nodes.push(Node {
                    term,
                    args,
                    shape,
                    extents,
                    form,
                    from,
                });
                tree.shares
                    .push(share.filter(|_| matches!(n.op, Op::Weight(_))));
                T(tree.nodes.len() as u32 - 1)
            };
            at[i] = Some(match &n.op {
                Op::Input { entry: e, port } if *e as usize == entry => {
                    push(Term::Leaf(Home::Input(*port)), args)
                }
                Op::Weight(w) => push(Term::Leaf(Home::Weight(*w)), args),
                Op::Input { .. } => return Err(Unsupported::EntryInput.into()),
                Op::Row(r) => {
                    let (mut cur, mut rest) = (args[0], args[1..].iter());
                    // One rank holds the whole value already.
                    let crossing = |s: &Stage| matches!(s, Stage::AllReduce | Stage::AllGather);
                    let summed = |s: &&Stage| ranks > 1 || !crossing(s);
                    for stage in r.0.iter().filter(summed) {
                        let operand = (stage.operands() == 1).then(|| rest.next());
                        let operand = operand.flatten().copied();
                        let args = [cur].into_iter().chain(operand).collect();
                        cur = push(Term::Stage(stage.clone()), args);
                    }
                    cur
                }
                op => push(Term::Op(op.clone()), args),
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
        for o in outputs {
            let t = tree.at[o].expect("outputs are needed");
            tree.readers[t.0 as usize] += 1;
            let shape = sized(g, sizes, upto, &g.nodes[o].ty.shape)?;
            tree.outputs
                .push((t, shape.iter().map(Extent::most).collect()));
        }
        Ok(tree)
    }

    fn add(&mut self, node: Node, share: Option<Share>) -> T {
        self.nodes.push(node);
        self.shares.push(share);
        T(self.nodes.len() as u32 - 1)
    }

    /// A rotated weight's leaves: codes, `suh` and `svh`.
    fn turned(
        &mut self,
        weight: u32,
        (shape, extents): (&[usize], &[Extent]),
        share: Option<Share>,
        codes: &'static dyn Load,
        from: NodeId,
    ) -> [T; 3] {
        let leaf = |part: u8, axis: usize, form: &'static dyn Load| Node {
            term: Term::Leaf(Home::Part(weight, part)),
            args: Vec::new(),
            shape: match part {
                0 => shape.to_vec(),
                _ => vec![shape[axis]],
            },
            extents: match part {
                0 => extents.to_vec(),
                _ => vec![extents[axis]],
            },
            form,
            from,
        };
        let along = |axis: u32| {
            let cut = |s: Share| Share {
                axis: 0,
                inner: 1,
                ..s
            };
            share.filter(|s| s.axis == axis).map(cut)
        };
        let f32 = crate::form::F32_FORM;
        [
            self.add(leaf(0, 0, codes), share),
            self.add(leaf(1, 1, f32), along(1)),
            self.add(leaf(2, 0, f32), along(0)),
        ]
    }

    /// `x` through a rotated matmul: scale, turn, codes, turn, scale.
    fn turn(
        &mut self,
        x: T,
        [codes, suh, svh]: [T; 3],
        (shape, extents): (&[usize], &[Extent]),
        from: NodeId,
    ) -> T {
        let (rows, wide) = (self.node(x).shape.clone(), self.node(x).extents.clone());
        let turn = Stage::Hadamard { block: 128 };
        let mut step = |term: Term, args: Vec<T>, out: bool| {
            let (shape, extents) = match out {
                true => (shape.to_vec(), extents.to_vec()),
                false => (rows.clone(), wide.clone()),
            };
            let form = crate::form::F32_FORM;
            let node = Node {
                term,
                args,
                shape,
                extents,
                form,
                from,
            };
            self.add(node, None)
        };
        let scaled = step(Term::Stage(Stage::Mul), vec![x, suh], false);
        let turned = step(Term::Stage(turn.clone()), vec![scaled], false);
        let product = step(Term::Op(Op::Matmul), vec![turned, codes], true);
        let back = step(Term::Stage(turn), vec![product], true);
        step(Term::Stage(Stage::Mul), vec![back, svh], true)
    }

    /// This rank's part of a split weight.
    pub fn share(&self, t: T) -> Option<Share> {
        self.shares[t.0 as usize]
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
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

    /// Whether `t` is kept rows: an `Append`.
    pub fn appends(&self, t: T) -> bool {
        matches!(self.node(t).term, Term::Op(Op::Append(_)))
    }

    fn output(&self, t: T) -> bool {
        self.outputs.iter().any(|(o, _)| *o == t)
    }

    /// Whether the value must have a buffer of its own.
    pub fn root(&self, t: T) -> bool {
        let kept = self.leaf(t) || self.appends(t);
        let end = self.ends[t.0 as usize];
        kept || end || self.readers[t.0 as usize] != 1 || self.output(t)
    }

    /// The pattern `P` at `t`, if the node shows it.
    pub fn site<P: Pattern>(&self, t: T) -> Option<P> {
        P::find(self, t)
    }

    /// Whether `t` is an elementwise stage only its reader needs.
    pub(crate) fn joins(&self, t: T) -> bool {
        let map = matches!(&self.node(t).term, Term::Stage(s) if map_stage(s).is_some());
        map && !self.root(t)
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

    /// The node appending to kept `index`, and rows per slot.
    pub fn kept_rows(&self, index: u32) -> Option<(T, usize)> {
        let append = Term::Op(Op::Append(index));
        let t = (0..self.len() as u32)
            .map(T)
            .find(|t| self.node(*t).term == append)?;
        let DimExpr::Dim(axis) = &self.g.kept[index as usize].shape.0[1] else {
            return None;
        };
        // A slot holds the axis's bound, whatever this launch sees.
        let most = self.g.dims.iter().find(|d| d.name == *axis);
        Some((t, most.map_or(0, |d| d.size.max() as usize)))
    }

    /// A dimension's size in this view.
    pub fn size(&self, name: DimName) -> Option<u64> {
        self.sizes.iter().find(|(n, _)| *n == name).map(|(_, v)| *v)
    }

    /// Roots inside a cover at `t`. `None` if one leaks.
    pub(crate) fn absorbs(&self, t: T, reads: &[T], writes: &[T]) -> Option<Vec<T>> {
        let inside = self.inside(t, reads);
        let roots: Vec<T> = inside
            .iter()
            .copied()
            .filter(|n| *n != t && self.root(*n))
            .collect();
        // Outside readers may read the outputs it writes.
        for &r in roots.iter().filter(|r| !writes.contains(r)) {
            let args = inside.iter().flat_map(|n| &self.node(*n).args);
            let seen = args.filter(|a| **a == r).count();
            if seen != self.readers[r.0 as usize] as usize {
                return None;
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
}
