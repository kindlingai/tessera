//! An entry as trees: values with one reader join it.

use crate::Error;
use crate::extent::Extent;
use crate::form::{Form, Pins};
use crate::kernels::MapStage;
use crate::select::Home;
use crate::sites::Pattern;
use graph::{Attend, DType, DimName, Graph, NodeId, Op, Select, Shape, Stage, View};

/// A node of the view. Arguments come before their readers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct T(pub u32);

/// A view node's computation.
#[derive(Clone, Debug, PartialEq)]
pub enum Term {
    /// A weight or an input, there before any launch.
    Leaf(Home),
    Matmul,
    /// One stage of a row.
    Stage(Stage),
    Attend(Attend),
    Gather,
    Top,
    /// New rows joining the kept rows of this index.
    Append(u32),
    /// Its argument under another shape.
    View,
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
    pub form: &'static dyn Form,
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
}

pub(crate) fn unsupported<T>(what: impl std::fmt::Debug) -> Result<T, Error> {
    Err(Error::Unsupported(format!("{what:?}")))
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
    let at = |e: &graph::DimExpr, vary: DimName, value: u64| {
        let size = |name| match name == vary {
            true => Some(value),
            false => sizes.iter().find(|(n, _)| *n == name).map(|(_, n)| *n),
        };
        e.value(&g.dims, &size).map(|v| v as i64)
    };
    let most = |e: &graph::DimExpr| {
        let size = |name| sizes.iter().find(|(n, _)| *n == name).map(|(_, n)| *n);
        e.value(&g.dims, &size).map(|n| n as usize)
    };
    let dim = |e: &graph::DimExpr| -> Option<Extent> {
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
    all.ok_or_else(|| Error::Size(s.to_string()))
}

impl<'a> Tree<'a> {
    /// Views `entry` at these sizes, one node per row stage.
    pub fn new(
        g: &'a Graph,
        entry: usize,
        sizes: &[(DimName, u64)],
        pins: &Pins,
    ) -> Result<Self, Error> {
        Self::upto(g, entry, sizes, &[], pins)
    }

    /// The same, with `upto` dimensions read at launch.
    pub fn upto(
        g: &'a Graph,
        entry: usize,
        sizes: &[(DimName, u64)],
        upto: &[DimName],
        pins: &Pins,
    ) -> Result<Self, Error> {
        let Some(e) = g.entries.get(entry) else {
            return unsupported(format_args!("entry {entry}"));
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
        };
        for port in &e.inputs {
            let shape = sized(g, sizes, upto, &port.ty.shape)?;
            let len = shape.iter().map(Extent::most).product();
            tree.ports.push((len, port.ty.dtype));
        }
        // Each graph node's value, as a view node.
        let mut at: Vec<Option<T>> = vec![None; g.nodes.len()];
        for i in (0..g.nodes.len()).filter(|i| needed[*i]) {
            let n = &g.nodes[i];
            let extents = sized(g, sizes, upto, &n.ty.shape)?;
            let shape: Vec<usize> = extents.iter().map(Extent::most).collect();
            let from = NodeId(i as u32);
            let missing = || Error::Unsupported(format!("node {i} reads an unsupported node"));
            let args: Result<Vec<T>, Error> =
                (n.args.iter().map(|a| at[a.0 as usize].ok_or_else(missing))).collect();
            let args = args?;
            let form = match &n.op {
                Op::Weight(w) => pins.form(&g.weights[*w as usize].path),
                _ => crate::form::of(n.ty.dtype),
            };
            // Kernels read whole blocks, so a row holds whole ones.
            if !shape.last().is_some_and(|k| k.is_multiple_of(form.align())) {
                let what = format!("node {i} in {} needs rows of {}", form.name(), form.align());
                return Err(Error::Unsupported(what));
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
                T(tree.nodes.len() as u32 - 1)
            };
            at[i] = Some(match &n.op {
                Op::Input { entry: e, port } if *e as usize == entry => {
                    push(Term::Leaf(Home::Input(*port)), args)
                }
                Op::Weight(w) => push(Term::Leaf(Home::Weight(*w)), args),
                Op::View(View::Reshape(_)) => push(Term::View, args),
                Op::Matmul => push(Term::Matmul, args),
                Op::Attend(a) => push(Term::Attend(a.clone()), args),
                Op::Select(Select::Gather) => push(Term::Gather, args),
                Op::Select(Select::TopK { .. }) => push(Term::Top, args),
                Op::Append(kept) => push(Term::Append(*kept), args),
                Op::Row(r) => {
                    let (mut cur, mut rest) = (args[0], args[1..].iter());
                    // One rank holds the whole sum already.
                    for stage in r.0.iter().filter(|s| **s != Stage::AllReduce) {
                        let operand = (stage.operands() == 1).then(|| rest.next());
                        let operand = operand.flatten().copied();
                        let args = [cur].into_iter().chain(operand).collect();
                        cur = push(Term::Stage(stage.clone()), args);
                    }
                    cur
                }
                other => return unsupported(other),
            });
        }
        tree.readers = vec![0; tree.nodes.len()];
        for n in &tree.nodes {
            for a in &n.args {
                tree.readers[a.0 as usize] += 1;
            }
        }
        for o in outputs {
            let t = at[o].expect("outputs are needed");
            tree.readers[t.0 as usize] += 1;
            let shape = sized(g, sizes, upto, &g.nodes[o].ty.shape)?;
            tree.outputs
                .push((t, shape.iter().map(Extent::most).collect()));
        }
        Ok(tree)
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

    fn output(&self, t: T) -> bool {
        self.outputs.iter().any(|(o, _)| *o == t)
    }

    /// Whether the value must have a buffer of its own.
    pub fn root(&self, t: T) -> bool {
        let kept = matches!(self.node(t).term, Term::Leaf(_) | Term::Append(_));
        kept || self.readers[t.0 as usize] != 1 || self.output(t)
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

    /// Tree nodes a cover at `t` computes, given its reads.
    pub fn inside(&self, t: T, reads: &[T]) -> Vec<T> {
        let (mut nodes, mut stack) = (Vec::new(), vec![t]);
        while let Some(n) = stack.pop() {
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
