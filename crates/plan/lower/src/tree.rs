//! An entry as trees: values with one reader join it.

use crate::Error;
use crate::extent::Extent;
use crate::form::Pins;
use crate::kernels::MapStage;
use crate::rule::{Attn, Site};
use crate::select::Home;
use graph::{Attend, DType, DimName, Graph, Mask, NodeId, Op, Select, Shape, Stage, View};

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
    pub form: DType,
    /// The graph node it came from.
    pub from: NodeId,
}

/// The nodes an entry needs, with their sizes and readers.
pub struct Tree<'a> {
    pub g: &'a Graph,
    nodes: Vec<Node>,
    /// Per node: the pattern rules see there, found once.
    sites: Vec<Option<Site>>,
    /// Per node: how many arguments and outputs name it.
    readers: Vec<u32>,
    /// The entry's outputs, with each one's graph shape.
    pub outputs: Vec<(T, Vec<usize>)>,
    /// Elements and type of each input port, used or not.
    pub ports: Vec<(usize, DType)>,
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
    let at = |e: &graph::DimExpr, value: Option<u64>| {
        let size = |name| match upto.contains(&name) {
            true => value,
            false => sizes.iter().find(|(n, _)| *n == name).map(|(_, n)| *n),
        };
        e.value(&g.dims, &size)
    };
    let most = |e: &graph::DimExpr| {
        let size = |name| sizes.iter().find(|(n, _)| *n == name).map(|(_, n)| *n);
        e.value(&g.dims, &size).map(|n| n as usize)
    };
    let dim = |e: &graph::DimExpr| -> Option<Extent> {
        let most = most(e)?;
        let (v0, v1, v2) = (at(e, Some(0))?, at(e, Some(1))?, at(e, Some(2))?);
        if upto.is_empty() || (v0 == v1 && v1 == v2) {
            return Some(Extent::Fixed(most));
        }
        // At most one runtime dimension, entering linearly.
        let (mul, add) = (v1 - v0, v0 as i64);
        (v2 - v1 == mul && upto.len() == 1).then_some(Extent::Dim {
            name: upto[0],
            mul: mul as usize,
            add,
            most,
        })
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
            sites: Vec::new(),
            readers: Vec::new(),
            outputs: Vec::new(),
            ports: Vec::new(),
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
                _ => n.ty.dtype,
            };
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
        // Every rule asks everywhere, so sites are found once.
        tree.sites = (0..tree.nodes.len() as u32)
            .map(|t| tree.find_site(T(t)))
            .collect();
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
        self.elements(t) * self.node(t).form.bytes()
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

    /// The pattern at `t`, as rules know it, if any.
    pub fn site(&self, t: T) -> Option<Site> {
        self.sites[t.0 as usize].clone()
    }

    fn find_site(&self, t: T) -> Option<Site> {
        let n = self.node(t);
        let wide = |a: T| &self.node(a).shape;
        match &n.term {
            Term::Leaf(_) | Term::View | Term::Append(_) => None,
            Term::Matmul => {
                let (x, w) = (n.args[0], n.args[1]);
                let [n, k] = wide(w)[..] else {
                    return None;
                };
                let (m, form) = (self.elements(x) / k, self.node(w).form);
                Some(Site::Matmul {
                    x,
                    w,
                    m,
                    n,
                    k,
                    form,
                })
            }
            Term::Stage(stage) if map_stage(stage).is_some() => self.chain(t),
            Term::Stage(stage) => {
                let x = n.args[0];
                let operand = n.args.get(1).map(|o| (*o, self.elements(*o)));
                let (len, width) = (self.elements(x), *wide(x).last()?);
                Some(Site::Stage {
                    stage: stage.clone(),
                    x,
                    operand,
                    len,
                    width,
                })
            }
            Term::Attend(a) => {
                if a.picked || a.mask != Mask::Causal {
                    return None;
                }
                let (q, k, v) = (n.args[0], n.args[1], n.args[2]);
                let (&[seqs, queries, heads, width], &[_, _, kv_heads, _]) =
                    (&wide(q)[..], &wide(k)[..])
                else {
                    return None;
                };
                let keys = self.node(k).extents[1];
                let sizes = Attn {
                    seqs,
                    queries,
                    keys,
                    heads,
                    kv_heads,
                    width,
                };
                let scale = a.scale;
                Some(Site::Attend {
                    scale,
                    q,
                    k,
                    v,
                    sizes,
                })
            }
            Term::Gather => {
                let (table, ids) = (n.args[0], n.args[1]);
                let (rows, width) = (
                    self.elements(ids),
                    self.elements(table) / wide(table).first()?,
                );
                let form = self.node(table).form;
                Some(Site::Gather {
                    table,
                    ids,
                    rows,
                    width,
                    form,
                })
            }
            Term::Top => {
                let x = n.args[0];
                let width = *wide(x).last()?;
                let (rows, k) = (
                    self.elements(x) / width,
                    self.elements(t) * width / self.elements(x),
                );
                Some(Site::Top { x, rows, width, k })
            }
        }
    }

    /// Whether `t` is an elementwise stage only its reader needs.
    fn joins(&self, t: T) -> bool {
        let map = matches!(&self.node(t).term, Term::Stage(s) if map_stage(s).is_some());
        map && !self.root(t)
    }

    /// The elementwise stages ending at `t`, back through first arguments.
    fn chain(&self, t: T) -> Option<Site> {
        let mut nodes = vec![t];
        while let Some(first) = self.node(nodes[nodes.len() - 1]).args.first() {
            if !self.joins(*first) {
                break;
            }
            nodes.push(*first);
        }
        nodes.reverse();
        let source = self.node(nodes[0]).args[0];
        let len = self.elements(source);
        let (mut stages, mut reads) = (Vec::new(), vec![source]);
        for n in nodes.iter().map(|n| self.node(*n)) {
            let Term::Stage(stage) = &n.term else {
                return None;
            };
            stages.push(map_stage(stage)?);
            reads.extend(&n.args[1..]);
        }
        // An operand of another size would need broadcasting.
        let fits = reads.iter().all(|r| self.elements(*r) == len);
        fits.then_some(Site::Map { stages, reads, len })
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
