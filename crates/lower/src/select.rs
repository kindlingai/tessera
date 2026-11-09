//! Picks kernels for graph nodes, as ops over values.

use crate::Error;
use crate::cost::Device;
use crate::kernel::KernelRef;
use crate::kernels::MapStage;
use crate::rule::{Attn, Draft, Read, Rule, Site, rules};
use graph::{Attend, DimName, Graph, Mask, NodeId, Op as Node, Select, Shape, Stage, View};
use std::collections::{BTreeSet, HashMap};

/// A value some op writes, or the plan is given.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Val(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Home {
    Weight(u32),
    Input(u32),
    /// Written by an op; its buffer may be reused.
    Temp,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Value {
    pub len: usize,
    pub home: Home,
}

/// One launch, before it has buffers.
#[derive(Clone, Debug, PartialEq)]
pub struct Op {
    pub kernel: KernelRef,
    /// The rule that proposed it.
    pub rule: &'static str,
    pub reads: Vec<Val>,
    pub write: Val,
    pub covers: Vec<NodeId>,
}

/// The kernels chosen for one entry, in traced order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Selected {
    pub values: Vec<Value>,
    pub ops: Vec<Op>,
    pub inputs: Vec<Val>,
    pub outputs: Vec<(Val, Vec<usize>)>,
    /// Rules that competed with another at some site.
    pub contested: BTreeSet<&'static str>,
}

/// Elementwise stages waiting for the node they fuse into.
struct Chain {
    source: usize,
    stages: Vec<MapStage>,
    operands: Vec<usize>,
    covers: Vec<NodeId>,
}

struct Picker<'a> {
    g: &'a Graph,
    sizes: &'a [(DimName, u64)],
    device: &'a Device,
    rules: Vec<&'static Rule>,
    out: Selected,
    /// Each lowered node's value and shape.
    vals: Vec<Option<(Val, Vec<usize>)>>,
    chains: HashMap<usize, Chain>,
    /// Per node: the node its work is emitted with.
    sink: Vec<usize>,
    /// Results that later equal launches may reuse.
    shared: HashMap<(String, Vec<Val>), Val>,
}

fn unsupported<T>(what: impl std::fmt::Debug) -> Result<T, Error> {
    Err(Error::Unsupported(format!("{what:?}")))
}

fn map_stage(s: &Stage) -> Option<MapStage> {
    match s {
        Stage::Add => Some(MapStage::Add),
        Stage::Mul => Some(MapStage::Mul),
        Stage::Scale(by) => Some(MapStage::Scale(*by)),
        Stage::Act(a) => Some(MapStage::Act(*a)),
        _ => None,
    }
}

/// Whether one elementwise kernel can run this whole op.
fn maps(op: &Node) -> bool {
    matches!(op, Node::Row(r) if r.0.iter().all(|s| map_stage(s).is_some()))
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

/// A map read once, as another map's row, joins it.
fn sinks(g: &Graph, needed: &[bool], outputs: &[usize]) -> Vec<usize> {
    let mut readers = vec![Vec::new(); g.nodes.len()];
    for (j, n) in g.nodes.iter().enumerate().filter(|(j, _)| needed[*j]) {
        for a in &n.args {
            readers[a.0 as usize].push(j);
        }
    }
    let mut sink: Vec<usize> = (0..g.nodes.len()).collect();
    for i in (0..g.nodes.len()).rev() {
        let alone = needed[i] && !outputs.contains(&i) && maps(&g.nodes[i].op);
        if let (true, [j]) = (alone, &readers[i][..]) {
            let reader = &g.nodes[*j];
            if maps(&reader.op) && reader.args[0].0 as usize == i {
                sink[i] = sink[*j];
            }
        }
    }
    sink
}

/// Chooses kernels for `entry`, sized by `sizes`, costed by `device`.
pub fn select(
    g: &Graph,
    entry: usize,
    sizes: &[(DimName, u64)],
    device: &Device,
) -> Result<Selected, Error> {
    let Some(e) = g.entries.get(entry) else {
        return unsupported(format_args!("entry {entry}"));
    };
    let outputs: Vec<usize> = e.outputs.iter().map(|o| o.node.0 as usize).collect();
    let needed = needed(g, &outputs);
    let mut p = Picker {
        g,
        sizes,
        device,
        rules: rules(),
        out: Selected::default(),
        vals: vec![None; g.nodes.len()],
        chains: HashMap::new(),
        sink: sinks(g, &needed, &outputs),
        shared: HashMap::new(),
    };
    for (port, decl) in e.inputs.iter().enumerate() {
        let len = p.shape(&decl.ty.shape)?.iter().product();
        let val = p.value(len, Home::Input(port as u32));
        p.out.inputs.push(val);
    }
    for i in (0..g.nodes.len()).filter(|i| needed[*i]) {
        p.node(i, entry)?;
    }
    for o in outputs {
        let out = p.val(o)?;
        p.out.outputs.push(out);
    }
    Ok(p.out)
}

impl Picker<'_> {
    fn shape(&self, s: &Shape) -> Result<Vec<usize>, Error> {
        let size = |name| {
            let bound = self.sizes.iter().find(|(n, _)| *n == name);
            bound.map(|(_, size)| *size)
        };
        let dim = |e: &graph::DimExpr| {
            let poly = e.poly(&self.g.dims).ok()?;
            Some(poly.eval(&size)? as usize)
        };
        let sized: Option<Vec<usize>> = s.0.iter().map(dim).collect();
        sized.ok_or_else(|| Error::Size(s.to_string()))
    }

    fn val(&self, node: usize) -> Result<(Val, Vec<usize>), Error> {
        let missing = || Error::Unsupported(format!("node {node} as an argument"));
        self.vals[node].clone().ok_or_else(missing)
    }

    fn len(&self, v: Val) -> usize {
        self.out.values[v.0 as usize].len
    }

    fn value(&mut self, len: usize, home: Home) -> Val {
        self.out.values.push(Value { len, home });
        Val(self.out.values.len() as u32 - 1)
    }

    /// Asks every rule, then emits the cheapest proposal.
    fn choose(&mut self, site: Site, covers: &[NodeId]) -> Result<Val, Error> {
        let mut best: Option<(f64, &'static str, Vec<Draft>)> = None;
        let mut offered = Vec::new();
        for rule in &self.rules {
            let Some(drafts) = (rule.propose)(&site) else {
                continue;
            };
            let seconds = |d: &Draft| self.device.seconds(&*d.kernel, d.len);
            let mut cost: f64 = drafts.iter().map(seconds).sum();
            // A favored rule wins wherever it applies.
            if self.device.favors(rule.name) {
                cost = f64::NEG_INFINITY;
            }
            offered.push(rule.name);
            if best.as_ref().is_none_or(|(least, ..)| cost < *least) {
                best = Some((cost, rule.name, drafts));
            }
        }
        if offered.len() > 1 {
            self.out.contested.extend(offered);
        }
        let Some((_, rule, drafts)) = best else {
            return unsupported(site);
        };
        let mut made: Vec<Val> = Vec::with_capacity(drafts.len());
        for d in drafts {
            let read = |r: &Read| match r {
                Read::Value(v) => *v,
                Read::Draft(at) => made[*at],
            };
            let reads: Vec<Val> = d.reads.iter().map(read).collect();
            made.push(self.emit(d.kernel, rule, reads, d.len, covers));
        }
        Ok(*made.last().expect("a rule proposes at least one launch"))
    }

    /// Adds an op writing a new value, or shares one.
    fn emit(
        &mut self,
        kernel: KernelRef,
        rule: &'static str,
        reads: Vec<Val>,
        len: usize,
        covers: &[NodeId],
    ) -> Val {
        let key = (format!("{kernel:?}"), reads.clone());
        if let Some(same) = self.shared.get(&key).filter(|_| kernel.shared()) {
            return *same;
        }
        let write = self.value(len, Home::Temp);
        if kernel.shared() {
            self.shared.insert(key, write);
        }
        let covers = covers.to_vec();
        self.out.ops.push(Op {
            kernel,
            rule,
            reads,
            write,
            covers,
        });
        write
    }

    fn node(&mut self, i: usize, entry: usize) -> Result<(), Error> {
        let n = &self.g.nodes[i];
        let shape = self.shape(&n.ty.shape)?;
        let len = shape.iter().product();
        let covers = [NodeId(i as u32)];
        let arg = |k: usize| n.args[k].0 as usize;
        let val = match &n.op {
            Node::Input { entry: e, port } if *e as usize == entry => {
                self.out.inputs[*port as usize]
            }
            Node::Weight(w) => self.value(len, Home::Weight(*w)),
            // A reshape is the same value under a new shape.
            Node::View(View::Reshape(_)) => self.val(arg(0))?.0,
            Node::Matmul => {
                let ((x, _), (w, wide)) = (self.val(arg(0))?, self.val(arg(1))?);
                let [n, k] = wide[..] else {
                    return unsupported("a batched matmul");
                };
                let m = self.len(x) / k;
                self.choose(Site::Matmul { x, w, m, n, k }, &covers)?
            }
            Node::Row(r) if maps(&n.op) => return self.map(i, &r.0),
            Node::Row(r) => self.row(i, &r.0)?,
            Node::Attend(a) => self.attend(i, a)?,
            Node::Select(Select::Gather) => {
                let ((table, wide), (ids, _)) = (self.val(arg(0))?, self.val(arg(1))?);
                let site = Site::Gather {
                    table,
                    ids,
                    rows: self.len(ids),
                    width: self.len(table) / wide[0],
                };
                self.choose(site, &covers)?
            }
            other => return unsupported(other),
        };
        self.vals[i] = Some((val, shape));
        Ok(())
    }

    /// Extends the chain its first argument left, or starts one.
    fn map(&mut self, i: usize, stages: &[Stage]) -> Result<(), Error> {
        let n = &self.g.nodes[i];
        let first = n.args[0].0 as usize;
        let mut chain = self.chains.remove(&first).unwrap_or(Chain {
            source: first,
            stages: Vec::new(),
            operands: Vec::new(),
            covers: Vec::new(),
        });
        chain.stages.extend(stages.iter().filter_map(map_stage));
        chain
            .operands
            .extend(n.args[1..].iter().map(|a| a.0 as usize));
        chain.covers.push(NodeId(i as u32));
        if self.sink[i] != i {
            self.chains.insert(i, chain);
            return Ok(());
        }
        let (x, shape) = self.val(chain.source)?;
        let mut reads = vec![x];
        for o in &chain.operands {
            let operand = self.val(*o)?.0;
            if self.len(operand) != self.len(x) {
                return unsupported("a broadcast operand");
            }
            reads.push(operand);
        }
        let site = Site::Map {
            stages: &chain.stages,
            reads: &reads,
            len: self.len(x),
        };
        let val = self.choose(site, &chain.covers)?;
        self.vals[i] = Some((val, shape));
        Ok(())
    }

    /// A row holding a norm or rotation, stage by stage.
    fn row(&mut self, i: usize, stages: &[Stage]) -> Result<Val, Error> {
        let n = &self.g.nodes[i];
        let covers = [NodeId(i as u32)];
        let (mut cur, shape) = self.val(n.args[0].0 as usize)?;
        let width = *shape.last().expect("verified rank");
        let len = self.len(cur);
        let mut rest = n.args[1..].iter().map(|a| a.0 as usize);
        for stage in stages {
            // One rank holds the whole sum already.
            if *stage == Stage::AllReduce {
                continue;
            }
            let mut operand = None;
            if stage.operands() == 1 {
                let val = self.val(rest.next().expect("verified"))?.0;
                operand = Some((val, self.len(val)));
            }
            let site = Site::Stage {
                stage,
                x: cur,
                operand,
                len,
                width,
            };
            cur = self.choose(site, &covers)?;
        }
        Ok(cur)
    }

    fn attend(&mut self, i: usize, a: &Attend) -> Result<Val, Error> {
        let n = &self.g.nodes[i];
        if a.picked || a.mask != Mask::Causal {
            return unsupported(a);
        }
        let arg = |k: usize| self.val(n.args[k].0 as usize);
        let ((q, wide), (k, deep), (v, _)) = (arg(0)?, arg(1)?, arg(2)?);
        let (&[seqs, queries, heads, width], &[_, keys, kv_heads, _]) = (&wide[..], &deep[..])
        else {
            return unsupported("attention below rank 4");
        };
        let sizes = Attn {
            seqs,
            queries,
            keys,
            heads,
            kv_heads,
            width,
        };
        let site = Site::Attend {
            scale: a.scale,
            q,
            k,
            v,
            sizes,
        };
        self.choose(site, &[NodeId(i as u32)])
    }
}
