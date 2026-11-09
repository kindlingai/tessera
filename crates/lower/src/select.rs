//! Picks kernels for graph nodes, as ops over values.

use crate::Error;
use crate::cost::Device;
use crate::plan::{Attn, Kernel, MapStage};
use graph::{Attend, DimName, Graph, Mask, NodeId, Op as Node, Pairs, Select, Shape, Stage, View};
use std::collections::HashMap;

/// The most score elements dense attention may hold.
pub const DENSE_SCORES: usize = 1 << 28;

/// The most rows the one-thread-per-column matmul takes.
pub const THIN_ROWS: usize = 32;

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
    pub kernel: Kernel,
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
    out: Selected,
    /// Each lowered node's value and shape.
    vals: Vec<Option<(Val, Vec<usize>)>>,
    chains: HashMap<usize, Chain>,
    /// Per node: the node its work is emitted with.
    sink: Vec<usize>,
    angles: HashMap<(Val, usize, u32), Val>,
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

/// Whether a generated elementwise kernel can run this whole op.
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
        out: Selected::default(),
        vals: vec![None; g.nodes.len()],
        chains: HashMap::new(),
        sink: sinks(g, &needed, &outputs),
        angles: HashMap::new(),
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

    /// Adds an op writing a new value of `len` elements.
    fn emit(&mut self, kernel: Kernel, reads: Vec<Val>, len: usize, covers: &[NodeId]) -> Val {
        let write = self.value(len, Home::Temp);
        let covers = covers.to_vec();
        self.out.ops.push(Op {
            kernel,
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
                let kernel = self.matmul(self.len(x) / k, n, k);
                self.emit(kernel, vec![x, w], len, &covers)
            }
            Node::Row(r) if maps(&n.op) => return self.map(i, &r.0),
            Node::Row(r) => self.row(i, &r.0)?,
            Node::Attend(a) => self.attend(i, a)?,
            Node::Select(Select::Gather) => {
                let ((table, wide), (ids, _)) = (self.val(arg(0))?, self.val(arg(1))?);
                let kernel = Kernel::Gather {
                    rows: self.len(ids),
                    width: self.len(table) / wide[0],
                };
                self.emit(kernel, vec![table, ids], len, &covers)
            }
            other => return unsupported(other),
        };
        self.vals[i] = Some((val, shape));
        Ok(())
    }

    /// The cheaper matmul kernel on this device, for this shape.
    fn matmul(&self, m: usize, n: usize, k: usize) -> Kernel {
        let tiled = Kernel::Matmul { m, n, k };
        let thin = Kernel::Thin { m, n, k };
        let seconds = |kernel: &Kernel| self.device.seconds(kernel, m * n);
        if m <= THIN_ROWS && seconds(&thin) < seconds(&tiled) {
            thin
        } else {
            tiled
        }
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
        let val = self.emit(Kernel::Map(chain.stages), reads, self.len(x), &chain.covers);
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
        for s in stages {
            let operand = (s.operands() == 1).then(|| rest.next().expect("verified"));
            match (s, operand) {
                (Stage::RmsNorm { eps }, Some(w)) => {
                    let kernel = Kernel::RmsNorm {
                        rows: len / width,
                        width,
                        eps: *eps,
                    };
                    let w = self.val(w)?.0;
                    cur = self.emit(kernel, vec![cur, w], len, &covers);
                }
                (Stage::Softmax, _) => {
                    let rows = len / width;
                    cur = self.emit(Kernel::Softmax { rows, width }, vec![cur], len, &covers);
                }
                (Stage::Rope { theta, pairs }, Some(at)) if *pairs == Pairs::Halves => {
                    let at = self.val(at)?.0;
                    let angles = self.turns(i, at, width, *theta);
                    let kernel = Kernel::Rope {
                        positions: self.len(at),
                        heads: len / (self.len(at) * width),
                        width,
                    };
                    cur = self.emit(kernel, vec![cur, angles], len, &covers);
                }
                // One rank holds the whole sum already.
                (Stage::AllReduce, _) => {}
                (other, _) => return unsupported(other),
            }
        }
        Ok(cur)
    }

    /// One table of angles serves every rotation that matches.
    fn turns(&mut self, i: usize, at: Val, width: usize, theta: f32) -> Val {
        let key = (at, width, theta.to_bits());
        if let Some(angles) = self.angles.get(&key) {
            return *angles;
        }
        let kernel = Kernel::Angles { width, theta };
        let len = self.len(at) * width;
        let angles = self.emit(kernel, vec![at], len, &[NodeId(i as u32)]);
        self.angles.insert(key, angles);
        angles
    }

    /// Key by key, or as two matmuls: whichever costs less.
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
        let covers = [NodeId(i as u32)];
        let side = kv_heads * width;
        let scale = Kernel::Map(vec![MapStage::Scale(a.scale)]);
        let join = Kernel::Join {
            left: side,
            right: side,
        };
        let causal = Kernel::Causal {
            queries,
            keys,
            scale: a.scale,
        };
        let (out, joined) = (self.len(q), 2 * self.len(k));
        let pairs = seqs * heads * queries * keys;
        let cost = |ops: [(&Kernel, usize); 3]| -> f64 {
            let each = ops.map(|(kernel, len)| self.device.seconds(kernel, len));
            each.iter().sum()
        };
        let stream = cost([
            (&scale, out),
            (&join, joined),
            (&Kernel::Attend(sizes), out),
        ]);
        let (first, last) = (Kernel::Scores(sizes), Kernel::Values(sizes));
        let dense = cost([(&first, pairs), (&causal, pairs), (&last, out)]);
        if pairs <= DENSE_SCORES && dense < stream {
            let scores = self.emit(first, vec![q, k], pairs, &covers);
            let probs = self.emit(causal, vec![scores], pairs, &covers);
            return Ok(self.emit(last, vec![probs, v], out, &covers));
        }
        // The streaming kernel wants scaled queries, and keys beside values.
        let scaled = self.emit(scale, vec![q], out, &covers);
        let both = self.emit(join, vec![k, v], joined, &covers);
        Ok(self.emit(Kernel::Attend(sizes), vec![scaled, both], out, &covers))
    }
}
