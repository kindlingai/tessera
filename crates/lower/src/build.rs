//! Lowers one graph entry, at fixed sizes, to a plan.

use crate::Error;
use crate::plan::{Kernel, MapStage, Output, Plan, Slot, Step};
use graph::{Attend, DimName, Graph, Mask, NodeId, Op, Pairs, Select, Shape, Stage, View};
use std::collections::HashMap;

/// A last use that never comes: weights, inputs and outputs.
const FOREVER: usize = usize::MAX;

#[derive(Clone)]
struct Value {
    slot: Slot,
    shape: Vec<usize>,
}

/// Elementwise stages waiting for the node they fuse into.
struct Chain {
    source: usize,
    stages: Vec<MapStage>,
    operands: Vec<usize>,
    covers: Vec<NodeId>,
}

struct Builder<'a> {
    g: &'a Graph,
    sizes: &'a [(DimName, u64)],
    plan: Plan,
    values: Vec<Option<Value>>,
    chains: HashMap<usize, Chain>,
    /// Per node: the last node whose launch reads it.
    last_use: Vec<usize>,
    /// Per node: the node its work is emitted with.
    sink: Vec<usize>,
    /// Per buffer: the last node that reads it.
    ends: Vec<usize>,
    free: Vec<u32>,
    angles: HashMap<(u32, usize, u32), Slot>,
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
fn maps(op: &Op) -> bool {
    matches!(op, Op::Row(r) if r.0.iter().all(|s| map_stage(s).is_some()))
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

/// Lowers `entry`, with every bounded dimension sized by `sizes`.
pub fn lower(g: &Graph, entry: usize, sizes: &[(DimName, u64)]) -> Result<Plan, Error> {
    let Some(e) = g.entries.get(entry) else {
        return unsupported(format_args!("entry {entry}"));
    };
    let outputs: Vec<usize> = e.outputs.iter().map(|o| o.node.0 as usize).collect();
    let needed = needed(g, &outputs);
    let sink = sinks(g, &needed, &outputs);
    let mut last_use = vec![0; g.nodes.len()];
    for (j, n) in g.nodes.iter().enumerate().filter(|(j, _)| needed[*j]) {
        for a in &n.args {
            let a = a.0 as usize;
            last_use[a] = last_use[a].max(sink[j]);
        }
    }
    for &o in &outputs {
        last_use[o] = FOREVER;
    }
    let mut b = Builder {
        g,
        sizes,
        plan: Plan::default(),
        values: vec![None; g.nodes.len()],
        chains: HashMap::new(),
        last_use,
        sink,
        ends: Vec::new(),
        free: Vec::new(),
        angles: HashMap::new(),
    };
    for port in &e.inputs {
        let len = b.shape(&port.ty.shape)?.iter().product();
        let slot = b.alloc(len, FOREVER);
        b.plan.inputs.push(slot);
    }
    for i in (0..g.nodes.len()).filter(|i| needed[*i]) {
        b.node(i, entry)?;
        b.release(i);
    }
    for o in outputs {
        let Value { slot, shape } = b.value(o)?;
        b.plan.outputs.push(Output { slot, shape });
    }
    Ok(b.plan)
}

impl Builder<'_> {
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

    fn value(&self, node: usize) -> Result<Value, Error> {
        let missing = || Error::Unsupported(format!("node {node} as an argument"));
        self.values[node].clone().ok_or_else(missing)
    }

    /// A buffer of `len` elements, free again after node `end`.
    fn alloc(&mut self, len: usize, end: usize) -> Slot {
        let fits = |b: &u32| self.plan.buffers[*b as usize] == len;
        let buffer = match self.free.iter().position(fits) {
            Some(at) => self.free.swap_remove(at),
            None => {
                self.plan.buffers.push(len);
                self.ends.push(end);
                self.ends.len() as u32 - 1
            }
        };
        self.ends[buffer as usize] = end;
        Slot { buffer, len }
    }

    /// Frees the buffers nothing reads after node `i`.
    fn release(&mut self, i: usize) {
        for (b, end) in self.ends.iter().enumerate() {
            if *end == i && !self.free.contains(&(b as u32)) {
                self.free.push(b as u32);
            }
        }
    }

    fn step(&mut self, kernel: Kernel, reads: Vec<Slot>, write: Slot, covers: &[NodeId]) {
        let covers = covers.to_vec();
        self.plan.steps.push(Step {
            kernel,
            reads,
            write,
            covers,
        });
    }

    /// Emits a step that writes node `i`'s own fresh buffer.
    fn emit(&mut self, i: usize, kernel: Kernel, reads: Vec<Slot>, shape: Vec<usize>) {
        let slot = self.alloc(shape.iter().product(), self.last_use[i]);
        self.step(kernel, reads, slot, &[NodeId(i as u32)]);
        self.values[i] = Some(Value { slot, shape });
    }

    fn node(&mut self, i: usize, entry: usize) -> Result<(), Error> {
        let n = &self.g.nodes[i];
        let shape = self.shape(&n.ty.shape)?;
        let arg = |k: usize| n.args[k].0 as usize;
        match &n.op {
            Op::Input { entry: e, port } if *e as usize == entry => {
                let slot = self.plan.inputs[*port as usize];
                self.values[i] = Some(Value { slot, shape });
            }
            Op::Weight(w) => {
                let slot = self.alloc(shape.iter().product(), FOREVER);
                self.plan.weights.push((*w, slot));
                self.values[i] = Some(Value { slot, shape });
            }
            // A reshape is the same buffer under a new shape.
            Op::View(View::Reshape(_)) => {
                let slot = self.value(arg(0))?.slot;
                let end = &mut self.ends[slot.buffer as usize];
                *end = (*end).max(self.last_use[i]);
                self.values[i] = Some(Value { slot, shape });
            }
            Op::Matmul => {
                let (x, w) = (self.value(arg(0))?, self.value(arg(1))?);
                let [n, k] = w.shape[..] else {
                    return unsupported("a batched matmul");
                };
                let m = x.slot.len / k;
                self.emit(i, Kernel::Matmul { m, n, k }, vec![x.slot, w.slot], shape);
            }
            Op::Row(r) if maps(&n.op) => self.map(i, &r.0)?,
            Op::Row(r) => self.row(i, &r.0)?,
            Op::Attend(a) => self.attend(i, a)?,
            Op::Select(Select::Gather) => {
                let (table, ids) = (self.value(arg(0))?, self.value(arg(1))?);
                let width = table.slot.len / table.shape[0];
                let kernel = Kernel::Gather {
                    rows: ids.slot.len,
                    width,
                };
                self.emit(i, kernel, vec![table.slot, ids.slot], shape);
            }
            other => return unsupported(other),
        }
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
        let x = self.value(chain.source)?;
        let mut reads = vec![x.slot];
        for o in &chain.operands {
            let operand = self.value(*o)?.slot;
            if operand.len != x.slot.len {
                return unsupported("a broadcast operand");
            }
            reads.push(operand);
        }
        let slot = self.alloc(x.slot.len, self.last_use[i]);
        self.step(Kernel::Map(chain.stages), reads, slot, &chain.covers);
        let shape = x.shape;
        self.values[i] = Some(Value { slot, shape });
        Ok(())
    }

    /// A row holding a norm or rotation, stage by stage.
    fn row(&mut self, i: usize, stages: &[Stage]) -> Result<(), Error> {
        let n = &self.g.nodes[i];
        let covers = [NodeId(i as u32)];
        let mut cur = self.value(n.args[0].0 as usize)?;
        let mut rest = n.args[1..].iter().map(|a| a.0 as usize);
        for (at, s) in stages.iter().enumerate() {
            // Only the last stage's value outlives this node.
            let end = if at + 1 == stages.len() {
                self.last_use[i]
            } else {
                i
            };
            let operand = (s.operands() == 1).then(|| rest.next().expect("verified"));
            match (s, operand) {
                (Stage::RmsNorm { eps }, Some(w)) => {
                    let width = *cur.shape.last().expect("verified rank");
                    let kernel = Kernel::RmsNorm {
                        rows: cur.slot.len / width,
                        width,
                        eps: *eps,
                    };
                    let slot = self.alloc(cur.slot.len, end);
                    self.step(kernel, vec![cur.slot, self.value(w)?.slot], slot, &covers);
                    cur.slot = slot;
                }
                (Stage::Rope { theta, pairs }, Some(pos)) if *pairs == Pairs::Halves => {
                    cur.slot = self.rope(i, &cur, pos, *theta, end)?;
                }
                // One rank holds the whole sum already.
                (Stage::AllReduce, _) => {}
                (other, _) => return unsupported(other),
            }
        }
        let end = &mut self.ends[cur.slot.buffer as usize];
        *end = (*end).max(self.last_use[i]);
        self.values[i] = Some(cur);
        Ok(())
    }

    /// Rotation is in place, so a shared input is copied.
    fn rope(
        &mut self,
        i: usize,
        x: &Value,
        pos: usize,
        theta: f32,
        end: usize,
    ) -> Result<Slot, Error> {
        let covers = [NodeId(i as u32)];
        let pos = self.value(pos)?.slot;
        let width = *x.shape.last().expect("verified rank");
        let key = (pos.buffer, width, theta.to_bits());
        let angles = match self.angles.get(&key) {
            Some(slot) => *slot,
            None => {
                let slot = self.alloc(pos.len * width, FOREVER);
                self.step(Kernel::Angles { width, theta }, vec![pos], slot, &covers);
                self.angles.insert(key, slot);
                slot
            }
        };
        let mut slot = x.slot;
        if self.ends[slot.buffer as usize] == i {
            self.ends[slot.buffer as usize] = end;
        } else {
            slot = self.alloc(x.slot.len, end);
            self.step(Kernel::Map(Vec::new()), vec![x.slot], slot, &covers);
        }
        let kernel = Kernel::Rope {
            positions: pos.len,
            heads: x.slot.len / (pos.len * width),
            width,
        };
        self.step(kernel, vec![slot, angles], slot, &covers);
        Ok(slot)
    }

    /// The kernel wants scaled queries, and keys beside values.
    fn attend(&mut self, i: usize, a: &Attend) -> Result<(), Error> {
        let n = &self.g.nodes[i];
        if a.picked || a.mask != Mask::Causal {
            return unsupported(a);
        }
        let arg = |k: usize| self.value(n.args[k].0 as usize);
        let (q, k, v) = (arg(0)?, arg(1)?, arg(2)?);
        let (&[seqs, queries, heads, width], &[_, keys, kv_heads, _]) =
            (&q.shape[..], &k.shape[..])
        else {
            return unsupported("attention below rank 4");
        };
        let covers = [NodeId(i as u32)];
        let scaled = self.alloc(q.slot.len, i);
        let scale = Kernel::Map(vec![MapStage::Scale(a.scale)]);
        self.step(scale, vec![q.slot], scaled, &covers);
        let joined = self.alloc(2 * k.slot.len, i);
        let side = kv_heads * width;
        let join = Kernel::Join {
            left: side,
            right: side,
        };
        self.step(join, vec![k.slot, v.slot], joined, &covers);
        let kernel = Kernel::Attend {
            seqs,
            queries,
            keys,
            heads,
            kv_heads,
            width,
        };
        self.emit(i, kernel, vec![scaled, joined], q.shape);
        Ok(())
    }
}
