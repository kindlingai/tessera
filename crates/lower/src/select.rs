//! Picks kernels by covering each tree, as ops over values.

use crate::Error;
use crate::cost::Device;
use crate::kernel::KernelRef;
use crate::kernels::Append;
use crate::rule::{Draft, Read, Rule, rules};
use crate::tree::{T, Term, Tree, unsupported};
use graph::{DimExpr, DimName, Graph, NodeId};
use std::collections::{BTreeSet, HashMap};

/// A value some op writes, or the plan is given.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Val(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Home {
    Weight(u32),
    Input(u32),
    /// Kept rows; `room` elements hold the longest context.
    Kept {
        index: u32,
        room: usize,
    },
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

/// The cheapest cover found at one node.
struct Choice {
    rule: &'static str,
    drafts: Vec<Draft>,
    /// Seconds for the drafts and for all they read.
    cost: f64,
    /// Every rule that proposed here.
    offered: Vec<&'static str>,
}

/// A cover's reads of tree nodes.
fn reads(drafts: &[Draft]) -> Vec<T> {
    let node = |r: &Read| match r {
        Read::Node(t) => Some(*t),
        Read::Draft(_) => None,
    };
    drafts
        .iter()
        .flat_map(|d| d.reads.iter().filter_map(node))
        .collect()
}

struct Cover<'a> {
    tree: &'a Tree<'a>,
    device: &'a Device,
    rules: Vec<&'static Rule>,
    /// Per solved node: its choice, if a rule covers it.
    best: Vec<Option<Option<Choice>>>,
}

impl Cover<'_> {
    /// Seconds to have `t` in a buffer, within one tree.
    fn need(&mut self, t: T) -> Option<f64> {
        // Another tree's root is paid for there.
        match self.tree.root(t) {
            true => Some(0.0),
            false => self.solve(t),
        }
    }

    /// Remembers the cheapest cover of `t`'s subtree.
    fn solve(&mut self, t: T) -> Option<f64> {
        if self.best[t.0 as usize].is_none() {
            let choice = self.choose(t);
            self.best[t.0 as usize] = Some(choice);
        }
        let solved = self.best[t.0 as usize].as_ref()?;
        solved.as_ref().map(|c| c.cost)
    }

    /// Asks every rule, and keeps the cheapest proposal.
    fn choose(&mut self, t: T) -> Option<Choice> {
        let n = self.tree.node(t);
        // A view or kept rows only pass their argument on.
        if matches!(n.term, Term::View | Term::Append(_)) {
            let cost = self.need(n.args[0])?;
            return Some(Choice {
                rule: "append",
                drafts: Vec::new(),
                cost,
                offered: Vec::new(),
            });
        }
        let (mut best, mut offered): (Option<Choice>, _) = (None, Vec::new());
        for rule in self.rules.clone() {
            let Some(drafts) = (rule.propose)(self.tree, t) else {
                continue;
            };
            offered.push(rule.name);
            let seconds = |d: &Draft| self.device.seconds(&*d.kernel, d.len);
            let mut cost: f64 = drafts.iter().map(seconds).sum();
            let below: Option<f64> = reads(&drafts).into_iter().map(|r| self.need(r)).sum();
            let Some(below) = below else {
                continue;
            };
            cost += below;
            // A favored rule wins here. Readers see its true cost.
            let favored = self.device.favors(rule.name);
            let beaten = |b: &Choice| match self.device.favors(b.rule) {
                true => false,
                false => favored || cost < b.cost,
            };
            if best.as_ref().is_none_or(beaten) {
                best = Some(Choice {
                    rule: rule.name,
                    drafts,
                    cost,
                    offered: Vec::new(),
                });
            }
        }
        best.map(|b| Choice { offered, ..b })
    }

    /// The reads of the chosen cover at `t`.
    fn below(&self, t: T) -> Vec<T> {
        let n = self.tree.node(t);
        match &self.best[t.0 as usize] {
            Some(Some(c)) if c.drafts.is_empty() => vec![n.args[0]],
            Some(Some(c)) => reads(&c.drafts),
            _ => Vec::new(),
        }
    }
}

/// The ops written, and the values they use.
#[derive(Default)]
struct Emit {
    out: Selected,
    /// Results that later equal launches may reuse.
    shared: HashMap<(String, Vec<Val>), Val>,
}

impl Emit {
    fn value(&mut self, len: usize, home: Home) -> Val {
        self.out.values.push(Value { len, home });
        Val(self.out.values.len() as u32 - 1)
    }

    fn len(&self, v: Val) -> usize {
        self.out.values[v.0 as usize].len
    }

    /// Adds an op writing a new value, or shares one.
    fn op(&mut self, d: Draft, rule: &'static str, reads: Vec<Val>, covers: &[NodeId]) -> Val {
        let key = (format!("{:?}", d.kernel), reads.clone());
        if let Some(same) = self.shared.get(&key).filter(|_| d.kernel.shared()) {
            return *same;
        }
        let write = self.value(d.len, Home::Temp);
        if d.kernel.shared() {
            self.shared.insert(key, write);
        }
        self.out.ops.push(Op {
            kernel: d.kernel,
            rule,
            reads,
            write,
            covers: covers.to_vec(),
        });
        write
    }

    /// Copies new rows in after those already kept.
    fn append(&mut self, tree: &Tree, t: T, index: u32, rows: Val) -> Result<Val, Error> {
        let (n, all) = (tree.node(t), &tree.node(t).shape);
        let DimExpr::Dim(axis) = &tree.g.kept[index as usize].shape.0[1] else {
            return unsupported("kept rows along a computed axis");
        };
        // Sequences would move as their rows grow.
        if all[0] != 1 {
            return unsupported("kept rows for several sequences");
        }
        let most = tree.g.dims.iter().find(|d| d.name == *axis);
        let most = most.map_or(0, |d| d.size.max() as usize);
        let len: usize = all.iter().product();
        let room = most * (len / all[1]);
        let write = self.value(len, Home::Kept { index, room });
        let (skip, len) = (len - self.len(rows), self.len(rows));
        self.out.ops.push(Op {
            kernel: KernelRef::new(Append { skip, len }),
            rule: "append",
            reads: vec![rows],
            write,
            covers: vec![n.from],
        });
        Ok(write)
    }
}

/// Chooses kernels for `entry`, sized by `sizes`, costed by `device`.
pub fn select(
    g: &Graph,
    entry: usize,
    sizes: &[(DimName, u64)],
    device: &Device,
) -> Result<Selected, Error> {
    let tree = Tree::new(g, entry, sizes)?;
    let nodes = (0..tree.len() as u32).map(T);
    let mut cover = Cover {
        tree: &tree,
        device,
        rules: rules(),
        best: (0..tree.len()).map(|_| None).collect(),
    };
    // Roots get buffers. So does whatever a chosen cover reads.
    let mut own = vec![false; tree.len()];
    for t in nodes.clone().rev().filter(|t| !tree.leaf(*t)) {
        if !(tree.root(t) || own[t.0 as usize]) {
            continue;
        }
        own[t.0 as usize] = true;
        if cover.solve(t).is_none() {
            return unsupported(&tree.node(t).term);
        }
        for r in cover.below(t) {
            own[r.0 as usize] = true;
        }
    }
    let mut e = Emit::default();
    for len in &tree.ports {
        let port = e.out.inputs.len() as u32;
        let val = e.value(*len, Home::Input(port));
        e.out.inputs.push(val);
    }
    // In traced order, so equal graphs give equal plans.
    let mut vals: Vec<Option<Val>> = vec![None; tree.len()];
    for t in nodes {
        let n = tree.node(t);
        let val = |a: T| vals[a.0 as usize].expect("arguments come first");
        let made = match &n.term {
            Term::Leaf(Home::Input(port)) => e.out.inputs[*port as usize],
            Term::Leaf(home) => e.value(tree.elements(t), *home),
            _ if !own[t.0 as usize] => continue,
            Term::View => val(n.args[0]),
            Term::Append(index) => e.append(&tree, t, *index, val(n.args[0]))?,
            _ => {
                let choice = cover.best[t.0 as usize].take().flatten();
                let choice = choice.expect("solved above");
                if choice.offered.len() > 1 {
                    e.out.contested.extend(&choice.offered);
                }
                let covers = tree.covered(t, &reads(&choice.drafts));
                let mut made: Vec<Val> = Vec::with_capacity(choice.drafts.len());
                for d in choice.drafts {
                    let read = |r: &Read| match r {
                        Read::Node(a) => val(*a),
                        Read::Draft(at) => made[*at],
                    };
                    let reads: Vec<Val> = d.reads.iter().map(read).collect();
                    made.push(e.op(d, choice.rule, reads, &covers));
                }
                *made.last().expect("a rule proposes at least one launch")
            }
        };
        vals[t.0 as usize] = Some(made);
    }
    for (t, shape) in &tree.outputs {
        let val = vals[t.0 as usize].expect("outputs have values");
        e.out.outputs.push((val, shape.clone()));
    }
    Ok(e.out)
}
