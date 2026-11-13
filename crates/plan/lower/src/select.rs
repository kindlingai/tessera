//! Picks kernels by covering each tree, as ops over values.

use crate::Error;
use crate::cost::Device;
use crate::form::Stored;
use crate::kernel::{Kernel, KernelRef};
use crate::kernels::{Append, Move, Widen};
use crate::rule::{Draft, Read, Rule, rules};
use crate::tree::{T, Term, Tree, unsupported};
use graph::{DType, DimExpr, DimName, Graph, NodeId};
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Value {
    /// Elements it holds.
    pub len: usize,
    pub home: Home,
    /// The zone it lies in.
    pub zone: u32,
    pub dtype: DType,
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
    pub zone: u32,
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

/// The cheapest cover found at one node, in one zone.
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

/// Leaves and outputs lie where the runner feeds and reads.
const EDGE: u32 = 0;

struct Cover<'a> {
    tree: &'a Tree<'a>,
    device: &'a Device,
    rules: Vec<&'static Rule>,
    /// The rule the device favors, if it names one.
    favorite: Option<&'static Rule>,
    /// Per node and zone: the choice there, if any.
    best: Vec<Vec<Option<Option<Choice>>>>,
    /// Per root: the zones its readers run in.
    wanted: Vec<Vec<u32>>,
}

impl Cover<'_> {
    fn zones(&self) -> std::ops::Range<u32> {
        0..self.device.zones()
    }

    /// Seconds to have `t` in `zone`, within one tree.
    fn need(&mut self, t: T, zone: u32) -> Option<f64> {
        if self.tree.leaf(t) {
            return self.device.transfer(EDGE, zone, self.tree.bytes(t));
        }
        // Another tree's root pays its own moves once placed.
        match self.tree.root(t) {
            true => Some(0.0),
            false => self.place(t, zone).map(|(_, cost)| cost),
        }
    }

    /// Where to make `t` cheapest for a reader in `zone`.
    fn place(&mut self, t: T, zone: u32) -> Option<(u32, f64)> {
        let len = self.tree.bytes(t);
        let mut best: Option<(u32, f64)> = None;
        for z in self.zones() {
            let (Some(made), Some(moved)) = (self.solve(t, z), self.device.transfer(z, zone, len))
            else {
                continue;
            };
            if best.is_none_or(|(_, b)| made + moved < b) {
                best = Some((z, made + moved));
            }
        }
        best
    }

    /// A root's cheapest zone, its readers' moves included.
    fn settle(&mut self, t: T) -> Option<u32> {
        let (len, wanted) = (self.tree.bytes(t), self.wanted[t.0 as usize].clone());
        let mut best: Option<(u32, f64)> = None;
        for z in self.zones() {
            let moves = wanted.iter().map(|w| self.device.transfer(z, *w, len));
            let (Some(made), Some(moved)) = (self.solve(t, z), moves.sum::<Option<f64>>()) else {
                continue;
            };
            if best.is_none_or(|(_, b)| made + moved < b) {
                best = Some((z, made + moved));
            }
        }
        best.map(|(z, _)| z)
    }

    /// Remembers the cheapest cover of `t`'s subtree in `zone`.
    fn solve(&mut self, t: T, zone: u32) -> Option<f64> {
        let (t, z) = (t.0 as usize, zone as usize);
        if self.best[t][z].is_none() {
            let choice = self.choose(T(t as u32), zone);
            self.best[t][z] = Some(choice);
        }
        let solved = self.best[t][z].as_ref()?;
        solved.as_ref().map(|c| c.cost)
    }

    /// Asks every rule, and keeps the cheapest proposal.
    fn choose(&mut self, t: T, zone: u32) -> Option<Choice> {
        let n = self.tree.node(t);
        // A view or kept rows only pass their argument on.
        if matches!(n.term, Term::View | Term::Append(_)) {
            if matches!(n.term, Term::Append(_)) && zone != EDGE {
                return None;
            }
            let cost = self.need(n.args[0], zone)?;
            return Some(Choice {
                rule: "append",
                drafts: Vec::new(),
                cost,
                offered: Vec::new(),
            });
        }
        let kind = self.device.cluster.kind(zone as usize);
        let (mut best, mut offered): (Option<Choice>, _) = (None, Vec::new());
        for rule in self.rules.clone() {
            let Some(drafts) = (rule.propose)(self.tree, t) else {
                continue;
            };
            if drafts.iter().any(|d| d.kernel.runs_on() != kind) {
                continue;
            }
            offered.push(rule.name);
            if self.swallows(rule, t, &drafts) {
                continue;
            }
            let seconds = |d: &Draft| self.device.seconds_in(zone, &*d.kernel, d.len);
            let mut cost: f64 = drafts.iter().map(seconds).sum();
            cost += self.widening(zone, &drafts);
            let below: Option<f64> = reads(&drafts).into_iter().map(|r| self.need(r, zone)).sum();
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

    /// Seconds to widen reads the drafts cannot take as stored.
    fn widening(&self, zone: u32, drafts: &[Draft]) -> f64 {
        let mut seconds = 0.0;
        for d in drafts {
            for (i, r) in d.reads.iter().enumerate() {
                let Read::Node(t) = r else { continue };
                let form = self.tree.node(*t).form;
                if form.narrow() && !d.kernel.accepts(i, form) {
                    let len = self.tree.elements(*t);
                    let widen = Widen { len, form };
                    seconds += self.device.seconds_in(zone, &widen, len);
                }
            }
        }
        seconds
    }

    /// Whether a cover hides a node the favored rule takes.
    fn swallows(&self, rule: &Rule, t: T, drafts: &[Draft]) -> bool {
        let Some(favorite) = self.favorite.filter(|f| f.name != rule.name) else {
            return false;
        };
        let inside = self.tree.inside(t, &reads(drafts));
        let below = inside.into_iter().filter(|n| *n != t);
        below
            .into_iter()
            .any(|n| (favorite.propose)(self.tree, n).is_some())
    }

    /// The reads of `t`'s cover in `zone`, each placed.
    fn below(&mut self, t: T, zone: u32) -> Vec<(T, u32)> {
        let n = self.tree.node(t);
        let reads = match &self.best[t.0 as usize][zone as usize] {
            Some(Some(c)) if c.drafts.is_empty() => vec![n.args[0]],
            Some(Some(c)) => reads(&c.drafts),
            _ => Vec::new(),
        };
        let mut placed = Vec::with_capacity(reads.len());
        for r in reads {
            let z = match self.tree.leaf(r) || self.tree.root(r) {
                true => zone,
                false => self.place(r, zone).expect("solved above").0,
            };
            placed.push((r, z));
        }
        placed
    }
}

/// The ops written, and the values they use.
#[derive(Default)]
struct Emit {
    out: Selected,
    /// Results that later equal launches may reuse.
    shared: HashMap<(String, Vec<Val>), Val>,
    /// Each value's copy in another zone, once made.
    moved: HashMap<(Val, u32), Val>,
    /// Each narrow value as floats, once made.
    widened: HashMap<Val, Val>,
}

impl Emit {
    fn value(&mut self, len: usize, home: Home, zone: u32, dtype: DType) -> Val {
        self.out.values.push(Value {
            len,
            home,
            zone,
            dtype,
        });
        Val(self.out.values.len() as u32 - 1)
    }

    fn len(&self, v: Val) -> usize {
        self.out.values[v.0 as usize].len
    }

    /// The value in `zone`, moved there if it lies elsewhere.
    fn bring(&mut self, v: Val, zone: u32) -> Val {
        if self.out.values[v.0 as usize].zone == zone {
            return v;
        }
        if let Some(copy) = self.moved.get(&(v, zone)) {
            return *copy;
        }
        let (len, dtype) = (self.len(v), self.out.values[v.0 as usize].dtype);
        let write = self.value(len, Home::Temp, zone, dtype);
        self.out.ops.push(Op {
            kernel: KernelRef::new(Move { len }),
            rule: "move",
            reads: vec![v],
            write,
            covers: Vec::new(),
            zone,
        });
        self.moved.insert((v, zone), write);
        write
    }

    /// The value as floats, unless `k` reads it as stored.
    fn widen(&mut self, v: Val, input: usize, k: &dyn Kernel) -> Val {
        let Value {
            len, zone, dtype, ..
        } = self.out.values[v.0 as usize];
        if !dtype.narrow() || k.accepts(input, dtype) {
            return v;
        }
        if let Some(wide) = self.widened.get(&v) {
            return *wide;
        }
        let write = self.value(len, Home::Temp, zone, DType::F32);
        self.out.ops.push(Op {
            kernel: KernelRef::new(Widen { len, form: dtype }),
            rule: "widen",
            reads: vec![v],
            write,
            covers: Vec::new(),
            zone,
        });
        self.widened.insert(v, write);
        write
    }

    /// Adds an op writing a new value, or shares one.
    fn op(
        &mut self,
        d: Draft,
        rule: &'static str,
        reads: Vec<Val>,
        covers: &[NodeId],
        zone: u32,
        dtype: DType,
    ) -> Val {
        let mut taken = Vec::with_capacity(reads.len());
        for (i, v) in reads.into_iter().enumerate() {
            let v = self.bring(v, zone);
            taken.push(self.widen(v, i, &*d.kernel));
        }
        let reads = taken;
        // Printing kernels is slow, so shared ones alone are keyed.
        let key = d
            .kernel
            .shared()
            .then(|| (format!("{:?}", d.kernel), reads.clone()));
        if let Some(same) = key.as_ref().and_then(|k| self.shared.get(k)) {
            return *same;
        }
        let write = self.value(d.len, Home::Temp, zone, dtype);
        if let Some(key) = key {
            self.shared.insert(key, write);
        }
        self.out.ops.push(Op {
            kernel: d.kernel,
            rule,
            reads,
            write,
            covers: covers.to_vec(),
            zone,
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
        let write = self.value(len, Home::Kept { index, room }, EDGE, DType::F32);
        let rows = self.bring(rows, EDGE);
        // The new rows go after those kept so far.
        let width = len / all[1];
        let skip = n.extents[1].times(width).less(self.len(rows));
        let len = self.len(rows);
        self.out.ops.push(Op {
            kernel: KernelRef::new(Append { skip, len }),
            rule: "append",
            reads: vec![rows],
            write,
            covers: vec![n.from],
            zone: EDGE,
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
    select_upto(g, entry, sizes, &[], device)
}

/// The same, with `upto` dimensions read at launch.
pub fn select_upto(
    g: &Graph,
    entry: usize,
    sizes: &[(DimName, u64)],
    upto: &[DimName],
    device: &Device,
) -> Result<Selected, Error> {
    let tree = Tree::upto(g, entry, sizes, upto, &device.pins)?;
    let nodes = (0..tree.len() as u32).map(T);
    let rules = rules();
    let mut cover = Cover {
        tree: &tree,
        device,
        favorite: rules.iter().copied().find(|r| device.favors(r.name)),
        rules,
        best: (0..tree.len())
            .map(|_| (0..device.zones()).map(|_| None).collect())
            .collect(),
        wanted: vec![Vec::new(); tree.len()],
    };
    for (t, _) in &tree.outputs {
        cover.wanted[t.0 as usize].push(EDGE);
    }
    // Roots get buffers. So does whatever a chosen cover reads.
    let mut own: Vec<Option<u32>> = vec![None; tree.len()];
    for t in nodes.clone().rev().filter(|t| !tree.leaf(*t)) {
        let zone = match (tree.root(t), own[t.0 as usize]) {
            (true, _) => cover.settle(t),
            (false, placed) => placed,
        };
        let Some(zone) = zone else {
            if tree.root(t) {
                return unsupported(&tree.node(t).term);
            }
            continue;
        };
        own[t.0 as usize] = Some(zone);
        for (r, z) in cover.below(t, zone) {
            match tree.root(r) {
                true => cover.wanted[r.0 as usize].push(zone),
                false => own[r.0 as usize] = Some(z),
            }
        }
    }
    let mut e = Emit::default();
    for (port, &(len, dtype)) in tree.ports.iter().enumerate() {
        let val = e.value(len, Home::Input(port as u32), EDGE, dtype);
        e.out.inputs.push(val);
    }
    // In traced order, so equal graphs give equal plans.
    let mut vals: Vec<Option<Val>> = vec![None; tree.len()];
    for t in nodes {
        let n = tree.node(t);
        let val = |a: T| vals[a.0 as usize].expect("arguments come first");
        let made = match (&n.term, own[t.0 as usize]) {
            (Term::Leaf(Home::Input(port)), _) => e.out.inputs[*port as usize],
            (Term::Leaf(home), _) => e.value(tree.elements(t), *home, EDGE, n.form),
            (_, None) => continue,
            (Term::View, _) => val(n.args[0]),
            (Term::Append(index), _) => e.append(&tree, t, *index, val(n.args[0]))?,
            (_, Some(zone)) => {
                let choice = cover.best[t.0 as usize][zone as usize].take().flatten();
                let choice = choice.expect("solved above");
                if choice.offered.len() > 1 {
                    e.out.contested.extend(&choice.offered);
                }
                let covers = tree.covered(t, &reads(&choice.drafts));
                let mut made: Vec<Val> = Vec::with_capacity(choice.drafts.len());
                let last = choice.drafts.len() - 1;
                for (i, d) in choice.drafts.into_iter().enumerate() {
                    let read = |r: &Read| match r {
                        Read::Node(a) => val(*a),
                        Read::Draft(at) => made[*at],
                    };
                    let reads: Vec<Val> = d.reads.iter().map(read).collect();
                    // The last launch writes the node's own type.
                    let dtype = if i == last { n.form } else { DType::F32 };
                    made.push(e.op(d, choice.rule, reads, &covers, zone, dtype));
                }
                *made.last().expect("a rule proposes at least one launch")
            }
        };
        vals[t.0 as usize] = Some(made);
    }
    for (t, shape) in &tree.outputs {
        let val = vals[t.0 as usize].expect("outputs have values");
        let val = e.bring(val, EDGE);
        e.out.outputs.push((val, shape.clone()));
    }
    Ok(e.out)
}
