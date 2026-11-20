//! Picks kernels for an entry: covers, then ops over values.

mod cover;
mod emit;

use cover::{Cover, EDGE, reads};
use emit::Emit;

use crate::cost::Device;
use crate::error::Uncovered;
use crate::form::{self, Load};
use crate::kernel::KernelRef;
use crate::plan::Share;
use crate::rule::{Read, Rule, linked};
use crate::tree::{T, Term, Tree};
use crate::{Error, Unsupported};
use sensai_graph::{DimName, Graph, NodeId, View};
use std::collections::{BTreeMap, BTreeSet};

/// A value some op writes, or the plan is given.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Val(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Home {
    Weight(u32),
    /// Part of a weight: codes, `suh` or `svh`.
    Part(u32, u8),
    Input(u32),
    /// Kept rows. `room` elements hold the longest context.
    Kept {
        index: u32,
        room: usize,
    },
    /// Written by an op. Its buffer may be reused.
    Temp,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Value {
    /// Elements it holds.
    pub len: usize,
    pub home: Home,
    /// The zone it lies in.
    pub zone: u32,
    pub form: &'static dyn Load,
    /// A split weight's part on this rank.
    pub share: Option<Share>,
}

/// One launch, before it has buffers.
#[derive(Clone, Debug, PartialEq)]
pub struct Op {
    pub kernel: KernelRef,
    /// The proposing rule.
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
    /// Page table entries, when kept rows are paged.
    pub table: usize,
    /// Rules that competed with another at some site.
    pub contested: BTreeSet<&'static str>,
    /// Each rule's runnable ways, as the kernels they launch.
    pub ways: BTreeMap<&'static str, BTreeSet<Vec<&'static str>>>,
    /// Ranks that run together, and which this is.
    pub ranks: u32,
    pub rank: u32,
}

/// Names node `t`, and the root above it.
fn uncovered(tree: &Tree, t: T, root: T) -> Unsupported {
    let node = tree.node(t);
    Unsupported::Uncovered(Box::new(Uncovered {
        term: node.term.clone(),
        shape: node.shape.clone(),
        reads: node
            .args
            .iter()
            .map(|a| tree.node(*a).shape.clone())
            .collect(),
        under: (t != root).then(|| tree.node(root).term.clone()),
    }))
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
pub(crate) fn select_upto(
    g: &Graph,
    entry: usize,
    sizes: &[(DimName, u64)],
    upto: &[DimName],
    device: &Device,
) -> Result<Selected, Error> {
    let rules = device.rules();
    linked(&rules)?;
    let (parts, rank) = (device.division(), device.rank);
    let mut tree = Tree::upto(
        g,
        entry,
        sizes,
        upto,
        &device.pins,
        device.kept,
        &parts,
        rank,
    )?;
    let whole: Vec<&Rule> = rules
        .iter()
        .copied()
        .filter(|r| r.propose.regions())
        .collect();
    tree.end_regions(&whole);
    let nodes = (0..tree.len() as u32).map(T);
    let mut cover = Cover {
        tree: &tree,
        device,
        favorite: rules.iter().copied().find(|r| device.favors(r.name)),
        rules,
        best: (0..tree.len())
            .map(|_| (0..device.zones()).map(|_| None).collect())
            .collect(),
        wanted: vec![Vec::new(); tree.len()],
        bare: Vec::new(),
    };
    for (t, _) in &tree.outputs {
        cover.wanted[t.0 as usize].push(EDGE);
    }
    // Roots get buffers. So does whatever a chosen cover reads.
    let mut own: Vec<Option<u32>> = vec![None; tree.len()];
    // A chosen cover computes the roots inside it.
    let mut absorbed = vec![false; tree.len()];
    for t in nodes.clone().rev().filter(|t| !tree.leaf(*t)) {
        if absorbed[t.0 as usize] {
            continue;
        }
        let zone = match (tree.root(t), own[t.0 as usize]) {
            (true, _) => cover.settle(t),
            (false, placed) => placed,
        };
        let Some(zone) = zone else {
            if tree.root(t) {
                return Err(uncovered(&tree, cover.uncovered(t), t).into());
            }
            continue;
        };
        own[t.0 as usize] = Some(zone);
        for r in cover.absorbed(t, zone) {
            absorbed[r.0 as usize] = true;
        }
        for (r, z) in cover.below(t, zone) {
            match tree.root(r) {
                true => cover.wanted[r.0 as usize].push(zone),
                false => own[r.0 as usize] = Some(z),
            }
        }
    }
    let mut e = Emit::default();
    for (port, &(len, dtype)) in tree.ports.iter().enumerate() {
        let val = e.value(
            len,
            Home::Input(port as u32),
            EDGE,
            sensai_codegen::form::of(dtype),
        );
        e.out.inputs.push(val);
    }
    // In traced order, so equal graphs give equal plans.
    let mut vals: Vec<Option<Val>> = vec![None; tree.len()];
    for t in nodes {
        let n = tree.node(t);
        let val = |a: T| vals[a.0 as usize].expect("arguments come first");
        let made = match (&n.term, own[t.0 as usize]) {
            (Term::Leaf(Home::Input(port)), _) => e.out.inputs[*port as usize],
            (Term::Leaf(home), _) => {
                let val = e.value(tree.elements(t), *home, EDGE, n.form);
                e.out.values[val.0 as usize].share = tree.share(t);
                val
            }
            (_, None) => continue,
            (Term::Op(sensai_graph::Op::View(View::Reshape(_))), _) => val(n.args[0]),
            (Term::Op(sensai_graph::Op::Append(index)), _) => {
                let args = [n.args[0], n.args[1], n.args[2]].map(val);
                e.append(&tree, t, *index, args)?
            }
            (_, Some(zone)) => {
                let choice = cover.best[t.0 as usize][zone as usize].take().flatten();
                let choice = choice.expect("solved above");
                if choice.offered.len() > 1 {
                    e.out.contested.extend(&choice.offered);
                }
                for (rule, way) in choice.ways {
                    e.out.ways.entry(rule).or_default().insert(way);
                }
                let covers = tree.covered(t, &reads(&choice.drafts));
                let mut made: Vec<Val> = Vec::with_capacity(choice.drafts.len());
                let mut written = Vec::new();
                let last = choice.drafts.len() - 1;
                for (i, d) in choice.drafts.into_iter().enumerate() {
                    let mut reads = Vec::with_capacity(d.reads.len());
                    for r in &d.reads {
                        reads.push(match r {
                            Read::Node(a) => val(*a),
                            Read::Draft(at) => made[*at],
                            Read::Kept(index) => e.kept(&tree, *index)?,
                        });
                    }
                    // The last launch writes the node's own type.
                    let form = match d.writes {
                        Some(w) => tree.node(w).form,
                        None if i == last => n.form,
                        None => form::F32_FORM,
                    };
                    let writes = d.writes;
                    made.push(e.op(d, choice.rule, reads, &covers, zone, form));
                    if let Some(w) = writes {
                        written.push((w, made[i]));
                    }
                }
                let mut own = *made.last().expect("a rule proposes at least one launch");
                for (w, v) in written {
                    match w == t {
                        true => own = v,
                        false => vals[w.0 as usize] = Some(v),
                    }
                }
                own
            }
        };
        vals[t.0 as usize] = Some(made);
    }
    for (t, shape) in &tree.outputs {
        let val = vals[t.0 as usize].expect("outputs have values");
        let val = e.bring(val, EDGE);
        e.out.outputs.push((val, shape.clone()));
    }
    e.out.ranks = tree.parts.ranks() as u32;
    e.out.rank = tree.rank;
    Ok(e.out)
}
