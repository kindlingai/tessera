//! Picks kernels for an entry: covers, then ops over values.

mod cover;
mod emit;

use cover::{Cover, EDGE, reads};
use emit::Emit;

use crate::Error;
use crate::cost::Device;
use crate::kernel::KernelRef;
use crate::rule::{Read, rules};
use crate::tree::{T, Term, Tree, unsupported};
use graph::{DType, DimName, Graph, NodeId};
use std::collections::BTreeSet;

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
            (Term::Append(index), _) => {
                let args = [n.args[0], n.args[1], n.args[2]].map(val);
                e.append(&tree, t, *index, args)?
            }
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
