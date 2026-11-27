//! Picks kernels for an entry: covers, then ops over values.

mod cover;
mod emit;
mod report;

use cover::{Cover, EDGE, reads};
use emit::Emit;
pub use report::Report;
use report::{finish_report, scan_cover};

use crate::cost::Device;
use crate::error::Uncovered;
use crate::form::{self, Load};
use crate::kernel::KernelRef;
use crate::plan::{KeptInfo, Share, pack_shape};
use crate::rule::{Read, Rule, linked};
use crate::tree::{T, Term, Tree, Wants};
use crate::{Error, Unsupported};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use tessera_graph::{DimName, Graph, NodeId};

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
        info: KeptInfo,
    },
    /// Written by an op. Its buffer may be reused.
    Temp,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Value {
    /// Its element count.
    pub len: usize,
    pub home: Home,
    /// The zone it lies in.
    pub zone: u32,
    pub form: &'static dyn Load,
    /// A split weight's part on this rank.
    pub share: Option<Share>,
    pub dims: u8,
    pub shape: [u32; 4],
}

/// One launch, before it has buffers.
#[derive(Clone, Debug, PartialEq)]
pub struct Op {
    pub kernel: KernelRef,
    /// The proposing rule.
    pub rule: &'static str,
    pub site: Option<String>,
    pub reads: Vec<Val>,
    pub write: Val,
    /// Extra values this launch also writes.
    pub also: Vec<Val>,
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
    /// Physical pages in the kept pool, when paged.
    pub pool_pages: usize,
    /// Secondary page table entries for dense index keys.
    pub index_table: usize,
    /// Physical pages in the dense index pool.
    pub index_pool_pages: usize,
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
    select_upto(g, entry, Wants::ALL, sizes, &[], device)
}

/// The same, with `upto` dimensions read at launch.
pub(crate) fn select_upto(
    g: &Graph,
    entry: usize,
    wants: Wants,
    sizes: &[(DimName, u64)],
    upto: &[DimName],
    device: &Device,
) -> Result<Selected, Error> {
    Ok(select_with(g, entry, wants, sizes, upto, device, false)?.0)
}

/// Selects kernels and builds a fast-path diagnostic report.
pub(crate) fn select_report(
    g: &Graph,
    entry: usize,
    wants: Wants,
    sizes: &[(DimName, u64)],
    upto: &[DimName],
    device: &Device,
) -> Result<(Selected, Report), Error> {
    let (selected, report) = select_with(g, entry, wants, sizes, upto, device, true)?;
    Ok((selected, report.expect("explain produces report")))
}

fn select_with(
    g: &Graph,
    entry: usize,
    wants: Wants,
    sizes: &[(DimName, u64)],
    upto: &[DimName],
    device: &Device,
    explain: bool,
) -> Result<(Selected, Option<Report>), Error> {
    let rules = device.rules();
    linked(&rules)?;
    let (parts, rank) = (device.division(), device.rank);
    let mut tree = Tree::upto(
        g,
        entry,
        wants,
        sizes,
        upto,
        &device.pins,
        device.kept,
        &parts,
        rank,
    )?
    .with_compute(device.compute);
    tree.config = device.config.clone();
    tree.knobs = device.knobs.clone();
    tree.kept_rows = device.kept_rows;
    tree.index_rows = device.index_rows;
    let labels = (device.scoped() || !g.regions.is_empty()).then(|| crate::recipe::labels(g));
    if device.scoped()
        && let Some(labels) = &labels
    {
        tree.computes = labels.iter().map(|l| device.compute_at(l)).collect();
    }
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
        streams: device.streams(tree.weight_bytes()),
        labels,
        priced: explain.then(HashMap::new),
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
        let live = tree.appends(t) || !cover.wanted[t.0 as usize].is_empty();
        let zone = match (tree.root(t), own[t.0 as usize]) {
            (true, _) if live => cover.settle(t),
            (true, _) => continue,
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
    let scan = explain.then(|| scan_cover(&cover, &own));
    let mut e = Emit::with_labels(cover.labels.as_deref());
    for (port, (len, shape, dtype)) in tree.ports.iter().enumerate() {
        let val = e.value(
            *len,
            Home::Input(port as u32),
            EDGE,
            tessera_codegen::form::of(*dtype),
            pack_shape(shape, *len),
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
                let len = tree.elements(t);
                let val = e.value(len, *home, EDGE, n.form, pack_shape(&n.shape, len));
                e.out.values[val.0 as usize].share = tree.share(t);
                val
            }
            (_, None) => continue,
            _ if tree.alias(t) => val(n.args[0]),
            (Term::Op(tessera_graph::Op::Append(index)), Some(zone)) => {
                let choice = cover.best[t.0 as usize][zone as usize].take().flatten();
                let choice = choice.expect("solved above");
                if choice.drafts.is_empty() {
                    if choice.offered.len() > 1 {
                        e.out.contested.extend(&choice.offered);
                    }
                    let args = [n.args[0], n.args[1], n.args[2]].map(val);
                    e.append(&tree, t, *index, args)?
                } else {
                    emit_choice(&mut e, &tree, t, choice, zone, Some(*index), &mut vals)?
                }
            }
            (_, Some(zone)) => {
                let choice = cover.best[t.0 as usize][zone as usize].take().flatten();
                let choice = choice.expect("solved above");
                emit_choice(&mut e, &tree, t, choice, zone, None, &mut vals)?
            }
        };
        vals[t.0 as usize] = Some(made);
    }
    for (t, shape) in &tree.outputs {
        let val = vals[t.0 as usize].expect("outputs have values");
        let val = e.bring(val, EDGE);
        let want = tree.node(*t).form;
        let val = e.as_form(val, want);
        e.out.outputs.push((val, shape.clone()));
    }
    if let Some(favor) = device.required_favor()
        && !e.out.ops.iter().any(|op| op.rule == favor)
    {
        return Err(Error::FavoredUnused(favor.to_string()));
    }
    e.out.ranks = tree.parts.ranks() as u32;
    e.out.rank = tree.rank;
    let report = scan.map(|s| finish_report(s, &cover, &e.out));
    Ok((e.out, report))
}

/// Emits one chosen cover's drafts, optionally targeting kept rows.
fn emit_choice(
    e: &mut Emit<'_>,
    tree: &Tree,
    t: T,
    choice: cover::Choice,
    zone: u32,
    append: Option<u32>,
    vals: &mut [Option<Val>],
) -> Result<Val, Error> {
    if choice.offered.len() > 1 {
        e.out.contested.extend(&choice.offered);
    }
    for (rule, way) in choice.ways {
        e.out.ways.entry(rule).or_default().insert(way);
    }
    let n = tree.node(t);
    let covers = tree.covered(t, &reads(&choice.drafts));
    let mut made: Vec<Val> = Vec::with_capacity(choice.drafts.len());
    let mut written = Vec::new();
    let last = choice.drafts.len() - 1;
    for (i, d) in choice.drafts.into_iter().enumerate() {
        let site = cover::graph_site(tree, e.labels, d.writes.unwrap_or(t));
        let mut args = Vec::with_capacity(d.reads.len());
        for r in &d.reads {
            args.push(match r {
                Read::Node(a) => vals[a.0 as usize].expect("arguments come first"),
                Read::Draft(at) => made[*at],
                Read::Kept(index) => e.kept(tree, *index)?,
            });
        }
        // The last launch writes the node's own type.
        let (form, shape) = match d.writes {
            Some(w) => {
                let node = tree.node(w);
                (node.form, pack_shape(&node.shape, d.len))
            }
            None if i == last => (n.form, pack_shape(&n.shape, d.len)),
            None => (form::F32_FORM, pack_shape(&[d.len], d.len)),
        };
        let target = match append {
            Some(index) if i == last => Some(e.kept(tree, index)?),
            _ => None,
        };
        let writes = d.writes;
        let also_nodes: Vec<T> = d.also.iter().map(|(w, _)| *w).collect();
        let also_specs: Vec<_> = d
            .also
            .iter()
            .map(|&(w, len)| {
                let node = tree.node(w);
                (len, node.form, pack_shape(&node.shape, len))
            })
            .collect();
        let (primary, extras) = e.op(
            d,
            choice.rule,
            Some(site.as_str()),
            args,
            &covers,
            zone,
            form,
            shape,
            &also_specs,
            target,
        );
        made.push(primary);
        if let Some(w) = writes {
            written.push((w, made[i]));
        }
        for (w, v) in also_nodes.into_iter().zip(extras) {
            written.push((w, v));
        }
    }
    let mut own = *made.last().expect("a rule proposes at least one launch");
    for (w, v) in written {
        match w == t {
            true => own = v,
            false => vals[w.0 as usize] = Some(v),
        }
    }
    Ok(own)
}
