//! Sites for the module, and its offers as drafts.

use crate::abi::{Count, Entry, Input, Kept, Kind, Offer, Place, Site};
use crate::text::{Fixed, Text};
use tessera_graph::{Composite, DimName, PathSeg};
use tessera_lower::kernels::AllReduce;
use tessera_lower::{Arg, Draft, Extent, Home, Read, T, Term, Tree, sites};

/// Data a site's offers may read.
pub(crate) struct Reads {
    pub nodes: Vec<T>,
    pub kept: Vec<sites::Appended>,
    /// Per rotated input: its place, then `suh` and `svh`.
    pub parts: Vec<(usize, [T; 2])>,
    /// Region outputs. Entries name the one they write.
    pub outputs: Vec<T>,
    /// Elements of each all-reduce the offers must place.
    pub sums: Vec<usize>,
}

impl Reads {
    fn scales(&self, input: usize) -> Option<[T; 2]> {
        let found = self.parts.iter().find(|(place, _)| *place == input)?;
        Some(found.1)
    }
}

/// The site for `kind` at `at`, and its reads.
pub(crate) fn find(tree: &Tree, at: T, kind: Kind) -> Option<(Site, Reads)> {
    let wanted = match kind {
        Kind::Matmul => return matmul(tree, at),
        Kind::Mlp => Composite::Mlp,
        Kind::Moe => Composite::Moe,
        Kind::Attention => Composite::Attention,
        Kind::Layer => Composite::Layer,
        Kind::Model => Composite::Model,
    };
    let r = sites::Region::of(tree, at, Some(wanted))?;
    let outputs = r.outputs.iter().map(|t| tree.elements(*t)).collect();
    let reads = Reads {
        nodes: r.inputs,
        kept: r.kept,
        parts: r.parts,
        outputs: r.outputs,
        sums: r.sums,
    };
    let mut inputs = Vec::with_capacity(reads.nodes.len());
    for (i, t) in reads.nodes.iter().enumerate() {
        let mut input = input(tree, *t);
        input.parts = reads.scales(i).map(|s| s.map(|t| tree.elements(t)));
        inputs.push(input);
    }
    let kept = reads.kept.iter().map(|k| rows(tree, k)).collect();
    let site = Site::Region {
        kind,
        inputs,
        outputs,
        kept,
        sums: reads.sums.clone(),
    };
    Some((site, reads))
}

fn matmul(tree: &Tree, at: T) -> Option<(Site, Reads)> {
    let m: sites::Matmul = tree.site(at)?;
    let site = Site::Matmul {
        m: m.m,
        n: m.n,
        k: m.k,
        form: m.form.name().into(),
    };
    let reads = Reads {
        nodes: vec![m.x, m.w],
        kept: Vec::new(),
        parts: Vec::new(),
        outputs: Vec::new(),
        sums: Vec::new(),
    };
    m.picked.is_none().then_some((site, reads))
}

fn rows(tree: &Tree, k: &sites::Appended) -> Kept {
    Kept {
        index: k.index,
        slots: k.slots,
        context: k.context,
        width: k.width,
        layout: tree.kept.name().into(),
        page: tree.kept.page(),
        params: tree.kept.params().iter().map(|p| p.to_string()).collect(),
        values: tree.kept.values(k.slots, k.context),
        by: k.by,
    }
}

fn input(tree: &Tree, t: T) -> Input {
    let node = tree.node(t);
    let dims = node.extents.iter().map(|e| match e {
        Extent::Dim { name, .. } => Some(format!("{name:?}")),
        Extent::Fixed(_) => None,
    });
    let weight = match node.term {
        Term::Leaf(Home::Weight(w) | Home::Part(w, 0)) => {
            Some(PathSeg::dotted(&tree.g.weights[w as usize].path))
        }
        _ => None,
    };
    let share = tree.share(t);
    Input {
        shape: node.shape.clone(),
        dims: dims.collect(),
        form: node.form.name().into(),
        weight,
        share: share.map(|s| [s.axis as usize, s.first, s.count, s.whole]),
        parts: None,
    }
}

/// A count's argument. Varying dimensions wait for launch.
fn arg(tree: &Tree, nodes: &[T], count: Count) -> Option<Arg> {
    let (dim, mul): (DimName, usize) = match count {
        Count::Fixed(n) => return Some(Arg::count(n as usize)),
        Count::Dim { dim, mul } => (dim, mul),
    };
    let most = tree.size(dim)? as usize * mul;
    let varies = |t: &T| {
        let extents = &tree.node(*t).extents;
        extents
            .iter()
            .any(|e| matches!(e, Extent::Dim { name, .. } if *name == dim))
    };
    let extent = match nodes.iter().any(varies) {
        true => Extent::Dim {
            name: dim,
            mul,
            add: 0,
            most,
        },
        false => Extent::Fixed(most),
    };
    Some(Arg::size(extent))
}

/// The entries as drafts. `None` if they do not fit.
pub(crate) fn drafts(
    entries: Vec<Entry>,
    kernels: &'static [Fixed],
    tree: &Tree,
    from: &Reads,
) -> Option<Vec<Draft>> {
    let mut drafts: Vec<Draft> = Vec::with_capacity(entries.len());
    let (mut sums, mut written) = (0, vec![None; from.outputs.len()]);
    for entry in entries {
        let (draft, writes) = match entry {
            Entry::Launch(offer) => {
                // A resident grid runs every sum itself.
                if offer.resident.is_some() {
                    sums += from.sums.len();
                }
                let made = drafts.len();
                (launch(&offer, kernels, tree, from, made)?, offer.writes)
            }
            Entry::Sum(sum) => {
                let len = drafts.get(sum.all_reduce)?.len;
                // Ranks pair sums by order, so each length must match.
                if from.sums.get(sums) != Some(&len) {
                    return None;
                }
                sums += 1;
                let read = [Read::Draft(sum.all_reduce)];
                (Draft::new(AllReduce { len }, read, len), sum.writes)
            }
        };
        if let Some(o) = writes
            && written.get_mut(o)?.replace(drafts.len()).is_some()
        {
            return None;
        }
        drafts.push(draft);
    }
    if sums != from.sums.len() {
        return None;
    }
    // One output defaults to the last entry.
    if let [last @ None] = written.as_mut_slice() {
        *last = drafts.len().checked_sub(1);
    }
    for (o, d) in written.into_iter().enumerate() {
        drafts[d?].writes = Some(from.outputs[o]);
    }
    Some(drafts)
}

/// One offer's launch, after `made` earlier drafts.
fn launch(
    offer: &Offer,
    kernels: &'static [Fixed],
    tree: &Tree,
    from: &Reads,
    made: usize,
) -> Option<Draft> {
    let fixed = kernels.iter().find(|k| k.name == offer.kernel)?;
    let mut reads = Vec::with_capacity(offer.reads.len());
    let mut kept = None;
    for place in &offer.reads {
        reads.push(match *place {
            Place::Read(i) => Read::Node(*from.nodes.get(i)?),
            Place::Draft { draft } if draft < made => Read::Draft(draft),
            Place::Draft { .. } => return None,
            Place::Part { part: [i, 0] } => Read::Node(*from.nodes.get(i)?),
            Place::Part { part: [i, p] } => Read::Node(*from.scales(i)?.get(p - 1)?),
            Place::Kept { kept: i } => {
                let k = from.kept.get(i)?;
                kept = Some(*k);
                Read::Kept(k.index)
            }
        });
    }
    let mut args = Vec::with_capacity(offer.counts.len());
    for count in &offer.counts {
        args.push(arg(tree, &from.nodes, *count)?);
    }
    let mut text = Text::new(fixed, offer, args);
    text.kept = kept.map(|k| {
        let values = tree.kept.values(k.slots, k.context);
        let values = values.into_iter().map(|v| Arg::count(v as usize));
        (tree.kept, values.collect())
    });
    Some(Draft::new(text, reads, offer.len))
}
