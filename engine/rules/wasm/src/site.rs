//! Sites for the module, and its offers as drafts.

use crate::abi::{Count, Input, Kind, Offer, Place, Site};
use crate::text::{Fixed, Text};
use sensai_graph::{Composite, DimName, PathSeg};
use sensai_lower::{Arg, Draft, Extent, Home, Read, T, Term, Tree, sites};

/// The site for `kind` at `at`, and its reads.
pub(crate) fn find(tree: &Tree, at: T, kind: Kind) -> Option<(Site, Vec<T>)> {
    if kind == Kind::Matmul {
        let m: sites::Matmul = tree.site(at)?;
        let site = Site::Matmul {
            m: m.m,
            n: m.n,
            k: m.k,
            form: m.form.name(),
        };
        return m.picked.is_none().then_some((site, vec![m.x, m.w]));
    }
    let r: sites::Region = tree.site(at)?;
    let wanted = match kind {
        Kind::Mlp => Composite::Mlp,
        _ => Composite::Moe,
    };
    if r.kind != wanted {
        return None;
    }
    let inputs = r.inputs.iter().map(|t| input(tree, *t)).collect();
    let len = r.len;
    Some((Site::Region { kind, inputs, len }, r.inputs))
}

fn input(tree: &Tree, t: T) -> Input {
    let node = tree.node(t);
    let dims = node.extents.iter().map(|e| match e {
        Extent::Dim { name, .. } => Some(*name),
        Extent::Fixed(_) => None,
    });
    let weight = match node.term {
        Term::Leaf(Home::Weight(w)) => Some(dotted(&tree.g.weights[w as usize].path)),
        _ => None,
    };
    let share = tree.share(t);
    Input {
        shape: node.shape.clone(),
        dims: dims.collect(),
        form: node.form.name(),
        weight,
        share: share.map(|s| [s.axis as usize, s.first, s.count, s.whole]),
    }
}

fn dotted(path: &[PathSeg]) -> String {
    let seg = |s: &PathSeg| match s {
        PathSeg::Field(f) => f.clone(),
        PathSeg::Index(i) => i.to_string(),
    };
    path.iter().map(seg).collect::<Vec<_>>().join(".")
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

/// Drafts for the offers. `None` on unknown names.
pub(crate) fn drafts(
    offers: Vec<Offer>,
    kernels: &'static [Fixed],
    tree: &Tree,
    nodes: &[T],
) -> Option<Vec<Draft>> {
    let mut drafts = Vec::with_capacity(offers.len());
    for offer in offers {
        let fixed = kernels.iter().find(|k| k.name == offer.kernel)?;
        let mut reads = Vec::with_capacity(offer.reads.len());
        for place in &offer.reads {
            reads.push(match *place {
                Place::Read(i) => Read::Node(*nodes.get(i)?),
                Place::Draft { draft } if draft < drafts.len() => Read::Draft(draft),
                Place::Draft { .. } => return None,
            });
        }
        let mut args = Vec::with_capacity(offer.counts.len());
        for count in &offer.counts {
            args.push(arg(tree, nodes, *count)?);
        }
        let text = Text::new(fixed, &offer, args);
        drafts.push(Draft::new(text, reads, offer.len));
    }
    Some(drafts)
}
