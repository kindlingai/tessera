//! Sites for the module, and its offers as drafts.

use crate::abi::{Count, Entry, Input, Kept, Kind, Offer, Place, Scalar, Site};
use crate::text::{Fixed, Text};
use tessera_graph::{Composite, DimName, Op, Pairs, PathSeg, Select, Stage};
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
    /// Each all-reduce the offers must place.
    pub sums: Vec<T>,
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
    let scalars = scalars(tree, tree.inside(at, &r.inputs));
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
        sums: reads.sums.iter().map(|t| tree.elements(*t)).collect(),
        scalars,
    };
    Some((site, reads))
}

/// The distinct op constants of `nodes`, in order.
fn scalars(tree: &Tree, nodes: Vec<T>) -> Vec<Scalar> {
    let mut out: Vec<Scalar> = Vec::new();
    for n in nodes {
        let (op, value) = match &tree.node(n).term {
            Term::Stage(Stage::RmsNorm { eps } | Stage::Rms { eps }) => ("rms_norm", *eps),
            Term::Stage(Stage::Rope { theta, pairs, .. }) => match pairs {
                Pairs::Adjacent => ("rope_adjacent", *theta),
                Pairs::Halves => ("rope_halves", *theta),
            },
            Term::Stage(Stage::Scale(by)) => ("scale", *by),
            Term::Op(Op::Attend(a)) => ("attend", a.scale),
            Term::Op(Op::Select(Select::TopK { .. })) => {
                let k = tree.node(n).shape.last().copied().unwrap_or(0);
                ("top_k", k as f32)
            }
            _ => continue,
        };
        let scalar = Scalar {
            op: op.into(),
            value,
        };
        if !out.contains(&scalar) {
            out.push(scalar);
        }
    }
    out
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
fn arg(tree: &Tree, nodes: &[T], count: Count) -> Result<Arg, String> {
    let (dim, mul): (DimName, usize) = match count {
        Count::Fixed(n) => return Ok(Arg::count(n as usize)),
        Count::Dim { dim, mul } => (dim, mul),
    };
    let Some(size) = tree.size(dim) else {
        return Err(format!("count dim {dim:?} has no size at site"));
    };
    let most = size as usize * mul;
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
    Ok(Arg::size(extent))
}

/// The entries as drafts, or why they did not fit.
pub(crate) fn drafts(
    entries: Vec<Entry>,
    kernels: &'static [Fixed],
    tree: &Tree,
    from: &Reads,
) -> Result<Vec<Draft>, String> {
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
                let Some(prev) = drafts.get(sum.all_reduce) else {
                    return Err(format!(
                        "all_reduce draft {} out of range ({} earlier drafts)",
                        sum.all_reduce,
                        drafts.len()
                    ));
                };
                let len = prev.len;
                // Ranks pair sums by order, so each length must match.
                let Some(&at) = from.sums.get(sums) else {
                    return Err(format!(
                        "extra all_reduce #{sums} (site expects {})",
                        from.sums.len()
                    ));
                };
                let want = tree.elements(at);
                if want != len {
                    return Err(format!(
                        "all_reduce #{sums} length {len} does not match site length {want}"
                    ));
                }
                sums += 1;
                let read = [Read::Draft(sum.all_reduce)];
                let Some(wire) = tree.wire(at) else {
                    return Err(format!("all_reduce #{sums} has no sum wire dtype at site"));
                };
                (Draft::new(AllReduce { len, wire }, read, len), sum.writes)
            }
        };
        if let Some(o) = writes {
            let Some(slot) = written.get_mut(o) else {
                return Err(format!(
                    "writes index {o} out of range (site has {} outputs)",
                    from.outputs.len()
                ));
            };
            if slot.replace(drafts.len()).is_some() {
                return Err(format!("output {o} written more than once"));
            }
        }
        drafts.push(draft);
    }
    if sums != from.sums.len() {
        return Err(format!(
            "offer placed {sums} all_reduce(s), site requires {}",
            from.sums.len()
        ));
    }
    // One output defaults to the last entry.
    if let [last @ None] = written.as_mut_slice() {
        *last = drafts.len().checked_sub(1);
    }
    for (o, d) in written.into_iter().enumerate() {
        let Some(idx) = d else {
            return Err(format!(
                "output {o} was not written (site has {} outputs, {} entries)",
                from.outputs.len(),
                drafts.len()
            ));
        };
        drafts[idx].writes = Some(from.outputs[o]);
    }
    Ok(drafts)
}

/// One offer's launch, after `made` earlier drafts.
fn launch(
    offer: &Offer,
    kernels: &'static [Fixed],
    tree: &Tree,
    from: &Reads,
    made: usize,
) -> Result<Draft, String> {
    let Some(fixed) = kernels.iter().find(|k| k.name == offer.kernel) else {
        let names: Vec<_> = kernels.iter().map(|k| k.name).collect();
        return Err(format!(
            "unknown kernel {:?} (manifest declares: [{}])",
            offer.kernel,
            names.join(", ")
        ));
    };
    let mut reads = Vec::with_capacity(offer.reads.len());
    let mut kept = None;
    for place in &offer.reads {
        reads.push(match *place {
            Place::Read(i) => {
                let Some(&node) = from.nodes.get(i) else {
                    return Err(format!(
                        "kernel {}: read index {i} out of range (site has {} inputs)",
                        offer.kernel,
                        from.nodes.len()
                    ));
                };
                Read::Node(node)
            }
            Place::Draft { draft } if draft < made => Read::Draft(draft),
            Place::Draft { draft } => {
                return Err(format!(
                    "kernel {}: draft index {draft} out of range ({made} earlier drafts)",
                    offer.kernel
                ));
            }
            Place::Part { part: [i, 0] } => {
                let Some(&node) = from.nodes.get(i) else {
                    return Err(format!(
                        "kernel {}: part [{i}, 0] out of range (site has {} inputs)",
                        offer.kernel,
                        from.nodes.len()
                    ));
                };
                Read::Node(node)
            }
            Place::Part { part: [i, p] } => {
                let Some(scales) = from.scales(i) else {
                    return Err(format!(
                        "kernel {}: input {i} has no scale parts",
                        offer.kernel
                    ));
                };
                let Some(&node) = scales.get(p - 1) else {
                    return Err(format!(
                        "kernel {}: part [{i}, {p}] out of range (expected part 0..=2)",
                        offer.kernel
                    ));
                };
                Read::Node(node)
            }
            Place::Kept { kept: i } => {
                let Some(k) = from.kept.get(i) else {
                    return Err(format!(
                        "kernel {}: kept index {i} out of range (site has {} kept buffers)",
                        offer.kernel,
                        from.kept.len()
                    ));
                };
                kept = Some(*k);
                Read::Kept(k.index)
            }
        });
    }
    let mut args = Vec::with_capacity(offer.counts.len());
    for count in &offer.counts {
        let built = arg(tree, &from.nodes, *count).map_err(|e| format!("{}: {e}", offer.kernel))?;
        args.push(built);
    }
    let mut text = Text::new(fixed, offer, args);
    text.kept = kept.map(|k| {
        let values = tree.kept.values(k.slots, k.context);
        let values = values.into_iter().map(|v| Arg::count(v as usize));
        (tree.kept, values.collect())
    });
    Ok(Draft::new(text, reads, offer.len))
}
