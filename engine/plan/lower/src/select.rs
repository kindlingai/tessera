//! Picks kernels for an entry: covers, then ops over values.

mod cover;
mod emit;

use cover::{Cover, EDGE, reads};
use emit::Emit;
pub use report::{Finding, Rejection, Report};
use report::{finish_report, scan_cover};

use crate::cost::Device;
use crate::error::Uncovered;
use crate::form::{self, Load};
use crate::kernel::KernelRef;
use crate::plan::{KeptInfo, Share, pack_shape};
use crate::rule::{Read, Rule, linked};
use crate::tree::{T, Term, Tree, Wants};
use crate::{Error, Unsupported};
use std::collections::{BTreeMap, BTreeSet};
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
    let mut e = Emit::default();
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
            (Term::Op(tessera_graph::Op::Append(index)), _) => {
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
                    let (form, shape) = match d.writes {
                        Some(w) => {
                            let node = tree.node(w);
                            (node.form, pack_shape(&node.shape, d.len))
                        }
                        None if i == last => (n.form, pack_shape(&n.shape, d.len)),
                        None => (form::F32_FORM, pack_shape(&[d.len], d.len)),
                    };
                    let writes = d.writes;
                    made.push(e.op(d, choice.rule, reads, &covers, zone, form, shape));
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

mod report {
    use super::cover::{Cover, EDGE, fits, reads, writes};
    use super::{Home, Selected, Value};
    use crate::cost::Device;
    use crate::kernels::AllReduce;
    use crate::rule::{Draft, Rule, ways};
    use crate::tree::{T, Term, Tree};
    use std::collections::BTreeMap;
    use std::fmt::Write as _;
    use tessera_engine_settings::Backend;
    use tessera_graph::Op;
    use tessera_platform::Platform;
    use tessera_sim::ZoneKind;

    /// Why one candidate fast-path rule did not cover a site.
    #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    pub struct Rejection {
        pub rule: &'static str,
        pub reason: String,
    }

    /// One grouped slow, unfused, or widened site in a plan.
    #[derive(Clone, Debug, PartialEq)]
    pub struct Finding {
        pub site: String,
        pub rule: &'static str,
        pub kernel: String,
        pub count: usize,
        pub ms: f64,
        pub region: bool,
        pub rejections: Vec<Rejection>,
    }

    /// Summary of fast-path misses and weight widenings for a plan.
    #[derive(Clone, Debug, Default, PartialEq)]
    pub struct Report {
        pub label: String,
        pub total_ops: usize,
        pub total_ms: f64,
        pub unused_favor: Option<String>,
        pub weight_widens: usize,
        pub findings: Vec<Finding>,
    }

    impl Report {
        /// Whether any site fell off fast path or widened weights.
        pub fn has_fallbacks(&self) -> bool {
            !self.findings.is_empty() || self.unused_favor.is_some() || self.weight_widens > 0
        }

        /// Count of fallback sites, excluding overlapping region summaries.
        pub fn fallback_sites(&self) -> usize {
            let non_region: usize = self
                .findings
                .iter()
                .filter(|f| !f.region)
                .map(|f| f.count)
                .sum();
            if non_region > 0 {
                return non_region;
            }
            self.findings.iter().map(|f| f.count).sum()
        }

        /// Estimated milliseconds spent on fallback sites.
        pub fn fallback_ms(&self) -> f64 {
            let non_region: f64 = self
                .findings
                .iter()
                .filter(|f| !f.region)
                .map(|f| f.ms)
                .sum();
            let regions: f64 = self
                .findings
                .iter()
                .filter(|f| f.region)
                .map(|f| f.ms)
                .sum();
            let ms = non_region.max(regions);
            if self.total_ms > 0.0 {
                ms.min(self.total_ms)
            } else {
                ms
            }
        }

        /// One-line boot warning, or `None` when on fast path.
        pub fn warning(&self, flag: &str) -> Option<String> {
            if !self.has_fallbacks() {
                return None;
            }
            let mut parts = Vec::new();
            if let Some(favor) = &self.unused_favor {
                parts.push(format!("favored rule `{favor}` unused"));
            }
            let sites = self.fallback_sites();
            if sites > 0 {
                parts.push(format!("{sites} fallback sites"));
            }
            if self.weight_widens > 0 {
                parts.push(format!("{} weight widens", self.weight_widens));
            }
            let ms = self.fallback_ms();
            if self.total_ms > 0.0 && ms > 0.0 {
                let pct = (ms / self.total_ms * 100.0).min(100.0);
                if self.label.is_empty() {
                    parts.push(format!("est {pct:.0}% of step"));
                } else {
                    parts.push(format!("est {pct:.0}% of {} step", self.label));
                }
            }
            let detail = parts.join(", ");
            Some(format!(
                "warning: lowering fell off fast path ({detail}) - pass {flag} for details"
            ))
        }

        /// Detailed multi-line report showing up to `top` fallback site groups.
        pub fn format(&self, top: usize) -> String {
            let head = if self.label.is_empty() {
                "lowering".to_string()
            } else {
                format!("lowering ({})", self.label)
            };
            if !self.has_fallbacks() {
                return format!(
                    "{head}: on fast path ({} ops, est {:.2} ms)",
                    self.total_ops, self.total_ms
                );
            }
            let mut out = String::new();
            let ms = self.fallback_ms();
            let pct = if self.total_ms > 0.0 {
                (ms / self.total_ms * 100.0).min(100.0)
            } else {
                0.0
            };
            let _ = writeln!(
                out,
                "{head}: {} fallback sites, {} ops, est {:.2} ms ({pct:.0}% fallback)",
                self.fallback_sites(),
                self.total_ops,
                self.total_ms,
            );
            if let Some(favor) = &self.unused_favor {
                let _ = writeln!(out, "  favored rule `{favor}` covered 0 steps");
            }
            for (i, f) in self.findings.iter().take(top).enumerate() {
                let share = if self.total_ms > 0.0 && f.ms > 0.0 {
                    format!(", {:.0}%", (f.ms / self.total_ms * 100.0).min(100.0))
                } else {
                    String::new()
                };
                let _ = writeln!(
                    out,
                    "  {}. {}x {} -> {}/{} (est {:.2} ms{share})",
                    i + 1,
                    f.count,
                    f.site,
                    f.rule,
                    f.kernel,
                    f.ms,
                );
                for r in &f.rejections {
                    let _ = writeln!(out, "       {}: {}", r.rule, r.reason);
                }
            }
            if self.findings.len() > top {
                let rem = self.findings.len() - top;
                let _ = writeln!(out, "  ... and {rem} more fallback site groups");
            }
            out.trim_end().to_string()
        }
    }

    /// Grouping key that merges identical repeated layers into one finding.
    type GroupKey = (bool, String, &'static str, String, Vec<Rejection>);

    /// Whether `rule` targets the active `kind` and `platform` backend.
    fn applicable(rule: &Rule, kind: ZoneKind, platform: &Platform) -> bool {
        if kind != ZoneKind::Gpu {
            return false;
        }
        let backend = Backend::named(&platform.backend);
        backend.is_none_or(|b| !rule.propose.lacks().contains(&b))
    }

    /// Self-cost of one cover's drafts, including widens and narrows.
    fn self_secs(cover: &Cover<'_>, zone: u32, t: T, drafts: &[Draft]) -> Option<f64> {
        let mut secs = 0.0;
        for d in drafts {
            secs += cover.seconds(zone, d)?;
        }
        secs += cover.widening(zone, drafts) + cover.narrowing(zone, t, drafts);
        Some(secs)
    }

    /// Explains why candidate `rule` did not win at `(t, zone)`.
    fn check_rule(
        cover: &Cover<'_>,
        rule: &Rule,
        t: T,
        zone: u32,
        chosen_rule: &'static str,
        chosen_secs: Option<f64>,
    ) -> Option<Rejection> {
        let kind = cover.device.cluster.kind(zone as usize);
        let platform = &cover.device.platform;
        let Some(drafts) = rule.propose.at(cover.tree, t) else {
            let reason = rule.propose.why_not(cover.tree, t)?;
            return Some(Rejection {
                rule: rule.name,
                reason,
            });
        };
        if !drafts.iter().all(|d| fits(d, kind)) {
            let want = drafts
                .iter()
                .find(|d| !fits(d, kind))
                .map_or(ZoneKind::Gpu, |d| d.kernel.runs_on());
            return Some(Rejection {
                rule: rule.name,
                reason: format!("runs on {want:?}, plan zone is {kind:?}"),
            });
        }
        let mut needs_err: Option<String> = None;
        let mut swallowed = false;
        let mut leaked = false;
        let mut best_secs: Option<f64> = None;
        for way in ways(drafts) {
            if let Some(err) = way.iter().find_map(|d| d.kernel.needs().mismatch(platform)) {
                if needs_err.is_none() {
                    needs_err = Some(err);
                }
                continue;
            }
            if cover.swallows(rule, t, &way, kind) {
                swallowed = true;
                continue;
            }
            if cover.tree.absorbs(t, &reads(&way), &writes(&way)).is_none() {
                leaked = true;
                continue;
            }
            if let Some(secs) = self_secs(cover, zone, t, &way)
                && best_secs.is_none_or(|b| secs < b)
            {
                best_secs = Some(secs);
            }
        }
        let reason = if let Some(cand) = best_secs {
            if cover.device.favors(chosen_rule) {
                format!("outbid by favored rule {chosen_rule}")
            } else if let Some(ch) = chosen_secs
                && cand > ch
            {
                format!(
                    "estimated slower than {chosen_rule} ({:.3} ms vs {:.3} ms)",
                    cand * 1e3,
                    ch * 1e3
                )
            } else {
                format!("higher subtree cost than {chosen_rule}")
            }
        } else if let Some(err) = needs_err {
            err
        } else if swallowed {
            "blocked by favored rule".into()
        } else if leaked {
            "absorbed value used outside cover".into()
        } else {
            "no runnable way on device".into()
        };
        Some(Rejection {
            rule: rule.name,
            reason,
        })
    }

    /// Concise human description of tree node `t`.
    fn site_label(tree: &Tree<'_>, t: T) -> String {
        let n = tree.node(t);
        match &n.term {
            Term::Op(Op::Matmul) => {
                let (x, w) = (tree.node(n.args[0]), tree.node(n.args[1]));
                format!("matmul {:?} x {:?} ({})", x.shape, w.shape, w.form.name())
            }
            Term::Op(Op::Attend(a)) => {
                let q = tree.node(n.args[0]);
                let k = tree.node(n.args[1]);
                let kind = match (a.ragged, a.picked) {
                    (true, true) => "attend ragged+listed",
                    (true, false) => "attend ragged",
                    (false, true) => "attend listed",
                    (false, false) => "attend",
                };
                format!("{kind} {:?} k={:?} ({})", q.shape, k.shape, k.form.name())
            }
            Term::Op(op) => format!("{op:?} {:?} ({})", n.shape, n.form.name()),
            Term::Stage(s) => format!("{s:?} {:?} ({})", n.shape, n.form.name()),
            Term::Leaf(_) => format!("leaf {:?} ({})", n.shape, n.form.name()),
        }
    }

    /// Formats a `Value`'s packed shape as `[d0, d1, ...]`.
    fn value_shape(v: &Value) -> String {
        let dims = v.dims as usize;
        format!("{:?}", &v.shape[..dims.min(4)])
    }

    /// Joins unique kernel names for one cover.
    fn kernel_names(drafts: &[Draft]) -> String {
        let mut names = Vec::with_capacity(drafts.len());
        for d in drafts {
            let name = d.kernel.name();
            if !names.contains(&name) {
                names.push(name);
            }
        }
        names.join("+")
    }

    /// Inserts or accumulates a finding in `groups`.
    fn merge_finding(
        groups: &mut BTreeMap<GroupKey, (usize, f64)>,
        region: bool,
        site: String,
        rule: &'static str,
        kernel: String,
        ms: f64,
        mut rejections: Vec<Rejection>,
    ) {
        rejections.sort();
        rejections.dedup();
        let key = (region, site, rule, kernel, rejections);
        let entry = groups.entry(key).or_insert((0, 0.0));
        entry.0 += 1;
        entry.1 += ms;
    }

    /// Estimates total seconds across all emitted ops in `selected`.
    fn plan_secs(selected: &Selected, device: &Device, streams: bool) -> f64 {
        let mut total = 0.0;
        for op in &selected.ops {
            let out = &selected.values[op.write.0 as usize];
            if op.kernel.name() == AllReduce::NAME {
                total += device
                    .all_reduce(out.form.bytes(out.len))
                    .unwrap_or_default();
                continue;
            }
            let reads_weight = op.reads.iter().any(|r| {
                matches!(
                    selected.values[r.0 as usize].home,
                    Home::Weight(_) | Home::Part(..)
                )
            });
            let cold = streams && reads_weight;
            total += device.seconds_in(op.zone, &*op.kernel, out.len, cold);
        }
        total
    }

    /// Records findings for emitted weight-widening ops.
    fn add_widens(
        groups: &mut BTreeMap<GroupKey, (usize, f64)>,
        selected: &Selected,
        device: &Device,
        streams: bool,
    ) -> usize {
        let mut count = 0;
        for (idx, op) in selected.ops.iter().enumerate() {
            if op.rule != "widen" {
                continue;
            }
            let Some(&src) = op.reads.first() else {
                continue;
            };
            let val = &selected.values[src.0 as usize];
            if !matches!(val.home, Home::Weight(_) | Home::Part(..)) {
                continue;
            }
            count += 1;
            let out = &selected.values[op.write.0 as usize];
            let secs = device.seconds_in(op.zone, &*op.kernel, out.len, streams);
            let consumer = selected.ops[idx + 1..]
                .iter()
                .find(|later| later.reads.contains(&op.write));
            let (c_rule, reason) = match consumer {
                Some(c) => (
                    c.rule,
                    format!(
                        "kernel {} does not accept {} weight",
                        c.kernel.name(),
                        val.form.name()
                    ),
                ),
                None => (
                    "widen",
                    format!("weight form {} widened to f32", val.form.name()),
                ),
            };
            let site = format!("weight {} -> f32 {}", val.form.name(), value_shape(val));
            let rej = vec![Rejection {
                rule: c_rule,
                reason,
            }];
            merge_finding(
                groups,
                false,
                site,
                "widen",
                op.kernel.name().to_string(),
                secs * 1e3,
                rej,
            );
        }
        count
    }

    /// Records findings for graph regions declined by registered region rules.
    fn add_regions(
        groups: &mut BTreeMap<GroupKey, (usize, f64)>,
        cover: &Cover<'_>,
        covered_by_region: &[bool],
        node_secs: &[f64],
    ) {
        let kind = cover.device.cluster.kind(EDGE as usize);
        let region_rules: Vec<&Rule> = cover
            .rules
            .iter()
            .copied()
            .filter(|r| {
                r.propose.regions()
                    && (applicable(r, kind, &cover.device.platform) || cover.device.favors(r.name))
            })
            .collect();
        if region_rules.is_empty() {
            return;
        }
        for region in &cover.tree.g.regions {
            let Some(anchor) = cover.tree.anchor(region) else {
                continue;
            };
            if covered_by_region[anchor.0 as usize] {
                continue;
            }
            let mut rejections = Vec::new();
            for rule in &region_rules {
                if let Some(rej) = check_rule(cover, rule, anchor, EDGE, "unfused", None) {
                    rejections.push(rej);
                }
            }
            if rejections.is_empty() {
                continue;
            }
            let mut secs = 0.0;
            for id in region.start.0..region.end.0 {
                if let Some(t) = cover.tree.of(tessera_graph::NodeId(id)) {
                    secs += node_secs[t.0 as usize];
                }
            }
            let site = format!("region {:?}", region.kind);
            merge_finding(
                groups,
                true,
                site,
                "unfused",
                "fallback ops".into(),
                secs * 1e3,
                rejections,
            );
        }
    }

    /// Intermediate cover scan captured before choices are consumed.
    pub(crate) struct Scan {
        groups: BTreeMap<GroupKey, (usize, f64)>,
        covered_by_region: Vec<bool>,
        node_secs: Vec<f64>,
    }

    /// Scans chosen covers before emit consumes `cover.best`.
    pub(crate) fn scan_cover(cover: &Cover<'_>, own: &[Option<u32>]) -> Scan {
        let mut groups: BTreeMap<GroupKey, (usize, f64)> = BTreeMap::new();
        let mut covered_by_region = vec![false; cover.tree.len()];
        let mut node_secs = vec![0.0f64; cover.tree.len()];
        for (idx, zone) in own.iter().enumerate() {
            let Some(zone) = *zone else {
                continue;
            };
            let t = T(idx as u32);
            let n = cover.tree.node(t);
            if cover.tree.alias(t) || matches!(n.term, Term::Op(Op::Append(_))) {
                continue;
            }
            let Some(Some(choice)) = &cover.best[idx][zone as usize] else {
                continue;
            };
            let chosen_rule = cover.rules.iter().copied().find(|r| r.name == choice.rule);
            if chosen_rule.is_some_and(|r| r.propose.regions()) {
                for inner in cover.tree.inside(t, &reads(&choice.drafts)) {
                    covered_by_region[inner.0 as usize] = true;
                }
            }
            let secs = self_secs(cover, zone, t, &choice.drafts).unwrap_or(0.0);
            node_secs[idx] = secs;
            let on_fast = chosen_rule.is_some_and(|r| r.propose.fast());
            let kind = cover.device.cluster.kind(zone as usize);
            let mut rejections = Vec::new();
            for rule in &cover.rules {
                if rule.name == choice.rule || rule.propose.regions() {
                    continue;
                }
                let favored = cover.device.favors(rule.name);
                if on_fast && !favored {
                    continue;
                }
                let wants_check = favored
                    || (rule.propose.fast() && applicable(rule, kind, &cover.device.platform));
                if !wants_check {
                    continue;
                }
                if let Some(rej) = check_rule(cover, rule, t, zone, choice.rule, Some(secs)) {
                    rejections.push(rej);
                }
            }
            if !rejections.is_empty() {
                let site = site_label(cover.tree, t);
                let kernel = kernel_names(&choice.drafts);
                merge_finding(
                    &mut groups,
                    false,
                    site,
                    choice.rule,
                    kernel,
                    secs * 1e3,
                    rejections,
                );
            }
        }
        Scan {
            groups,
            covered_by_region,
            node_secs,
        }
    }

    /// Finishes the lowering report once `selected` ops are emitted.
    pub(crate) fn finish_report(mut scan: Scan, cover: &Cover<'_>, selected: &Selected) -> Report {
        let weight_widens = add_widens(&mut scan.groups, selected, cover.device, cover.streams);
        add_regions(
            &mut scan.groups,
            cover,
            &scan.covered_by_region,
            &scan.node_secs,
        );
        let mut findings: Vec<Finding> = scan
            .groups
            .into_iter()
            .map(
                |((region, site, rule, kernel, rejections), (count, ms))| Finding {
                    site,
                    rule,
                    kernel,
                    count,
                    ms,
                    region,
                    rejections,
                },
            )
            .collect();
        findings.sort_by(|a, b| {
            b.ms.total_cmp(&a.ms)
                .then_with(|| b.count.cmp(&a.count))
                .then_with(|| a.site.cmp(&b.site))
        });
        let unused_favor = cover.device.favored().and_then(|favor| {
            let used = selected.ops.iter().any(|op| op.rule == favor);
            (!used).then(|| favor.to_string())
        });
        Report {
            label: String::new(),
            total_ops: selected.ops.len(),
            total_ms: plan_secs(selected, cover.device, cover.streams) * 1e3,
            unused_favor,
            weight_widens,
            findings,
        }
    }
}
