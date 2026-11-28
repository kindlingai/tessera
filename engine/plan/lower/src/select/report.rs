use super::cover::{Cover, EDGE, Variants, fits, grid_mismatch, reads, site_label, writes};
use super::{Home, Selected, Value};
use crate::cost::Device;
use crate::kernels::{AllReduce, GatherRows, ReduceScatter, moved};
use crate::rule::{Draft, Rule, ways};
use crate::tree::T;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use tessera_engine_settings::Backend;
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
    /// Rules that proposed at each chosen graph node.
    pub offered: Vec<(tessera_graph::NodeId, Vec<&'static str>)>,
    /// Costed ways at chosen nodes with more than one.
    pub variants: Vec<Variants>,
}

impl Report {
    /// Whether any site fell off fast path or widened weights.
    pub fn has_fallbacks(&self) -> bool {
        !self.findings.is_empty() || self.unused_favor.is_some() || self.weight_widens > 0
    }

    fn site_count(&self, region: bool) -> usize {
        let mut n = 0;
        for f in &self.findings {
            if f.region == region {
                n += f.count;
            }
        }
        n
    }

    fn site_ms(&self, region: bool) -> f64 {
        let mut ms = 0.0;
        for f in &self.findings {
            if f.region == region {
                ms += f.ms;
            }
        }
        ms
    }

    fn percent(&self, ms: f64) -> f64 {
        (ms / self.total_ms * 100.0).min(100.0)
    }

    fn tag(&self, open: &str, close: &str) -> String {
        match self.label.is_empty() {
            true => String::new(),
            false => format!("{open}{}{close}", self.label),
        }
    }

    /// Count of fallback sites, excluding overlapping region summaries.
    pub fn fallback_sites(&self) -> usize {
        match self.site_count(false) {
            0 => self.site_count(true),
            n => n,
        }
    }

    /// Estimated milliseconds spent on fallback sites.
    pub fn fallback_ms(&self) -> f64 {
        let ms = self.site_ms(false).max(self.site_ms(true));
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
            let pct = self.percent(ms);
            parts.push(format!("est {pct:.0}% of{} step", self.tag(" ", "")));
        }
        let detail = parts.join(", ");
        Some(format!(
            "warning: lowering fell off fast path ({detail}) - pass {flag} for details"
        ))
    }

    /// The detailed report under `explain`, else the warning.
    pub fn notice(&self, explain: bool, flag: &str) -> Option<String> {
        match explain {
            true => Some(self.format(12)),
            false => self.warning(flag),
        }
    }

    /// Detailed multi-line report showing up to `top` fallback site groups.
    pub fn format(&self, top: usize) -> String {
        let head = format!("lowering{}", self.tag(" (", ")"));
        if !self.has_fallbacks() {
            return format!(
                "{head}: on fast path ({} ops, est {:.2} ms)",
                self.total_ops, self.total_ms
            );
        }
        let mut out = String::new();
        let ms = self.fallback_ms();
        let pct = if self.total_ms > 0.0 {
            self.percent(ms)
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
                format!(", {:.0}%", self.percent(f.ms))
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
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct GroupKey {
    region: bool,
    site: String,
    rule: &'static str,
    kernel: String,
    rejections: Vec<Rejection>,
}

type Groups = BTreeMap<GroupKey, (usize, f64)>;

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
        let mismatch = |d: &Draft| {
            let needs = d.kernel.needs().mismatch(platform);
            needs.or_else(|| grid_mismatch(d, kind))
        };
        if let Some(err) = way.iter().find_map(mismatch) {
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
fn merge_finding(groups: &mut Groups, mut key: GroupKey, ms: f64) {
    key.rejections.sort();
    key.rejections.dedup();
    let entry = groups.entry(key).or_insert((0, 0.0));
    entry.0 += 1;
    entry.1 += ms;
}

/// Estimates total seconds across all emitted ops in `selected`.
fn plan_secs(selected: &Selected, device: &Device, streams: bool) -> f64 {
    let mut total = 0.0;
    for op in &selected.ops {
        let out = &selected.values[op.write.0 as usize];
        match op.kernel.name() {
            AllReduce::NAME => {
                let bytes = moved(&*op.kernel, out.len, out.form.bytes(out.len));
                total += device.all_reduce(bytes).unwrap_or_default();
                continue;
            }
            ReduceScatter::NAME => {
                let total_elems = op.kernel.sizes()[1];
                let bytes = moved(&*op.kernel, total_elems, out.form.bytes(total_elems));
                total += device.half_ring(bytes).unwrap_or_default();
                continue;
            }
            GatherRows::NAME => {
                total += device
                    .half_ring(out.form.bytes(out.len))
                    .unwrap_or_default();
                continue;
            }
            _ => {}
        }
        let reads_weight = op.reads.iter().any(|r| {
            matches!(
                selected.values[r.0 as usize].home,
                Home::Weight(_) | Home::Part(..)
            )
        });
        let cold = streams && reads_weight;
        total += device.seconds_ref(op.zone, &op.kernel, out.len, cold);
    }
    total
}

/// Records findings for emitted weight-widening ops.
fn add_widens(groups: &mut Groups, selected: &Selected, device: &Device, streams: bool) -> usize {
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
        let secs = device.seconds_ref(op.zone, &op.kernel, out.len, streams);
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
        let key = GroupKey {
            region: false,
            site,
            rule: "widen",
            kernel: op.kernel.name().to_string(),
            rejections: rej,
        };
        merge_finding(groups, key, secs * 1e3);
    }
    count
}

/// Records findings for graph regions declined by registered region rules.
fn add_regions(
    groups: &mut Groups,
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
        let key = GroupKey {
            region: true,
            site,
            rule: "unfused",
            kernel: "fallback ops".into(),
            rejections,
        };
        merge_finding(groups, key, secs * 1e3);
    }
}

/// Intermediate cover scan captured before choices are consumed.
pub(crate) struct Scan {
    groups: Groups,
    covered_by_region: Vec<bool>,
    node_secs: Vec<f64>,
    offered: Vec<(tessera_graph::NodeId, Vec<&'static str>)>,
    variants: Vec<Variants>,
}

/// Scans chosen covers before emit consumes `cover.best`.
pub(crate) fn scan_cover(cover: &Cover<'_>, own: &[Option<u32>]) -> Scan {
    let mut groups: Groups = BTreeMap::new();
    let mut covered_by_region = vec![false; cover.tree.len()];
    let mut node_secs = vec![0.0f64; cover.tree.len()];
    let mut offered = Vec::new();
    let mut variants = Vec::new();
    for (idx, zone) in own.iter().enumerate() {
        let Some(zone) = *zone else {
            continue;
        };
        let t = T(idx as u32);
        if cover.tree.alias(t) {
            continue;
        }
        let Some(Some(choice)) = &cover.best[idx][zone as usize] else {
            continue;
        };
        offered.push((cover.tree.node(t).from, choice.offered.clone()));
        variants.extend(cover.variants(t, choice));
        if choice.drafts.is_empty() {
            continue;
        }
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
            let wants_check =
                favored || (rule.propose.fast() && applicable(rule, kind, &cover.device.platform));
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
            let key = GroupKey {
                region: false,
                site,
                rule: choice.rule,
                kernel,
                rejections,
            };
            merge_finding(&mut groups, key, secs * 1e3);
        }
    }
    Scan {
        groups,
        covered_by_region,
        node_secs,
        offered,
        variants,
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
        .map(|(key, (count, ms))| Finding {
            site: key.site,
            rule: key.rule,
            kernel: key.kernel,
            count,
            ms,
            region: key.region,
            rejections: key.rejections,
        })
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
        offered: scan.offered,
        variants: scan.variants,
    }
}
