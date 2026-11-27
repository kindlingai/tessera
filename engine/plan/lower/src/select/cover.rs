//! Covers each tree node with the cheapest rule, per zone.

use crate::abi::{Shape, Source};
use crate::cost::Device;
use crate::form::{F32_FORM, in_scope};
use crate::kernels::{
    AllGather, AllReduce, GatherRows, Narrow, ReduceScatter, Scatter, Widen, moved,
};
use crate::plan::{Slot, Step};
use crate::rule::{Draft, Read, Rule, ways};
use crate::select::Home;
use crate::tree::{T, Term, Tree};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use tessera_graph::{Op, PathSeg, Stage};
use tessera_platform::limits::{GRID_ACROSS, GRID_DOWN};
use tessera_sim::ZoneKind;

/// The cheapest cover found at one node, in one zone.
pub(crate) struct Choice {
    pub(crate) rule: &'static str,
    pub(crate) drafts: Vec<Draft>,
    /// Seconds for the drafts and for all they read.
    pub(crate) cost: f64,
    /// Every rule that proposed here.
    pub(crate) offered: Vec<&'static str>,
    /// Each runnable way here: its rule and kernel names.
    pub(crate) ways: Vec<Way>,
}

/// One way a rule offers: its rule, then its kernels.
pub(crate) type Way = (&'static str, Vec<&'static str>);

/// A cover's reads of tree nodes.
pub(crate) fn reads(drafts: &[Draft]) -> Vec<T> {
    let node = |r: &Read| match r {
        Read::Node(t) => Some(*t),
        Read::Draft(_) | Read::Kept(_) => None,
    };
    drafts
        .iter()
        .flat_map(|d| d.reads.iter().filter_map(node))
        .collect()
}

/// The outputs a cover writes, besides its own root.
pub(crate) fn writes(drafts: &[Draft]) -> Vec<T> {
    drafts
        .iter()
        .flat_map(|d| d.writes.into_iter().chain(d.also.iter().map(|(t, _)| *t)))
        .collect()
}

/// Leaves and outputs lie where the runner feeds and reads.
pub(crate) const EDGE: u32 = 0;

pub(crate) struct Cover<'a> {
    pub(crate) tree: &'a Tree<'a>,
    pub(crate) device: &'a Device,
    pub(crate) rules: Vec<&'static Rule>,
    /// The rule the device favors, if it names one.
    pub(crate) favorite: Option<&'static Rule>,
    /// Per node and zone: the choice there, if any.
    pub(crate) best: Vec<Vec<Option<Option<Choice>>>>,
    /// Per root: the zones its readers run in.
    pub(crate) wanted: Vec<Vec<u32>>,
    /// Nodes without a proposing rule, deepest first.
    pub(crate) bare: Vec<T>,
    /// Whether weights are too many for the cache.
    pub(crate) streams: bool,
    /// Each graph node's region label, when the recipe scopes choices.
    pub(crate) labels: Option<Vec<String>>,
    /// Each way's rule, kernels and seconds by node, when explaining.
    pub(crate) priced: Option<HashMap<T, Vec<Priced>>>,
}

/// A way's rule, kernels and estimated cost.
pub(crate) type Priced = (&'static str, Vec<&'static str>, f64, usize);

/// The costed ways at one chosen node, cheapest first.
#[derive(Clone, Debug, PartialEq)]
pub struct Variants {
    pub node: tessera_graph::NodeId,
    pub site: String,
    /// The kernels the plan chose here.
    pub chosen: Vec<&'static str>,
    /// Each way's rule, kernels, estimated milliseconds and graph ops covered.
    pub ways: Vec<Priced>,
}

impl<'a> Cover<'a> {
    /// The rule and kernel favored at `t`, and names barred.
    fn scope(&self, t: T) -> (Option<&'a str>, Option<&'a str>, Vec<&'a str>) {
        let (mut favor, mut pick, mut barred) = (None, None, Vec::new());
        let Some(labels) = &self.labels else {
            return (favor, pick, barred);
        };
        let label = &labels[self.tree.node(t).from.0 as usize];
        for s in &self.device.scoped {
            if in_scope(&s.at, label) {
                favor = s.favor.as_deref().or(favor);
                pick = s.pick.as_deref().or(pick);
                barred.extend(s.barred.iter().map(String::as_str));
            }
        }
        (favor, pick, barred)
    }

    /// The costed ways at `t`, when there are several.
    pub(crate) fn variants(&self, t: T, choice: &Choice) -> Option<Variants> {
        let mut ways = self.priced.as_ref()?.get(&t)?.clone();
        if ways.len() < 2 {
            return None;
        }
        ways.sort_by(|a, b| a.2.total_cmp(&b.2));
        let chosen = choice.drafts.iter().map(|d| d.kernel.name()).collect();
        let node = self.tree.node(t).from;
        Some(Variants {
            node,
            site: site_label(self.tree, t),
            chosen,
            ways,
        })
    }

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

    /// A root's cheapest zone, counting its readers' moves.
    pub(crate) fn settle(&mut self, t: T) -> Option<u32> {
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
        // A zero-copy view passes its argument on.
        if self.tree.alias(t) {
            let cost = self.need(n.args[0], zone)?;
            return Some(Choice {
                rule: "append",
                drafts: Vec::new(),
                cost,
                offered: Vec::new(),
                ways: Vec::new(),
            });
        }
        let append = match n.term {
            Term::Op(Op::Append(index)) => Some(index),
            _ => None,
        };
        if append.is_some() && zone != EDGE {
            return None;
        }
        if let Term::Stage(
            stage
            @ (Stage::AllReduce | Stage::AllGather | Stage::ReduceScatter | Stage::GatherRows),
        ) = &n.term
        {
            let len = self.tree.elements(t);
            let src = n.args[0];
            let (rule, draft, moved_bytes) = match stage {
                Stage::AllReduce => {
                    let wire = self.tree.wire(t)?;
                    let sum = AllReduce { len, wire };
                    let bytes = moved(&sum, len, self.tree.bytes(t));
                    (AllReduce::NAME, Draft::new(sum, [src], len), bytes)
                }
                Stage::AllGather => {
                    let rows = len / n.shape.last().copied().unwrap_or(1).max(1);
                    let gather = AllGather { len, rows };
                    let bytes = moved(&gather, len, self.tree.bytes(t));
                    (AllGather::NAME, Draft::new(gather, [src], len), bytes)
                }
                Stage::ReduceScatter => {
                    let wire = self.tree.wire(t)?;
                    let total = self.tree.elements(src);
                    let whole_rows = self.tree.node(src).shape[0];
                    let width = n.shape[1..].iter().product::<usize>();
                    let start = self.tree.parts.rows(whole_rows, self.tree.rank as usize).0 * width;
                    let rs = ReduceScatter { start, total, wire };
                    let bytes = moved(&rs, total, self.tree.bytes(src));
                    (ReduceScatter::NAME, Draft::new(rs, [src], len), bytes)
                }
                Stage::GatherRows => {
                    let whole_rows = n.shape[0];
                    let width = n.shape[1..].iter().product::<usize>();
                    let start = self.tree.parts.rows(whole_rows, self.tree.rank as usize).0 * width;
                    let gr = GatherRows { len, start };
                    let bytes = self.tree.bytes(t);
                    (GatherRows::NAME, Draft::new(gr, [src], len), bytes)
                }
                _ => unreachable!(),
            };
            let moved = match stage {
                Stage::ReduceScatter | Stage::GatherRows => self.device.half_ring(moved_bytes)?,
                _ => self.device.all_reduce(moved_bytes)?,
            };
            let drafts = vec![draft];
            let cast = self.widening(zone, &drafts) + self.narrowing(zone, t, &drafts);
            let cost = moved + cast + self.need(src, zone)?;
            return Some(Choice {
                rule,
                drafts,
                cost,
                offered: vec![rule],
                ways: Vec::new(),
            });
        }
        let kind = self.device.cluster.kind(zone as usize);
        let (favor, pick, barred) = self.scope(t);
        let device = self.device;
        let picked =
            |drafts: &[Draft]| pick.is_some_and(|p| drafts.iter().any(|d| d.kernel.name() == p));
        let favors = |name: &str| device.favors(name) || favor == Some(name);
        // The winning cover gets a grid check. Tall ones retry.
        let mut skip = Vec::new();
        let (best, offered, runnable, proposed) = loop {
            let (mut best, mut fallback): (Option<Choice>, Option<Choice>) = (None, None);
            let (mut offered, mut runnable) = (Vec::new(), Vec::new());
            let mut proposed = false;
            let (mut won, mut fall_won): (Option<usize>, Option<usize>) = (None, None);
            let mut seen = 0;
            // Default scatter cover for kept appends, competing with fused rules.
            if let Some(index) = append {
                proposed = true;
                if let Some(below) = self.need(n.args[0], zone)
                    && let Some(scatter) = Scatter::at(self.tree, t, index)
                {
                    let len = scatter.rows * scatter.width;
                    let cost = below + self.device.seconds_in(zone, &scatter, len, false);
                    offered.push("append");
                    best = Some(Choice {
                        rule: "append",
                        drafts: Vec::new(),
                        cost,
                        offered: Vec::new(),
                        ways: Vec::new(),
                    });
                }
            }
            for rule in self.rules.clone() {
                let Some(drafts) = rule.propose.at(self.tree, t) else {
                    continue;
                };
                proposed = true;
                if !drafts.iter().all(|d| fits(d, kind)) {
                    continue;
                }
                offered.push(rule.name);
                for drafts in ways(drafts) {
                    seen += 1;
                    if skip.contains(&seen) {
                        continue;
                    }
                    // Kept state lives in EDGE so mutations stay in place.
                    let touches_kept = drafts
                        .iter()
                        .any(|d| d.reads.iter().any(|r| matches!(r, Read::Kept(_))));
                    if touches_kept && zone != EDGE {
                        continue;
                    }
                    // Fused appends write kept rows directly in their form.
                    if append.is_some() {
                        let Some(last) = drafts.last() else {
                            continue;
                        };
                        if last.kernel.overwrites()
                            || (n.form.narrow() && !last.kernel.emits_at(0, n.form))
                            || !last.also.is_empty()
                        {
                            continue;
                        }
                    }
                    let names =
                        std::iter::once(rule.name).chain(drafts.iter().map(|d| d.kernel.name()));
                    if names.into_iter().any(|name| barred.contains(&name)) {
                        continue;
                    }
                    // A kernel the device cannot run is not a way.
                    let platform = &self.device.platform;
                    if drafts.iter().any(|d| !d.kernel.needs().fits(platform)) {
                        continue;
                    }
                    let swallowed = self.swallows(rule, t, &drafts, kind);
                    if !swallowed {
                        runnable
                            .push((rule.name, drafts.iter().map(|d| d.kernel.name()).collect()));
                    }
                    let Some(mut cost) = drafts.iter().map(|d| self.seconds(zone, d)).sum() else {
                        continue;
                    };
                    cost += self.widening(zone, &drafts) + self.narrowing(zone, t, &drafts);
                    let below = reads(&drafts).into_iter().map(|r| self.need(r, zone));
                    let Some(below) = below.sum::<Option<f64>>() else {
                        continue;
                    };
                    cost += below;
                    // Roots a cover absorbs no longer cost their own.
                    let Some(absorbed) = self.tree.absorbs(t, &reads(&drafts), &writes(&drafts))
                    else {
                        continue;
                    };
                    if !rule.propose.regions() && absorbed.iter().any(|r| self.tree.end(*r)) {
                        continue;
                    }
                    // The zone must also cover each root the drafts absorb.
                    let own = absorbed.into_iter().map(|r| self.solve(r, zone));
                    let Some(own) = own.sum::<Option<f64>>() else {
                        continue;
                    };
                    cost -= own;
                    let ops = self
                        .priced
                        .is_some()
                        .then(|| self.tree.covered(t, &reads(&drafts)).len());
                    if let (Some(priced), Some(ops)) = (&mut self.priced, ops) {
                        let kernels: Vec<_> = drafts.iter().map(|d| d.kernel.name()).collect();
                        let ways = priced.entry(t).or_default();
                        match ways.iter_mut().find(|w| w.0 == rule.name && w.1 == kernels) {
                            Some(w) => w.2 = w.2.min(cost * 1e3),
                            None => ways.push((rule.name, kernels, cost * 1e3, ops)),
                        }
                    }
                    // A picked kernel wins, then a favored rule, then cost.
                    let (mine, favored) = (picked(&drafts), favors(rule.name));
                    let beaten = |b: &Choice| match (picked(&b.drafts), mine) {
                        (true, false) => false,
                        (false, true) => true,
                        _ => match (b.rule == rule.name, favors(b.rule)) {
                            (true, _) => cost < b.cost,
                            (false, true) => false,
                            (false, false) => favored || cost < b.cost,
                        },
                    };
                    let (slot, tag) = if swallowed {
                        (&mut fallback, &mut fall_won)
                    } else {
                        (&mut best, &mut won)
                    };
                    if slot.as_ref().is_none_or(beaten) {
                        *tag = Some(seen);
                        *slot = Some(Choice {
                            rule: rule.name,
                            drafts,
                            cost,
                            offered: Vec::new(),
                            ways: Vec::new(),
                        });
                    }
                }
            }
            let (picked, picked_won) = match best {
                Some(b) => (Some(b), won),
                None => (fallback, fall_won),
            };
            let tall = |c: &Choice| c.drafts.iter().any(|d| grid_mismatch(d, kind).is_some());
            match picked_won {
                Some(at) if picked.as_ref().is_some_and(tall) => skip.push(at),
                _ => break (picked, offered, runnable, proposed),
            }
        };
        if !proposed && !self.bare.contains(&t) {
            self.bare.push(t);
        }
        let ways = runnable;
        best.map(|b| Choice { offered, ways, ..b })
    }

    /// The deepest node below `root` without a proposing rule.
    pub(crate) fn uncovered(&self, root: T) -> T {
        let mut below = vec![false; self.tree.len()];
        let mut stack = vec![root];
        while let Some(t) = stack.pop() {
            below[t.0 as usize] = true;
            for &a in &self.tree.node(t).args {
                if !self.tree.leaf(a) && !self.tree.root(a) {
                    stack.push(a);
                }
            }
        }
        let found = self.bare.iter().find(|t| below[t.0 as usize]);
        found.copied().unwrap_or(root)
    }

    /// Seconds for one draft. A module's sum costs the engine's.
    pub(crate) fn seconds(&self, zone: u32, d: &Draft) -> Option<f64> {
        if d.kernel.name() == AllReduce::NAME {
            let form = d.writes.map_or(F32_FORM, |w| self.tree.node(w).form);
            return self
                .device
                .all_reduce(moved(&*d.kernel, d.len, form.bytes(d.len)));
        }
        let weight = |r: &Read| matches!(r, Read::Node(t) if self.tree.weighed(*t));
        let cold = self.streams && d.reads.iter().any(weight);
        Some(self.device.seconds_ref(zone, &d.kernel, d.len, cold))
    }

    /// Seconds to widen reads the drafts cannot take as stored.
    pub(crate) fn widening(&self, zone: u32, drafts: &[Draft]) -> f64 {
        let mut seconds = 0.0;
        for d in drafts {
            for (i, r) in d.reads.iter().enumerate() {
                let (form, len, cold) = match r {
                    Read::Node(t) => (
                        self.tree.node(*t).form,
                        self.tree.elements(*t),
                        self.streams && self.tree.weighed(*t),
                    ),
                    Read::Kept(index) => match self.tree.kept_rows(*index) {
                        Some((t, _)) => (self.tree.node(t).form, self.tree.elements(t), false),
                        None => continue,
                    },
                    Read::Draft(_) => continue,
                };
                if form.narrow() && !d.kernel.accepts(i, form) {
                    let widen = Widen { len, form };
                    seconds += self.device.seconds_in(zone, &widen, len, cold);
                }
            }
        }
        seconds
    }

    /// Seconds to narrow outputs the drafts write as floats.
    pub(crate) fn narrowing(&self, zone: u32, t: T, drafts: &[Draft]) -> f64 {
        let mut seconds = 0.0;
        let last = drafts.len().saturating_sub(1);
        for (i, d) in drafts.iter().enumerate() {
            let form = match d.writes {
                Some(w) => self.tree.node(w).form,
                None if i == last => self.tree.node(t).form,
                None => F32_FORM,
            };
            if form.narrow() && !d.kernel.emits_at(0, form) {
                let narrow = Narrow { len: d.len, form };
                seconds += self.device.seconds_in(zone, &narrow, d.len, false);
            }
            for (k, &(w, len)) in d.also.iter().enumerate() {
                let form = self.tree.node(w).form;
                if form.narrow() && !d.kernel.emits_at(k + 1, form) {
                    let narrow = Narrow { len, form };
                    seconds += self.device.seconds_in(zone, &narrow, len, false);
                }
            }
        }
        seconds
    }

    /// Whether a cover hides a node the favorite takes here.
    pub(crate) fn swallows(&self, rule: &Rule, t: T, drafts: &[Draft], kind: ZoneKind) -> bool {
        let Some(favorite) = self.favorite.filter(|f| f.name != rule.name) else {
            return false;
        };
        let inside = self.tree.inside(t, &reads(drafts));
        for n in inside {
            if n == t {
                continue;
            }
            let Some(drafts) = favorite.propose.at(self.tree, n) else {
                continue;
            };
            let platform = &self.device.platform;
            let runs = |d: &Draft| {
                fits(d, kind) && d.kernel.needs().fits(platform) && grid_mismatch(d, kind).is_none()
            };
            if drafts.iter().all(runs) {
                return true;
            }
        }
        false
    }

    /// Roots the chosen cover of `t` in `zone` computes.
    pub(crate) fn absorbed(&self, t: T, zone: u32) -> Vec<T> {
        match &self.best[t.0 as usize][zone as usize] {
            Some(Some(c)) if !c.drafts.is_empty() => {
                let (reads, writes) = (reads(&c.drafts), writes(&c.drafts));
                self.tree.absorbs(t, &reads, &writes).unwrap_or_default()
            }
            _ => Vec::new(),
        }
    }

    /// The reads of `t`'s cover in `zone`, each placed.
    pub(crate) fn below(&mut self, t: T, zone: u32) -> Vec<(T, u32)> {
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

/// A tree runs on a CPU too, in lockstep.
pub(crate) fn fits(d: &Draft, kind: ZoneKind) -> bool {
    d.kernel.runs_on() == kind || (kind == ZoneKind::Cpu && made(d))
}

/// Why a draft's grid is too large for a GPU.
pub(crate) fn grid_mismatch(d: &Draft, kind: ZoneKind) -> Option<String> {
    if kind != ZoneKind::Gpu {
        return None;
    }
    let shapes = match d.kernel.shape() {
        Some(shape) => vec![shape],
        None => stand_in(d)
            .launches()
            .into_iter()
            .map(|l| l.shape)
            .collect(),
    };
    for shape in shapes {
        let Shape::Grid { across, down, .. } = shape else {
            continue;
        };
        if down > GRID_DOWN {
            return Some(format!("grid height {down} exceeds {GRID_DOWN}"));
        }
        if across > GRID_ACROSS {
            return Some(format!("grid width {across} exceeds {GRID_ACROSS}"));
        }
    }
    None
}

/// A step whose shapes read sizes, not buffers.
fn stand_in(d: &Draft) -> Step {
    let slot = Slot {
        buffer: 0,
        offset: 0,
        len: d.len,
        form: F32_FORM,
        dims: 0,
        shape: [0; 4],
    };
    Step {
        kernel: d.kernel.clone(),
        rule: "",
        site: None,
        reads: vec![slot; d.reads.len()],
        write: slot,
        also: vec![slot; d.also.len() + 1],
        covers: Vec::new(),
        zone: 0,
        lane: 0,
        table: Some(slot),
        index_table: None,
    }
}

/// Whether a kernel is a tree, asked once per name.
fn made(d: &Draft) -> bool {
    // Building a body costs more than the cover.
    static MADE: OnceLock<Mutex<HashMap<&'static str, bool>>> = OnceLock::new();
    let name = d.kernel.name();
    let mut known = MADE.get_or_init(Mutex::default).lock().expect("made");
    *known
        .entry(name)
        .or_insert_with(|| matches!(d.kernel.source(), Source::Made(_)))
}

/// Concise human description of tree node `t`.
pub(crate) fn site_label(tree: &Tree<'_>, t: T) -> String {
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

/// Stable graph location for `t`: region, weight, and node id.
pub(crate) fn graph_site(tree: &Tree<'_>, labels: Option<&[String]>, t: T) -> String {
    let n = tree.node(t);
    let id = n.from.0;
    let region = labels
        .filter(|_| !tree.g.regions.is_empty())
        .and_then(|l| l.get(id as usize))
        .map(String::as_str);
    let weight = n.args.iter().find_map(|&a| match tree.node(a).term {
        Term::Leaf(Home::Weight(w) | Home::Part(w, _)) => {
            let p = &tree.g.weights[w as usize].path;
            (!p.is_empty()).then(|| PathSeg::dotted(p))
        }
        _ => None,
    });
    match (region, weight) {
        (Some(r), Some(w)) => format!("{r}:{w}@n{id}"),
        (Some(r), None) => format!("{r}@n{id}"),
        (None, Some(w)) => format!("{w}@n{id}"),
        (None, None) => format!("n{id}"),
    }
}
