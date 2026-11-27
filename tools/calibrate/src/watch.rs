//! Runs a plan again whenever a kernel file changes.

use crate::grid::{bounded, codes};
use crate::routing::{DEFAULT_SKEW, extract_router, prefill_kept, routed_feeds};
use crate::watch_args::{
    ServeDims, USAGE, WatchArgs, build_options, build_recipe, parse_watch, user_rules,
};
use crate::watch_report::{
    Meta, Report, Row, build_report, format_diff, format_table, load_report, write_json,
};
use std::error::Error;
use std::path::{Path, PathBuf};
use std::thread::sleep;
use std::time::{Duration, Instant};
use tessera_bundle::names::Rule as NameRule;
use tessera_bundle::parts::{self, Part, Spec};
use tessera_engine::comm::{Error as CommError, Transport};
use tessera_graph::DimName::{self, Context, Rows, Seqs};
use tessera_graph::{DType, Graph, Op, Parts, PathSeg, Region, Size, shard_mode};
use tessera_lower::form::{BF16_FORM, F32_FORM, FP8_FORM, Load, named};
use tessera_lower::{
    Cluster, Device, Launching, Pins, Plan, Recipe, Settings, Source, Step, Tensor, View, Weight,
    Wires, ZoneKind, cut_region, lower,
};
use tessera_run::{Outcome, Rank, Reload, Run, open};
use tessera_serve::start::Options;
use tessera_testkit::graphs::{attention, gqa_attend_rows, linear, mla_attend_rows, moe_routed};
use tessera_testkit::{Noise, weights};

/// Time between checks for changed files.
const POLL: Duration = Duration::from_millis(20);
/// Device seconds to spend timing one step.
const SPEND: f64 = 0.01;
const MOST: f64 = 64.0;
/// Seconds of whole runs first, so clocks rise from idle.
const WARM: Duration = Duration::from_millis(100);

/// A graph at fixed sizes, with its weight forms.
pub(crate) struct Subject {
    pub(crate) g: Graph,
    pub(crate) sizes: Vec<(DimName, u64)>,
    pub(crate) pins: Pins,
    /// Weights from a checkpoint. Without them the run makes noise.
    pub(crate) loads: Option<Vec<Weight>>,
    /// Zipf skew exponent for MoE routing feeds.
    pub(crate) skew: f64,
}

/// Numbers in `args`, or the usage.
fn numbers(args: &[String]) -> Result<Vec<u64>, Box<dyn Error>> {
    let mut n = Vec::new();
    for arg in args {
        n.push(arg.parse::<u64>().map_err(|_| USAGE)?);
    }
    Ok(n)
}

/// `Rows=64` or `rows=64` as a size.
fn sized(arg: &str) -> Option<(DimName, u64)> {
    let (name, n) = arg.split_once('=')?;
    let mut chars = name.chars();
    let first = chars.next()?.to_uppercase();
    let name = serde_json::Value::String(first.chain(chars).collect());
    Some((serde_json::from_value(name).ok()?, n.parse().ok()?))
}

/// Each region as `<kind>.<n>`, counted by kind in graph order.
fn regions(g: &Graph) -> Vec<String> {
    let mut names = Vec::new();
    for (at, r) in g.regions.iter().enumerate() {
        let n = g.regions[..at].iter().filter(|o| o.kind == r.kind).count();
        names.push(format!("{:?}.{n}", r.kind).to_lowercase());
    }
    names
}

/// The path that the weights `r` reads share.
fn label(g: &Graph, r: &Region) -> String {
    let mut shared: Option<&[PathSeg]> = None;
    for id in &r.inputs {
        let Op::Weight(w) = g.node(*id).op else {
            continue;
        };
        let path = &g.weights[w as usize].path[..];
        let same = shared.map_or(path.len(), |s| {
            s.iter().zip(path).take_while(|(a, b)| a == b).count()
        });
        shared = Some(&path[..same]);
    }
    PathSeg::dotted(shared.unwrap_or_default())
}

/// A bundle's error, which may cross threads.
type Failed = Box<dyn Error + Send + Sync>;

/// Prints each part's dims and regions.
fn list_parts(dir: &Path, names: &[String], spec: &Spec<'_>) -> Result<(), Failed> {
    for name in names {
        let g = parts::open_with(dir, name, spec)?.graph;
        println!(
            "{name}: {} weights, {} nodes",
            g.weights.len(),
            g.nodes.len()
        );
        let mut dims = Vec::new();
        for d in &g.dims {
            dims.push(match d.size {
                Size::Bounded(n) => format!("{:?}<={n}", d.name),
                Size::Fixed(n) => format!("{:?}={n}", d.name),
            });
        }
        println!("  {}", dims.join(" "));
        for (r, region) in g.regions.iter().zip(regions(&g)) {
            let nodes = r.end.0 - r.start.0;
            println!("  {region:<14} {nodes:>6} nodes  {}", label(&g, r));
        }
    }
    Ok(())
}

/// Pins unpinned sums in `g` using recipe wires and `--sum-wire`.
fn resolve_wires(g: Graph, wires: &Wires, wire: Option<DType>) -> Graph {
    let g = wires.pin(g);
    g.wired(wire.unwrap_or(DType::F32))
}

/// Applies recipe, rank topology and shares to `base`.
fn configure_device(base: Device, recipe: &Recipe, parts: &Parts, rank: u32, gpu: bool) -> Device {
    let mut d = base.with(recipe);
    if parts.ranks() > 1 {
        let kind = if gpu { ZoneKind::Gpu } else { ZoneKind::Cpu };
        let ranks = parts.ranks() as u32;
        d = d
            .zoned(Cluster::mesh(ranks as usize, kind))
            .ranked((0..ranks).collect())
            .uneven(parts.0.clone())
            .as_rank(rank);
    }
    d
}

/// A prepared watch target ready to lower and load weights.
pub(crate) struct Target {
    pub(crate) g: Graph,
    pub(crate) sizes: Vec<(DimName, u64)>,
    pub(crate) recipe: Recipe,
    part: Option<Part>,
    loads: Option<Vec<Weight>>,
    skew: f64,
    region: Option<String>,
}

/// Parses model args into a sharded, region-cut target.
pub(crate) fn model_target(
    dir: &Path,
    rest: &[String],
    options: &Options,
    dims: ServeDims,
    user: &[NameRule],
    args: &WatchArgs,
    parts: &Parts,
) -> Result<Option<Target>, Failed> {
    let (mut part, mut region, mut listed, mut given) = (None, None, false, Vec::new());
    let mut skew = DEFAULT_SKEW;
    let mut it = rest.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--list" => listed = true,
            "--region" => region = Some(it.next().ok_or(USAGE)?.clone()),
            "--skew" => skew = it.next().ok_or(USAGE)?.parse::<f64>().map_err(|_| USAGE)?,
            _ if arg.contains('=') => given.push(sized(arg).ok_or(USAGE)?),
            _ if part.is_none() => part = Some(arg.clone()),
            _ => return Err(USAGE.into()),
        }
    }
    // Precedence: positional Dim=value > CLI serve flags > recipe serve flags.
    let mut resolved: Vec<(DimName, u64)> = Vec::new();
    if let Some(slots) = dims.slots {
        resolved.push((Seqs, slots));
    }
    if let Some(context) = dims.context {
        resolved.push((Context, context));
    }
    for &(d, v) in &given {
        if let Some(entry) = resolved.iter_mut().find(|(k, _)| *k == d) {
            entry.1 = v;
        } else {
            resolved.push((d, v));
        }
    }
    let size = |n: DimName| resolved.iter().find(|(d, _)| *d == n).map(|(_, v)| *v);
    let spec = Spec {
        overrides: &options.config,
        user,
        manifest: options.manifest(),
        module: options.module.as_deref(),
        size: &size,
    };
    let all = parts::names_with(dir, &options.config, options.module.as_deref())?;
    if listed {
        list_parts(dir, &part.map_or(all, |p| vec![p]), &spec)?;
        return Ok(None);
    }
    let part_name = match (part, all.as_slice()) {
        (Some(p), _) => p,
        (None, [one]) => one.clone(),
        _ => return Err(format!("pick a part: {}", all.join(", ")).into()),
    };
    let opened = parts::open_with(dir, &part_name, &spec)?;
    let mut recipe = build_recipe(options, opened.pins.clone(), opened.config.clone(), args);
    // Shard the full model graph before cutting one region.
    let whole = if parts.ranks() > 1 {
        let shares = recipe.pins.shares(&opened.graph);
        let sharded = shard_mode(&opened.graph, parts, &shares, options.ranks.moe_tp())?;
        resolve_wires(sharded.graph, &recipe.sums, options.ranks.sum_wire())
    } else {
        opened.graph.clone()
    };
    let g = match &region {
        Some(name) => {
            let at = regions(&whole).iter().position(|n| n == name);
            let at = at.ok_or_else(|| format!("{part_name} has no region {name}; see --list"))?;
            cut_region(&whole, &whole.regions[at])
        }
        None => whole,
    };
    let (loads, part_holder) = if opened.pins.is_some() {
        (None, Some(opened))
    } else {
        let (weights, pins) = opened.weights(&g)?;
        recipe = build_recipe(options, Some(pins), opened.config.clone(), args);
        (Some(weights), None)
    };
    let mut sizes = resolved;
    for d in &g.dims {
        if let Size::Bounded(n) = d.size
            && !sizes.iter().any(|(k, _)| *k == d.name)
        {
            let value = match (d.name, dims.max_batch_rows) {
                (Rows, Some(limit)) => limit.min(n),
                _ => n,
            };
            sizes.push((d.name, value));
        }
    }
    Ok(Some(Target {
        g,
        sizes,
        recipe,
        part: part_holder,
        loads,
        skew,
        region,
    }))
}

/// A part of the model in `args[0]`, or one region.
fn model(args: &[String]) -> Result<Option<Subject>, Failed> {
    let (dir, rest) = args.split_first().ok_or(USAGE)?;
    let options = Options::default();
    let empty = WatchArgs::default();
    let Some(t) = model_target(
        Path::new(dir),
        rest,
        &options,
        ServeDims::default(),
        &[],
        &empty,
        &Parts::even(1),
    )?
    else {
        return Ok(None);
    };
    let (loads, pins) = match (t.loads, t.part) {
        (Some(w), _) => (w, t.recipe.pins),
        (None, Some(p)) => p.weights(&t.g)?,
        (None, None) => unreachable!(),
    };
    Ok(Some(Subject {
        g: t.g,
        sizes: t.sizes,
        pins,
        loads: Some(loads),
        skew: t.skew,
    }))
}

fn kv_form(name: &str) -> Result<&'static dyn Load, Box<dyn Error>> {
    match name {
        "f32" => Ok(F32_FORM),
        "bf16" => Ok(BF16_FORM),
        "fp8" => Ok(FP8_FORM),
        _ => Err(format!("unknown kv form `{name}` (use f32, bf16, fp8)").into()),
    }
}

fn parse_kv_flag(rest: &[String]) -> Result<&'static dyn Load, Box<dyn Error>> {
    match rest {
        [] => Ok(BF16_FORM),
        [flag, value] if flag == "--kv" => kv_form(value),
        _ => Err(USAGE.into()),
    }
}

fn moe_subject(rest: &[String]) -> Result<Subject, Box<dyn Error>> {
    if rest.len() < 5 {
        return Err(USAGE.into());
    }
    let nums = numbers(&rest[..5])?;
    let [experts, top_k, hidden, ffn, rows] = nums.as_slice() else {
        return Err(USAGE.into());
    };
    let (mut form_name, mut skew, mut gated) = ("bf16".to_string(), DEFAULT_SKEW, true);
    let mut it = rest[5..].iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--form" => form_name = it.next().ok_or(USAGE)?.clone(),
            "--skew" => skew = it.next().ok_or(USAGE)?.parse::<f64>().map_err(|_| USAGE)?,
            "--plain" => gated = false,
            _ => return Err(USAGE.into()),
        }
    }
    let form = named(&form_name).ok_or_else(|| format!("unknown form `{form_name}`"))?;
    let pins = match form.name() {
        "bf16" => Pins::all(BF16_FORM)
            .activations(BF16_FORM)
            .outputs(BF16_FORM),
        _ => Pins::all(form),
    };
    Ok(Subject {
        g: moe_routed(&bounded(&[Rows]), [*experts, *top_k, *hidden, *ffn], gated),
        sizes: vec![(Rows, *rows)],
        pins,
        loads: None,
        skew,
    })
}

/// Builds synthetic weights, packing forms as needed.
pub(crate) fn make_loads(g: &Graph, pins: &Pins, noise: &mut Noise) -> Vec<Weight> {
    let mut out = Vec::with_capacity(g.weights.len());
    for (w, data) in g.weights.iter().zip(weights(g, noise)) {
        let form = pins.form(&w.path);
        match form.packs() {
            true => out.push(Weight::Floats(data)),
            false => out.push(Weight::Packed(form, codes(form, &data, noise))),
        }
    }
    out
}

pub(crate) fn subject(args: &[String]) -> Result<Option<Subject>, Box<dyn Error>> {
    let (kind, rest) = args.split_first().ok_or(USAGE)?;
    match kind.as_str() {
        "model" => return model(rest).map_err(|e| e as Box<dyn Error>),
        "moe" => return Ok(Some(moe_subject(rest)?)),
        "mla" => {
            if rest.len() < 3 {
                return Err(USAGE.into());
            }
            let nums = numbers(&rest[..3])?;
            let kv = parse_kv_flag(&rest[3..])?;
            let pins = Pins {
                kept: kv,
                ..Pins::default()
            };
            return Ok(Some(Subject {
                g: mla_attend_rows(&bounded(&[Rows]), nums[0], nums[2], 1),
                sizes: vec![(Rows, nums[1]), (Context, nums[2])],
                pins,
                loads: None,
                skew: DEFAULT_SKEW,
            }));
        }
        "gqa" => {
            if rest.len() < 4 {
                return Err(USAGE.into());
            }
            let nums = numbers(&rest[..4])?;
            let kv = parse_kv_flag(&rest[4..])?;
            let pins = Pins {
                kept: kv,
                ..Pins::default()
            };
            return Ok(Some(Subject {
                g: gqa_attend_rows(&bounded(&[Rows]), [nums[0], nums[1], 128], nums[3], 1),
                sizes: vec![(Rows, nums[2]), (Context, nums[3])],
                pins,
                loads: None,
                skew: DEFAULT_SKEW,
            }));
        }
        _ => {}
    }
    let (default, matmul, attend) = (Pins::default(), [Rows], [Seqs, Rows, Context]);
    Ok(Some(match (kind.as_str(), numbers(rest)?.as_slice()) {
        ("gemm", [m, k, width]) => Subject {
            g: linear(&bounded(&matmul), *k, *width),
            sizes: vec![(Rows, *m)],
            pins: default,
            loads: None,
            skew: DEFAULT_SKEW,
        },
        ("attend", [heads, kv, width, queries, keys]) => Subject {
            g: attention(&bounded(&attend), [*heads, *kv, *width]),
            sizes: vec![(Seqs, 1), (Rows, *queries), (Context, *keys)],
            pins: default,
            loads: None,
            skew: DEFAULT_SKEW,
        },
        _ => return Err(USAGE.into()),
    }))
}

fn entry(step: &Step) -> String {
    match step.kernel.source() {
        Source::Fixed { entry, .. } => entry.to_string(),
        Source::Made(body) => body.entry,
    }
}

/// A row per distinct entry, in plan order.
fn rows(plan: &Plan, device: &Device) -> Vec<Row> {
    let mut rows: Vec<Row> = Vec::new();
    for (i, step) in plan.steps.iter().enumerate() {
        let name = entry(step);
        let at = match rows.iter().position(|r| r.entry == name) {
            Some(at) => at,
            None => {
                rows.push(Row {
                    entry: name,
                    rule: step.rule,
                    steps: Vec::new(),
                    seconds: 0.0,
                    planned: 0.0,
                    bound: Some(0.0),
                    loaded: false,
                    attrs: None,
                });
                rows.len() - 1
            }
        };
        let (kernel, len) = (&*step.kernel, step.write.len);
        let row = &mut rows[at];
        row.steps.push(i);
        row.planned += device.step_seconds(plan, step).unwrap_or(0.0);
        row.bound = row.bound.zip(device.bound(kernel, len)).map(|(a, b)| a + b);
    }
    rows
}

/// Single-device loopback transport for timing one sharded rank.
struct DryRank {
    rank: u32,
    ranks: u32,
}

impl Transport for DryRank {
    fn rank(&self) -> u32 {
        self.rank
    }

    fn ranks(&self) -> u32 {
        self.ranks
    }

    fn exchange(&mut self, mine: &[u8]) -> Result<Vec<Vec<u8>>, CommError> {
        Ok(vec![mine.to_vec(); self.ranks as usize])
    }
}

/// Populates paged KV tables with valid sequential page ids.
fn init_tables(runner: &mut dyn Run, plan: &Plan) -> Result<(), Box<dyn Error>> {
    for (slot, pool) in [
        (plan.table, plan.pool_pages),
        (plan.index_table, plan.index_pool_pages),
    ] {
        let Some(slot) = slot else { continue };
        let cap = if pool > 0 { pool } else { slot.len }.max(1);
        let ids: Vec<u32> = (0..slot.len).map(|i| (i % cap) as u32).collect();
        let tensor = Tensor::u32(vec![slot.len], ids);
        runner.write(slot, tensor.view())?;
    }
    Ok(())
}

/// Opens a single-rank or dry-ranked runner and loads weights.
fn open_runner(
    plan: Plan,
    loads: &[Weight],
    settings: &Settings,
    launch: Launching,
) -> Result<Box<dyn Run>, Box<dyn Error>> {
    let (rank, ranks) = (plan.rank, plan.ranks);
    let mut runner: Box<dyn Run> = if ranks > 1 {
        Box::new(Rank::open(plan, settings, DryRank { rank, ranks })?)
    } else {
        open(plan, settings)?
    };
    runner.launching(launch);
    for (at, w) in loads.iter().enumerate() {
        runner.load_weight(at as u32, w)?;
    }
    let current = runner.plan().clone();
    init_tables(&mut *runner, &current)?;
    Ok(runner)
}

/// A runner for `plan` with `loads` as its weights.
pub(crate) fn opened(
    plan: Plan,
    loads: &[Weight],
    settings: &Settings,
) -> Result<Box<dyn Run>, Box<dyn Error>> {
    open_runner(plan, loads, settings, Launching::default())
}

/// Each output's values, as floats.
fn outputs(runner: &dyn Run, plan: &Plan) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let mut all = Vec::new();
    for out in &plan.outputs {
        let floats = match runner.read(out.slot)?.view().to_f32() {
            Ok(floats) => floats,
            Err(_) => runner
                .ids(out.slot)?
                .into_iter()
                .map(|i| i as f32)
                .collect(),
        };
        all.push(floats);
    }
    Ok(all)
}

/// The largest gap from `want`, and its largest value.
fn gap(got: &[Vec<f32>], want: &[Vec<f32>]) -> (f32, f32) {
    let (mut worst, mut most) = (0.0f32, 0.0f32);
    for (got, want) in got.iter().zip(want) {
        for (a, b) in got.iter().zip(want) {
            let d = (a - b).abs();
            // A NaN against a number is the worst gap.
            worst = if d.is_nan() && !b.is_nan() {
                f32::INFINITY
            } else {
                worst.max(d)
            };
            most = most.max(b.abs());
        }
    }
    (worst, most)
}

/// One step's device seconds, repeated to fill `SPEND`.
fn timed(runner: &mut dyn Run, step: usize) -> Result<f64, Box<dyn Error>> {
    let once = runner.repeat(step, 1)?;
    let times = (SPEND / once.max(1e-7)).clamp(1.0, MOST);
    Ok(runner.repeat(step, times as usize)?)
}

/// A runner with its plan, inputs and timings.
struct Watch {
    runner: Box<dyn Run>,
    plan: Plan,
    feeds: Vec<Tensor>,
    want: Vec<Vec<f32>>,
    rows: Vec<Row>,
    meta: Meta,
    json: Option<PathBuf>,
    baseline: Option<Report>,
}

impl Watch {
    /// Notes what each reload runs. Returns the rows to time.
    fn note(&mut self, reloads: &[Reload], to_stderr: bool) -> Vec<usize> {
        let mut changed = Vec::new();
        for reload in reloads {
            if to_stderr {
                eprintln!("{reload}");
            } else {
                println!("{reload}");
            }
            let stem = reload.path.file_stem().map(|s| s.to_string_lossy());
            for (at, row) in self.rows.iter_mut().enumerate() {
                if stem.as_deref() != Some(row.entry.as_str()) {
                    continue;
                }
                match reload.outcome {
                    Outcome::Loaded => row.loaded = true,
                    Outcome::Planned => row.loaded = false,
                    Outcome::Failed(_) => continue,
                }
                changed.push(at);
            }
        }
        changed
    }

    /// Times `which` rows and the whole plan, then checks outputs.
    fn report(&mut self, which: &[usize]) -> Result<(), Box<dyn Error>> {
        let runner = &mut *self.runner;
        let views: Vec<View> = self.feeds.iter().map(Tensor::view).collect();
        let all = 0..self.plan.steps.len();
        runner.feed(&views)?;
        let start = Instant::now();
        while start.elapsed() < WARM {
            runner.submit(all.clone())?;
        }
        for &at in which {
            let mut seconds = 0.0;
            for &step in &self.rows[at].steps {
                seconds += timed(runner, step)?;
            }
            self.rows[at].seconds = seconds;
        }
        // Steps alone may have overwritten what the plan reads.
        runner.feed(&views)?;
        let start = Instant::now();
        let clocked = runner.clocked(all)?;
        let whole = clocked.unwrap_or_else(|| start.elapsed().as_secs_f64());
        let (worst, most) = gap(&outputs(runner, &self.plan)?, &self.want);
        for row in &mut self.rows {
            row.attrs = runner.step_attrs(row.steps[0]);
        }
        let report = build_report(&self.meta, &self.rows, whole, worst, most);
        let mut text = format_table(&report);
        for &at in which {
            let row = &self.rows[at];
            let Some(found) = runner.marks(row.steps[0])? else {
                continue;
            };
            for region in &found.regions {
                let (middle, most) = region.spread();
                let (from, to, times) = (region.from, region.to, region.seconds.len());
                text.push_str(&format!(
                    "    {} lines {from}-{to}: {:.2} us middle, {:.2} us most, {times} times\n",
                    row.entry,
                    middle * 1e6,
                    most * 1e6
                ));
            }
            if found.lost > 0 {
                text.push_str(&format!(
                    "    {}: {} marks past room, not kept\n",
                    row.entry, found.lost
                ));
            }
        }
        if let Some(base) = &self.baseline {
            text.push_str(&format_diff(base, &report));
        }
        if self.json.as_deref() == Some(Path::new("-")) {
            eprint!("{text}");
        } else {
            print!("{text}");
        }
        if let Some(path) = &self.json {
            write_json(path, &report)?;
        }
        Ok(())
    }
}

/// Resolves the watch target from CLI args and recipe options.
fn resolve_target(
    args: &WatchArgs,
    options: &Options,
    dims: ServeDims,
    recipe_weights: Option<&Path>,
    user: &[NameRule],
    parts: &Parts,
) -> Result<Option<Target>, Box<dyn Error>> {
    let first = args.subject_args.first().map(String::as_str);
    if first == Some("model") {
        let (dir, rest) = args.subject_args[1..].split_first().ok_or(USAGE)?;
        return model_target(Path::new(dir), rest, options, dims, user, args, parts)
            .map_err(|e| e as Box<dyn Error>);
    }
    let known = matches!(first, Some("gemm" | "attend" | "mla" | "gqa" | "moe"));
    if !known && let Some(dir) = recipe_weights {
        return model_target(dir, &args.subject_args, options, dims, user, args, parts)
            .map_err(|e| e as Box<dyn Error>);
    }
    let Some(s) = subject(&args.subject_args)? else {
        return Ok(None);
    };
    let recipe = build_recipe(options, Some(s.pins), None, args);
    let g = if parts.ranks() > 1 {
        let shares = recipe.pins.shares(&s.g);
        let sharded = shard_mode(&s.g, parts, &shares, options.ranks.moe_tp())?;
        resolve_wires(sharded.graph, &recipe.sums, options.ranks.sum_wire())
    } else {
        s.g
    };
    Ok(Some(Target {
        g,
        sizes: s.sizes,
        recipe,
        part: None,
        loads: s.loads,
        skew: s.skew,
        region: None,
    }))
}

/// Plans `args`, writes its kernel files, then reruns on change.
pub fn watch(settings: &Settings, args: &[String]) -> Result<(), Box<dyn Error>> {
    tessera_engine::flatten_overrides();
    let parsed = parse_watch(args)?;
    let to_stderr = parsed.json.as_deref() == Some(Path::new("-"));
    let has_model = parsed.subject_args.first().map(String::as_str) == Some("model");
    let (options, dims, recipe_weights) = build_options(&parsed, settings, has_model)?;
    let parts = match parsed.parts.clone() {
        Some(p) => p,
        None if !options.ranks.parts().is_empty() => Parts(options.ranks.parts().to_vec()),
        None => Parts::even(1),
    };
    if parts.0.is_empty() || parts.0.contains(&0) {
        return Err("--parts shares must be > 0".into());
    }
    let rank = parsed.rank.unwrap_or_else(|| options.ranks.rank());
    if rank as usize >= parts.ranks() {
        return Err(format!("--rank {rank} out of range for {} parts", parts.ranks()).into());
    }
    let baseline = match &parsed.against {
        Some(path) => Some(load_report(path)?),
        None => None,
    };
    let user = user_rules(&options, settings)?;
    let Some(mut target) = resolve_target(
        &parsed,
        &options,
        dims,
        recipe_weights.as_deref(),
        &user,
        &parts,
    )?
    else {
        return Ok(());
    };
    let gpu = settings.gpu().is_some();
    let guess = configure_device(Device::guess(settings), &target.recipe, &parts, rank, gpu);
    let initial = lower(&target.g, 0, &target.sizes, &guess)?;
    let mut noise = Noise(7);
    let loads: Vec<Weight> = match (target.loads.take(), target.part.take()) {
        (Some(w), _) => w,
        (None, Some(part)) => {
            part.weights_shared(&target.g, &|at| initial.share(at))
                .map_err(|e| e as Box<dyn Error>)?
                .0
        }
        (None, None) => make_loads(&target.g, &target.recipe.pins, &mut noise),
    };
    let router = extract_router(&target.g, &loads);
    let prefill_seed = 19u64;
    // The reference runs the plan's own kernels.
    let plain = Settings {
        kernels: None,
        ..settings.clone()
    };
    let mut runner = open_runner(initial, &loads, &plain, target.recipe.launch)?;
    let stored =
        Device::stored(runner.platform(), settings).unwrap_or_else(|_| Device::guess(settings));
    let device = configure_device(stored, &target.recipe, &parts, rank, gpu);
    let mut p_noise = Noise(prefill_seed);
    let pos_offset = prefill_kept(
        &mut *runner,
        &target.g,
        &target.sizes,
        &device,
        &mut p_noise,
    )?;
    let (fed, route_stats) = routed_feeds(
        &target.g,
        &target.sizes,
        router.as_deref(),
        target.skew,
        pos_offset,
        &mut noise,
    )?;
    if let Some(stats) = route_stats {
        if to_stderr {
            eprintln!("routing: {stats}");
        } else {
            println!("routing: {stats}");
        }
    }
    let views: Vec<View> = fed.iter().map(Tensor::view).collect();
    let plan = lower(&target.g, 0, &target.sizes, &device)?;
    runner.replan(&plan)?;
    init_tables(&mut *runner, &plan)?;
    runner.feed(&views)?;
    runner.submit(0..plan.steps.len())?;
    let want = outputs(&*runner, &plan)?;
    drop(runner);
    let files = Settings {
        kernels: Some(parsed.dir.clone()),
        ..settings.clone()
    };
    let mut runner = open_runner(plan.clone(), &loads, &files, target.recipe.launch)?;
    // A checkpoint's weights now live on the device.
    drop(loads);
    let mut p_noise = Noise(prefill_seed);
    let _ = prefill_kept(
        &mut *runner,
        &target.g,
        &target.sizes,
        &device,
        &mut p_noise,
    )?;
    runner.replan(&plan)?;
    init_tables(&mut *runner, &plan)?;
    for path in runner.dump()? {
        if to_stderr {
            eprintln!("wrote {}", path.display());
        } else {
            println!("wrote {}", path.display());
        }
    }
    let meta = Meta {
        device: runner.device(),
        parts: parts.0.clone(),
        rank,
        region: target.region,
        recipe: options.name.or(parsed.recipe),
    };
    let rows = rows(&plan, &device);
    let mut watch = Watch {
        runner,
        plan,
        feeds: fed,
        want,
        rows,
        meta,
        json: parsed.json,
        baseline,
    };
    let reloads = watch.runner.refresh()?;
    watch.note(&reloads, to_stderr);
    if parsed.once {
        for r in &reloads {
            if let Outcome::Failed(why) = &r.outcome {
                return Err(format!("override failed ({}): {why}", r.path.display()).into());
            }
        }
    }
    let msg = format!(
        "timing {} kernels on {}",
        watch.rows.len(),
        watch.runner.device()
    );
    if to_stderr {
        eprintln!("{msg}");
    } else {
        println!("{msg}");
    }
    let all: Vec<usize> = (0..watch.rows.len()).collect();
    watch.report(&all)?;
    if parsed.once {
        return Ok(());
    }
    loop {
        let reloads = watch.runner.refresh()?;
        if reloads.is_empty() {
            sleep(POLL);
            continue;
        }
        let start = Instant::now();
        let changed = watch.note(&reloads, to_stderr);
        if changed.is_empty() {
            continue;
        }
        watch.report(&changed)?;
        let ms = start.elapsed().as_secs_f64() * 1e3;
        if to_stderr {
            eprintln!("in {ms:.0} ms");
        } else {
            println!("in {ms:.0} ms");
        }
    }
}
