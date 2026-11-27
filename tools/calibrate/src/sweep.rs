//! Times a subject's plan across the sizes of one dimension.

use crate::routing::{extract_router, prefill_kept, routed_feeds};
use crate::watch::{Subject, make_loads, opened, subject};
use crate::{BURST, TRIES, bursts, low, measure};
use std::collections::BTreeMap;
use std::error::Error;
use tessera_graph::DimName::{self, Rows};
use tessera_graph::Size;
use tessera_lower::{Device, Knobs, Plan, Settings, Tensor, View, Weight, lower};
use tessera_run::Run;
use tessera_testkit::Noise;

const USAGE: &str = "usage: tessera-calibrate sweep [--over <Dim>] [--sizes <n,n,..>] \
    [--skew <n>] [--shape <kernel>:<knob>=<n>] [--learn] \
    gemm <m> <k> <n> | attend <heads> <kv heads> <width> <queries> <keys> | \
    mla <heads> <rows> <context> [--kv f32|bf16|fp8] | \
    gqa <heads> <kv heads> <rows> <context> [--kv f32|bf16|fp8] | \
    moe <experts> <top k> <hidden> <ffn> <rows> [--form <name>] [--skew <n>] [--plain] | \
    model <model dir> [<part>] [--region <kind>.<n>] [<Dim>=<n> ...]";
/// The first size of a sweep with no sizes given.
const FIRST: u64 = 16;
/// The last such size, where the dimension reaches it.
const LAST: u64 = 2048;

/// One kernel's launches in a plan, all together.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Spent {
    pub launches: usize,
    pub seconds: f64,
    /// Multiply-adds, or elements touched.
    pub work: f64,
    /// The least seconds the work needs, where loads say.
    pub bound: Option<f64>,
}

/// A plan at one size, timed step by step.
#[derive(Clone, Debug, PartialEq)]
pub struct Point {
    pub size: u64,
    /// Seconds of every step, each timed alone.
    pub seconds: f64,
    /// Seconds the cost model gives the plan.
    pub estimate: f64,
    /// Bytes of the plan's buffers, weights apart.
    pub bytes: usize,
    pub kernels: BTreeMap<&'static str, Spent>,
}

impl Point {
    pub fn per_size(&self) -> f64 {
        self.seconds / self.size.max(1) as f64
    }
}

/// The power of size a value follows between two points.
pub fn growth(a: (u64, f64), b: (u64, f64)) -> Option<f64> {
    let grew = a.0 != b.0 && a.0 > 0 && b.0 > 0 && a.1 > 0.0 && b.1 > 0.0;
    grew.then(|| (b.1 / a.1).ln() / (b.0 as f64 / a.0 as f64).ln())
}

/// The size where `seconds` per unit of size is least.
pub fn pick(points: &[Point], seconds: fn(&Point) -> f64) -> Option<u64> {
    let each = |p: &Point| seconds(p) / p.size.max(1) as f64;
    let least = points.iter().min_by(|a, b| each(a).total_cmp(&each(b)));
    least.map(|p| p.size)
}

/// Doubling sizes from `FIRST` to the bound of `over`.
fn doubling(s: &Subject, over: DimName) -> Vec<u64> {
    let bound =
        s.g.dims
            .iter()
            .find(|d| d.name == over)
            .map(|d| match d.size {
                Size::Bounded(n) | Size::Fixed(n) => n,
            });
    let most = bound.unwrap_or(LAST).min(LAST);
    let mut sizes = Vec::new();
    let mut n = FIRST.min(most);
    while n < most {
        sizes.push(n);
        n *= 2;
    }
    sizes.push(most);
    sizes
}

/// Seconds of each step, equal launches timed once.
fn step_seconds(runner: &mut dyn Run, plan: &Plan) -> Result<Vec<f64>, tessera_engine::Error> {
    let mut known: Vec<(String, usize, f64)> = Vec::new();
    let mut all = Vec::with_capacity(plan.steps.len());
    for (i, step) in plan.steps.iter().enumerate() {
        let key = format!("{:?}", step.kernel);
        let same = |k: &&(String, usize, f64)| k.0 == key && k.1 == step.write.len;
        let seconds = match known.iter().find(same) {
            Some(k) => k.2,
            None => {
                let seconds = low(bursts(runner, i, TRIES, BURST)?);
                known.push((key, step.write.len, seconds));
                seconds
            }
        };
        all.push(seconds);
    }
    Ok(all)
}

/// The plan's steps summed by kernel name.
fn by_kernel(plan: &Plan, seconds: &[f64], device: &Device) -> BTreeMap<&'static str, Spent> {
    let mut by: BTreeMap<&'static str, Spent> = BTreeMap::new();
    for (step, seconds) in plan.steps.iter().zip(seconds) {
        let (kernel, len) = (&*step.kernel, step.write.len);
        let bound = device.bound(kernel, len);
        let fresh = Spent {
            bound: Some(0.0),
            ..Spent::default()
        };
        let spent = by.entry(kernel.name()).or_insert(fresh);
        spent.launches += step.launches().len();
        spent.seconds += seconds;
        spent.work += kernel.work(len);
        spent.bound = spent.bound.zip(bound).map(|(sum, b)| sum + b);
    }
    by
}

/// Times `s` at each size of `over`. Learns if asked.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn points(
    s: &mut Subject,
    over: DimName,
    sizes: &[u64],
    settings: &Settings,
    learn: bool,
) -> Result<Vec<Point>, Box<dyn Error>> {
    points_with(s, over, sizes, settings, &Knobs::default(), learn)
}

/// Times `s` across `over` with `knobs`, learning if asked.
pub(crate) fn points_with(
    s: &mut Subject,
    over: DimName,
    sizes: &[u64],
    settings: &Settings,
    knobs: &Knobs,
    learn: bool,
) -> Result<Vec<Point>, Box<dyn Error>> {
    let mut noise = Noise(7);
    let loads: Vec<Weight> = match s.loads.take() {
        Some(loads) => loads,
        None => make_loads(&s.g, &s.pins, &mut noise),
    };
    let router = extract_router(&s.g, &loads);
    let at = |n: u64| -> Vec<(DimName, u64)> {
        let rest = s.sizes.iter().filter(|(d, _)| *d != over).copied();
        rest.chain([(over, n)]).collect()
    };
    let first = *sizes.first().ok_or(USAGE)?;
    let mut guess = Device::guess(settings).pinned(s.pins.clone());
    guess.knobs = knobs.clone();
    let mut runner = opened(lower(&s.g, 0, &at(first), &guess)?, &loads, settings)?;
    drop(loads);
    let stored = Device::stored(runner.platform(), settings);
    let mut device = stored.map_or(guess, |d| {
        let mut d = d.pinned(s.pins.clone());
        d.knobs = knobs.clone();
        d
    });
    let mut all = Vec::new();
    for &size in sizes {
        let dims = at(size);
        let mut p_noise = Noise(19);
        let pos_offset = prefill_kept(&mut *runner, &s.g, &dims, &device, &mut p_noise)?;
        let (fed, route_stats) = routed_feeds(
            &s.g,
            &dims,
            router.as_deref(),
            s.skew,
            pos_offset,
            &mut noise,
        )?;
        if let Some(stats) = route_stats {
            println!("routing ({over:?}={size}): {stats}");
        }
        let views: Vec<View> = fed.iter().map(Tensor::view).collect();
        if learn {
            let graph = (&s.g, 0);
            measure(
                &mut *runner,
                graph,
                &dims,
                &views,
                &mut device,
                (settings, false),
            )?;
        }
        let plan = lower(&s.g, 0, &dims, &device)?;
        runner.replan(&plan)?;
        runner.feed(&views)?;
        // The first run pays for anything set up lazily.
        runner.submit(0..plan.steps.len())?;
        let seconds = step_seconds(&mut *runner, &plan)?;
        all.push(Point {
            size,
            seconds: seconds.iter().sum(),
            estimate: device.cost(&plan).seconds,
            bytes: plan.buffers.iter().sum(),
            kernels: by_kernel(&plan, &seconds, &device),
        });
    }
    if let (true, Some(file)) = (learn, Device::path(settings, &device.platform)) {
        device.save(&file)?;
        println!("{} samples in {}", device.samples(), file.display());
    }
    Ok(all)
}

/// A kernel's `value` at its first and last sizes.
fn ends(points: &[Point], kernel: &str, value: fn(&Spent) -> f64) -> Option<[(u64, f64); 2]> {
    let ran = |p: &Point| p.kernels.get(kernel).map(|s| (p.size, value(s)));
    let mut ran = points.iter().filter_map(ran);
    let first = ran.next()?;
    Some([first, ran.next_back().unwrap_or(first)])
}

fn power(ends: Option<[(u64, f64); 2]>) -> String {
    let power = ends.and_then(|[a, b]| growth(a, b));
    power.map_or("    -".into(), |p| format!("{p:5.2}"))
}

/// The sweep as a table by kernel, then by size.
pub fn table(points: &[Point], over: DimName) -> String {
    use std::fmt::Write as _;
    let mut out = format!("ms by kernel at each {over:?}; then how time and work grow\n");
    let mut names: Vec<(&str, f64)> = Vec::new();
    for p in points {
        for (name, s) in &p.kernels {
            match names.iter_mut().find(|(n, _)| n == name) {
                Some((_, total)) => *total += s.seconds,
                None => names.push((name, s.seconds)),
            }
        }
    }
    names.sort_by(|a, b| b.1.total_cmp(&a.1));
    let _ = write!(out, "{:28}", "kernel");
    for p in points {
        let _ = write!(out, " {:>9}", p.size);
    }
    let _ = writeln!(out, "  time  work  over best");
    for (name, _) in &names {
        let _ = write!(out, "{name:28}");
        for p in points {
            match p.kernels.get(name) {
                Some(s) => write!(out, " {:9.3}", s.seconds * 1e3),
                None => write!(out, " {:>9}", "-"),
            }
            .ok();
        }
        let time = power(ends(points, name, |s| s.seconds));
        let work = power(ends(points, name, |s| s.work));
        let last = points.iter().rev().find_map(|p| p.kernels.get(name));
        let over = last.and_then(|s| Some(s.seconds / s.bound.filter(|b| *b > 0.0)?));
        let over = over.map_or("    -".into(), |o| format!("{o:5.1}x"));
        let _ = writeln!(out, " {time} {work} {over}");
    }
    let _ = writeln!(
        out,
        "{:>9} {:>10} {:>10} {:>10} {:>10}",
        "size", "ms", "us/size", "est ms", "MB"
    );
    for p in points {
        let (ms, each) = (p.seconds * 1e3, p.per_size() * 1e6);
        let (est, mb) = (p.estimate * 1e3, p.bytes as f64 / 1e6);
        let _ = writeln!(
            out,
            "{:>9} {ms:10.3} {each:10.2} {est:10.3} {mb:10.1}",
            p.size
        );
    }
    let picks = [
        ("timed", pick(points, |p| p.seconds)),
        ("estimated", pick(points, |p| p.estimate)),
    ];
    for (how, best) in picks {
        if let Some(best) = best {
            let _ = writeln!(out, "least {how} time per {over:?}: {best}");
        }
    }
    out
}

fn dim(name: &str) -> Option<DimName> {
    serde_json::from_value(serde_json::Value::String(name.into())).ok()
}

/// Sweeps the subject in `args` and prints the table.
pub fn sweep(settings: &Settings, args: &[String]) -> Result<(), Box<dyn Error>> {
    let (mut over, mut sizes, mut learn, mut skew, mut knobs, mut at) =
        (Rows, None, false, None, Knobs::default(), 0);
    while let Some(arg) = args.get(at) {
        match arg.as_str() {
            "--over" => over = args.get(at + 1).and_then(|d| dim(d)).ok_or(USAGE)?,
            "--sizes" => {
                let given = args.get(at + 1).ok_or(USAGE)?.split(',');
                let parsed: Result<Vec<u64>, _> = given.map(str::parse).collect();
                sizes = Some(parsed.map_err(|_| USAGE)?);
            }
            "--skew" => {
                let val = args.get(at + 1).ok_or(USAGE)?.parse::<f64>();
                skew = Some(val.map_err(|_| USAGE)?);
            }
            "--shape" => {
                let spec = args.get(at + 1).ok_or(USAGE)?;
                knobs.parse(spec)?;
            }
            "--learn" => learn = true,
            _ => break,
        }
        at += if arg == "--learn" { 1 } else { 2 };
    }
    let Some(mut s) = subject(&args[at..])? else {
        return Ok(());
    };
    if let Some(alpha) = skew {
        s.skew = alpha;
    }
    let sizes = sizes.unwrap_or_else(|| doubling(&s, over));
    let points = points_with(&mut s, over, &sizes, settings, &knobs, learn)?;
    print!("{}", table(&points, over));
    Ok(())
}
