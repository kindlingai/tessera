//! Runs a plan again whenever a kernel file changes.

use crate::grid::bounded;
use crate::plans::{feeds, sizes, subjects};
use sensai_bundle::flux2::transformer;
use sensai_graph::DimName::{self, Context, Rows, Seqs};
use sensai_graph::Graph;
use sensai_lower::{Device, Pins, Plan, Settings, Source, Step, Tensor, View, Weight, lower};
use sensai_run::{Outcome, Reload, Run, open};
use sensai_testkit::graphs::{attention, linear};
use sensai_testkit::{Noise, weights};
use std::error::Error;
use std::path::{Path, PathBuf};
use std::thread::sleep;
use std::time::{Duration, Instant};

const USAGE: &str = "usage: tessera-calibrate watch <dir> \
    gemm <m> <k> <n> | attend <heads> <kv heads> <width> <queries> <keys> | \
    qwen3 <rows> [<context>] | eg2 <rows> [<context>] | \
    flux2 <model dir> <image rows> [<text rows>]";
/// Time between checks for changed files.
const POLL: Duration = Duration::from_millis(20);
/// Device seconds to spend timing one step.
const SPEND: f64 = 0.01;
const MOST: f64 = 64.0;
/// Seconds of whole runs first, so clocks rise from idle.
const WARM: Duration = Duration::from_millis(100);
/// Text rows klein pads every prompt to.
const TEXT: u64 = 512;

/// A graph at fixed sizes, with its weight forms.
struct Subject {
    g: Graph,
    sizes: Vec<(DimName, u64)>,
    pins: Pins,
    /// Weights from a checkpoint. Without them the run makes noise.
    loads: Option<Vec<Weight>>,
}

/// Numbers in `args`, or the usage.
fn numbers(args: &[String]) -> Result<Vec<u64>, Box<dyn Error>> {
    let mut n = Vec::new();
    for arg in args {
        n.push(arg.parse::<u64>().map_err(|_| USAGE)?);
    }
    Ok(n)
}

/// A FLUX.2 transformer from a diffusers directory, with its weights.
fn flux2(args: &[String]) -> Result<Subject, Box<dyn Error>> {
    let (dir, rest) = args.split_first().ok_or(USAGE)?;
    let (rows, text) = match numbers(rest)?.as_slice() {
        [rows] => (*rows, TEXT),
        [rows, text] => (*rows, *text),
        _ => return Err(USAGE.into()),
    };
    let t = transformer(Path::new(dir), [1, text, rows], &Default::default())
        .map_err(|e| e as Box<dyn Error>)?;
    Ok(Subject {
        g: t.graph,
        sizes: vec![(Seqs, 1), (Rows, rows)],
        pins: t.pins,
        loads: Some(t.weights),
    })
}

fn subject(args: &[String]) -> Result<Subject, Box<dyn Error>> {
    if args.first().map(String::as_str) == Some("flux2") {
        return flux2(&args[1..]);
    }
    let n = numbers(args.get(1..).unwrap_or_default())?;
    let (default, matmul, attend) = (Pins::default(), [Rows], [Seqs, Rows, Context]);
    Ok(match (args.first().map(String::as_str), n.as_slice()) {
        (Some("gemm"), [m, k, width]) => Subject {
            g: linear(&bounded(&matmul), *k, *width),
            sizes: vec![(Rows, *m)],
            pins: default,
            loads: None,
        },
        (Some("attend"), [heads, kv, width, queries, keys]) => Subject {
            g: attention(&bounded(&attend), [*heads, *kv, *width]),
            sizes: vec![(Seqs, 1), (Rows, *queries), (Context, *keys)],
            pins: default,
            loads: None,
        },
        (Some(model), [rows, context @ ..]) if context.len() < 2 => {
            let found = subjects().into_iter().find(|s| s.name == model);
            let s = found.ok_or(USAGE)?;
            let context = context.first().unwrap_or(rows);
            Subject {
                g: s.graph,
                sizes: sizes(*rows, *context).to_vec(),
                pins: s.pins,
                loads: None,
            }
        }
        _ => return Err(USAGE.into()),
    })
}

/// One kernel entry: its steps and what they cost.
struct Row {
    entry: String,
    steps: Vec<usize>,
    seconds: f64,
    planned: f64,
    bound: Option<f64>,
    /// Whether a kernel file runs in place of the plan's.
    loaded: bool,
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
                    steps: Vec::new(),
                    seconds: 0.0,
                    planned: 0.0,
                    bound: Some(0.0),
                    loaded: false,
                });
                rows.len() - 1
            }
        };
        let (kernel, len) = (&*step.kernel, step.write.len);
        let row = &mut rows[at];
        row.steps.push(i);
        row.planned += device.seconds(kernel, len);
        row.bound = row.bound.zip(device.bound(kernel, len)).map(|(a, b)| a + b);
    }
    rows
}

/// A runner for `plan` with `loads` as its weights.
fn opened(
    plan: Plan,
    loads: &[Weight],
    settings: &Settings,
) -> Result<Box<dyn Run>, Box<dyn Error>> {
    let mut runner = open(plan, settings)?;
    for (at, w) in loads.iter().enumerate() {
        runner.load_weight(at as u32, w)?;
    }
    Ok(runner)
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
}

impl Watch {
    /// Notes what each reload runs. Returns the rows to time.
    fn note(&mut self, reloads: &[Reload]) -> Vec<usize> {
        let mut changed = Vec::new();
        for reload in reloads {
            println!("{reload}");
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
        let mut order: Vec<&Row> = self.rows.iter().collect();
        order.sort_by(|a, b| b.seconds.total_cmp(&a.seconds));
        println!(
            "{:>9} {:>9} {:>9} {:>5}  kernel",
            "us", "planned", "bound", "steps"
        );
        for r in order {
            let bound = r.bound.map_or("-".into(), |b| format!("{:.1}", b * 1e6));
            let mark = if r.loaded { "*" } else { " " };
            let (us, planned) = (r.seconds * 1e6, r.planned * 1e6);
            let steps = r.steps.len();
            println!(
                "{us:>9.1} {planned:>9.1} {bound:>9} {steps:>5} {mark}{}",
                r.entry
            );
        }
        for &at in which {
            let row = &self.rows[at];
            let Some(found) = runner.marks(row.steps[0])? else {
                continue;
            };
            for region in &found.regions {
                let (middle, most) = region.spread();
                let (from, to, times) = (region.from, region.to, region.seconds.len());
                println!(
                    "    {} lines {from}-{to}: {:.2} us middle, {:.2} us most, {times} times",
                    row.entry,
                    middle * 1e6,
                    most * 1e6
                );
            }
            if found.lost > 0 {
                println!(
                    "    {}: {} marks past room, not kept",
                    row.entry, found.lost
                );
            }
        }
        let alone: f64 = self.rows.iter().map(|r| r.seconds).sum();
        println!(
            "plan {:.1} us, steps alone {:.1} us, error {worst:.3e} of {most:.3e}",
            whole * 1e6,
            alone * 1e6
        );
        Ok(())
    }
}

/// Plans `args`, writes its kernel files, then reruns on change.
pub fn watch(settings: &Settings, args: &[String]) -> Result<(), Box<dyn Error>> {
    let (dir, rest) = args.split_first().ok_or(USAGE)?;
    let mut s = subject(rest)?;
    let mut noise = Noise(7);
    let noisy = |noise: &mut Noise| weights(&s.g, noise).into_iter().map(Weight::Floats);
    let loads: Vec<Weight> = match s.loads.take() {
        Some(loads) => loads,
        None => noisy(&mut noise).collect(),
    };
    let fed = feeds(&s.g, &s.sizes, &mut noise)?;
    let views: Vec<View> = fed.iter().map(Tensor::view).collect();
    // The reference runs the plan's own kernels.
    let plain = Settings {
        kernels: None,
        ..settings.clone()
    };
    let guess = Device::guess(settings).pinned(s.pins.clone());
    let mut runner = opened(lower(&s.g, 0, &s.sizes, &guess)?, &loads, &plain)?;
    let stored = Device::stored(runner.platform(), settings);
    let device = stored.map_or(guess, |d| d.pinned(s.pins.clone()));
    let plan = lower(&s.g, 0, &s.sizes, &device)?;
    runner.replan(&plan)?;
    runner.feed(&views)?;
    runner.submit(0..plan.steps.len())?;
    let want = outputs(&*runner, &plan)?;
    drop(runner);
    let files = Settings {
        kernels: Some(PathBuf::from(dir)),
        ..settings.clone()
    };
    let runner = opened(plan.clone(), &loads, &files)?;
    // A checkpoint's weights now live on the device.
    drop(loads);
    for path in runner.dump()? {
        println!("wrote {}", path.display());
    }
    let rows = rows(&plan, &device);
    let mut watch = Watch {
        runner,
        plan,
        feeds: fed,
        want,
        rows,
    };
    let reloads = watch.runner.refresh()?;
    watch.note(&reloads);
    println!(
        "timing {} kernels on {}",
        watch.rows.len(),
        watch.runner.device()
    );
    let all: Vec<usize> = (0..watch.rows.len()).collect();
    watch.report(&all)?;
    loop {
        let reloads = watch.runner.refresh()?;
        if reloads.is_empty() {
            sleep(POLL);
            continue;
        }
        let start = Instant::now();
        let changed = watch.note(&reloads);
        if changed.is_empty() {
            continue;
        }
        watch.report(&changed)?;
        println!("in {:.0} ms", start.elapsed().as_secs_f64() * 1e3);
    }
}
