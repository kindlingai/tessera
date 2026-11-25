//! Device kernels against the host and the rules they displace.

use crate::grid::{bounded, codes, fill};
use std::collections::BTreeSet;
use tessera_engine::{Error, block, trellis};
use tessera_graph::DimName::Rows;
use tessera_graph::{Act, Graph};
use tessera_lower::form::{BF16_FORM, Load};
use tessera_lower::{Backend, Device, Pick, Pins, Plan, Settings, Tensor, View, lower, select};
use tessera_run::{Run, open};
use tessera_testkit::graphs::{blocks, finished, linear, picked};
use tessera_testkit::{Noise, max_abs_diff};

/// Launches of one step in each timed try.
const REPS: usize = 100;
/// Tries of each plan, taken in turn.
const ROUNDS: usize = 7;
/// Largest output error, over the largest host output.
const OFF: f32 = 1e-2;
/// The plan may trail a forced rule by this share.
const SLACK: f64 = 0.1;

/// A graph whose shapes `rule` claims, and rows to run.
struct Case {
    rule: &'static str,
    what: String,
    g: Graph,
    pins: Pins,
    rows: &'static [u64],
}

fn cases() -> Vec<Case> {
    let one = bounded(&[Rows]);
    let case = |rule, what: String, g, pins: &Pins, rows| Case {
        rule,
        what,
        g,
        pins: pins.clone(),
        rows,
    };
    let bf16 = Pins::all(BF16_FORM);
    let mut all = Vec::new();
    // Decode widths. 640 and 1000 end on a part trip.
    let gemv = [[2560, 512], [2560, 12288], [6144, 2560], [640, 2560]];
    for [k, n] in gemv.into_iter().chain([[1000, 1024], [10240, 24]]) {
        all.push(case(
            "fast_gemv",
            format!("{k}x{n}"),
            linear(&one, k, n),
            &bf16,
            &[1],
        ));
    }
    for act in [Act::Silu, Act::Sigmoid] {
        let g = finished(&one, [2560, 640], act);
        let what = format!("2560x640 {act:?}");
        all.push(case("fast_gemv", what, g, &bf16, &[1]));
    }
    for form in [&block::MXFP8 as &dyn Load, &block::INT4_G32] {
        let pins = Pins::all(form);
        let what = format!("4096x4096 {}", form.name());
        all.push(case(
            "fast_gemv",
            what,
            linear(&one, 4096, 4096),
            &pins,
            &[1],
        ));
    }
    let g = linear(&one, 2560, 2560);
    all.push(case(
        "fast_mm",
        "2560x2560".into(),
        g.clone(),
        &bf16,
        &[16, 64, 256],
    ));
    let g3 = Pins::all(&trellis::TRELLIS_K3_PMCGX);
    all.push(case("trellis_mma", "2560x2560".into(), g, &g3, &[1, 16]));
    let g = blocks(&one, [8, 128, 128]);
    all.push(case("fast_stacked", "8x128x128".into(), g, &bf16, &[1, 16]));
    let books = [
        &trellis::TRELLIS_K3_PMCGX as &dyn Load,
        &trellis::TRELLIS_K2,
        &trellis::TRELLIS_K4_MUL1,
    ];
    for book in books {
        let pins = Pins::default().pin("set", book);
        let g = picked(&one, [16, 2560, 640]);
        let what = format!("16x2560x640 {}", book.name());
        all.push(case("picked_trellis", what, g, &pins, &[1, 4]));
    }
    all
}

/// Loads noise weights. Returns the host copy of each.
fn load(g: &Graph, runner: &mut dyn Run, noise: &mut Noise) -> Result<Vec<Tensor>, Error> {
    let mut host = Vec::new();
    for (at, w) in g.weights.iter().enumerate() {
        let at = at as u32;
        let data = fill(g, &w.ty, &Vec::new(), noise)?;
        let loaded = runner.plan().weights.iter().find(|l| l.weight == at);
        let floats = match loaded.map(|l| l.slot.form) {
            Some(form) if !form.packs() => {
                let bytes = codes(form, &data, noise);
                runner.load_packed(at, &bytes)?;
                form.unpack_all(&bytes)
            }
            Some(form) => {
                runner.load(at, data.view())?;
                form.roundtrip(floats(&data)?)
            }
            None => {
                runner.load(at, data.view())?;
                floats(&data)?.to_vec()
            }
        };
        host.push(Tensor::f32(data.shape.clone(), floats));
    }
    Ok(host)
}

/// The host's outputs for these weights and feeds.
fn reference(
    g: &Graph,
    rows: u64,
    weights: &[Tensor],
    feeds: &[View],
    settings: &Settings,
) -> Result<Vec<Tensor>, Error> {
    let cpu = Settings {
        backend: Pick::Cpu,
        ..settings.clone()
    };
    let plan = lower(g, 0, &[(Rows, rows)], &Device::guess(&cpu))?;
    let mut runner = open(plan, &cpu)?;
    for (at, w) in weights.iter().enumerate() {
        runner.load(at as u32, w.view())?;
    }
    Ok(runner.run(feeds)?)
}

fn floats(t: &Tensor) -> Result<&[f32], Error> {
    Ok(t.floats().map_err(tessera_lower::Error::from)?)
}

/// The worst output error, over the largest host output.
fn error(got: &[Tensor], want: &[Tensor]) -> Result<f32, Error> {
    let mut worst = 0.0f32;
    for (g, w) in got.iter().zip(want) {
        let (g, w) = (floats(g)?, floats(w)?);
        let scale = w.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
        worst = worst.max(max_abs_diff(g, w) / scale);
    }
    Ok(worst)
}

/// A plan under test: its label, plan and best seconds.
struct Way {
    label: String,
    plan: Plan,
    seconds: f64,
}

/// The plan as chosen, then one per rule that competed.
fn ways(c: &Case, rows: u64, device: &Device, seen: &mut BTreeSet<&'static str>) -> Vec<Way> {
    let sizes = [(Rows, rows)];
    let Ok(chosen) = select(&c.g, 0, &sizes, device) else {
        return Vec::new();
    };
    seen.extend(chosen.contested.iter().copied());
    seen.extend(chosen.ops.iter().map(|o| o.rule));
    let forced = chosen.contested.iter().map(|r| {
        let label = format!("favoring {r}");
        (label, device.clone().favoring(*r))
    });
    let mut out: Vec<Way> = Vec::new();
    for (label, d) in [("as planned".to_string(), device.clone())]
        .into_iter()
        .chain(forced)
    {
        let plan = match lower(&c.g, 0, &sizes, &d) {
            Ok(plan) => plan,
            Err(e) => {
                println!("    {label:24} does not lower: {e}");
                continue;
            }
        };
        if out.iter().any(|w| names(&w.plan) == names(&plan)) {
            continue;
        }
        let seconds = f64::INFINITY;
        out.push(Way {
            label,
            plan,
            seconds,
        });
    }
    out
}

fn names(plan: &Plan) -> String {
    let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
    names.join("+")
}

/// Seconds for every step of the runner's plan.
fn timed(runner: &mut dyn Run) -> Result<f64, Error> {
    let mut total = 0.0;
    for i in 0..runner.plan().steps.len() {
        total += runner.repeat(i, REPS)?;
    }
    Ok(total)
}

/// Runs one case at one size. Pushes what missed.
fn check(
    c: &Case,
    rows: u64,
    (runner, device): (&mut Box<dyn Run>, &Device),
    (host, noise): (&[Tensor], &mut Noise),
    (seen, misses): (&mut BTreeSet<&'static str>, &mut Vec<String>),
    settings: &Settings,
) -> Result<(), Error> {
    let sizes = vec![(Rows, rows)];
    let ports = c.g.entries[0].inputs.iter();
    let fed: Result<Vec<Tensor>, Error> = ports.map(|p| fill(&c.g, &p.ty, &sizes, noise)).collect();
    let fed = fed?;
    let feeds: Vec<View> = fed.iter().map(Tensor::view).collect();
    let want = reference(&c.g, rows, host, &feeds, settings)?;
    let mut ways = ways(c, rows, device, seen);
    let label = format!("{} {} rows={rows}", c.rule, c.what);
    for w in &ways {
        runner.replan(&w.plan)?;
        let off = error(&runner.run(&feeds)?, &want)?;
        println!("    {:24} error {off:.1e}  {}", w.label, names(&w.plan));
        if off.is_nan() || off > OFF {
            misses.push(format!("{label} {}: error {off:.1e}", w.label));
        }
    }
    for _ in 0..ROUNDS {
        for w in &mut ways {
            runner.replan(&w.plan)?;
            runner.run(&feeds)?;
            w.seconds = w.seconds.min(timed(&mut **runner)?);
        }
    }
    for w in &ways {
        let guess = device.cost(&w.plan).seconds * 1e6;
        let us = w.seconds * 1e6;
        println!("    {:24} {us:9.2} us, estimated {guess:9.2} us", w.label);
    }
    let Some((planned, forced)) = ways.split_first() else {
        return Ok(());
    };
    let best = forced.iter().min_by(|a, b| a.seconds.total_cmp(&b.seconds));
    if let Some(b) = best.filter(|b| planned.seconds > b.seconds * (1.0 + SLACK)) {
        let (p, f) = (planned.seconds * 1e6, b.seconds * 1e6);
        misses.push(format!("{label}: planned {p:.2} us, {} {f:.2} us", b.label));
    }
    Ok(())
}

/// Checks every case. Returns what missed.
pub fn check_all(settings: &Settings) -> Result<Vec<String>, Error> {
    let mut noise = Noise(31);
    let (mut seen, mut misses) = (BTreeSet::new(), Vec::new());
    let mut backend = None;
    let all = cases();
    for c in &all {
        let first = [(Rows, c.rows[0])];
        let guess = Device::guess(settings).pinned(c.pins.clone());
        let mut runner = open(lower(&c.g, 0, &first, &guess)?, settings)?;
        let host = load(&c.g, &mut *runner, &mut noise)?;
        let platform = runner.platform();
        backend = Backend::named(&platform.backend);
        let device = Device::stored(platform, settings).map_err(tessera_lower::Error::from)?;
        let device = device.pinned(c.pins.clone());
        println!("{} {} on {}:", c.rule, c.what, device.name);
        for &rows in c.rows {
            let held = (&mut runner, &device);
            let lists = (&mut seen, &mut misses);
            check(c, rows, held, (&host, &mut noise), lists, settings)?;
        }
    }
    // A rule this backend has must compete somewhere.
    let rules = tessera_lower::rules();
    for c in &all {
        let Some(rule) = rules.iter().find(|r| r.name == c.rule) else {
            continue;
        };
        let lacked = backend.is_none_or(|b| rule.propose.lacks().contains(&b));
        if !lacked && !seen.contains(c.rule) {
            misses.push(format!("{}: proposed for no shape", c.rule));
            seen.insert(c.rule);
        }
    }
    Ok(misses)
}
