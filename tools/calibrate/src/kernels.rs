//! Device kernels against the host and the rules they displace.

use crate::grid::{bounded, codes, fill};
use crate::routing::{DEFAULT_SKEW, prefill_kept, routed_feeds};
use std::collections::BTreeSet;
use tessera_engine::{Error, block, trellis};
use tessera_graph::DimName::{self, Rows};
use tessera_graph::{Act, Graph, Size};
use tessera_lower::form::{BF16_FORM, FP8_FORM, Load};
use tessera_lower::{
    Backend, Device, Knobs, Pick, Pins, Plan, Settings, Tensor, View, lower, select,
};
use tessera_run::{Run, open};
use tessera_testkit::graphs::{
    blocks, finished, gqa_attend_rows, linear, mla_attend_rows, moe_routed, picked,
};
use tessera_testkit::{Noise, max_abs_diff};

/// Row counts from 16 up to twice the old cap.
const TALL: &[u64] = &[16, 64, 256, 640, 1024, 2048, 4096];

/// Launches of one step in each timed try.
const REPS: usize = 100;
/// Tries of each plan, taken in turn.
const ROUNDS: usize = 7;
/// Largest output error, over the largest host output.
const OFF: f32 = 1e-2;
/// Larger tolerance for FP8 KV cache against f32 reference.
const OFF_FP8: f32 = 6e-2;
/// The plan may trail a forced rule by this share.
const SLACK: f64 = 0.1;

/// A graph whose shapes `rule` claims, and rows to run.
struct Case {
    rule: &'static str,
    what: String,
    g: Graph,
    pins: Pins,
    rows: &'static [u64],
    /// Kernel shape values the device sets.
    knobs: Knobs,
    /// Relative error tolerance against the host reference.
    tol: f32,
}

fn cases() -> Vec<Case> {
    let one = bounded(&[Rows]);
    let case = |rule, what: String, g, pins: &Pins, rows| Case {
        rule,
        what,
        g,
        pins: pins.clone(),
        rows,
        knobs: Knobs::default(),
        tol: OFF,
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
    let mxfp8 = Pins::all(&block::MXFP8);
    all.push(case("fast_mm", "2560x2560".into(), g.clone(), &bf16, TALL));
    let what = "2560x2560 mxfp8".into();
    all.push(case("fast_mm", what, g.clone(), &mxfp8, TALL));
    // Forced splits. bf16 splits need n%32 > 0.
    for (spot, [k, n], pins) in [
        ("fast_mm_bf16_sk", [2560, 2544], &bf16),
        ("fast_mm_mxfp8_sk", [4096, 2560], &mxfp8),
    ] {
        let (g, what) = (linear(&one, k, n), format!("{k}x{n} {spot}=4"));
        let mut split = case("fast_mm", what, g, pins, TALL);
        split
            .knobs
            .set(&format!("{spot}:splits"), 4)
            .expect("a knob");
        all.push(split);
    }
    let g3 = Pins::all(&trellis::TRELLIS_K3_PMCGX);
    all.push(case("trellis_mma", "2560x2560".into(), g, &g3, &[1, 16]));
    let g = blocks(&one, [8, 128, 128]);
    all.push(case(
        "fast_stacked",
        "8x128x128".into(),
        g,
        &bf16,
        &[1, 16, 64],
    ));
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
    let trellis_skew = Pins::all(&trellis::TRELLIS_K3_PMCGX);
    let g = moe_routed(&one, [64, 8, 2560, 640], false);
    all.push(case(
        "picked_trellis",
        "64e top8 2560x640 skewed".into(),
        g,
        &trellis_skew,
        &[1, 16],
    ));
    let bf16_moe = Pins::all(BF16_FORM)
        .activations(BF16_FORM)
        .outputs(BF16_FORM);
    let g = moe_routed(&one, [64, 8, 1024, 512], true);
    all.push(case(
        "picked_bf16",
        "64e top8 1024x512 swiglu".into(),
        g,
        &bf16_moe,
        &[16, 64],
    ));
    for (kv_name, kv, tol) in [("bf16", BF16_FORM, OFF), ("fp8", FP8_FORM, OFF_FP8)] {
        let pins = Pins {
            kept: kv,
            ..Pins::default()
        };
        let mut mla = case(
            "fast_mla_attend",
            format!("16h ctx512 {kv_name}"),
            mla_attend_rows(&one, 16, 512, 1),
            &pins,
            &[1, 16],
        );
        mla.tol = tol;
        all.push(mla);
        let mut gqa = case(
            "fast_gqa_attend",
            format!("32h/4kv ctx512 {kv_name}"),
            gqa_attend_rows(&one, [32, 4, 128], 512, 1),
            &pins,
            &[1, 16],
        );
        gqa.tol = tol;
        all.push(gqa);
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

/// Runtime sizes for `g` at `rows`, filling bounded dims.
fn case_sizes(g: &Graph, rows: u64) -> Vec<(DimName, u64)> {
    let mut sizes = vec![(Rows, rows)];
    for d in &g.dims {
        if d.name != Rows
            && let Size::Bounded(n) = d.size
        {
            sizes.push((d.name, n));
        }
    }
    sizes
}

/// The host's outputs for these weights and feeds.
fn reference(
    g: &Graph,
    rows: u64,
    weights: &[Tensor],
    feeds: &[View],
    prefill_seed: Option<u64>,
    settings: &Settings,
) -> Result<Vec<Tensor>, Error> {
    let cpu = Settings {
        backend: Pick::Cpu,
        ..settings.clone()
    };
    let cpu_device = Device::guess(&cpu);
    let sizes = case_sizes(g, rows);
    let plan = lower(g, 0, &sizes, &cpu_device)?;
    let mut runner = open(plan.clone(), &cpu)?;
    for (at, w) in weights.iter().enumerate() {
        runner.load(at as u32, w.view())?;
    }
    if let Some(seed) = prefill_seed {
        let mut p_noise = Noise(seed);
        let _ = prefill_kept(&mut *runner, g, &sizes, &cpu_device, &mut p_noise)?;
        runner.replan(&plan)?;
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
    let sizes = case_sizes(&c.g, rows);
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
    let sizes = case_sizes(&c.g, rows);
    let prefill_seed = (!c.g.kept.is_empty()).then_some(19 + rows);
    let pos_offset = match prefill_seed {
        Some(seed) => {
            let mut p_noise = Noise(seed);
            prefill_kept(&mut **runner, &c.g, &sizes, device, &mut p_noise)?
        }
        None => 0,
    };
    let (fed, route_stats) = routed_feeds(&c.g, &sizes, None, DEFAULT_SKEW, pos_offset, noise)?;
    if let Some(stats) = route_stats {
        println!("    routing: {stats}");
    }
    let feeds: Vec<View> = fed.iter().map(Tensor::view).collect();
    let want = reference(&c.g, rows, host, &feeds, prefill_seed, settings)?;
    let mut ways = ways(c, rows, device, seen);
    let label = format!("{} {} rows={rows}", c.rule, c.what);
    for w in &ways {
        runner.replan(&w.plan)?;
        let off = error(&runner.run(&feeds)?, &want)?;
        println!("    {:24} error {off:.1e}  {}", w.label, names(&w.plan));
        if off.is_nan() || off > c.tol {
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

/// Checks cases matching `filter` (or all when `None`).
pub fn check_filtered(settings: &Settings, filter: Option<&str>) -> Result<Vec<String>, Error> {
    let mut noise = Noise(31);
    let (mut seen, mut misses) = (BTreeSet::new(), Vec::new());
    let mut backend = None;
    let all: Vec<Case> = cases()
        .into_iter()
        .filter(|c| filter.is_none_or(|f| c.rule.contains(f) || c.what.contains(f)))
        .collect();
    if all.is_empty()
        && let Some(f) = filter
    {
        misses.push(format!("filter `{f}` matched 0 cases"));
        return Ok(misses);
    }
    for c in &all {
        let first = case_sizes(&c.g, c.rows[0]);
        let mut guess = Device::guess(settings).pinned(c.pins.clone());
        guess.knobs = c.knobs.clone();
        let mut runner = open(lower(&c.g, 0, &first, &guess)?, settings)?;
        let host = load(&c.g, &mut *runner, &mut noise)?;
        let platform = runner.platform();
        backend = Backend::named(&platform.backend);
        let device = Device::stored(platform, settings).map_err(tessera_lower::Error::from)?;
        let mut device = device.pinned(c.pins.clone());
        device.knobs = c.knobs.clone();
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
