//! Times whole model plans against the cost model.

use crate::{headroom, measure};
use sensai_engine::Error;
use sensai_engine::{Engine, Request};
use sensai_graph::DimName::{self, Context, Rows, Segments, Seqs};
use sensai_graph::{Graph, Role};
use sensai_lower::form::BF16_FORM;
use sensai_lower::{Device, Pins, Plan, Recipe, Settings, Tensor, View, Weight, lower, select};
use sensai_models::decoder::Swiglu;
use sensai_models::{decoder, eg2};
use sensai_run::{Run, open};
use sensai_testkit::{Noise, fill, weights};
use sensai_trace::{Linear, Norm};
use std::collections::BTreeMap;
use std::time::Instant;

/// Rows in each forward launch, short to long.
const LENGTHS: [u64; 3] = [16, 64, 256];
/// Tokens a decode run makes after its prompt.
const NEW: usize = 32;
/// Rows the decode graph keeps.
const CONTEXT: usize = 128;

/// A model at a few layers, with random weights.
struct Subject {
    name: &'static str,
    graph: Graph,
    pins: Pins,
}

/// Checks that missed. The run reports them and fails.
type Misses = Vec<String>;

fn qwen3_config(max_context: u64) -> decoder::Config {
    let mut c = decoder::Config::qwen3(1, max_context);
    (c.stack.layers, c.stack.vocab) = (8, 32_768);
    c
}

fn subjects() -> [Subject; 2] {
    let mut c = eg2::Config::release(1, 1024);
    (c.stack.layers, c.stack.vocab) = (6, 16_384);
    [
        Subject {
            name: "qwen3",
            graph: decoder::program::<Norm, Swiglu>(&qwen3_config(1024)),
            // The real checkpoint is bf16.
            pins: Pins::all(BF16_FORM),
        },
        Subject {
            name: "eg2",
            graph: eg2::program::<Linear>(&c),
            pins: Pins::default(),
        },
    ]
}

type Sizes = [(DimName, u64); 4];

/// One sequence's launch of `rows`, reaching back `context`.
fn sizes(rows: u64, context: u64) -> Sizes {
    [(Seqs, 1), (Rows, rows), (Context, context), (Segments, 1)]
}

/// A plan's time, whole and by step.
struct Timed {
    plan: Plan,
    /// Seconds per step, each alone.
    steps: Vec<f64>,
    /// Seconds for the plan, its output read back.
    whole: f64,
}

/// A runner with weights loaded, and its device.
struct Bench<'a> {
    s: &'a Subject,
    settings: &'a Settings,
    runner: Box<dyn Run>,
    device: Device,
    /// Launches this run timed for lack of a stored sample.
    learned: usize,
}

impl<'a> Bench<'a> {
    fn open(
        s: &'a Subject,
        settings: &'a Settings,
        first: Sizes,
        noise: &mut Noise,
    ) -> Result<Self, Error> {
        let guess = Device::guess(settings).pinned(s.pins.clone());
        let plan = lower(&s.graph, 0, &first, &guess)?;
        let mut runner = open(plan, settings)?;
        for (at, w) in weights(&s.graph, noise).iter().enumerate() {
            runner.load(at as u32, w.view())?;
        }
        let device =
            Device::stored(runner.platform(), settings).map_err(sensai_lower::Error::from)?;
        let device = device.pinned(s.pins.clone());
        Ok(Self {
            s,
            settings,
            runner,
            device,
            learned: 0,
        })
    }

    fn feeds(&self, at: &Sizes, noise: &mut Noise) -> Result<Vec<Tensor>, Error> {
        let g = &self.s.graph;
        let ports = g.entries[0].inputs.iter();
        let fed = ports.map(|p| {
            let unsized_shape = || {
                Error::from(sensai_lower::Error::from(sensai_lower::Size::Shape(
                    p.ty.shape.clone(),
                )))
            };
            let t = fill(g, &p.ty, at, noise).ok_or_else(unsized_shape)?;
            // One sequence: noise slots would read past its rows.
            Ok(match p.role {
                Role::Slots => Tensor::zeros(t.shape.clone(), t.dtype()),
                _ => t,
            })
        });
        fed.collect()
    }

    /// Runs `rows` as planned for `device`, and times each launch.
    fn run(&mut self, at: &Sizes, feeds: &[View], device: &Device) -> Result<Timed, Error> {
        let plan = lower(&self.s.graph, 0, at, device)?;
        self.runner.replan(&plan)?;
        self.runner.feed(feeds)?;
        let (all, out) = (0..plan.steps.len(), plan.outputs[0].slot);
        // The first run pays for anything set up lazily.
        self.runner.submit(all.clone())?;
        let start = Instant::now();
        self.runner.submit(all.clone())?;
        self.runner.read(out)?;
        let whole = start.elapsed().as_secs_f64();
        let mut steps = Vec::new();
        for i in all {
            steps.push(self.runner.repeat(i, 1)?);
        }
        Ok(Timed { plan, steps, whole })
    }

    /// Times this launch under each competing rule, where unknown.
    fn learn(&mut self, at: &Sizes, feeds: &[View]) -> Result<(), Error> {
        let args = ((&self.s.graph, 0), at, feeds);
        let added = measure(
            &mut *self.runner,
            args.0,
            args.1,
            args.2,
            &mut self.device,
            self.settings,
        )?;
        self.learned += added.len();
        Ok(())
    }

    fn save(&self) -> Result<(), Error> {
        let Some(file) = Device::path(self.settings, &self.device.platform) else {
            return Ok(());
        };
        let (learned, all) = (self.learned, self.device.samples());
        println!("  timed {learned} launches; {all} samples known");
        if learned > 0 {
            self.device.save(&file).map_err(sensai_lower::Error::from)?;
            println!("  stored in {}", file.display());
        }
        Ok(())
    }
}

/// Launches and seconds, by kernel.
fn table(plan: &Plan, steps: &[f64]) {
    let mut by: BTreeMap<&str, (usize, f64)> = BTreeMap::new();
    for (step, seconds) in plan.steps.iter().zip(steps) {
        let (count, total) = by.entry(step.kernel.name()).or_default();
        (*count, *total) = (*count + step.launches().len(), *total + seconds);
    }
    for (name, (count, seconds)) in by {
        println!("    {name:8} {count:4} launches {seconds:.4} s");
    }
}

/// Each kernel's time over the least its work needs.
fn room(device: &Device, t: &Timed) {
    let Some(fit) = device.fit else {
        return println!("    no chip was fitted; run tessera-calibrate for headroom");
    };
    let timed: Vec<_> = t
        .plan
        .steps
        .iter()
        .cloned()
        .zip(t.steps.iter().copied())
        .collect();
    let all: f64 = t.steps.iter().sum();
    for h in headroom(&fit.pool("fitted"), &timed) {
        let Some(bound) = h.bound else { continue };
        let (share, over) = (100.0 * h.seconds / all, h.seconds / bound);
        println!(
            "    {:8} {share:4.1}% of the time, {over:.1}x its best case",
            h.kernel
        );
    }
}

/// Prints estimate and time. Misses if far apart.
fn report(label: &str, t: &Timed, device: &Device, misses: &mut Misses) {
    let timed: f64 = t.steps.iter().sum();
    let cost = device.cost(&t.plan);
    let (all, estimate, whole) = (cost.launches, cost.seconds, t.whole);
    println!("  {label}: {whole:.4} s with the output read back");
    println!("    {all} launches: estimated {estimate:.4} s, timed {timed:.4} s");
    table(&t.plan, &t.steps);
    room(device, t);
    let ratio = estimate / timed;
    if !(0.5..2.0).contains(&ratio) {
        misses.push(format!("{label}: the estimate is off by {ratio:.2}x"));
    }
}

/// The plan should not lose to favoring any one rule.
fn compare(b: &mut Bench, at: &Sizes, feeds: &[View], misses: &mut Misses) -> Result<(), Error> {
    let timed = |t: &Timed| t.steps.iter().sum::<f64>();
    let pinned = Device::guess(b.settings).pinned(b.s.pins.clone());
    let contested = select(&b.s.graph, 0, at, &pinned)?.contested;
    let mut least = f64::INFINITY;
    for rule in contested {
        let device = b.device.clone().favoring(rule);
        let forced = timed(&b.run(at, feeds, &device)?);
        println!("    favoring {rule:8} {forced:.4} s");
        least = least.min(forced);
    }
    let device = b.device.clone();
    let mixed = timed(&b.run(at, feeds, &device)?);
    println!("    as planned        {mixed:.4} s");
    // Run-to-run spread at small sizes reaches a third.
    if mixed >= 1.5 * least {
        misses.push(format!(
            "{} {} rows: the plan lost to a forced rule",
            b.s.name, at[1].1
        ));
    }
    Ok(())
}

/// Times each launch, stores samples and reports.
fn launches(
    b: &mut Bench,
    at: &[Sizes],
    noise: &mut Noise,
    misses: &mut Misses,
) -> Result<(), Error> {
    for at in at {
        let fed = b.feeds(at, noise)?;
        let feeds: Vec<View> = fed.iter().map(Tensor::view).collect();
        b.learn(at, &feeds)?;
        let device = b.device.clone();
        let t = b.run(at, &feeds, &device)?;
        report(
            &format!("{} rows after {}", at[1].1, at[2].1 - at[1].1),
            &t,
            &device,
            misses,
        );
        compare(b, at, &feeds, misses)?;
    }
    b.save()
}

fn forward(s: &Subject, settings: &Settings, misses: &mut Misses) -> Result<(), Error> {
    let mut noise = Noise(17);
    let all: Vec<Sizes> = LENGTHS.iter().map(|n| sizes(*n, *n)).collect();
    let mut b = Bench::open(s, settings, all[0], &mut noise)?;
    println!("{} forward on {}:", s.name, b.runner.device());
    launches(&mut b, &all, &mut noise, misses)
}

/// Tokens per second over a prompt and `NEW` tokens, greedy.
fn rate(g: &Graph, w: &[Weight], recipe: &Recipe, settings: &Settings) -> Result<f64, Error> {
    let mut engine = Engine::open(g.clone(), CONTEXT, recipe, settings)?;
    engine.load(w)?;
    let request = || Request {
        tokens: vec![1, 2, 3, 4, 5, 6, 7, 1],
        max_tokens: NEW,
        ..Request::default()
    };
    // The first pass compiles what the plan newly picked.
    engine.generate(request(), &mut |_| {})?;
    let start = Instant::now();
    engine.generate(request(), &mut |_| {})?;
    Ok(NEW as f64 / start.elapsed().as_secs_f64())
}

/// Greedy decoding on qwen3, one launch a token.
fn decode(settings: &Settings, misses: &mut Misses) -> Result<(), Error> {
    let graph = decoder::stepper::<Norm, Swiglu>(&qwen3_config(CONTEXT as u64));
    let pins = Pins::all(BF16_FORM);
    let s = Subject {
        name: "qwen3 decode",
        graph,
        pins: pins.clone(),
    };
    let floats = weights(&s.graph, &mut Noise(17));
    let w: Vec<Weight> = floats.into_iter().map(Weight::Floats).collect();
    let recipe = Recipe {
        pins: pins.clone(),
        ..Recipe::default()
    };
    // Plans for these launches need stored samples.
    let launched = [
        sizes(8, 8),
        sizes(1, 16),
        sizes(1, 17),
        sizes(1, 32),
        sizes(1, 47),
    ];
    let mut noise = Noise(17);
    let mut b = Bench::open(&s, settings, launched[0], &mut noise)?;
    println!("{} on {}:", s.name, b.runner.device());
    launches(&mut b, &launched, &mut noise, misses)?;
    let planned = rate(&s.graph, &w, &recipe, settings)?;
    println!("  {NEW} tokens after a prompt: {planned:.1} tokens/s");
    let device = Device::guess(settings).pinned(pins);
    let mut most = 0.0f64;
    for rule in select(&s.graph, 0, &sizes(1, 16), &device)?.contested {
        let favor = Some(rule.to_string());
        let recipe = Recipe {
            favor,
            ..recipe.clone()
        };
        let forced = rate(&s.graph, &w, &recipe, settings)?;
        println!("  favoring {rule:8} {forced:.1} tokens/s");
        most = most.max(forced);
    }
    if 1.25 * planned <= most {
        misses.push("qwen3 decode: the plan lost to a forced rule".into());
    }
    Ok(())
}

/// Runs every timing check. Returns the ones that missed.
pub fn check(settings: &Settings) -> Result<Misses, Error> {
    let mut misses = Misses::new();
    for s in &subjects() {
        forward(s, settings, &mut misses)?;
    }
    decode(settings, &mut misses)?;
    Ok(misses)
}
