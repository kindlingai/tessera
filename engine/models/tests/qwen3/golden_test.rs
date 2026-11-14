//! Qwen3-0.6B on the GPU: timing, cost and decoding.

use super::{GOLDEN, release};
use crate::common::{
    Case, Reference, best, floats, interpreted, kernels, loaded, seconds, table, tokens,
};
use calibrate::{headroom, measure};
use checkpoint::File;
use gpukit::testing::max_abs_diff;
use interp::{Tensor, bind};
use lower::{Cache, Device, Pins, Plan, Settings, View, lower, select};
use models::qwen3::{Config, stepper, tensor_name};
use run::Run;
use std::time::Instant;
use trace::DimName::{self, Context, Rows, Segments, Seqs};
use trace::{Graph, PathSeg};

/// Read from the directory `TESSERA_QWEN3` names.
struct Fixture {
    settings: Settings,
    weights: Vec<Tensor>,
    tokens: Vec<u32>,
    want: Vec<f32>,
}

/// The reference the suite's golden test reads.
pub fn read(g: &Graph) -> Option<Reference> {
    let dir = settings::read().qwen3?;
    let file = File::open(&dir.join("model.safetensors")).unwrap();
    let find = |p: &[PathSeg]| {
        let v = file.floats(&tensor_name(p)?).ok()?;
        Some(Tensor::f32(vec![v.len()], v))
    };
    let tokens = tokens(&dir.join("tokens.txt"));
    let want = floats(&dir.join("logits.f32"));
    Some(Reference {
        weights: bind(g, find).unwrap(),
        cases: vec![Case::one("reference", tokens, want)],
    })
}

fn fixture(g: &Graph) -> Option<Fixture> {
    let mut reference = read(g)?;
    let one = reference.cases.remove(0);
    Some(Fixture {
        settings: settings::read(),
        weights: reference.weights,
        tokens: one.tokens,
        want: one.want,
    })
}

/// Rows of logits `tokens` give on the interpreter.
fn logits(g: &Graph, c: &Fixture, tokens: &[u32]) -> Vec<f32> {
    interpreted(g, &c.weights, 1, tokens)
}

/// Each position must pick the reference's top token.
fn close(label: &str, got: &[f32], c: &Fixture) {
    GOLDEN.tolerance.check(label, got, &c.want, c.tokens.len());
}

#[test]
fn gpu_checks() {
    if settings::read().gpu().is_none() {
        return eprintln!("no GPU here; skipped");
    }
    let g = release();
    let Some(c) = fixture(&g) else {
        return eprintln!("TESSERA_QWEN3 is not set; skipped");
    };
    let mut gpu = Gpu::open(&g, &c);
    forward::check(&mut gpu, &g);
    decode::check(&mut gpu, &g);
    gpu.save();
}

/// The checkpoint is bf16, so plans read it as stored.
fn pinned(settings: &Settings) -> Device {
    Device::guess(settings).pinned(Pins::all(lower::form::BF16_FORM))
}

/// Plans for either entry on one runner. Weights load once.
struct Gpu<'a> {
    c: &'a Fixture,
    runner: Box<dyn Run>,
    /// This machine's stored samples, and what this run adds.
    device: Device,
    /// Launches this run timed for lack of a stored sample.
    learned: usize,
    /// Plans by sizes, reused where a launch repeats.
    plans: Cache,
}

/// One sequence's launch: its new tokens after `past` rows.
#[derive(Clone, Copy)]
struct Span<'a> {
    past: usize,
    tokens: &'a [u32],
}

impl Span<'_> {
    /// Sizes for either entry: the forward one ignores `Rows`.
    fn sizes(&self) -> [(DimName, u64); 4] {
        let (rows, all) = (self.tokens.len(), self.past + self.tokens.len());
        [
            (Seqs, 1),
            (Rows, rows as u64),
            (Context, all as u64),
            (Segments, 1),
        ]
    }

    fn positions(&self) -> Vec<u32> {
        (self.past as u32..(self.past + self.tokens.len()) as u32).collect()
    }

    /// Tokens, positions and slots as columns, all in slot 0.
    fn tensors(&self) -> [Tensor; 3] {
        let shape = vec![self.tokens.len()];
        let tokens = Tensor::u32(shape.clone(), self.tokens.to_vec());
        let slots = Tensor::u32(shape.clone(), vec![0; self.tokens.len()]);
        [tokens, Tensor::u32(shape, self.positions()), slots]
    }
}

/// A whole prompt for the forward entry.
fn prompt(tokens: &[u32]) -> Span<'_> {
    Span { past: 0, tokens }
}

/// Each kernel's time over the least its work needs.
fn room(device: &Device, plan: &Plan, steps: &[f64]) {
    let Some(fit) = device.fit else {
        return eprintln!("  no chip was fitted; run tessera-calibrate for headroom");
    };
    let timed: Vec<_> = plan
        .steps
        .iter()
        .cloned()
        .zip(steps.iter().copied())
        .collect();
    let all: f64 = steps.iter().sum();
    for h in headroom(&fit.pool("fitted"), &timed) {
        let Some(bound) = h.bound else { continue };
        let (share, over) = (100.0 * h.seconds / all, h.seconds / bound);
        eprintln!(
            "  {:8} {share:4.1}% of the time, {over:.1}x its best case",
            h.kernel
        );
    }
}

impl<'a> Gpu<'a> {
    fn open(g: &Graph, c: &'a Fixture) -> Self {
        let first = prompt(&c.tokens);
        let plan = lower(g, 0, &first.sizes(), &pinned(&c.settings)).unwrap();
        let runner = loaded(plan, g, &c.weights, &c.settings);
        let device = Device::stored(runner.platform(), &c.settings).unwrap();
        let device = device.pinned(pinned(&c.settings).pins);
        let (name, samples) = (runner.device(), device.samples());
        eprintln!("{name}: {samples} stored samples");
        Self {
            c,
            runner,
            device,
            learned: 0,
            plans: Cache::default(),
        }
    }

    /// Times this launch under each competing rule, where unknown.
    fn learn(&mut self, g: &Graph, rows: Span) {
        let (sizes, tensors) = (rows.sizes(), rows.tensors());
        // The forward entry has no slots column.
        let ports = g.entries[0].inputs.len();
        let feeds: Vec<View> = tensors[..ports].iter().map(Tensor::view).collect();
        // What the stored samples alone would launch.
        let before = kernels(&lower(g, 0, &sizes, &self.device).unwrap());
        let device = &mut self.device;
        let settings = &self.c.settings;
        let added = measure(&mut *self.runner, (g, 0), &sizes, &feeds, device, settings).unwrap();
        self.learned += added.len();
        let after = kernels(&lower(g, 0, &sizes, &self.device).unwrap());
        if before != after {
            let (new, past) = (rows.tokens.len(), rows.past);
            eprintln!("  {new} rows after {past}: timing changed the plan");
            eprintln!("    from {before:?}\n    to   {after:?}");
        }
    }

    /// Lowers this launch for `device` and feeds it.
    fn plan(&mut self, g: &Graph, rows: Span, device: &Device) {
        let plan = self.plans.lower(g, 0, &rows.sizes(), device).unwrap();
        self.runner.replan(&plan).unwrap();
        let (tensors, ports) = (rows.tensors(), g.entries[0].inputs.len());
        let feeds: Vec<View> = tensors[..ports].iter().map(Tensor::view).collect();
        self.runner.feed(&feeds).unwrap();
    }

    fn save(&self) {
        let file = Device::path(&self.c.settings, &self.device.name).unwrap();
        let (learned, all) = (self.learned, self.device.samples());
        eprintln!("timed {learned} launches; {all} samples known");
        let (kept, hits) = (self.plans.len(), self.plans.hits);
        eprintln!("plans: {kept} lowered, {hits} reused");
        if learned > 0 {
            self.device.save(&file).unwrap();
            eprintln!("  stored in {}", file.display());
        }
    }
}

/// The forward entry as a plan, wholly on the GPU.
mod forward {
    use super::*;

    struct Timed {
        plan: Plan,
        out: Vec<f32>,
        /// Seconds per launch, by the device's own clock.
        steps: Vec<f64>,
        /// Seconds for the whole plan, logits read back.
        whole: f64,
    }

    fn repeat(c: &Fixture, n: usize) -> Vec<u32> {
        c.tokens.iter().cycle().take(n).copied().collect()
    }

    /// Runs `tokens` as planned for `device`, and times each launch.
    fn run(gpu: &mut Gpu, g: &Graph, tokens: &[u32], device: &Device) -> Timed {
        gpu.plan(g, prompt(tokens), device);
        let all = 0..gpu.runner.plan().steps.len();
        let out = gpu.runner.plan().outputs[0].slot;
        // The first run pays for anything set up lazily.
        gpu.runner.submit(all.clone()).unwrap();
        let start = Instant::now();
        gpu.runner.submit(all).unwrap();
        let out = gpu.runner.read(out).unwrap().floats().unwrap().to_vec();
        let whole = start.elapsed().as_secs_f64();
        Timed {
            plan: gpu.runner.plan().clone(),
            out,
            steps: seconds(&mut *gpu.runner),
            whole,
        }
    }

    /// Prints what the plan chose, its estimate and its time.
    fn report(r: &Timed, device: &Device) {
        let timed: f64 = r.steps.iter().sum();
        let cost = device.cost(&r.plan);
        let (n, whole) = (r.out.len() / 151_936, r.whole);
        eprintln!("plan, {n} tokens: {whole:.3} s with logits read back");
        let (all, estimate) = (cost.launches, cost.seconds);
        eprintln!("  {all} launches: estimated {estimate:.3} s, timed {timed:.3} s");
        let (moved, scratch) = (cost.bytes as f64 / 1e9, r.plan.scratch() as f64 * 4.0 / 1e6);
        eprintln!("  moves {moved:.2} GB; scratch {scratch:.1} MB");
        table(&r.plan, &r.steps);
        room(device, &r.plan, &r.steps);
        let ratio = estimate / timed;
        assert!(
            (0.5..2.0).contains(&ratio),
            "the estimate is off by {ratio:.2}x"
        );
    }

    /// The plan should not lose to favoring any one rule.
    fn compare(gpu: &mut Gpu, g: &Graph, planned: &Timed) {
        let timed = |r: &Timed| r.steps.iter().sum::<f64>();
        let tokens = repeat(gpu.c, planned.out.len() / 151_936);
        let sizes = prompt(&tokens).sizes();
        let contested = select(g, 0, &sizes, &pinned(&gpu.c.settings))
            .unwrap()
            .contested;
        let mut least = f64::INFINITY;
        for rule in contested {
            let device = gpu.device.clone().favoring(rule);
            let forced = timed(&run(gpu, g, &tokens, &device));
            eprintln!("  favoring {rule:8} {forced:.3} s");
            least = least.min(forced);
        }
        // A second timing, warm like the forced runs before it.
        let device = gpu.device.clone();
        let mixed = timed(&run(gpu, g, &tokens, &device));
        eprintln!("  as planned        {mixed:.3} s");
        // Metal's run-to-run spread at this size reaches a third.
        assert!(mixed < 1.5 * least, "the plan lost to a forced choice");
    }

    pub fn check(gpu: &mut Gpu, g: &Graph) {
        let c = gpu.c;
        let (longer, long) = (repeat(c, 64), repeat(c, 512));
        for tokens in [&c.tokens, &longer, &long] {
            gpu.learn(g, prompt(tokens));
        }
        let device = gpu.device.clone();
        let short = run(gpu, g, &c.tokens, &device);
        report(&short, &device);
        compare(gpu, g, &short);
        close("plan", &short.out, c);
        // Past the reference's length, the CPU is the check.
        let planned = run(gpu, g, &longer, &device);
        let off = max_abs_diff(&planned.out, &logits(g, c, &longer));
        eprintln!("plan vs interpreter, 64 tokens: max_abs {off:.3e}");
        assert!(off < 1e-3, "the plan and the interpreter disagree");
        // Causal, so a long run's first rows match the reference.
        let long = run(gpu, g, &long, &device);
        report(&long, &device);
        compare(gpu, g, &long);
        close("plan, start of 512", &long.out[..c.want.len()], c);
    }

    /// The logits of `tokens` as planned, for other checks.
    pub fn logits_of(gpu: &mut Gpu, g: &Graph, tokens: &[u32]) -> Vec<f32> {
        let device = gpu.device.clone();
        run(gpu, g, tokens, &device).out
    }
}

/// Greedy decoding: one plan per launch, rows kept on device.
mod decode {
    use super::*;

    /// Tokens decoded after the prompt.
    const NEW: usize = 32;

    /// A prompt and the tokens decoded after it.
    struct Decoded {
        tokens: Vec<u32>,
        /// The index chosen after each prompt token.
        prompt: Vec<u32>,
        /// Seconds running the prompt.
        prefill: f64,
        /// Seconds lowering and binding the later plans.
        planning: f64,
        /// Seconds feeding and running them, and reading indices.
        running: f64,
    }

    impl Decoded {
        fn rate(&self) -> f64 {
            (NEW - 1) as f64 / (self.planning + self.running)
        }
    }

    /// Runs one launch. Returns indices, then planning and running seconds.
    fn launch(gpu: &mut Gpu, g: &Graph, rows: Span, device: &Device) -> (Vec<u32>, [f64; 2]) {
        let start = Instant::now();
        // One plan per context range, its length bound at launch.
        let upto = [Context];
        let (plan, bound) = gpu.plans.upto(g, 0, &rows.sizes(), &upto, device).unwrap();
        let out = plan.outputs[0].slot;
        gpu.runner.replan(plan).unwrap();
        gpu.runner.bind_all(&bound);
        let planning = start.elapsed().as_secs_f64();
        let tensors = rows.tensors();
        let feeds: Vec<View> = tensors.iter().map(Tensor::view).collect();
        let start = Instant::now();
        gpu.runner.feed(&feeds).unwrap();
        gpu.runner.submit(0..gpu.runner.plan().steps.len()).unwrap();
        let ids = gpu.runner.ids(out).unwrap();
        (ids, [planning, start.elapsed().as_secs_f64()])
    }

    /// The prompt at once, then one row per launch.
    fn decode(gpu: &mut Gpu, g: &Graph, device: &Device) -> Decoded {
        let mut tokens = gpu.c.tokens.clone();
        let (prompt, [_, prefill]) = launch(gpu, g, super::prompt(&tokens), device);
        tokens.push(*prompt.last().unwrap());
        let (mut planning, mut running) = (0.0, 0.0);
        for _ in 1..NEW {
            let (past, last) = (tokens.len() - 1, [tokens[tokens.len() - 1]]);
            let rows = Span {
                past,
                tokens: &last,
            };
            let (ids, [planned, ran]) = launch(gpu, g, rows, device);
            (planning, running) = (planning + planned, running + ran);
            tokens.push(ids[0]);
        }
        Decoded {
            tokens,
            prompt,
            prefill,
            planning,
            running,
        }
    }

    /// `chosen` must be a row's best, or tie with it.
    fn likeliest(label: &str, row: &[f32], chosen: u32) {
        let lead = row[best(row)] - row[chosen as usize];
        assert!(lead < 1e-3, "{label}: {chosen} trails the best by {lead}");
    }

    pub fn check(gpu: &mut Gpu, forward: &Graph) {
        let text = stepper(&Config::small(1, 1024)).to_json();
        let g = Graph::from_json(&text).unwrap();
        assert_eq!(g.weights, forward.weights);
        let c = gpu.c;
        // A prompt, then single rows at a few lengths.
        gpu.learn(&g, prompt(&c.tokens));
        for past in [15, 16, 31, 46] {
            let tokens = &[0];
            gpu.learn(&g, Span { past, tokens });
        }
        let device = gpu.device.clone();
        // The first pass compiles what the plan newly picked.
        decode(gpu, &g, &device);
        let planned = decode(gpu, &g, &device);
        let (ms, rate) = (planned.prefill * 1e3, planned.rate());
        eprintln!("prefill, {} tokens: {ms:.1} ms", c.tokens.len());
        eprintln!("decode, {} tokens: {rate:.1} tokens/s", NEW - 1);
        let per = |seconds: f64| seconds * 1e3 / (NEW - 1) as f64;
        let (plan_ms, run_ms) = (per(planned.planning), per(planned.running));
        eprintln!("  per token: planning {plan_ms:.2} ms, device {run_ms:.2} ms");
        let last = gpu.runner.plan().clone();
        let (estimate, steps) = (device.cost(&last).seconds * 1e3, seconds(&mut *gpu.runner));
        let timed = steps.iter().sum::<f64>() * 1e3;
        eprintln!("  last token: estimated {estimate:.2} ms, timed {timed:.2} ms");
        table(&last, &steps);
        room(&device, &last, &steps);
        let vocab = c.want.len() / c.tokens.len();
        for (row, id) in c.want.chunks(vocab).zip(&planned.prompt) {
            likeliest("prefill", row, *id);
        }
        // One pass over everything must choose the same tokens.
        let tokens = &planned.tokens;
        eprintln!("  decoded {:?}", &tokens[c.tokens.len()..]);
        let whole = forward::logits_of(gpu, forward, &tokens[..tokens.len() - 1]);
        let rows = whole.chunks(vocab).zip(&tokens[1..]);
        for (row, id) in rows.skip(c.tokens.len() - 1) {
            likeliest("decode", row, *id);
        }
        // The plan should not lose to favoring any one rule.
        let one = Span {
            past: 15,
            tokens: &[0],
        };
        let pinned = pinned(&gpu.c.settings);
        let contested = select(&g, 0, &one.sizes(), &pinned).unwrap().contested;
        let mut most = 0.0f64;
        for rule in contested {
            let forced = decode(gpu, &g, &device.clone().favoring(rule)).rate();
            eprintln!("  favoring {rule:8} {forced:.1} tokens/s");
            most = most.max(forced);
        }
        assert!(1.25 * rate > most, "the plan lost to a forced choice");
    }
}
