//! Qwen3-0.6B: traced, loaded from its checkpoint, and run.

use checkpoint::File;
use interp::{Cpu, Data, Tensor, bind, run};
use models::qwen3::{Config, program, stepper, tensor_name};
use ops::testing::{cosine_f32, gpu_available, max_abs_diff};
use std::path::PathBuf;
use std::time::Instant;
use trace::{Graph, PathSeg};

/// The interpreter only ever sees the JSON text.
fn graph() -> Graph {
    Graph::from_json(&program(&Config::small(8, 40_960)).to_json()).unwrap()
}

#[test]
fn traces() {
    let g = graph();
    graph::verify(&g).unwrap();
    assert_eq!(g.regions.len(), 2 * 28);
    let count = |w: &graph::Weight| w.ty.shape.elements(&g.dims).unwrap().as_const().unwrap();
    let parameters: u64 = g.weights.iter().map(count).sum();
    assert_eq!(parameters, 596_049_920);
}

/// Read from the directory `TESSERA_QWEN3` names.
struct Case {
    weights: Vec<Tensor>,
    tokens: Vec<u32>,
    want: Vec<f32>,
}

fn case(g: &Graph) -> Option<Case> {
    let dir = PathBuf::from(std::env::var_os("TESSERA_QWEN3")?);
    let file = File::open(&dir.join("model.safetensors")).unwrap();
    let find = |p: &[PathSeg]| Some(Data::F32(file.floats(&tensor_name(p)?).ok()?));
    let tokens = std::fs::read_to_string(dir.join("tokens.txt")).unwrap();
    let floats = std::fs::read(dir.join("logits.f32")).unwrap();
    let wide = |b: &[u8]| f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    Some(Case {
        weights: bind(g, find).unwrap(),
        tokens: tokens
            .split_whitespace()
            .map(|t| t.parse().unwrap())
            .collect(),
        want: floats.chunks_exact(4).map(wide).collect(),
    })
}

/// One sequence of `tokens` through the CPU interpreter.
fn logits(g: &Graph, c: &Case, tokens: &[u32]) -> Vec<f32> {
    let shape = vec![1, tokens.len()];
    let positions = (0..tokens.len() as u32).collect();
    let ids = |data| Tensor {
        shape: shape.clone(),
        data: Data::U32(data),
    };
    let inputs = [ids(tokens.to_vec()), ids(positions)];
    let start = Instant::now();
    let out = run(g, &mut Cpu, &c.weights, 0, &inputs).unwrap();
    let seconds = start.elapsed().as_secs_f32();
    eprintln!("interpreter, {} tokens: {seconds:.3} s", tokens.len());
    out[0].floats().unwrap().to_vec()
}

fn best(row: &[f32]) -> usize {
    let top = |a: &(usize, &f32), b: &(usize, &f32)| a.1.total_cmp(b.1);
    row.iter().enumerate().max_by(top).unwrap().0
}

/// Each position must pick the reference's top token.
fn close(label: &str, got: &[f32], c: &Case) {
    let vocab = got.len() / c.tokens.len();
    let (max, mut cos) = (max_abs_diff(got, &c.want), 1.0f32);
    for (g, w) in got.chunks(vocab).zip(c.want.chunks(vocab)) {
        cos = cos.min(cosine_f32(g, w));
        assert_eq!(best(g), best(w), "{label}: top token differs");
    }
    eprintln!("{label}: max_abs {max:.3e} worst cos {cos:.7}");
    assert!(max < 1e-3 && cos > 0.99999, "{label}: {max:.3e} {cos}");
}

#[test]
fn matches_reference() {
    let g = graph();
    let Some(c) = case(&g) else {
        return eprintln!("TESSERA_QWEN3 is not set; skipped");
    };
    close("cpu", &logits(&g, &c, &c.tokens), &c);
    if !gpu_available() {
        return eprintln!("no GPU here; skipped");
    }
    let samples = plan::check(&g, &c);
    decode::check(&g, &c, samples);
}

/// The same model as a plan, wholly on the GPU.
mod plan {
    use super::*;
    use lower::kernels::{Matmul, Thin};
    use lower::{Device, Feed, Kernel, Plan, assign, lower, open, select};
    use std::collections::BTreeMap;
    use trace::DimName::{Context, Seqs};

    /// Timed launches, for fitting a device.
    pub type Samples = Vec<(lower::Step, f64)>;

    pub struct Run {
        plan: Plan,
        pub out: Vec<f32>,
        /// Seconds per launch, waited on one by one.
        steps: Vec<f64>,
        /// Seconds for every launch in one command buffer.
        whole: f64,
        /// Scratch elements had the ops kept their traced order.
        traced: usize,
    }

    fn repeat(c: &Case, n: usize) -> Vec<u32> {
        c.tokens.iter().cycle().take(n).copied().collect()
    }

    /// Lowers for `tokens` on `device`, runs, and times each launch.
    pub fn run(g: &Graph, c: &Case, tokens: &[u32], device: &Device) -> Run {
        let sizes = [(Seqs, 1), (Context, tokens.len() as u64)];
        let picked = select(g, 0, &sizes, device).unwrap();
        let in_order: Vec<usize> = (0..picked.ops.len()).collect();
        let traced = assign(&picked, &in_order).scratch();
        let plan = lower(g, 0, &sizes, device).unwrap();
        let runner = open(plan.clone()).unwrap();
        for (at, w) in c.weights.iter().enumerate() {
            runner.load(at as u32, w.floats().unwrap()).unwrap();
        }
        let positions: Vec<u32> = (0..tokens.len() as u32).collect();
        let feeds = [Feed::Ids(tokens), Feed::Ids(&positions)];
        // The first run pays for anything Metal sets up lazily.
        runner.run(&feeds).unwrap();
        let start = Instant::now();
        let out = runner.run(&feeds).unwrap().remove(0);
        let whole = start.elapsed().as_secs_f64();
        let steps = runner.run_timed(&feeds).unwrap();
        Run {
            plan,
            out,
            steps,
            whole,
            traced,
        }
    }

    /// Rules that competed somewhere when lowering `tokens`.
    fn contested(g: &Graph, tokens: usize, device: &Device) -> Vec<&'static str> {
        let sizes = [(Seqs, 1), (Context, tokens as u64)];
        let picked = select(g, 0, &sizes, device).unwrap();
        picked.contested.into_iter().collect()
    }

    /// Times every kernel: cost only chooses what was measured.
    fn calibrate(g: &Graph, c: &Case) -> (Device, Samples) {
        let mut runs = Vec::new();
        for n in [8, 32, 128, 384] {
            // Favoring each contested rule measures every candidate.
            for rule in contested(g, n, &Device::guess()) {
                let device = Device::guess().favoring(rule);
                runs.push(run(g, c, &repeat(c, n), &device));
            }
        }
        let timed = |r: &'_ Run| r.plan.steps.clone().into_iter().zip(r.steps.clone());
        let samples: Samples = runs.iter().flat_map(timed).collect();
        let mut device = Device::guess();
        device.fit(samples.iter().map(|(step, seconds)| (step, *seconds)));
        // The head matmul, as estimated for each kernel.
        let (m, n, k) = (c.tokens.len(), 151_936, 1024);
        let ms = |kernel: &dyn Kernel| device.seconds(kernel, m * n) * 1e3;
        let (tiled, thin) = (ms(&Matmul { m, n, k }), ms(&Thin { m, n, k }));
        eprintln!("  head matmul, {m} rows: tiled {tiled:.1} ms, thin {thin:.1} ms");
        (device, samples)
    }

    /// Prints what the plan chose, its estimate and its time.
    fn report(r: &Run, device: &Device) {
        let waited: f64 = r.steps.iter().sum();
        let cost = device.cost(&r.plan);
        let (n, whole) = (r.out.len() / 151_936, r.whole);
        eprintln!("plan, {n} tokens: {whole:.3} s in one command buffer");
        let (all, estimate) = (cost.launches, cost.seconds);
        eprintln!("  {all} launches: estimated {estimate:.3} s, measured {waited:.3} s");
        let mb = |elements: usize| elements as f64 * 4.0 / 1e6;
        let (moved, scratch, traced) =
            (cost.bytes as f64 / 1e9, mb(r.plan.scratch()), mb(r.traced));
        eprintln!("  moves {moved:.2} GB; scratch {scratch:.1} MB, {traced:.1} MB in traced order");
        table(&r.plan, &r.steps);
        let ratio = estimate / waited;
        assert!(
            (0.33..3.0).contains(&ratio),
            "the estimate is off by {ratio:.2}x"
        );
    }

    /// Launches and seconds, by kernel.
    pub fn table(plan: &Plan, steps: &[f64]) {
        let mut by: BTreeMap<&str, (usize, f64)> = BTreeMap::new();
        for (step, seconds) in plan.steps.iter().zip(steps) {
            let (count, total) = by.entry(step.kernel.name()).or_default();
            (*count, *total) = (*count + step.launches().len(), *total + seconds);
        }
        for (name, (count, seconds)) in by {
            eprintln!("  {name:8} {count:4} launches {seconds:.4} s");
        }
    }

    /// The plan should not lose to favoring any one rule.
    fn compare(g: &Graph, c: &Case, planned: &Run, device: &Device) {
        let waited = |r: &Run| r.steps.iter().sum::<f64>();
        let tokens = repeat(c, planned.out.len() / 151_936);
        let (mixed, mut least) = (waited(planned), f64::INFINITY);
        for rule in contested(g, tokens.len(), device) {
            let forced = waited(&run(g, c, &tokens, &device.clone().favoring(rule)));
            eprintln!("  favoring {rule:8} {forced:.3} s");
            least = least.min(forced);
        }
        eprintln!("  as planned        {mixed:.3} s");
        assert!(mixed < 1.25 * least, "the plan lost to a forced choice");
    }

    /// Returns its timed launches, for plans of other shapes.
    pub fn check(g: &Graph, c: &Case) -> Samples {
        eprintln!("calibrating:");
        let (device, samples) = calibrate(g, c);
        let short = run(g, c, &c.tokens, &device);
        report(&short, &device);
        compare(g, c, &short, &device);
        close("plan", &short.out, c);
        // Past the reference's length, the CPU is the check.
        let longer = repeat(c, 64);
        let planned = run(g, c, &longer, &device);
        let off = max_abs_diff(&planned.out, &logits(g, c, &longer));
        eprintln!("plan vs interpreter, 64 tokens: max_abs {off:.3e}");
        assert!(off < 1e-3, "the plan and the interpreter disagree");
        // Causal, so a long run's first rows match the reference.
        let long = run(g, c, &repeat(c, 512), &device);
        report(&long, &device);
        compare(g, c, &long, &device);
        close("plan, start of 512", &long.out[..c.want.len()], c);
        samples
    }
}

/// Greedy decoding: one plan per launch, rows kept on device.
mod decode {
    use super::plan::Samples;
    use super::*;
    use lower::{Device, Feed, Run, lower, open, select};
    use trace::DimName::{Context, Rows, Seqs};

    /// Tokens decoded after the prompt.
    const NEW: usize = 32;

    struct Decoder<'a> {
        g: &'a Graph,
        c: &'a Case,
        runner: Option<Box<dyn Run>>,
        /// The last launch's tokens and positions.
        fed: [Vec<u32>; 2],
    }

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

    fn sizes(past: usize, rows: usize) -> [(trace::DimName, u64); 3] {
        let all = (past + rows) as u64;
        [(Seqs, 1), (Rows, rows as u64), (Context, all)]
    }

    impl Decoder<'_> {
        /// Runs `rows` after `past` kept rows. Returns indices, seconds.
        fn launch(&mut self, past: usize, rows: &[u32], device: &Device) -> (Vec<u32>, [f64; 2]) {
            let start = Instant::now();
            let plan = lower(self.g, 0, &sizes(past, rows.len()), device).unwrap();
            let out = plan.outputs[0].slot;
            let first = self.runner.is_none();
            match &mut self.runner {
                Some(r) => r.replan(plan).unwrap(),
                None => self.runner = Some(open(plan).unwrap()),
            }
            let planning = start.elapsed().as_secs_f64();
            let r = self.runner.as_ref().unwrap();
            // Later plans find the weights already there.
            for (at, w) in self.c.weights.iter().enumerate().filter(|_| first) {
                r.load(at as u32, w.floats().unwrap()).unwrap();
            }
            let positions = (past as u32..(past + rows.len()) as u32).collect();
            self.fed = [rows.to_vec(), positions];
            let start = Instant::now();
            let feeds = [Feed::Ids(&self.fed[0]), Feed::Ids(&self.fed[1])];
            r.feed(&feeds).unwrap();
            r.submit(0..r.plan().steps.len()).unwrap();
            let ids = r.ids(out).unwrap();
            (ids, [planning, start.elapsed().as_secs_f64()])
        }

        /// Runs the last launch again, waiting on every step.
        fn timed(&self) -> Samples {
            let r = self.runner.as_ref().unwrap();
            let feeds = [Feed::Ids(&self.fed[0]), Feed::Ids(&self.fed[1])];
            let seconds = r.run_timed(&feeds).unwrap();
            r.plan().steps.iter().cloned().zip(seconds).collect()
        }

        /// The prompt at once, then one row per launch.
        fn decode(&mut self, device: &Device) -> Decoded {
            let mut tokens = self.c.tokens.clone();
            let (prompt, [_, prefill]) = self.launch(0, &tokens, device);
            tokens.push(*prompt.last().unwrap());
            let (mut planning, mut running) = (0.0, 0.0);
            for _ in 1..NEW {
                let (past, last) = (tokens.len() - 1, [tokens[tokens.len() - 1]]);
                let (ids, [planned, ran]) = self.launch(past, &last, device);
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

        /// Rules that compete when one row is decoded.
        fn contested(&self) -> Vec<&'static str> {
            let picked = select(self.g, 0, &sizes(15, 1), &Device::guess()).unwrap();
            picked.contested.into_iter().collect()
        }

        /// Adds a prompt and single rows under each contested rule.
        fn calibrate(&mut self, mut samples: Samples) -> Device {
            for rule in self.contested() {
                let device = Device::guess().favoring(rule);
                let prompt = self.c.tokens.clone();
                self.launch(0, &prompt, &device);
                samples.extend(self.timed());
                for past in prompt.len()..prompt.len() + 3 {
                    self.launch(past, &[0], &device);
                    samples.extend(self.timed());
                }
            }
            let mut device = Device::guess();
            device.fit(samples.iter().map(|(step, seconds)| (step, *seconds)));
            device
        }
    }

    /// `chosen` must be a row's best, or tie with it.
    fn likeliest(label: &str, row: &[f32], chosen: u32) {
        let lead = row[best(row)] - row[chosen as usize];
        assert!(lead < 1e-3, "{label}: {chosen} trails the best by {lead}");
    }

    pub fn check(forward: &Graph, c: &Case, samples: Samples) {
        let text = stepper(&Config::small(1, 1024)).to_json();
        let g = Graph::from_json(&text).unwrap();
        assert_eq!(g.weights, forward.weights);
        let (runner, fed) = (None, [Vec::new(), Vec::new()]);
        let mut d = Decoder {
            g: &g,
            c,
            runner,
            fed,
        };
        let device = d.calibrate(samples);
        let planned = d.decode(&device);
        let (ms, rate) = (planned.prefill * 1e3, planned.rate());
        eprintln!("prefill, {} tokens: {ms:.1} ms", c.tokens.len());
        eprintln!("decode, {} tokens: {rate:.1} tokens/s", NEW - 1);
        let per = |seconds: f64| seconds * 1e3 / (NEW - 1) as f64;
        let (plan_ms, run_ms) = (per(planned.planning), per(planned.running));
        eprintln!("  per token: planning {plan_ms:.2} ms, device {run_ms:.2} ms");
        let (steps, seconds): (Vec<_>, Vec<f64>) = d.timed().into_iter().unzip();
        let last = d.runner.as_ref().unwrap().plan();
        assert_eq!(last.steps, steps);
        plan::table(last, &seconds);
        let vocab = c.want.len() / c.tokens.len();
        for (row, id) in c.want.chunks(vocab).zip(&planned.prompt) {
            likeliest("prefill", row, *id);
        }
        // One pass over everything must choose the same tokens.
        let tokens = &planned.tokens;
        eprintln!("  decoded {:?}", &tokens[c.tokens.len()..]);
        let whole = plan::run(forward, c, &tokens[..tokens.len() - 1], &device).out;
        let rows = whole.chunks(vocab).zip(&tokens[1..]);
        for (row, id) in rows.skip(c.tokens.len() - 1) {
            likeliest("decode", row, *id);
        }
        // The plan should not lose to favoring any one rule.
        let mut most = 0.0f64;
        for rule in d.contested() {
            let forced = d.decode(&device.clone().favoring(rule)).rate();
            eprintln!("  favoring {rule:8} {forced:.1} tokens/s");
            most = most.max(forced);
        }
        assert!(1.25 * rate > most, "the plan lost to a forced choice");
    }
}
