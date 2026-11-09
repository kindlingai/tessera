//! Qwen3-0.6B: traced, loaded from its checkpoint, and run.

use checkpoint::File;
use interp::{Cpu, Data, Tensor, bind, run};
use models::qwen3::{Config, program, tensor_name};
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
    plan::check(&g, &c);
}

/// The same model as a plan, wholly on the GPU.
mod plan {
    use super::*;
    use lower::{Device, Feed, Kernel, Plan, assign, launches, lower, open, select};
    use std::collections::BTreeMap;
    use trace::DimName::{Context, Seqs};

    struct Run {
        plan: Plan,
        out: Vec<f32>,
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
    fn run(g: &Graph, c: &Case, tokens: &[u32], device: &Device) -> Run {
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

    /// Times every kernel: cost only chooses what was measured.
    fn calibrate(g: &Graph, c: &Case) -> Device {
        let tiled = || Device::guess().without("thin");
        let stream = tiled().without("scores");
        let thin = Device::guess().without("matmul");
        let dense = tiled().without("attend");
        let long = vec![8, 32, 128, 384];
        let cases = [
            (&stream, long.clone()),
            (&thin, vec![8, 16, 32]),
            (&dense, long),
        ];
        let mut runs = Vec::new();
        for (device, lengths) in cases {
            for n in lengths {
                runs.push(run(g, c, &repeat(c, n), device));
            }
        }
        let timed = |r: &'_ Run| r.plan.steps.clone().into_iter().zip(r.steps.clone());
        let samples: Vec<_> = runs.iter().flat_map(timed).collect();
        let mut device = Device::guess();
        device.fit(samples.iter().map(|(step, seconds)| (step, *seconds)));
        // The head matmul, as estimated for each kernel.
        let (m, n, k) = (c.tokens.len(), 151_936, 1024);
        let ms = |kernel| device.seconds(&kernel, m * n) * 1e3;
        let (tiled, thin) = (ms(Kernel::Matmul { m, n, k }), ms(Kernel::Thin { m, n, k }));
        eprintln!("  head matmul, {m} rows: tiled {tiled:.1} ms, thin {thin:.1} ms");
        device
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
        let mut by: BTreeMap<&str, (usize, f64)> = BTreeMap::new();
        for (step, seconds) in r.plan.steps.iter().zip(&r.steps) {
            let (count, total) = by.entry(step.kernel.name()).or_default();
            (*count, *total) = (*count + launches(step).len(), *total + seconds);
        }
        for (name, (count, seconds)) in by {
            eprintln!("  {name:8} {count:4} launches {seconds:.3} s");
        }
        let ratio = estimate / waited;
        assert!(
            (0.33..3.0).contains(&ratio),
            "the estimate is off by {ratio:.2}x"
        );
    }

    /// The plan should not lose to a forced choice.
    fn compare(g: &Graph, c: &Case, planned: &Run, device: &Device, without: [&'static str; 2]) {
        let waited = |r: &Run| r.steps.iter().sum::<f64>();
        let tokens = repeat(c, planned.out.len() / 151_936);
        let lacking = |kernel| waited(&run(g, c, &tokens, &device.clone().without(kernel)));
        let (a, b, mixed) = (lacking(without[0]), lacking(without[1]), waited(planned));
        let [x, y] = without;
        eprintln!("  without {x} {a:.3} s, without {y} {b:.3} s, planned {mixed:.3} s");
        assert!(mixed < 1.25 * a.min(b), "the plan lost to a forced choice");
    }

    pub fn check(g: &Graph, c: &Case) {
        eprintln!("calibrating:");
        let device = calibrate(g, c);
        let short = run(g, c, &c.tokens, &device);
        report(&short, &device);
        compare(g, c, &short, &device, ["thin", "matmul"]);
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
        compare(g, c, &long, &device, ["scores", "attend"]);
        close("plan, start of 512", &long.out[..c.want.len()], c);
    }
}
