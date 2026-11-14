//! `tessera-eval --model <path> [--recipe <json>] [--m 1,4]`: JSON out.

use checkpoint::exlr8::Release;
use checkpoint::{File, Gguf, Offer};
use engine::{Engine, Row};
use lower::{Pins, Recipe};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

/// Anything that stops a run.
type Failure = Box<dyn std::error::Error + Send + Sync>;

const USAGE: &str = "usage: tessera-eval --model <path> [--recipe <json>] [--m <list>] [--new <n>] \
[--prompt <text>] [--reference <dir>] [--context <rows>] [--slices <json>]";

/// A checkpoint's slices, as a manifest for the model tests.
fn manifest(model: &Path) -> Result<String, Failure> {
    let offer: Box<dyn Offer> = if model.is_file() {
        Box::new(Gguf::open(model)?)
    } else if model.join("exlr8-vocab.safetensors").is_file() {
        Box::new(Release::open(model))
    } else {
        Box::new(File::open(&model.join("model.safetensors"))?)
    };
    // One slice a line, so a manifest diffs by tensor.
    let mut lines = Vec::new();
    for s in offer.slices() {
        let entry = serde_json::json!(
            {"name": s.name, "shape": s.shape, "stored": s.stored.name()});
        lines.push(format!("  {}", serde_json::to_string(&entry)?));
    }
    Ok(format!("[\n{}\n]\n", lines.join(",\n")))
}

struct Args {
    model: PathBuf,
    recipe: Option<PathBuf>,
    /// Sequences decoded side by side, one run per value.
    m: Vec<usize>,
    /// Tokens decoded after the prompt.
    new: usize,
    prompt: String,
    /// A directory with `tokens.txt` and `logits.f32` to check against.
    reference: Option<PathBuf>,
    context: usize,
    /// Write the file's slices as JSON, then stop.
    slices: Option<PathBuf>,
}

fn args() -> Option<Args> {
    let mut args = Args {
        model: PathBuf::new(),
        recipe: None,
        m: vec![1],
        new: 32,
        prompt: "The capital of France is".into(),
        reference: None,
        context: 4096,
        slices: None,
    };
    let mut given = std::env::args().skip(1);
    while let Some(flag) = given.next() {
        let value = given.next()?;
        match flag.as_str() {
            "--model" => args.model = value.into(),
            "--recipe" => args.recipe = Some(value.into()),
            "--m" => {
                args.m = value
                    .split(',')
                    .map(|v| v.parse())
                    .collect::<Result<_, _>>()
                    .ok()?
            }
            "--new" => args.new = value.parse().ok()?,
            "--prompt" => args.prompt = value,
            "--reference" => args.reference = Some(value.into()),
            "--context" => args.context = value.parse().ok()?,
            "--slices" => args.slices = Some(value.into()),
            _ => return None,
        }
    }
    (!args.model.as_os_str().is_empty()).then_some(args)
}

/// One batch size's measurements.
#[derive(Serialize)]
struct Run {
    m: usize,
    prefill_ms: f64,
    decode_tokens_per_s: f64,
    per_token_ms: f64,
    kernels: BTreeMap<&'static str, usize>,
}

/// How many prefill positions picked the reference's top token.
#[derive(Serialize)]
struct Parity {
    positions: usize,
    matched: usize,
}

#[derive(Serialize)]
struct Report {
    device: String,
    recipe: Recipe,
    prompt_tokens: usize,
    runs: Vec<Run>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parity: Option<Parity>,
}

/// `prompt` in `m` slots, then `new` rows each, timed twice.
fn run(engine: &mut Engine, prompt: &[u32], m: usize, new: usize) -> Result<Run, lower::Error> {
    let row = |slot: usize, token: u32, position: usize| Row {
        slot: slot as u32,
        token,
        position: position as u32,
    };
    let mut report = None;
    // The first pass compiles what the plans newly pick.
    for _ in 0..2 {
        let rows: Vec<Row> = (0..m)
            .flat_map(|s| prompt.iter().enumerate().map(move |(p, t)| row(s, *t, p)))
            .collect();
        let start = Instant::now();
        let mut last = engine.step(&rows)?;
        let prefill = start.elapsed().as_secs_f64();
        // Each slot's last prompt row chose its first token.
        last = (0..m).map(|s| last[(s + 1) * prompt.len() - 1]).collect();
        let start = Instant::now();
        for i in 0..new {
            let rows: Vec<Row> = (0..m).map(|s| row(s, last[s], prompt.len() + i)).collect();
            last = engine.step(&rows)?;
        }
        let decode = start.elapsed().as_secs_f64();
        report = Some(Run {
            m,
            prefill_ms: prefill * 1e3,
            decode_tokens_per_s: (m * new) as f64 / decode,
            per_token_ms: decode * 1e3 / new as f64,
            kernels: engine.kernels(),
        });
    }
    Ok(report.expect("two passes"))
}

/// The reference prompt, and each position's top token.
fn reference(dir: &Path) -> Result<(Vec<u32>, Vec<u32>), Failure> {
    let tokens: Vec<u32> = std::fs::read_to_string(dir.join("tokens.txt"))?
        .split_whitespace()
        .map(|t| t.parse())
        .collect::<Result<_, _>>()?;
    let bytes = std::fs::read(dir.join("logits.f32"))?;
    let wide = |b: &[u8; 4]| f32::from_le_bytes(*b);
    let logits: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(wide).collect();
    let vocab = logits.len() / tokens.len();
    let best = |row: &[f32]| {
        let top = |a: &(usize, &f32), b: &(usize, &f32)| a.1.total_cmp(b.1);
        row.iter()
            .enumerate()
            .max_by(top)
            .map_or(0, |(i, _)| i as u32)
    };
    Ok((tokens, logits.chunks(vocab).map(best).collect()))
}

fn evaluate(args: &Args) -> Result<Report, Failure> {
    let slots = args.m.iter().copied().max().unwrap_or(1) as u64;
    let settings = settings::read();
    let rules = match &settings.names {
        Some(path) => bundle::names::rules(&std::fs::read_to_string(path)?)?,
        None => Vec::new(),
    };
    let loaded = bundle::load_named(&args.model, slots, args.context as u64, &rules)?;
    let recipe = match &args.recipe {
        Some(path) => Recipe::load(path)?,
        None => Recipe {
            pins: loaded
                .pins
                .unwrap_or_else(|| Pins::all(lower::form::BF16_FORM)),
            ..Recipe::default()
        },
    };
    let mut engine = Engine::open(loaded.graph, args.context, &recipe, &settings)?;
    engine.load(&loaded.weights)?;
    let (prompt, wanted) = match &args.reference {
        Some(dir) => {
            let (tokens, best) = reference(dir)?;
            (tokens, Some(best))
        }
        None => (loaded.tokenizer.encode(args.prompt.as_str())?, None),
    };
    let parity = match wanted {
        Some(best) => {
            let rows: Vec<Row> = prompt
                .iter()
                .enumerate()
                .map(|(p, t)| Row {
                    slot: 0,
                    token: *t,
                    position: p as u32,
                })
                .collect();
            let got = engine.step(&rows)?;
            let matched = got.iter().zip(&best).filter(|(g, w)| g == w).count();
            Some(Parity {
                positions: prompt.len(),
                matched,
            })
        }
        None => None,
    };
    let mut runs = Vec::new();
    for m in &args.m {
        runs.push(run(&mut engine, &prompt, *m, args.new)?);
    }
    Ok(Report {
        device: engine.device(),
        recipe,
        prompt_tokens: prompt.len(),
        runs,
        parity,
    })
}

fn main() -> ExitCode {
    let Some(args) = args() else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    if let Some(out) = &args.slices {
        return match manifest(&args.model).and_then(|text| Ok(std::fs::write(out, text)?)) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("tessera-eval: {e}");
                ExitCode::FAILURE
            }
        };
    }
    match evaluate(&args) {
        Ok(report) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&report).expect("plain data")
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("tessera-eval: {e}");
            ExitCode::FAILURE
        }
    }
}
