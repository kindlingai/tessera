//! A model's graph, its lowering per rank, tokens moving.

// Linking these registers their lowering rules.
use tessera_rules as _;

mod dot;
mod html;
mod lowering;
mod model;
mod pump;
mod ranks;
mod stages;

#[cfg(test)]
mod pump_test;

use model::Model;
use ranks::{Spec, Wire, spec};
use stages::{Pass, Setup};
use std::error::Error;
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "tessera-inspect [--model NAME] [--full] [--graph FILE.json]
  [--ranks strix:3,gb10:1] [--prompt N] [--tokens N] [--form q8_0]
  [--latency SECONDS] [--rate BYTES_PER_SECOND] [--out DIR]
  A rank is a preset (strix, gb10, m3, cpu) or BACKEND/ARCH, then :PART.
  BACKEND is cuda, rocm, metal or cpu. No device is opened.";

struct Args {
    model: String,
    full: bool,
    graph: Option<String>,
    specs: Vec<Spec>,
    prompt: u64,
    tokens: u64,
    form: Option<String>,
    wire: Wire,
    out: PathBuf,
}

fn parse(mut given: impl Iterator<Item = String>) -> Option<Args> {
    let mut args = Args {
        model: "qwen3".into(),
        full: false,
        graph: None,
        specs: ["strix:3", "gb10:1"]
            .iter()
            .map(|s| spec(s))
            .collect::<Option<_>>()?,
        prompt: 4,
        tokens: 3,
        form: None,
        wire: Wire {
            latency: 5e-5,
            rate: 1.25e9,
        },
        out: "inspect".into(),
    };
    while let Some(flag) = given.next() {
        if flag == "--full" {
            args.full = true;
            continue;
        }
        let value = given.next()?;
        match flag.as_str() {
            "--model" => args.model = value,
            "--graph" => args.graph = Some(value),
            "--ranks" => args.specs = value.split(',').map(spec).collect::<Option<_>>()?,
            "--prompt" => args.prompt = value.parse().ok().filter(|n| *n > 0)?,
            "--tokens" => args.tokens = value.parse().ok()?,
            "--form" => args.form = Some(value),
            "--latency" => args.wire.latency = value.parse().ok()?,
            "--rate" => args.wire.rate = value.parse().ok()?,
            "--out" => args.out = value.into(),
            _ => return None,
        }
    }
    Some(args)
}

fn main() -> ExitCode {
    let Some(args) = parse(std::env::args().skip(1)) else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    if let Err(e) = run(&args) {
        eprintln!("tessera-inspect: {e}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

/// The prompt's pass, then one pass per new token.
fn passes(setup: &Setup, prompt: u64, tokens: u64) -> Result<Vec<Pass>, Box<dyn Error>> {
    let mut out = vec![setup.pass("prompt".into(), prompt, prompt)?];
    for t in 0..tokens {
        let context = prompt + t + 1;
        let rows = match setup.model.step_entry() {
            true => 1,
            false => context,
        };
        out.push(setup.pass(format!("token {}", t + 1), rows, context)?);
    }
    Ok(out)
}

fn run(args: &Args) -> Result<(), Box<dyn Error>> {
    let (name, source) = match &args.graph {
        Some(path) => (path.clone(), model::read(path)?),
        None => {
            let g = model::named(&args.model, args.full)?;
            (args.model.clone(), g)
        }
    };
    let parts = tessera_graph::Parts(args.specs.iter().map(|s| s.part).collect());
    let model = Model::new(name, source, &parts)?;
    for w in &model.whole {
        let path = &model.source.weights[w.weight as usize].path;
        eprintln!("{path:?} stays whole: {}", w.why);
    }
    let pins = match &args.form {
        Some(form) => tessera_lower::Pins::all(
            tessera_lower::form::named(form).ok_or_else(|| format!("no form `{form}`"))?,
        ),
        None => tessera_lower::Pins::default(),
    };
    let setup = Setup {
        model: &model,
        specs: &args.specs,
        wire: args.wire,
        pins,
    };
    let passes = passes(&setup, args.prompt, args.tokens)?;
    let laps = pump::laps(&passes)?;
    let names: Vec<String> = args.specs.iter().map(|s| s.name.clone()).collect();
    let out = &args.out;
    std::fs::create_dir_all(out)?;
    std::fs::write(out.join("graph.json"), model.source.to_json())?;
    std::fs::write(out.join("graph.dot"), dot::draw(&model, false))?;
    std::fs::write(out.join("sharded.dot"), dot::draw(&model, true))?;
    std::fs::write(
        out.join("lowering.html"),
        lowering::page(&model, &passes[0]),
    )?;
    if let Some(token) = passes.get(1) {
        std::fs::write(
            out.join("lowering-token.html"),
            lowering::page(&model, token),
        )?;
    }
    std::fs::write(
        out.join("pump.html"),
        pump::page(&model.name, &names, &laps),
    )?;
    print!("{}", pump::terminal(&laps, &names));
    eprintln!("wrote {}", out.display());
    Ok(())
}
