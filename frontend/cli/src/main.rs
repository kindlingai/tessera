//! `sensai (--recipe <src> | --model <dir | hf:<owner>/<name>[@<revision>]>) <verb>`.

mod fetch;
mod serve;
mod source;
#[cfg(test)]
mod source_test;
mod task;

use sensai_serve::ranks::FLAGS as RANKS;
use sensai_serve::start::FLAGS;
use source::{Source, Stop, split};
use std::process::ExitCode;

const USAGE: &str = "usage: sensai (--recipe <dir | git+<url>#<commit>> | --model <dir>) <verb>
  serve [flags]  load the model and answer OpenAI-shaped requests
  fetch          download the recipe and its weights, then print where they are
  task <task>    run one task on the model";

fn run(given: &[String]) -> Result<(), Stop> {
    let (origin, args) = split(given)?;
    let Some((verb, rest)) = args.split_first() else {
        return Err(Stop::Usage);
    };
    match verb.as_str() {
        "serve" => serve::run(Source::open(origin)?, rest),
        "fetch" => fetch::run(Source::open(origin)?, rest),
        "task" => task::run(origin, rest),
        _ => Err(Stop::Usage),
    }
}

fn main() -> ExitCode {
    let given: Vec<String> = std::env::args().skip(1).collect();
    match run(&given) {
        Ok(()) => ExitCode::SUCCESS,
        Err(Stop::Usage) => {
            eprintln!("{USAGE}\nserve flags: {FLAGS} {RANKS}");
            ExitCode::from(2)
        }
        Err(Stop::Tasks(unknown)) => {
            if let Some(name) = unknown {
                eprintln!("sensai: unknown task {name}");
            }
            eprintln!("{}", task::LIST);
            ExitCode::from(2)
        }
        Err(Stop::Failed(e)) => {
            eprintln!("sensai: {e}");
            ExitCode::FAILURE
        }
    }
}
