//! `tesserax [--recipe <src> | --model <dir | hf:<owner>/<name>[@<revision>]>] <verb>`.

mod fetch;
mod serve;
mod source;
#[cfg(test)]
mod source_test;
mod task;

use source::{Source, Stop, split};
use std::process::ExitCode;
use tessera_serve::ranks::FLAGS as RANKS;
use tessera_serve::start::flags;

const USAGE: &str = "usage: tesserax [--recipe <dir | git+<url>#<commit>> | --model <dir | hf:<owner>/<name>[@<revision>]>] <verb>
  serve [flags]  load the model and answer OpenAI-shaped requests
  fetch          download the recipe and its weights, then print where they are
  task <task>    run one task on the model";

fn run(given: &[String]) -> Result<(), Stop> {
    let (origin, args) = split(given)?;
    let Some((verb, rest)) = args.split_first() else {
        return Err(Stop::Usage);
    };
    match verb.as_str() {
        "serve" => serve::run(Source::open_or_default(origin)?, rest),
        "fetch" => fetch::run(Source::open_or_default(origin)?, rest),
        "task" => task::run(origin, rest),
        _ => Err(Stop::Usage),
    }
}

fn main() -> ExitCode {
    let given: Vec<String> = std::env::args().skip(1).collect();
    match run(&given) {
        Ok(()) => ExitCode::SUCCESS,
        Err(Stop::Usage) => {
            let default = source::DEFAULT.0;
            eprintln!("{USAGE}\nserve, fetch, ask and chat default to {default}");
            eprintln!("serve flags: {} {RANKS}", flags());
            ExitCode::from(2)
        }
        Err(Stop::Tasks(unknown)) => {
            if let Some(name) = unknown {
                eprintln!("tesserax: unknown task {name}");
            }
            eprintln!("{}", task::list());
            ExitCode::from(2)
        }
        Err(Stop::Failed(e)) => {
            eprintln!("tesserax: {e}");
            ExitCode::FAILURE
        }
    }
}
