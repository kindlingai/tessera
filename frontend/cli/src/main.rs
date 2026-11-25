//! `tesserax [--recipe <src> | --model <dir | hf:<owner>/<name>[@<revision>]>] <verb>`.

mod fetch;
mod serve;
mod source;
#[cfg(test)]
mod source_test;
mod task;
mod teleport;
#[cfg(test)]
mod teleport_test;
mod version;

use source::{Stop, split};
use std::process::ExitCode;
use tessera_serve::ranks::FLAGS as RANKS;
use tessera_serve::start::flags;

const USAGE: &str = "usage: tesserax [--recipe <dir | git+<url>#<commit>> | --model <dir | hf:<owner>/<name>[@<revision>]>] <verb>
  [--yes] [--max-tokens <n>] <prompt>...
                 answer one prompt. With no model flag, the default model may
                 pick a command for it instead, and asks first on a terminal
  <command>      run one of the commands below
  task <task>    run one of the tasks below on the model
  teleport [--from <dir>] [--root <dir>] <user@host | local>...
                 copy this build to other machines over ssh
  --version      print version, commit, target and GPU features";

fn run(given: &[String]) -> Result<(), Stop> {
    if given == ["--version"] {
        println!("{}", version::line());
        return Ok(());
    }
    let (origin, args) = split(given)?;
    let Some((verb, rest)) = args.split_first() else {
        return Err(Stop::Usage);
    };
    match (verb.as_str(), task::find("", verb)) {
        // Words a command refuses may still be a prompt.
        (_, Some(command)) if origin.is_some() || (command.accepts)(rest) => {
            (command.call)(origin, rest, &mut task::Print::default())
        }
        ("task", _) => task::run(origin, rest),
        ("teleport", _) => teleport::run(rest),
        ("-h" | "--help" | "help", _) => Err(Stop::Usage),
        // Other words are a prompt for the default model.
        _ if origin.is_none() => task::route(&args),
        _ => task::run(origin, &[&["ask".to_string()], args.as_slice()].concat()),
    }
}

fn main() -> ExitCode {
    let given: Vec<String> = std::env::args().skip(1).collect();
    match run(&given) {
        Ok(()) => ExitCode::SUCCESS,
        Err(Stop::Usage) => {
            let default = source::default_model();
            eprintln!("{USAGE}\nwith neither flag, the model is {default}");
            eprintln!("{}", task::list());
            eprintln!("serve flags: {} {RANKS}", flags());
            ExitCode::from(2)
        }
        Err(Stop::Tasks(message)) => {
            if let Some(message) = message {
                eprintln!("tesserax: {message}");
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
