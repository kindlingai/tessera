//! `tessera-serve --model <dir>`: OpenAI-shaped routes over one model.

use sensai_serve::ranks::FLAGS as RANKS;
use sensai_serve::start::{FLAGS, Options};
use std::process::ExitCode;

fn args() -> Option<Options> {
    let mut options = Options::default();
    let mut given = std::env::args().skip(1);
    while let Some(flag) = given.next() {
        if options.switch(&flag) {
            continue;
        }
        let value = given.next()?;
        match flag.as_str() {
            "--model" => options.model = value.into(),
            _ => options.flag(&flag, value)?,
        }
    }
    (!options.model.as_os_str().is_empty()).then_some(options)
}

fn main() -> ExitCode {
    let Some(options) = args() else {
        eprintln!("usage: tessera-serve --model <dir> {FLAGS} {RANKS}");
        return ExitCode::from(2);
    };
    match sensai_serve::start::run(options) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("tessera-serve: {e}");
            ExitCode::FAILURE
        }
    }
}
