//! `tessera-serve --model <dir>`: OpenAI-shaped routes over one model.

use std::process::ExitCode;
use tessera_serve::ranks::FLAGS as RANKS;
use tessera_serve::start::{Options, flags};

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
        eprintln!("usage: tessera-serve --model <dir> {} {RANKS}", flags());
        return ExitCode::from(2);
    };
    match tessera_serve::start::run(options) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("tessera-serve: {e}");
            ExitCode::FAILURE
        }
    }
}
