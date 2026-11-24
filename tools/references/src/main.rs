//! `tessera-references [dir]`: the files the parity tests read.

mod fetch;

use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = match args.as_slice() {
        [] => tessera_testkit::references(),
        [dir] => Some(PathBuf::from(dir)),
        _ => {
            eprintln!("usage: tessera-references [dir]");
            return ExitCode::FAILURE;
        }
    };
    let Some(dir) = dir else {
        eprintln!("tessera-references: no home directory: name a dir");
        return ExitCode::FAILURE;
    };
    match fetch::fetch(&dir) {
        Ok(()) => {
            eprintln!("tessera-references: done in {}", dir.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("tessera-references: {e}");
            ExitCode::FAILURE
        }
    }
}
