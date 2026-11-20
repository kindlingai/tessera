//! Writes the merged crates.io crates to `target/publish` or `[dir]`.

use std::process::ExitCode;

fn main() -> ExitCode {
    match tessera_publish::run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("tessera-publish: {e}");
            ExitCode::FAILURE
        }
    }
}
