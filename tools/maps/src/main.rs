//! Checks and queries the task maps.

use std::path::PathBuf;
use std::process::ExitCode;

use tessera_maps::Atlas;

const USAGE: &str =
    "usage: tessera-maps [check | show <task> | cites <path> | registries | digest]";

fn main() -> ExitCode {
    tessera_findings::main_of("tessera-maps", run)
}

fn run(args: &[&str]) -> Result<bool, String> {
    let root = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."));
    let atlas = Atlas::open(&root)?;
    let out = match args {
        [] | ["check"] => {
            let faults = atlas.check();
            for f in &faults {
                println!("{}:{}: map: {}", f.path, f.line, f.message);
            }
            return Ok(faults.is_empty());
        }
        ["show", id] => atlas.show(id)?,
        ["cites", path] => atlas.cites(path),
        ["registries"] => atlas.inventory(),
        ["digest"] => atlas.digest(),
        _ => return Err(USAGE.into()),
    };
    print!("{out}");
    Ok(true)
}
