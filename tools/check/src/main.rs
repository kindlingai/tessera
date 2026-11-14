//! Runs the style checks and renders one report.

mod markdown;
#[cfg(test)]
mod markdown_test;
mod run;
mod terminal;

use findings::Report;
use run::{Changes, Dirs};
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

const USAGE: &str =
    "usage: tessera-check run [--staged | <base> <head>] [--markdown <file> [<link_base>]]
       tessera-check render <json>... [--markdown <file> [<link_base>]]
       tessera-check fetch

`run` checks lint over the repository, and style and simcheck when their
models are in target/check. It fails on any failure. `render` reads the
JSON each check writes. `fetch` downloads both models.";

fn root() -> Result<PathBuf, String> {
    let out = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output();
    let out = out.map_err(|e| format!("git: {e}"))?;
    Ok(PathBuf::from(String::from_utf8_lossy(&out.stdout).trim()))
}

/// Prints the reports, and writes Markdown when asked.
fn show(reports: &[Report], markdown: &[&str]) -> Result<(), String> {
    print!(
        "{}",
        terminal::render(reports, std::io::stdout().is_terminal())
    );
    if let [file, rest @ ..] = markdown {
        let md = markdown::render(reports, rest.first().copied()).map_err(|e| e.to_string())?;
        std::fs::write(file, md).map_err(|e| format!("{file}: {e}"))?;
    }
    Ok(())
}

/// Splits off a trailing `--markdown <file> [<link_base>]`.
fn split_markdown<'a>(args: &'a [&'a str]) -> (&'a [&'a str], &'a [&'a str]) {
    match args.iter().position(|a| *a == "--markdown") {
        Some(i) => (&args[..i], &args[i + 1..]),
        None => (args, &[]),
    }
}

fn run(args: &[&str]) -> Result<bool, String> {
    let (cmd, rest) = args.split_first().ok_or(USAGE)?;
    let (rest, markdown) = split_markdown(rest);
    match (*cmd, rest) {
        ("fetch", []) => run::fetch(&Dirs::new(&root()?)).map(|()| true),
        ("render", files) if !files.is_empty() => {
            let reports: Result<Vec<Report>, String> =
                files.iter().map(|f| Report::read(Path::new(f))).collect();
            show(&reports?, markdown).map(|()| true)
        }
        ("run", which) => {
            let changes = match which {
                ["--staged"] | [] => Changes::Staged,
                [base, head] => Changes::Range(base.to_string(), head.to_string()),
                _ => return Err(USAGE.into()),
            };
            let dirs = Dirs::new(&root()?);
            std::fs::create_dir_all(&dirs.out)
                .map_err(|e| format!("{}: {e}", dirs.out.display()))?;
            let reports = vec![
                run::lint(&dirs)?,
                run::style(&dirs, &changes)?,
                run::simcheck(&dirs, &changes)?,
            ];
            show(&reports, markdown)?;
            Ok(reports.iter().all(|r| r.fails() == 0))
        }
        _ => Err(USAGE.into()),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    match run(&args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("tessera-check: {e}");
            ExitCode::from(2)
        }
    }
}
