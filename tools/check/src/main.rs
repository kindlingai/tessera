//! Runs the style checks and renders one report.

mod markdown;
#[cfg(test)]
mod markdown_test;
mod run;
mod terminal;
mod tests;
#[cfg(test)]
mod tests_test;

use run::{Changes, Dirs};
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use tessera_findings::Report;

const USAGE: &str =
    "usage: tessera-check run [--staged | <base> <head>] [--markdown <file> [<link_base>]]
       tessera-check render <json>... [--markdown <file> [<link_base>]]
       tessera-check tests <junit_xml> <json>
       tessera-check message <commit_msg_file>
       tessera-check fetch

`run` checks lint's comment rules on the changed files, style and simcheck when
their models are in ~/.cache/tessera/check, and lint over the repository. It runs
the fastest checks first. With `--staged` it stops at the first failure, and a
range runs every check. `message` checks a commit message for style. `render`
reads the JSON each check writes. `tests` writes failed and slow tests from nextest's
JUnit file as such JSON. `fetch` downloads both models.";

fn root() -> Result<PathBuf, String> {
    let out = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output();
    let out = out.map_err(|e| format!("git: {e}"))?;
    Ok(PathBuf::from(String::from_utf8_lossy(&out.stdout).trim()))
}

/// Models in the shared check folder, reports in this checkout.
fn dirs() -> Result<Dirs, String> {
    Ok(Dirs::new(
        &root()?,
        tessera_frontend_settings::read().check.as_deref(),
    ))
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

/// Runs each check. Staged runs stop at the first failure.
fn gate(changes: &Changes, checks: &[run::Check], markdown: &[&str]) -> Result<bool, String> {
    let dirs = dirs()?;
    std::fs::create_dir_all(&dirs.out).map_err(|e| format!("{}: {e}", dirs.out.display()))?;
    let early = matches!(changes, Changes::Staged);
    let mut reports = Vec::new();
    for (i, check) in checks.iter().enumerate() {
        let report = check(&dirs, changes)?;
        let failed = report.fails() > 0;
        reports.push(report);
        if early && failed && i + 1 < checks.len() {
            eprintln!("tessera-check: stopped at the first failure; later checks did not run");
            break;
        }
    }
    show(&reports, markdown)?;
    Ok(reports.iter().all(|r| r.fails() == 0))
}

fn run(args: &[&str]) -> Result<bool, String> {
    let (cmd, rest) = args.split_first().ok_or(USAGE)?;
    let (rest, markdown) = split_markdown(rest);
    match (*cmd, rest) {
        ("tests", [xml, json]) => {
            let text = std::fs::read_to_string(xml).map_err(|e| format!("{xml}: {e}"))?;
            tests::report(&text)?.write(Path::new(json)).map(|()| true)
        }
        ("fetch", []) => run::fetch(&dirs()?).map(|()| true),
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
            gate(
                &changes,
                &[run::comments, run::style, run::lint, run::simcheck],
                markdown,
            )
        }
        ("message", [file]) => gate(&Changes::Message(file.to_string()), &[run::style], markdown),
        _ => Err(USAGE.into()),
    }
}

fn main() -> ExitCode {
    tessera_findings::main_of("tessera-check", run)
}
