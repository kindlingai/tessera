//! `tessera-rebase`: rebases when conflicts only add list lines.

mod git;
mod hunk;
mod lists;
mod replay;
mod seal;

#[cfg(test)]
mod lists_test;

use std::path::Path;
use std::process::ExitCode;

use replay::{Outcome, Plan, replay};

const USAGE: &str = "usage: tessera-rebase plan <base> <head> <work>
       tessera-rebase build <base> <head> <work> <out> [cargo feature flags]
       tessera-rebase seal <base> <head> <work> <from>
plan replays <head> on <base> in the new work tree <work>.
It joins the conflicts where both sides only add list lines.
build also formats and builds, and copies the changed files to <out>.
seal plans again and commits the files of a build in <from>.";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let (step, base, head, work, rest) = match args.as_slice() {
        [step, base, head, work, rest @ ..] => (*step, base, head, Path::new(work), rest),
        _ => return usage(),
    };
    let dir = match (step, rest) {
        ("plan", []) => None,
        ("build", [out, ..]) => Some(Path::new(out)),
        ("seal", [from]) => Some(Path::new(from)),
        _ => return usage(),
    };
    let outcome = replay(base, head, work).and_then(|outcome| match (outcome, dir) {
        (Outcome::Done(plan), Some(out)) if step == "build" => {
            seal::build(work, &plan, out, &rest[1..])?;
            Ok(Outcome::Done(plan))
        }
        (Outcome::Done(plan), Some(from)) => {
            let tip = seal::seal(work, &plan, from)?;
            Ok(Outcome::Done(Plan { tip, ..plan }))
        }
        (outcome, _) => Ok(outcome),
    });
    // The commits stay in the repository without the work tree.
    let dir = work.to_string_lossy();
    let removed = git::git(Path::new("."), &["worktree", "remove", "--force", &dir]);
    match outcome {
        Ok(Outcome::Done(plan)) => {
            println!("tip {}", plan.tip);
            for path in &plan.paths {
                println!("path {path}");
            }
            ExitCode::SUCCESS
        }
        Ok(Outcome::Refused(why)) => {
            println!("refused {why}");
            ExitCode::from(2)
        }
        Err(e) => {
            eprintln!("tessera-rebase: {e}");
            if let Err(e) = removed {
                eprintln!("tessera-rebase: {e}");
            }
            ExitCode::FAILURE
        }
    }
}

fn usage() -> ExitCode {
    eprintln!("{USAGE}");
    ExitCode::FAILURE
}
