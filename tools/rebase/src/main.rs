//! `tessera-rebase`: rebases when conflicts only add list lines.

mod code;
mod git;
mod hunk;
mod lists;
mod names;
mod replay;
mod seal;

#[cfg(test)]
mod lists_test;

use std::path::Path;
use std::process::ExitCode;

use replay::{Outcome, Plan, replay};

const USAGE: &str = "usage: tessera-rebase plan <base> <head> <work>
       tessera-rebase open <base> <head> <work>
       tessera-rebase seal <base> <head> <work> <from>
plan replays <head> on <base> in the new work tree <work>.
It joins the conflicts where both sides only add list entries.
open does the same and keeps <work> for a build.
seal plans again and commits the files of a build in <from>.";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let (step, base, head, work, from) = match args.as_slice() {
        [step @ ("plan" | "open"), base, head, work] => (*step, base, head, work, None),
        ["seal", base, head, work, from] => ("seal", base, head, work, Some(Path::new(from))),
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::FAILURE;
        }
    };
    let work = Path::new(work);
    let outcome = replay(base, head, work).and_then(|outcome| match (outcome, from) {
        (Outcome::Done(plan), Some(from)) => {
            let tip = seal::seal(work, &plan, from)?;
            Ok(Outcome::Done(Plan { tip, ..plan }))
        }
        (outcome, _) => Ok(outcome),
    });
    // The commits stay in the repository without the work tree.
    if step != "open" || !matches!(outcome, Ok(Outcome::Done(_))) {
        let dir = work.to_string_lossy();
        if let Err(e) = git::git(Path::new("."), &["worktree", "remove", "--force", &dir]) {
            eprintln!("tessera-rebase: {e}");
        }
    }
    match outcome {
        Ok(Outcome::Done(plan)) => {
            println!("tip {}\ntree {}", plan.tip, plan.tree);
            for path in &plan.paths {
                println!("path {path}");
            }
            for (name, version) in &plan.pins {
                println!("pin {name} {version}");
            }
            ExitCode::SUCCESS
        }
        Ok(Outcome::Refused(why)) => {
            println!("refused {why}");
            ExitCode::from(2)
        }
        Err(e) => {
            eprintln!("tessera-rebase: {e}");
            ExitCode::FAILURE
        }
    }
}
