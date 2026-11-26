//! Runs git and cargo in a work tree.

use std::path::Path;
use std::process::Command;

pub struct Output {
    pub ok: bool,
    pub stdout: String,
    pub stderr: String,
}

/// With these, git does not run work tree code.
const SAFE: [&str; 7] = [
    // `GIT_COMMITTER_NAME` replaces this default.
    "user.name=tessera-rebase",
    "user.email=tessera-rebase@users.noreply.github.com",
    "core.hooksPath=/dev/null",
    "core.editor=true",
    "rerere.enabled=false",
    "commit.gpgsign=false",
    "merge.conflictStyle=diff3",
];

pub fn git(dir: &Path, args: &[&str]) -> Result<Output, String> {
    let mut command = Command::new("git");
    command.arg("-C").arg(dir);
    for setting in SAFE {
        command.args(["-c", setting]);
    }
    let out = command
        .args(args)
        .output()
        .map_err(|e| format!("git: {e}"))?;
    Ok(Output {
        ok: out.status.success(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    })
}

/// The trimmed output of a git command that must pass.
pub fn must(dir: &Path, args: &[&str]) -> Result<String, String> {
    let out = git(dir, args)?;
    if out.ok {
        Ok(out.stdout.trim().to_string())
    } else {
        Err(format!("git {}: {}", args.join(" "), out.stderr.trim()))
    }
}

pub fn cargo(dir: &Path, args: &[&str]) -> Result<(), String> {
    let status = Command::new("cargo").current_dir(dir).args(args).status();
    match status {
        Ok(status) if status.success() => Ok(()),
        Ok(_) => Err(format!("cargo {} failed", args.join(" "))),
        Err(e) => Err(format!("cargo: {e}")),
    }
}
