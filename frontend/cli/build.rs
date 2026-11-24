//! Records the git commit for `tesserax --version`.

use std::path::Path;
use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    let text = String::from_utf8(out.stdout).ok()?.trim().to_string();
    (out.status.success() && !text.is_empty()).then_some(text)
}

fn main() {
    // Rebuild when HEAD or the branch it names moves.
    let mut names = vec!["HEAD".to_string()];
    names.extend(git(&["symbolic-ref", "-q", "HEAD"]));
    for name in names {
        if let Some(path) = git(&["rev-parse", "--git-path", &name])
            && Path::new(&path).exists()
        {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    if let Some(commit) = git(&["rev-parse", "HEAD"]) {
        println!("cargo:rustc-env=TESSERA_GIT_COMMIT={commit}");
    }
}
