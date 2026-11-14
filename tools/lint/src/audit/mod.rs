//! Rules that read the workspace's crates and what they name.

#[cfg(test)]
mod audit_test;
mod layers;
#[cfg(test)]
mod layers_test;
mod repeats;
#[cfg(test)]
mod repeats_test;
mod unused;
#[cfg(test)]
mod unused_test;
mod workspace;

use std::path::Path;

use findings::Level;

use crate::files::Source;
use crate::{LIMITS, Violation};

pub use layers::Allow;

/// Rules in `warn_rules` only advise.
pub fn level(rule: &str, warn: &[String]) -> Level {
    if warn.iter().any(|w| w == rule) {
        Level::Note
    } else {
        Level::Fail
    }
}

pub fn check(
    root: &Path,
    dirs: &[String],
    sources: &[Source],
    out: &mut Vec<Violation>,
) -> Result<(), String> {
    let pkgs = workspace::load(root)?;
    layers::check(&pkgs, &LIMITS.layers, &LIMITS.layer_allow, out);
    unused::check(dirs, sources, out);
    repeats::check(sources, out);
    Ok(())
}
