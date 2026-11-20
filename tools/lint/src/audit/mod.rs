//! Rules that read the workspace's crates and what they name.

#[cfg(test)]
mod audit_test;
mod deps;
mod layers;
#[cfg(test)]
mod layers_test;
mod repeats;
#[cfg(test)]
mod repeats_test;
mod suite;
mod unused;
#[cfg(test)]
mod unused_test;
mod workspace;

use std::path::Path;

use sensai_findings::Level;

use crate::files::Source;
use crate::{LIMITS, Violation};

pub use deps::Thin;
pub use layers::{Allow, Facade};
pub use repeats::Exprs;

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
    layers::facade(&pkgs, &LIMITS.facade, out);
    layers::placement(&pkgs, &LIMITS.layers, LIMITS.placement_gap, out);
    deps::check(&pkgs, sources, &LIMITS.layers[0], &LIMITS.thin_edge, out);
    unused::check(dirs, sources, out);
    suite::check(&pkgs, sources, out);
    repeats::check(sources, out);
    repeats::exprs(sources, &LIMITS.repeat_exprs, out);
    Ok(())
}
