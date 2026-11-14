//! Rules that read the workspace's crates and what they name.

mod layers;
#[cfg(test)]
mod layers_test;
mod workspace;

use std::path::Path;

use crate::{LIMITS, Violation};

pub use layers::Allow;

pub fn check(root: &Path, out: &mut Vec<Violation>) -> Result<(), String> {
    let pkgs = workspace::load(root)?;
    layers::check(&pkgs, &LIMITS.layers, &LIMITS.layer_allow, out);
    Ok(())
}
