//! Attention rules for the CPU and GPU. Linking registers them.

mod attend;
mod conv;
mod cpu;
mod delta;
mod flash;
mod listed;

use tessera_lower::kernels::Map;
use tessera_lower::{Draft, T, Tree, sites};

/// A listed site with all keys present, and its picks.
fn fixed_list(tree: &Tree, (site, ids): (sites::Attend, T)) -> Option<(sites::Attend, T, usize)> {
    if site.sizes.keys.runtime() {
        return None;
    }
    Some((site, ids, *tree.node(ids).shape.last()?))
}

/// GPU whole sequences: scaled queries and the output length.
fn scaled(site: &sites::Attend) -> Option<(Draft, usize)> {
    let a = &site.sizes;
    // Sequences would sit at moving offsets as the keys grow.
    if a.keys.runtime() && a.seqs > 1 {
        return None;
    }
    let out = a.seqs * site.out();
    let scale = Map::scaled(site.scale, site.q_form);
    Some((Draft::new(scale, [site.q], out), out))
}
