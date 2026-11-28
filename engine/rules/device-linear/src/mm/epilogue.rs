//! Epilogue matching shared by the matmul rules.

use super::*;

pub(crate) trait MmDims {
    fn dims(&self) -> (usize, usize);

    /// Sets a bf16 output and epilogue input, or refuses them.
    fn bf16_io(&mut self, y_bf16: bool, o_bf16: bool) -> bool {
        !y_bf16 && !o_bf16
    }
}

impl MmDims for MmSite {
    fn dims(&self) -> (usize, usize) {
        (self.m, self.n)
    }

    fn bf16_io(&mut self, y_bf16: bool, o_bf16: bool) -> bool {
        self.io[1] = y_bf16;
        self.io[2] = o_bf16;
        true
    }
}

fn is_plain_io(form: &dyn Stored) -> bool {
    matches!(form.name(), "f32" | "bf16")
}

/// Matches `x @ w + add`. `bias_bits` flags bf16 biases.
pub(crate) fn match_mm_add<S: MmDims>(
    tree: &Tree,
    at: T,
    pick: fn(&Tree, T) -> Option<S>,
    bias_bits: bool,
) -> Option<(S, u32, Option<T>)> {
    let node = tree.node(at);
    if node.term != Term::Stage(Stage::Add) || !is_plain_io(node.form) {
        return None;
    }
    let y_bf16 = node.form.name() == "bf16";
    let (a, b) = (node.args[0], *node.args.get(1)?);
    for (mat, add) in [(a, b), (b, a)] {
        if mat == add || tree.root(mat) || !is_plain_io(tree.node(add).form) {
            continue;
        }
        let Some(mut site) = pick(tree, mat) else {
            continue;
        };
        let (m, n) = site.dims();
        let add_node = tree.node(add);
        let add_elems = tree.elements(add);
        let o_bf16 = add_node.form.name() == "bf16";
        let (mode, o_bf16) = if add_elems == m * n && add_node.shape == node.shape {
            (1, o_bf16)
        } else if add_elems == n && add_node.shape.last() == Some(&n) {
            match bias_bits && o_bf16 {
                true => (2 | 16, false),
                false => (2, o_bf16),
            }
        } else {
            continue;
        };
        if site.bf16_io(y_bf16, o_bf16) {
            return Some((site, mode, Some(add)));
        }
    }
    None
}

fn match_mm_mul<S: MmDims>(
    tree: &Tree,
    at: T,
    pick: fn(&Tree, T) -> Option<S>,
) -> Option<(S, u32, Option<T>)> {
    let node = tree.node(at);
    if node.term != Term::Stage(Stage::Mul) || !is_plain_io(node.form) {
        return None;
    }
    let y_bf16 = node.form.name() == "bf16";
    let (a, b) = (node.args[0], *node.args.get(1)?);
    let silu = Term::Stage(Stage::Act(Act::Silu));
    for (cand, other) in [(a, b), (b, a)] {
        if cand == other || tree.root(cand) || !is_plain_io(tree.node(other).form) {
            continue;
        }
        let cnode = tree.node(cand);
        if cnode.term != silu || !is_plain_io(cnode.form) {
            continue;
        }
        let mat = cnode.args[0];
        if mat == other || tree.root(mat) {
            continue;
        }
        if let Some(site) = mul_site(tree, &node.shape, mat, other, y_bf16, pick) {
            return Some((site, 3, Some(other)));
        }
    }
    for (mat, other) in [(a, b), (b, a)] {
        if mat == other || tree.root(mat) || !is_plain_io(tree.node(other).form) {
            continue;
        }
        if let Some(site) = mul_site(tree, &node.shape, mat, other, y_bf16, pick) {
            return Some((site, 4, Some(other)));
        }
    }
    None
}

fn mul_site<S: MmDims>(
    tree: &Tree,
    shape: &[usize],
    mat: T,
    other: T,
    y_bf16: bool,
    pick: fn(&Tree, T) -> Option<S>,
) -> Option<S> {
    let mut site = pick(tree, mat)?;
    let (m, n) = site.dims();
    let o_bf16 = tree.node(other).form.name() == "bf16";
    let fits = tree.elements(other) == m * n && tree.node(other).shape[..] == *shape;
    (fits && site.bf16_io(y_bf16, o_bf16)).then_some(site)
}

pub(crate) fn match_mm_site<S: MmDims>(
    tree: &Tree,
    at: T,
    pick: fn(&Tree, T) -> Option<S>,
) -> Option<(S, u32, Option<T>)> {
    if let Some(hit) = match_mm_add(tree, at, pick, false) {
        return Some(hit);
    }
    if let Some(hit) = match_mm_mul(tree, at, pick) {
        return Some(hit);
    }
    let node = tree.node(at);
    if node.term == Term::Stage(Stage::Act(Act::Silu)) && is_plain_io(node.form) {
        let mat = node.args[0];
        if !tree.root(mat)
            && let Some(mut site) = pick(tree, mat)
            && site.bf16_io(node.form.name() == "bf16", false)
        {
            return Some((site, 5, None));
        }
    }
    Some((pick(tree, at)?, 0, None))
}
